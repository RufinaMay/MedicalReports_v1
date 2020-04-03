from torch import nn
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from sklearn.metrics import hamming_loss, roc_curve, roc_auc_score

from utils.utils import read_and_resize
from utils.constants import IMG_DIR, BATCH_SIZE
from models.mlc.attention_LSTM import f1_score, eval, adjust_learning_rate


class WeightedBCE(nn.Module):
    def __init__(self):
        super(WeightedBCE, self).__init__()

    def forward(self, inputs, targets):
        n = targets.shape[0]
        pos_sum, neg_sum = 0, 0
        P, N = 0, 0

        for i in range(targets.shape[0]):
            for j in range(targets.shape[1]):
                if targets[i, j] == 1:
                    P += 1
                    pos_sum -= torch.log(inputs[i, j])
                else:
                    N += 1
                    neg_sum -= torch.log(1 - inputs[i, j])

        beta_p = (P + N) / P
        beta_n = (P + N) / N
        return (beta_p * pos_sum + beta_n * neg_sum) / n


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class CNNModel(nn.Module):
    def __init__(self, encoder_name, encoder_dim, UNIQUE_TAGS, encoded_image_size=14):
        super(CNNModel, self).__init__()
        self.enc_image_size = encoded_image_size
        self.encoder_dim = encoder_dim

        if encoder_name == 'densenet':
            densenet = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
            self.cnn_encoder = densenet.features
        if encoder_name == 'shufflenet':
            shufflenet = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            self.cnn_encoder = nn.Sequential(*list(shufflenet.children())[:-2])
        if encoder_name == 'vgg':
            vgg = torchvision.models.vgg16(pretrained=True)
            self.cnn_encoder = vgg.features
        if encoder_name == 'resnet':
            resnet = torchvision.models.resnet101(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.cnn_encoder = nn.Sequential(*modules)
        if encoder_name == 'googlenet':
            googlenet = torchvision.models.googlenet(pretrained=True)
            self.cnn_encoder = nn.Sequential(*list(googlenet.children())[:-3])
        if encoder_name == 'mnasnet':  # encoder_dim=1280
            mnasnet = torchvision.models.mnasnet1_0(pretrained=True)
            self.cnn_encoder = mnasnet.layers

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fc = nn.Linear(encoder_dim * encoded_image_size * encoded_image_size, UNIQUE_TAGS)
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.cnn_encoder(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.view(-1, self.encoder_dim * self.enc_image_size * self.enc_image_size)
        out = torch.sigmoid(self.fc(out))
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.cnn_encoder.parameters():
            p.requires_grad = True
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.cnn_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


def batch(img_tag_mapping, tag_to_index, UNIQUE_TAGS, include_negatives=False):
    batch_IMGS, batch_TAGS = [], []
    b = 0
    for im_path in img_tag_mapping:
        im = read_and_resize(f'{IMG_DIR}/{im_path}.png')
        one_hot_tags = np.zeros(UNIQUE_TAGS)
        for tag in img_tag_mapping[im_path]:
            if tag in tag_to_index:
                one_hot_tags[tag_to_index[tag]] = 1

        include = True
        if not include_negatives and len(np.unique(one_hot_tags)) == 1:
            include = False

        if include:
            batch_IMGS.append(im), batch_TAGS.append(one_hot_tags)
        b += 1
        if b >= BATCH_SIZE:
            yield torch.stack(batch_IMGS), np.array(batch_TAGS)
            b = 0
            batch_IMGS, batch_TAGS = [], []
    if len(batch_IMGS) != 0:
        yield torch.stack(batch_IMGS), np.array(batch_TAGS)


def process_predictions(train_pred, y_train, UNIQUE_TAGS):
    true_overall, predicted_overall = [], []
    y_train = y_train.cpu().data.numpy()
    train_pred = train_pred.cpu().data.numpy()
    for true, predicted in zip(y_train, train_pred):
        p_idxs = np.argsort(predicted)[-10:]  # np.where(predicted >= 0.5)[0] # np.argsort(predicted)[:n]
        true_overall.append(true)
        vect = np.zeros(UNIQUE_TAGS)
        for p in p_idxs:
            vect[p] = 1
        predicted_overall.append(vect)
    return predicted_overall, true_overall


def train_step(x, y, model, device, optimizer, UNIQUE_TAGS, loss_name, training=True):
    if training:
        model.train()
    else:
        model.eval()
    x, y = x.to(device), torch.from_numpy(y).float().to(device)
    output = model(x)
    if loss_name == 'focal':
        loss = FocalLoss().forward(output, y)
    if loss_name == 'bce':
        criterion = nn.BCELoss()
        loss = criterion(output, y)
    if loss_name == 'weighted':
        loss = WeightedBCE().forward(output, y)
    if training:
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predicted, true = process_predictions(output, y, UNIQUE_TAGS)
    predicted_scores = output
    return loss.item(), predicted, true, predicted_scores.cpu().data.numpy(), model


def train_epoch(e, train_set, valid_set, test_set, model, tag_to_index, UNIQUE_TAGS, train_metrics, valid_metrics,
                device, optimizer, loss_name, include_negatives, verbose=True):
    T_loss, V_loss = [], []
    T_predicted, T_true, T_pred_scores, V_predicted, V_true, V_pred_scores = [], [], [], [], [], []
    # train step
    for imgs, caps in batch(train_set, tag_to_index, UNIQUE_TAGS, include_negatives):
        train_out = train_step(imgs, caps, model, device, optimizer, UNIQUE_TAGS, loss_name, training=True)
        model = train_out[4]
        T_loss.append(train_out[0])
        for pred, true, pred_scores in zip(train_out[1], train_out[2], train_out[3]):
            T_predicted.append(pred), T_true.append(true), T_pred_scores.append(pred_scores)

    pre, rec, ovpre, ovrec = eval(T_predicted, T_true)
    macroF1, microF1, instanceF1 = f1_score(T_predicted, T_true)
    ham_loss = hamming_loss(np.array(T_true), np.array(T_predicted))
    auc = roc_auc_score(T_true, T_pred_scores)
    train_metrics.append((np.mean(T_loss), pre, rec, ovpre, ovrec, macroF1, microF1, instanceF1, ham_loss, auc))
    if verbose:
        ro = round
        print(f'============================= epoch {e} =========================================')
        print(
            f'Tr: l {np.mean(T_loss)} pre {pre} rec {rec} overpre {ovpre} overrec {ovrec}'
            f'macroF1 {macroF1} microF1 {microF1} instanceF1 {instanceF1} ham loss {ham_loss} auc {auc}')

    # valid step
    for imgs, caps in batch(valid_set, tag_to_index, UNIQUE_TAGS, include_negatives=False):
        val_out = train_step(imgs, caps, model, device, optimizer, UNIQUE_TAGS, loss_name, training=False)
        V_loss.append(val_out[0])
        for pred, true, pred_scores in zip(val_out[1], val_out[2], val_out[3]):
            V_predicted.append(pred), V_true.append(true), V_pred_scores.append(pred_scores)

    pre, rec, ovpre, ovrec = eval(V_predicted, V_true)
    macroF1, microF1, instanceF1 = f1_score(V_predicted, V_true)
    ham_loss = hamming_loss(np.array(V_true), np.array(V_predicted))
    auc = roc_auc_score(V_true, V_pred_scores)
    valid_metrics.append((np.mean(V_loss), pre, rec, ovpre, ovrec, macroF1, microF1, instanceF1, ham_loss, auc))
    if verbose:
        print(
            f'Va: l {np.mean(V_loss)} pre {pre} rec {rec} overpre {ovpre} '
            f'overrec {ovrec} macroF1 {macroF1} microF1 {microF1} instanceF1 {instanceF1} '
            f'ham loss {ham_loss} auc {auc} ')

    # test set performance
    Test_predicted, Test_true, Test_pred_scores = [], [], []
    for imgs, caps in batch(test_set, tag_to_index, UNIQUE_TAGS, include_negatives=False):
        test_out = train_step(imgs, caps, model, device, optimizer, UNIQUE_TAGS, loss_name, training=False)
        for pred, true, pred_scores in zip(test_out[1], test_out[2], test_out[3]):
            Test_predicted.append(pred), Test_true.append(true), Test_pred_scores.append(pred_scores)
    pre, rec, ovpre, ovrec = eval(Test_predicted, Test_true)
    macroF1, microF1, instanceF1 = f1_score(Test_predicted, Test_true)
    ham_loss = hamming_loss(np.array(Test_true), np.array(Test_predicted))
    auc = roc_auc_score(Test_true, Test_pred_scores)
    test_metrics = [pre, rec, ovpre, ovrec, macroF1, microF1, instanceF1, ham_loss, auc]

    return np.mean(V_loss), train_metrics, valid_metrics, test_metrics, model


def train(start_epoch, end_epoch, train_set, valid_set, test_set, tag_to_index, UNIQUE_TAGS, model, optimizer,
          loss_name, device, include_negatives=True, verbose=True):
    epochs_since_improvement = 0
    train_metrics, valid_metrics, test_metrics = [], [], []
    best_loss = 1000
    for epoch in range(start_epoch, end_epoch):
        recent_loss, train_metrics, valid_metrics, test_metrics, model = train_epoch(
            epoch, train_set, valid_set, test_set, model, tag_to_index, UNIQUE_TAGS, train_metrics, valid_metrics,
            device, optimizer, loss_name, include_negatives=include_negatives, verbose=verbose)
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        if recent_loss < best_loss:
            best_loss = recent_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
    train_metrics = np.array(train_metrics)
    valid_metrics = np.array(valid_metrics)

    return train_metrics, valid_metrics, test_metrics, model
