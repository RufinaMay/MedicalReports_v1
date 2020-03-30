import torch
import torch.optim
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms
import cv2
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, precision_score, recall_score, classification_report, \
    roc_auc_score
from sklearn.metrics import hamming_loss
import torchvision
import os
import pickle
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import skimage.transform

from utils.constants import alpha_c, IMG_DIR, BATCH_SIZE


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14, encoder_name='densenet'):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

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

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.cnn_encoder(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.cnn_encoder.parameters():
            p.requires_grad = True


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, device):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def process_predictions(train_pred, y_train, tag_to_index, UNIQUE_TAGS):
    predicted_overall, true_overall, predicted_scores_overall = [], [], []
    for prediction in train_pred.cpu().data.numpy():
        # print(f'raw prediction {prediction}')
        predicted_tags = np.zeros(UNIQUE_TAGS - 3)
        predicted_scores = np.zeros(UNIQUE_TAGS - 3)
        predicted_idxs = np.argmax(prediction, axis=1)
        predicted_max = np.amax(prediction, axis=1)
        for i, idx in enumerate(predicted_idxs):
            if idx < tag_to_index['start']:
                predicted_tags[idx] = 1
                predicted_scores[idx] = predicted_max[i]
        predicted_overall.append(predicted_tags)
        predicted_scores_overall.append(predicted_scores)

    for true in y_train.cpu().data.numpy():
        true_tags = np.zeros(UNIQUE_TAGS - 3)
        for idx in true:
            if idx < tag_to_index['start']:
                true_tags[idx] = 1
        true_overall.append(true_tags)

    return predicted_overall, true_overall, predicted_scores_overall


def eval(predicted_overall, true_overall):
    overall_precision = precision_score(true_overall, predicted_overall, average='macro')
    overall_recall = precision_score(true_overall, predicted_overall, average='macro')
    precision = precision_score(true_overall, predicted_overall, average='micro')
    recall = precision_score(true_overall, predicted_overall, average='micro')

    return precision, recall, overall_precision, overall_recall
    # true_overall, predicted_overall = np.array(true_overall), np.array(predicted_overall)
    #     # precision, recall = 0, 0
    #     # precision_upper, recall_upper = 0, 0
    #     # overall_precision, overall_recall = [0, 0], [0, 0]
    #     # n = 0
    #     # for j in range(true_overall.shape[1] - 3):
    #     #     if np.sum(true_overall[:, j]) > 0:
    #     #         n += 1
    #     #         recall += np.sum(true_overall[:, j] * predicted_overall[:, j]) / np.sum(true_overall[:, j])
    #     #         if np.sum(predicted_overall[:, j]) > 0:
    #     #             precision += np.sum(true_overall[:, j] * predicted_overall[:, j]) / np.sum(predicted_overall[:, j])
    #     #         overall_recall[0] = overall_recall[0] + np.sum(true_overall[:, j] * predicted_overall[:, j])
    #     #         overall_recall[1] = overall_recall[1] + np.sum(true_overall[:, j])
    #     #         overall_precision[0] = overall_precision[0] + np.sum(true_overall[:, j] * predicted_overall[:, j])
    #     #         overall_precision[1] = overall_precision[1] + np.sum(predicted_overall[:, j])
    #     #
    #     # overall_precision = overall_precision[0] / overall_precision[1]
    #     # overall_recall = overall_recall[0] / overall_recall[1]
    #     #
    #     # return precision / n, recall / n, overall_precision, overall_recall


def f1_score(predicted_overall, true_overall):
    true_overall, predicted_overall = np.array(true_overall), np.array(predicted_overall)
    macroF1, microF1, instanceF1 = 0, 0, 0

    # macro
    n = 0
    for j in range(true_overall.shape[1] - 3):
        if np.sum(true_overall[:, j]) > 0:
            n += 1
            val = 2 * np.sum(predicted_overall[:, j] * true_overall[:, j])
            d = np.sum(predicted_overall[:, j]) + np.sum(true_overall[:, j])
            val /= d
            macroF1 += val

    # micro
    val1 = 2 * np.sum(predicted_overall * true_overall)
    val2 = np.sum(predicted_overall) + np.sum(true_overall)
    microF1 = val1 / val2

    # instance f1
    n = 0
    for i in range(true_overall.shape[0]):
        if np.sum(true_overall[i]) != 0:
            n += 1
            val = 2 * np.sum(true_overall[i] * predicted_overall[i])
            d = np.sum(true_overall[i]) + np.sum(predicted_overall[i])
            instanceF1 += val / d

    return macroF1 / n, microF1, instanceF1 / n


def one_error(y_train, predicted_scores):
    n = len(y_train)
    one_error = 0
    for true, predicted in zip(y_train, predicted_scores):
        p_idx = np.argsort(predicted)[-1]  # np.where(predicted >= 0.5)[0] # np.argsort(predicted)[:n]
        t_idxs = np.where(true == 1)[0]
        one_error += (p_idx in t_idxs) * 1
    return one_error / n


def ranking_loss(y_train, predicted_scores):
    n = len(y_train)
    ranking_error = 0
    for true, predicted in zip(y_train, predicted_scores):
        score = 0
        one_idxs = np.where(true == 1)[0]
        zeros_idxs = np.where(true == 0)[0]

        for one in one_idxs:
            for zero in zeros_idxs:
                if predicted[one] <= predicted[zero]:
                    score += 1
        ranking_error += score / (np.sum(true) * (len(true) - np.sum(true)))
    return ranking_error / n


def normalize(images):
    # return np.array(images)/255.
    return (np.array(images) - 127.5) / 127.5


def read_and_resize(filename):
    imgbgr = cv2.imread(filename)
    imgbgr = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    imgbgr = torch.FloatTensor(imgbgr / 255.)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(imgbgr)


def batch(img_tag_mapping, tag_to_index, UNIQUE_TAGS, include_negatives=False):
    batch_IMGS, batch_CAPS, batch_CAPLENS = [], [], []
    b = 0
    for im_path in img_tag_mapping:
        im = read_and_resize(f'{IMG_DIR}/{im_path}.png')
        caps = [tag_to_index['start']]
        for tag in img_tag_mapping[im_path]:
            if tag in tag_to_index:
                caps.append(tag_to_index[tag])
        caps.append(tag_to_index['end'])

        include = True
        if not include_negatives and len(caps) <= 2:
            include = False

        while len(caps) < UNIQUE_TAGS:
            caps.append(tag_to_index['pad'])

        if include:
            batch_IMGS.append(im), batch_CAPS.append(caps), batch_CAPLENS.append(len(img_tag_mapping[im_path]) + 2)
        b += 1
        if b >= BATCH_SIZE:
            yield torch.stack(batch_IMGS), np.array(batch_CAPS), np.array(batch_CAPLENS).reshape((-1, 1))
            # yield normalize(batch_IMGS).reshape((-1,3,600,600)), np.array(batch_CAPS), np.array(batch_CAPLENS).reshape((-1,1))
            b = 0
            batch_IMGS, batch_CAPS, batch_CAPLENS = [], [], []
    if len(batch_IMGS) != 0:
        yield torch.stack(batch_IMGS), np.array(batch_CAPS), np.array(batch_CAPLENS).reshape((-1, 1))
        # yield normalize(batch_IMGS).reshape((-1,3,600,600)), np.array(batch_CAPS), np.array(batch_CAPLENS).reshape((-1,1))


def train_step(imgs, caps, caplens, encoder, decoder, decoder_optimizer, encoder_optimizer, criterion, device,
               tag_to_index, UNIQUE_TAGS, training=True, attention=True):
    if training:
        encoder.train()
        decoder.train()
    else:
        decoder.eval()
        encoder.eval()

    imgs = imgs.to(device)
    caps = torch.from_numpy(caps).long().to(device)
    caplens = torch.from_numpy(caplens).long().to(device)

    # Forward prop.
    imgs = encoder(imgs)
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, device)
    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    targets = caps_sorted[:, 1:]
    targets_old = targets.clone()
    scores_old = scores.clone()
    # Remove timesteps that we didn't decode at, or are pads, pack_padded_sequence is an easy trick to do this
    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

    loss = criterion(scores.data, targets.data)
    # Add doubly stochastic attention regularization
    if attention:
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

    if training:
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()

    predicted, true, predicted_scores = process_predictions(scores_old, targets_old, tag_to_index, UNIQUE_TAGS)
    return loss.item(), predicted, true, predicted_scores, encoder, decoder, decoder_optimizer, encoder_optimizer


def train_epoch(e, train_set, valid_set, test_set, tag_to_index, UNIQUE_TAGS, encoder, decoder, decoder_optimizer,
                encoder_optimizer, criterion, device, verbose=True, include_negatives=True, attention=True):
    train_metrics, valid_metrics = [], []
    T_loss, V_loss = [], []
    T_predicted, T_true, T_pred_scores, V_predicted, V_true, V_pred_scores = [], [], [], [], [], []
    # train step
    for imgs, caps, caplens in batch(train_set, tag_to_index, UNIQUE_TAGS, include_negatives):
        train_out = train_step(imgs, caps, caplens, encoder, decoder, decoder_optimizer, encoder_optimizer, criterion,
                               device, tag_to_index, UNIQUE_TAGS, training=True, attention=attention)
        encoder, decoder, decoder_optimizer, encoder_optimizer = train_out[4], train_out[5], train_out[6], train_out[7]
        T_loss.append(train_out[0])
        for pred, true, pred_scores in zip(train_out[1], train_out[2], train_out[3]):
            T_predicted.append(pred), T_true.append(true), T_pred_scores.append(pred_scores)

    pre, rec, ovpre, ovrec = eval(T_predicted, T_true)
    macroF1, microF1, instanceF1 = f1_score(T_predicted, T_true)
    ham_loss = hamming_loss(np.array(T_true), np.array(T_predicted))
    auc = roc_auc_score(T_true, T_pred_scores)
    train_metrics.append((np.mean(T_loss), pre, rec, ovpre, ovrec, macroF1, microF1, instanceF1, ham_loss, auc))
    ro = round
    if verbose:
        print(f'============================= epoch {e} =========================================')
        print(
            f'Tr: l {ro(np.mean(T_loss), 3)} pre {ro(pre, 3)} rec {ro(rec, 3)} overpre {ro(ovpre, 3)} overrec {ovrec} '
            f'macroF1 {macroF1} microF1 {microF1} instanceF1 {instanceF1} ham loss {ham_loss} auc {auc}')

    # valid step
    for imgs, caps, caplens in batch(valid_set, tag_to_index, UNIQUE_TAGS):
        val_out = train_step(imgs, caps, caplens, encoder, decoder, decoder_optimizer, encoder_optimizer, criterion,
                             device, tag_to_index, UNIQUE_TAGS, training=False, attention=attention)
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
            f'Va: l {ro(np.mean(V_loss), 3)} pre {ro(pre, 3)} rec {ro(rec, 3)} overpre {ro(ovpre, 3)} overrec {ovrec} '
            f'macroF1 {macroF1} microF1 {microF1} instanceF1 {instanceF1} ham loss {ham_loss} auc {auc}')

    # test set performance
    Test_predicted, Test_true, Test_pred_scores = [], [], []
    for imgs, caps, caplens in batch(test_set, tag_to_index, UNIQUE_TAGS):
        test_out = train_step(imgs, caps, caplens, encoder, decoder, decoder_optimizer, encoder_optimizer, criterion,
                              device, tag_to_index, UNIQUE_TAGS, training=False, attention=attention)
        for pred, true, pred_scores in zip(test_out[1], test_out[2], test_out[3]):
            Test_predicted.append(pred), Test_true.append(true), Test_pred_scores.append(pred_scores)
    pre, rec, ovpre, ovrec = eval(Test_predicted, Test_true)
    macroF1, microF1, instanceF1 = f1_score(Test_predicted, Test_true)
    ham_loss = hamming_loss(np.array(Test_true), np.array(Test_predicted))
    auc = roc_auc_score(Test_true, Test_pred_scores)
    test_metrics = [pre, rec, ovpre, ovrec, macroF1, microF1, instanceF1, ham_loss, auc]
    return np.mean(
        V_loss), train_metrics, valid_metrics, test_metrics, encoder, decoder, decoder_optimizer, encoder_optimizer


def train(start_epoch, end_epoch, train_set, valid_set, test_set, tag_to_index, UNIQUE_TAGS, encoder, decoder,
          decoder_optimizer,
          encoder_optimizer, criterion, device, include_negatives=True, verbose=True, attention=True):
    epochs_since_improvement = 0
    train_metrics, valid_metrics, test_metrics = [], [], []
    best_loss = 100
    for epoch in range(start_epoch, end_epoch):
        recent_loss, train_metrics_out, valid_metrics_out, test_metrics, encoder, decoder, decoder_optimizer, encoder_optimizer = train_epoch(
            epoch, train_set, valid_set, test_set, tag_to_index, UNIQUE_TAGS, encoder, decoder, decoder_optimizer,
            encoder_optimizer, criterion, device, include_negatives=include_negatives, verbose=verbose,
            attention=attention)
        train_metrics.append(train_metrics_out)
        valid_metrics.append(valid_metrics_out)
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            adjust_learning_rate(encoder_optimizer, 0.8)

        if recent_loss < best_loss:
            best_loss = recent_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
    valid_metrics = np.array(valid_metrics)
    train_metrics = np.array(train_metrics)
    valid_metrics = valid_metrics.reshape(valid_metrics.shape[0], -1)
    train_metrics = train_metrics.reshape(train_metrics.shape[0], -1)
    return train_metrics, valid_metrics, test_metrics, encoder, decoder


def save_models(encoder, decoder, ENCODER_NAME, include_negatives):
    # save models
    dir_name = f'{ENCODER_NAME}_results'
    if include_negatives:
        dir_name = 'NegativeSampling_' + dir_name
    else:
        dir_name = 'NoNegativeSampling_' + dir_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    decoder_name = f'{ENCODER_NAME}_decoder'
    encoder_name = f'{ENCODER_NAME}_encoder'
    if include_negatives:
        decoder_name = 'NegativeSampling_' + decoder_name
        encoder_name = 'NegativeSampling_' + encoder_name
    else:
        decoder_name = 'NoNegativeSampling_' + decoder_name
        encoder_name = 'NoNegativeSampling_' + encoder_name

    torch.save(decoder, os.path.join(dir_name, decoder_name))
    torch.save(encoder, os.path.join(dir_name, encoder_name))


def save_metrics(train_metrics, valid_metrics, test_metrics, ENCODER_NAME, include_negatives):
    dir_name = f'{ENCODER_NAME}_results'
    if include_negatives:
        dir_name = 'NegativeSampling_' + dir_name
    else:
        dir_name = 'NoNegativeSampling_' + dir_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = 'train_metrics.pickle'
    with open(os.path.join(dir_name, file_name), 'wb') as f:
        pickle.dump(train_metrics, f)

    file_name = 'valid_metrics.pickle'
    with open(os.path.join(dir_name, file_name), 'wb') as f:
        pickle.dump(valid_metrics, f)

    file_name = 'test_metrics.pickle'
    with open(os.path.join(dir_name, file_name), 'wb') as f:
        pickle.dump(test_metrics, f)


def prediction_step(encoder, decoder, device, imgs, caps, caplens, tag_to_index, UNIQUE_TAGS):
    decoder.eval()
    encoder.eval()

    imgs = imgs.to(device)
    caps = torch.from_numpy(caps).long().to(device)
    caplens = torch.from_numpy(caplens).long().to(device)
    imgs = encoder(imgs)
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, device)
    targets = caps_sorted[:, 1:]

    predicted, true, predicted_scores = process_predictions(scores, targets, tag_to_index, UNIQUE_TAGS)
    return true, predicted, predicted_scores


def prediction(encoder, decoder, device, test_set, tag_to_index, UNIQUE_TAGS):
    true, predicted, predicted_scores = [], [], []
    for imgs, caps, caplens in batch(test_set, tag_to_index, UNIQUE_TAGS):
        test_out = prediction_step(encoder, decoder, device, imgs, caps, caplens, tag_to_index, UNIQUE_TAGS)
        for t, p, ps in zip(test_out[0], test_out[1], test_out[2]):
            true.append(t), predicted.append(p), predicted_scores.append(ps)

    return np.array(true), np.array(predicted), np.array(predicted_scores)


def extract_attention_weights(imgs, caps, caplens, encoder, decoder, device, tag_to_index, UNIQUE_TAGS):
    decoder.eval()
    encoder.eval()

    imgs = imgs.to(device)
    caps = torch.from_numpy(caps).long().to(device)
    caplens = torch.from_numpy(caplens).long().to(device)
    imgs = encoder(imgs)
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, device)
    targets = caps_sorted[:, 1:]
    targets_old = targets.clone()
    scores_old = scores.clone()
    predicted, true, predicted_scores = process_predictions(scores_old, targets_old, tag_to_index, UNIQUE_TAGS)
    return predicted, true, alphas


def visualize_attention_step(image_path, seq, alphas, rev_word_map, true, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.show()

    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    words = words[1:]
    print(f'True {[rev_word_map[ind] for ind in true]}')
    print(f'Predictions {words}')

    for t in range(len(words)):
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.cpu().numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [14 * 24, 14 * 24])
        plt.imshow(alpha, alpha=0.8)
        plt.axis('off')
        plt.show()


def visualize_attention(tag_to_index, img_tag_mapping, UNIQUE_TAGS, encoder, decoder, device):
    rev_word_map = {v: k for k, v in tag_to_index.items()}
    ii = 0

    def specisl_batch(img_tag_mapping, UNIQUE_TAGS):
        batch_IMGS, batch_CAPS, batch_CAPLENS = [], [], []
        b = 0
        for im_path in img_tag_mapping:
            im = read_and_resize(f'{IMG_DIR}/{im_path}.png')
            caps = [tag_to_index['start']]
            for tag in img_tag_mapping[im_path]:
                if tag in tag_to_index:
                    caps.append(tag_to_index[tag])
            caps.append(tag_to_index['end'])
            while len(caps) < UNIQUE_TAGS:
                caps.append(tag_to_index['pad'])

            batch_IMGS.append(im), batch_CAPS.append(caps), batch_CAPLENS.append(len(img_tag_mapping[im_path]) + 2)
            if len(batch_IMGS) != 0:
                yield torch.stack(batch_IMGS), np.array(batch_CAPS), np.array(batch_CAPLENS).reshape((-1, 1)), im_path

    for imgs, caps, caplens, img_path in specisl_batch(img_tag_mapping, UNIQUE_TAGS):
        ii += 1
        if ii > 50:
            break
        predicted, true, alphas = extract_attention_weights(imgs, caps, caplens, encoder, decoder, device, tag_to_index,
                                                            UNIQUE_TAGS)
        predicted = predicted[0]
        true = true[0]
        seq = [tag_to_index['start']]
        for i in range(len(predicted)):
            if predicted[i] == 1:
                seq.append(i)
        truth = []
        for i in range(len(true)):
            if true[i] == 1:
                truth.append(i)
        alphas = alphas.view(-1, 14, 14)
        print(alphas.shape)
        alphas = alphas.detach()
        visualize_attention_step(f'{IMG_DIR}/{img_path}.png', seq, alphas, rev_word_map, truth)
