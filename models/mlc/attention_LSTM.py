import torch
import torch.optim
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
import numpy as np
import torchvision

from utils.utils import train_test_split, prepare_data, read_and_resize, normalize

class Encoder(nn.Module):
    """
    CNN Encoder.
    """
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

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
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=256, dropout=0.5):
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

    def forward(self, encoder_out, encoded_captions, caption_lengths):
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

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

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


class AttentionLSTM:
    def __init__(self, img_tag_mapping_path='IMG_TAG.pickle', tag_to_index_path='TAG_TO_INDEX.pickle',
                 img_dir='data/chest_images', emb_dim=512, attention_dim=512, decoder_dim=512, dropout=0.5,
                 encoder_lr=1e-4, decoder_lr=4e-4, alpha_c=1.):
        """
        initialize models, optimizers and criterion to be used
        :param img_tag_mapping_path: path to image-tags mapping
        :param tag_to_index_path: path to tag-index dictionary
        :param img_dir: path to directory containing chest x-ray images
        :param emb_dim: dimension of word embeddings
        :param attention_dim: dimension of attention linear layers
        :param decoder_dim: dimension of decoder RNN
        :param dropout: dropout probability
        :param encoder_lr: learning rate for encoder if fine-tuning
        :param decoder_lr: learning rate for decoder
        """
        with open(img_tag_mapping_path, 'rb') as f:
            self.img_tag_mapping = pickle.load(f)
        with open(tag_to_index_path, 'rb') as f:
            tag_to_index = pickle.load(f)
        tag_to_index['start'] = len(tag_to_index)
        tag_to_index['end'] = len(tag_to_index)
        tag_to_index['pad'] = len(tag_to_index)
        self.unique_tags = len(tag_to_index)
        self.img_dir = img_dir
        self.tag_to_index = tag_to_index
        self.alpha_c = alpha_c
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=self.unique_tags,
                                       dropout=dropout)
        self.decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                  lr=decoder_lr)
        encoder = Encoder()
        self.encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                  lr=encoder_lr)

        self.decoder = decoder.to(self.device)
        self.encoder = encoder.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def process_predictions(self, pred, true):
        """
        Convert model predictions in one-hot encoding
        :param pred: predicted captions
        :param true: true captions
        :return: one-hot encoded predictions and true labels
        """
        predicted_overall, true_overall = [], []
        for prediction in pred.cpu().data.numpy():
            predicted_tags = np.zeros(self.tag_to_index - 3)
            predicted_idxs = np.argmax(prediction, axis=1)
            for idx in predicted_idxs:
                if idx < self.tag_to_index['start']:
                    predicted_tags[idx] = 1
            predicted_overall.append(predicted_tags)

        for true in true.cpu().data.numpy():
            true_tags = np.zeros(self.unique_tags - 3)
            for idx in true:
                if idx < self.tag_to_index['start']:
                    true_tags[idx] = 1
            true_overall.append(true_tags)

        return predicted_overall, true_overall

    def eval(self, predicted, true):
        """
        Computes precision and recall metrics
        :param predicted: one hot encoded model predictions
        :param true: one hot encoded true tags
        :return: per class precision, per class recall, overall precision, overall recall
        """
        true, predicted = np.array(true), np.array(predicted)
        precision, recall = 0, 0
        precision_upper, recall_upper = 0, 0
        overall_precision, overall_recall = [0, 0], [0, 0]
        n = 0
        for j in range(true.shape[1] - 3):
            if np.sum(true[:, j]) > 0:
                n += 1
                recall += np.sum(true[:, j] * predicted[:, j]) / np.sum(true[:, j])
                if np.sum(predicted[:, j]) > 0:
                    precision += np.sum(true[:, j] * predicted[:, j]) / np.sum(predicted[:, j])
                overall_recall[0] = overall_recall[0] + np.sum(true[:, j] * predicted[:, j])
                overall_recall[1] = overall_recall[1] + np.sum(true[:, j])
                overall_precision[0] = overall_precision[0] + np.sum(true[:, j] * predicted[:, j])
                overall_precision[1] = overall_precision[1] + np.sum(predicted[:, j])

        overall_precision = overall_precision[0] / overall_precision[1]
        overall_recall = overall_recall[0] / overall_recall[1]

        return precision / n, recall / n, overall_precision, overall_recall

    def f1_score(self, predicted_overall, true_overall):
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

    def batch(self, batch_size=8):
        """

        :param img_tag_mapping: dictionary of img-tags pairs
        :return: numpy array of images,indexed captions, caption lengths
        """
        batch_IMGS, batch_CAPS, batch_CAPLENS = [], [], []
        b = 0
        for im_path in self.img_tag_mapping:
            im = read_and_resize(f'{self.img_dir}/{im_path}.png')
            caps = [self.tag_to_index['start']]
            for tag in self.img_tag_mapping[im_path]:
                caps.append(self.tag_to_index[tag])
            caps.append(self.tag_to_index['end'])
            while len(caps) < self.unique_tags:
                caps.append(self.tag_to_index['pad'])

            batch_IMGS.append(im), batch_CAPS.append(caps), batch_CAPLENS.append(len(self.img_tag_mapping[im_path]) + 2)
            b += 1
            if b >= batch_size:
                yield normalize(batch_IMGS).reshape((-1, 3, 256, 256)), np.array(batch_CAPS), np.array(
                    batch_CAPLENS).reshape((-1, 1))
                b = 0
                batch_IMGS, batch_CAPS, batch_CAPLENS = [], [], []
        if len(batch_IMGS) != 0:
            yield normalize(batch_IMGS).reshape((-1, 3, 256, 256)), np.array(batch_CAPS), np.array(
                batch_CAPLENS).reshape((-1, 1))

    def train_step(self, imgs, caps, caplens, training=True):
        """

        :param imgs:
        :param caps:
        :param caplens:
        :param training:
        :return:
        """
        if training:
            self.encoder.train()
            self.decoder.train()
        else:
            self.decoder.eval()
            self.encoder.eval()

        imgs = torch.from_numpy(imgs).float().to(self.device)
        caps = torch.from_numpy(caps).long().to(self.device)
        caplens = torch.from_numpy(caplens).long().to(self.device)
        # Forward prop.
        imgs = self.encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, caps, caplens)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        targets_old = targets.clone()
        scores_old = scores.clone()
        # Remove timesteps that we didn't decode at, or are pads, pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = self.criterion(scores.data, targets.data)
        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        if training:
            self.decoder_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            loss.backward()
            self.decoder_optimizer.step()
            self.encoder_optimizer.step()

        predicted, true = self.process_predictions(scores_old, targets_old)
        return loss.item(), predicted, true

    def train_epoch(self, e, train_set, valid_set):
        """

        :param train_set:
        :param valid_set:
        :return:
        """
        T_loss, V_loss = [], []
        T_predicted, T_true, V_predicted, V_true = [], [], [], []

        # train step
        for imgs, caps, caplens in self.batch(train_set):
            train_out = self.train_step(imgs, caps, caplens, training=True)
            T_loss.append(train_out[0])
            for pred, true in zip(train_out[1], train_out[2]):
                T_predicted.append(pred), T_true.append(true)

        pre, rec, ovpre, ovrec = eval(T_predicted, T_true)
        macroF1, microF1, instanceF1 = self.f1_score(T_predicted, T_true)
        self.train_metrics.append((np.mean(T_loss), pre, rec, ovpre, ovrec, macroF1, microF1, instanceF1))
        ro = round
        print(f'======================== epoch {e} ================================')
        print(f'Tr: l {ro(np.mean(T_loss), 3)} pre {ro(pre, 3)} rec {ro(rec, 3)} overpre {ro(ovpre, 3)} overrec {ro(ovrec,3)} macroF1 {macroF1} microF1 {microF1} instanceF1 {instanceF1}')

        # valid step
        for imgs, caps, caplens in self.batch(valid_set):
            val_out = self.train_step(imgs, caps, caplens, training=False)
            V_loss.append(val_out[0])
            for pred, true in zip(val_out[1], val_out[2]):
                V_predicted.append(pred), V_true.append(true)

        pre, rec, ovpre, ovrec = eval(V_predicted, V_true)
        macroF1, microF1, instanceF1 = self.f1_score(V_predicted, V_true)
        self.valid_metrics.append((np.mean(V_loss), pre, rec, ovpre, ovrec, macroF1, microF1, instanceF1))
        print(f'Va: l {ro(np.mean(V_loss), 3)} pre {ro(pre, 3)} rec {ro(rec, 3)} overpre {ro(ovpre, 3)} overrec {ro(ovrec,3)} macroF1 {macroF1} microF1 {microF1} instanceF1 {instanceF1}')

        # # save model after each epoch
        # torch.save(self.decoder,
        #            f'/content/drive/My Drive/Colab Notebooks/THESIS/Image reports/Own_CNN_model/decoder_epoch{e}')
        # torch.save(self.encoder,
        #            f'/content/drive/My Drive/Colab Notebooks/THESIS/Image reports/Own_CNN_model/encoder_epoch{e}')
        #
        # with open(f'/content/drive/My Drive/Colab Notebooks/THESIS/Image reports/Own_CNN_model/train_metrics_epoch{e}',
        #           'wb') as f:
        #     pickle.dump(self.train_metrics, f)
        # with open(f'/content/drive/My Drive/Colab Notebooks/THESIS/Image reports/Own_CNN_model/valid_metrics_epoch{e}',
        #           'wb') as f:
        #     pickle.dump(self.valid_metrics, f)

    def train(self, eposhs):
        self.train_metrics, self.valid_metrics = [], []
        train_set, valid_set, test_set = prepare_data(self.img_tag_mapping)
        for epoch in range(eposhs):
            self.train_epoch(epoch, train_set, valid_set)