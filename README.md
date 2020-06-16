# Chest ALSTM-MLC: A Large Scale Multi-Label Chest X-ray Annotation and Localization Framework

The annotation of Chest X-ray images with one or more tags. We have 128 unique label in the dataset. This work combines the CNN model that is capable of visual features extraction, Linear model that is capable for "attention", e.g learning which parts of image are important while predicting a particular label, and LSTM network which is capable of predicting label based on information from two previous networks. We name this framework as a Chest Attention LSTM-Multi Label Classification model (Chest ALSTM-MLC). Given a chest X-ray image it is able to assign one, multiple or no labels out of 128 possible findings that we have in database. We introduce the concept of negative sampling and hierarchy during training which increases the model's performance. We used Indiana University Chest X-ray collection for training and evaluating the model. Our proposed model is achieving 78\% AUC score. 

## Run the code
The detailed instructions on how to run the code are located in the [main.ipynb](https://github.com/RufinaMay/MedicalReports_v1/blob/master/main.py). This file contains information not only on the best obtained model that we present here, but baseline models as well as Chest ALSTM-MLC models that are using different visual encoders as well. 

## Model
![The main parts of Chest ALSTM-MLC model that images are following through to get a prediction.](https://github.com/RufinaMay/MedicalReports_v1/raw/master/model_diagrams/methodology_overall.png)
The overall model design is presented in Figure above. One can note that the overall proposed model contains of three different modules, namely Encoder, Attention and Decoder. The input image first goes to normalization part, then it is passed to the visual features encoder. After than encoded image is passed into Attention network to obtain highlighted features given the embedding of previous label as input (embedding of *start* at the beginning), then image encoded with attention network and embedding of previous label are passed to the lstm unit to generate one more label description. The whole process is repeated until the $<end>$ tag will be generated which means model is done generating labels. In the following sections we describe what exactly each and every module does in more details.
  ### Encoder
  To obtain the encoded representation of an image we are using CNN network. We are taking advantage of transfer learning in this case and using the CNN model developed for ImageNet competition, in particular DenseNet. We are taking all layers of the network besides fully connected classification layers and fine tune them together with the model training. 
  ### Decoder
  
  In proposed MLCNet architecture, decoder is parameterized recurrent neural network variant LSTM to predict  multiple-labels of X-ray image.
  ![The internal architecture of single memory cell of LSTM network.](https://github.com/RufinaMay/MedicalReports_v1/raw/master/model_diagrams/decoder.png)
  Our implementation of LSTM is based on  [Show Attent and Tell](https://arxiv.org/abs/1502.03044) and implemented by following functions.

![formula](https://render.githubusercontent.com/render/math?math=g_t=\sigma(W_g\cdot{a_t}%2B{b_g}))
![formula](https://render.githubusercontent.com/render/math?math=f_t=\sigma({W_f}{[h_{t-1},g_t,e_{m,t}]}%2B{b_f}))
![formula](https://render.githubusercontent.com/render/math?math=i_t=\sigma({W_i}{[h_{t-1},g_t,e_{m,t}]}%2B{b_i}))
![formula](https://render.githubusercontent.com/render/math?math=\hat{C_t}=tanh(W_c{[h_{t-1},g_t,e_{m,t}]}%2B{b_c}))
![formula](https://render.githubusercontent.com/render/math?math=\hat{C_t}=f_t*C_{t-1}%2B{i_t}*\hat{C_t})
![formula](https://render.githubusercontent.com/render/math?math=o_t=\sigma(W_o{[h_{t-1},g_t,e_{m,t}]}%2B{b_o}))
![formula](https://render.githubusercontent.com/render/math?math=h_t=o_t*tanh(C_t))
![formula](https://render.githubusercontent.com/render/math?math=a_t=attention([encoder(image),h_{t-1}]))


Where *g_t* is attention gate, *f_t* is forget gate, *i_t* is an input gate, and *o_t* is output gate, $h_t$ is hidden state.Similarly, *W_g, W_f, W_i, W_c, W_o, b_g, b_f, b_i, b_c, b_o* are weight matrices and bias vectors respectively. *a_t* is attention weighted vector over encoded image and previous hidden state. Similarly,
![formula](https://render.githubusercontent.com/render/math?math=e_{m,t}\in R^D)

 is a embedding vector of ![formula](https://render.githubusercontent.com/render/math?math=m^{th}) tag from embedding matrix and
 ![formula](https://render.githubusercontent.com/render/math?math=E\in R^{D \times V})

 where *D* is the embedding dimension and *V* is the size of the vocabulary. 
 
 Visual feature encoder output is flattened and passed to LSTM hidden state with attention network together to highlight important regions of an image. An attention gate output is obtained by passing hidden state of LSTM to a fully connected layer with sigmoid activation function. The attention weighted encoding is obtained by multiplying the attention gate with attention weights. We obtain the embedding of previous decoder input to be passed to LSTM unit. All previously obtained variables are passed to LSTM unit to calculate the new hidden state and cell state. New hidden state is passed to another fully connected layer to obtain the scores over vocabulary for predicting the label.
  
  
### Attention
  ![The pipeline of attention network.](https://github.com/RufinaMay/MedicalReports_v1/raw/master/model_diagrams/attention.png)
Before going into details of attention network pipeline let's find out why do we need the attention network? We are adding attention mechanism at each time step of the decoder, so that decoder is able to "look" at different parts of the image at each time step. We can refer to attention mechanism as to the weighted average across encoded visual features, with the weights of the important features being greater. The weighted representation of the image is concatenated with the previously generated word at each time step to generate the next word. 

Attention pays attention to particular areas or objects rather than treating the whole image equally. Attention mechanism should consider the labels generated thus far, and attend to the part of the image that describes next label.
The Attention network is parameterized with fully connected network that computes weights. In this work we are using soft attention, where the weights of the pixels add up to 1, to avoid large numbers. If there are *M* features in encoded image, then at each time step t:

![formula](https://render.githubusercontent.com/render/math?math=\sum_{p=m}^{p=M}\alpha_{p,t}=1)

where ![formula](https://render.githubusercontent.com/render/math?math=\alpha_{p,t})
is the *p*-s weight of attention network at time step *t*. The overall attention mechanism is presented in Figure above. One can note that attention mechanism consists of three fully connected networks and data flows through the network in the following way: 

- Previous Decoder Output (previously generated label, *start* at the beginning) is passed to embedding layer of the model to obtain embedding of that unit.
- Encoded input image and embeded tag are passed to two different identically defined fully connected networks.
- The outputs of previous iteration are concatenated together and passed through ReLU activation function.
- After that data goes to another one unit linear layer with SoftMax activation function to obtain a weighted representation of the image. Since we use "soft" attention we are applying SoftMax activation function so that sum of the weights is equal to 1.

Attention network is applied at each time step of decoder tags generation process. 

### Loss function
As suggested in [Show, attend and tell: Neural image caption generation with visual attention](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf) we are including the Doubly Stochastic Attention Regularization that encourage the model to pay equal attention to every part of the image while predicting the labels, resulting in minimizing the following penalized negative log-likelihood: 

![formula](https://render.githubusercontent.com/render/math?math=loss=-log(p(y|x))+\lambda\sum_i^{L}(1-\sum_t^Ca_{ti})^2)

### Negative Sampling
In this work we refer to negative sampling as keeping images that have no labels associated with them. We explore the including and excluding the negative samples from the training set and see how it affects the resulting performance for models. We explore this with an assumption that including images that do not have any labels associated with them will help network to learn positive labels by showing them how this label does not look like. 

## Data set
Indiana University Chest X-ray collection ([images](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz), [reports](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz)) is used in this work. There are total 7,470 samples in the data set and 559 unique tags. Data was split on train, test and validation sets using stratified data split [On the stratification of multi-label data](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10). The data is stored in the following format [*image name*]-[*list of tags*], e.g. 'CXR960_IM-2451-4004': ['right', 'mild', 'scolioses'] and located in three following files:
- test_set.pickle - 1,940 samples, 471 unique tags
- valid_set.pickle - 862 samples, 372 unique tags
- train_set.pickle - 4,668 samples, 590 unique tags
