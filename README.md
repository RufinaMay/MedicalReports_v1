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
  In proposed MLCNet architecture, decoder is parameterized recurrent neural network variant LSTM to predict  multiple-labels of X-ray image. Our implementation of LSTM is based on  [Show Attent and Tell](https://arxiv.org/abs/1502.03044) and implemented by following functions.
  
  ![The internal architecture of single memory cell of LSTM network.](https://github.com/RufinaMay/MedicalReports_v1/raw/master/model_diagrams/decoder.png)
  

![formula](https://render.githubusercontent.com/render/math?math=g_t=\sigma(W_g\cdot{a_t}%2B{b_g}))

![formula](https://render.githubusercontent.com/render/math?math=f_t=\sigma({W_f}{[h_{t-1},g_t,e_{m,t}]}%2B{b_f}))

![formula](https://render.githubusercontent.com/render/math?math=i_t=\sigma({W_i}{[h_{t-1},g_t,e_{m,t}]}%2B{b_i}))

![formula](https://render.githubusercontent.com/render/math?math=\hat{C_t}=tanh(W_c{[h_{t-1},g_t,e_{m,t}]}%2B{b_c}))

![formula](https://render.githubusercontent.com/render/math?math=\hat{C_t}=f_t*C_{t-1}%2B{i_t}*\hat{C_t})

![formula](https://render.githubusercontent.com/render/math?math=o_t=\sigma(W_o{[h_{t-1},g_t,e_{m,t}]}%2B{b_o}))

![formula](https://render.githubusercontent.com/render/math?math=h_t=o_t*tanh(C_t))

![formula](https://render.githubusercontent.com/render/math?math=a_t=attention([encoder(image),h_{t-1}]))


Where ![formula](https://render.githubusercontent.com/render/math?math=g_t) is attention gate, ![formula](https://render.githubusercontent.com/render/math?math=f_t) is forget gate, ![formula](https://render.githubusercontent.com/render/math?math=i_t) is an input gate, and ![formula](https://render.githubusercontent.com/render/math?math=o_t) is output gate, ![formula](https://render.githubusercontent.com/render/math?math=h_t) is hidden state.Similarly, ![formula](https://render.githubusercontent.com/render/math?math=W_g,W_f,W_i,W_c,W_o,b_g,b_f,b_i,b_c,bo), are weight matrices and bias vectors respectively. *at* is attention weighted vector over encoded image and previous hidden state. Similarly,
![formula](https://render.githubusercontent.com/render/math?math=e_{m,t}\in{R^D}) is a embedding vector of ![formula](https://render.githubusercontent.com/render/math?math=m^{th}) tag from embedding matrix and ![formula](https://render.githubusercontent.com/render/math?math=E\in{R^{DxV}}), where *D* is the embedding dimension and *V* is the size of the vocabulary. 
 
 Visual feature encoder output is flattened and passed to LSTM hidden state with attention network together to highlight important regions of an image. An attention gate output is obtained by passing hidden state of LSTM to a fully connected layer with sigmoid activation function. The attention weighted encoding is obtained by multiplying the attention gate with attention weights. We obtain the embedding of previous decoder input to be passed to LSTM unit. All previously obtained variables are passed to LSTM unit to calculate the new hidden state and cell state. New hidden state is passed to another fully connected layer to obtain the scores over vocabulary for predicting the label.
  
  
### Attention
  ![The pipeline of attention network.](https://github.com/RufinaMay/MedicalReports_v1/raw/master/model_diagrams/attention.png)
  
Attention pays special attention to particular areas rather than treating the whole X-ray image equally. It also consider the labels generated so far, and attend the part of image that describes next label. Inside Attention, it has three fully connected networks. First, encoder attention network computes attention weights over the encoded image features. Second, decoder attention network to computes the attention weights over the previous hidden state of the decoder. Third, full attention network computes the final attention weights over the encoded image given the output of previous two networks.

The process of attention module is computed by the following equations:

![formula](https://render.githubusercontent.com/render/math?math=a_e=EncoderAttention(I))

![formula](https://render.githubusercontent.com/render/math?math=a_d=DecoderAttention(h_{t-1}))

![formula](https://render.githubusercontent.com/render/math?math=a_f=FullAttention(ReLU(a_e+a_d)))

![formula](https://render.githubusercontent.com/render/math?math=\alpha=SoftMax(a_f))

![formula](https://render.githubusercontent.com/render/math?math=Out=I*\alpha)

where *I* is the encoded image: ![formula](https://render.githubusercontent.com/render/math?math=I\in{R}^{size{x}size{x}dim})

Similarly, encoder attention, decoder attention and full attention are parameterized by linear networks. The *h t-1* is decoder's hidden state at previous time step. The ![formula](https://render.githubusercontent.com/render/math?math=a_e\in{R}^{AD}), ![formula](https://render.githubusercontent.com/render/math?math=a_d\in{R}^{AD}),  where *AD* is the dimension of the attention. ![formula](https://render.githubusercontent.com/render/math?math=a_f\in{R}^{1}), ![formula](https://render.githubusercontent.com/render/math?math=\alpha\in{R}^{size{X}size{X}dim}). The attention that we are using in this work is referred as soft-attention, hence at each time step t:

![formula](https://render.githubusercontent.com/render/math?math=\sum_{p=1}^{p=ED}\alpha_{p,t}=1)

Where ![formula](https://render.githubusercontent.com/render/math?math=\alpha_{p,t}) is the *p*-s weight of attention network at time step *t* and *ED* is the number of features in the encoded image.


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
