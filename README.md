# Image_Captioning-Data_Analytics

Group Name - Seven

Project Title - Image Caption Generation using LSTM </br>
Dataset - Flickr_8k Dataset (https://www.kaggle.com/adityajn105/flickr8k?select=Images)
```
Group Members -
  1.Arjun C Menon             PES2201800090
  2.Sameer Prasad Subhedar    PES2201800323
  3.Roshini Bhaskar K         PES2201800122
  ```
  
# Approach to the problem statement
    1.Understanding the Dataset
    2.Prepare Photo data using VGG16 Model
    3.Preparing Text Data
    4.Developing Model 
    5.Model evaluation
    6.Generating New Descriptions
    
# Feature Extraction
The dataset consists of 8091 images. Images tend to have high dimensions and noises. 
Feature extraction is an important process since the main goal of feature extraction is to obtain 
the most relevant information from the original data and represent that information in a lower dimensionality space.

# VGG16 Model
This model is used in extraction of features.</br>
Visual Geometry Group from University of Oxford developed VGG model.
Given an image, find object name in the image.
It can detect any one of 1000 images.
It takes input image of size 224 224 3 (RGB image) i.e 224 * 224 pixel image with 3 channels.</br>
VGG16 Model consists of:</br>
  1. 3×3 filters in all convolutional layers</br>
  2. 16 Layers Model</br>
  
# Developing the model
The model is in three parts:</br>

<b>Photo Feature Extractor </b>: This is a 16-layer VGG model pre-trained on the ImageNet dataset. We have pre-processed the photos with the VGG model (without the output layer) and will use the extracted features predicted by this model as input.</br>

<b>Sequence Processor </b>: This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.</br>

<b>Decoder </b>: Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction. The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. These are processed by a Dense layer to produce a 256 element representation of the photo.</br>

The Sequence Processor model expects input sequences with a pre-defined length (34 words) which are fed into an Embedding layer that uses a mask to ignore padded values. This is followed by an LSTM layer with 256 memory units.</br>

Both the input models produce a 256 element vector. Further, both input models use regularization in the form of 50% dropout. This is to reduce overfitting the training dataset, as this model configuration learns very fast.</br>

The Decoder model merges the vectors from both input models using an addition operation. This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.</br>
 
# Model Training
Model is trained using these main parameters - </br>
epchos - 20</br>
Descriptions for Photos: train= 6000</br>
Photos: train= 6000</br>
Vocabulary Size: 7579</br>
Description Length:  34</br>

# Model Evaluation
Evaluation of our model was implemented by generating descriptions for all the images in the test dataset and then evaluating those predictions with the standard cost functions.
This needs the generation of a description for an image using the model that we just trained also known as the predictions of our model. This involves passing the start description or word of the token ‘startseq‘, thus generating one word and then calling the model recursively with generated words as the  input, until the end of the sequence of the token is  ‘endseq‘ or the maximum caption/sentence length is reached.</br>
The actual and predicted descriptions are collected and evaluated collectively using the corpus BLEU score that summarizes and tells us  how close the generated text is to the expected text.</br>
BLEU is an algorithm that  is used for evaluating the quality of text which has been machine-translated from one natural language to another or simply predicted by a machine. There are many ways to calculate the BLEU score and the method used for our project is Cumulative N-Gram Scores.N-gram score is the evaluation of just matching grams (grams mean words ) in a specific order, such as single words (1-gram) or word pairs (2-gram or bigram).Cumulative scores refer to the calculation of individual n-gram scores at all orders from 1 to n and weighting them by calculating the weighted geometric mean.</br>
We then compare each generated description or caption against all of the reference descriptions from the dataset for the given picture. A score above 40% can be considered a good score.</br>
