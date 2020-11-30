# Image_Captioning-Data_Analytics

Group Name - Seven

Project Title - Image Caption Generation using LSTM </br>
Dataset - Flickr_8k Dataset (https://www.kaggle.com/adityajn105/flickr8k?select=Images)
```
Group Members -
  1.Arjun C Menon
  2.Sameer Prasad Subhedar
  3.Roshini Bhaskar K 
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
 
