# ECE228-Team40-Project

## Introduction
Chest X-Ray Images (Pneumonia) | Kaggle

Pneumonia has become one of the most deadly diseases in the world, and how to
quickly diagnose and predict pneumonia has become a top priority. To solve this
problem, we collect the chest X-ray dataset and then use machine learning and deep
learning to build the various models. They are VGG, ResNet, DenseNet121/161 and
Self-Attention. Through these models, we can identify pneumonia among the images
and compare their accuracy. The results show that the best model is DenseNet161
with transfer learning, in which the accuracy can achieve 88.14%. 

## Dataset and Processing
The dataset contains 5,863 X-Ray images of the chest and is separated into
categories of Pneumonia and Normal for us to apply machine learning and deep
learning-based models to it for classification. One of three labels is given to each
image: normal, bacterial pneumonia, and viral pneumonia, to compare our model
prediction with the actual result.

After data loading, we did data visualization. The data is imbalanced. To increase the
number of training examples, we use data augmentation like Center Crop and
Random Rotation to avoid overfitting problem. We split the images into 82%
(training), 1% (validation), and 10% (testing). The original dimension of image is
299*299*3, we resize the images to 224*224*3 to fit our models

## Methodology/Approach
We compare the accuracy among different models by using both machine learning
and deep learning. Since CNN models are powerful in classifying images, we select
some derivative models: VGG19, ResNet10/50, DenseNet121/161 and SelfAttention. Also, we add transfer learning method to see difference between these
models. In the following, we briefly introduce these models.

### VGG 19
VGG19 has 19 layers. It is composed of 5 blocks of convolutional layers; besides, it
accompanies pooling layers in the front and fully connected layers at the end.
Instead of using 11*11, 7*7, or 5*5 convolutional layers, VGG19 only uses 3*3
convolutional layers

### ResNet
ResNet uses network layers to fit a residual mapping instead of directly fitting a
desired underlying mapping. This shortcut structure can allow the gradient to update
the weights in upper layers more easily, which can prevent gradient vanishing in the
training process.

### DenseNet121/161
DenseNet connects each layer: that is, each layer is receiving collective knowledge
from all preceding layers. It can improve the declined accuracy because it could fix
the problem of vanishing gradient which happens very often in high-level neural
networks. We adopt DenseNet121/161 in the project.

### Self-Attention
Self-Attention is an attention mechanism relating different positions of a single
sequence, which allows the inputs to interact with each other (“self”) and find out
who they should pay more attention to (“attention”). The outputs are aggregates of
these interactions and attention scores.
