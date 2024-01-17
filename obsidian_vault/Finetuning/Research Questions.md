
## Methodology

#### Model
Pytorch Neural Network (torch.nn.Module)
#### Input
 - Encoding of image (Tensor with length 512)
 - Cosine distance between this image and the 211 Encodings of the string 'An image from the country {Country}'
 - 512 + 211 = 723
#### Output
 - An index of the maximum value of the 211 Nodes in the output layer
#### Optimizer
Adam Optimizer
#### Loss Function
Loss Function is a mixture of the region and country loss functions. If the correct country is classified, then the loss is 0; if the correct region is classified (and the wrong country), then the loss is 0.5; if the wrong country and region are classified, then the loss is 1.
#### Learning Rate
0.001
#### Layers (Fully Connected)
1 Input layer (Size = 723)
RELU
1 Hidden Layer (Size = 723)
RELU
1 Output Layer (Size = 211)
Softmax
#### Number of Epochs
50
#### Batch sizes
450
## Research Question

#### Training Sets
All of the training sets have to be the same size - this size is 80% of the Geoguessr Dataset. Step one is to randomly select and remove 20% of the Balanced Geoguessr Dataset. This test set should then be removed from the Original Geoguessr Dataset.

1. 80% of Balanced Geoguessr Dataset (Size = N)
2. A random selection of N of the Original Geoguessr Dataset (minus the samples that are in the testing set)
3. Start with training set 1, then replace (randomly) 10% of this dataset with Aerial Training Images and 20% with Tourist Images; such that the total amount of images remains the same
#### Test Set

The testing set will remain fixed throughout the entire experiment and none of these samples will have been seen in training. The testing set is made up of
 - 20% of the Balanced Geoguessr Dataset
 - 20% of the OpenAerialMap Dataset
 - 20% of the BigFoto Dataset

#### Hypothesis
Our Hypothesis is that the model trained on the balanced Geoguessr Dataset will perform better on the testing set as it has a more balanced distribution and will avoid overfitting to the USA. Furthermore, we believe that if we include Aerial Images and Tourist Images into our training set then the model will perform better overall on the testing data, but in particular on the Aerial and Tourist Images in the testing data.