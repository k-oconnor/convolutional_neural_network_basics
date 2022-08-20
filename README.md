# Building a Multi-Class Convolutional Neural Network with PyTorch
This README demonstrates how to set up a basic convolutional neural network for mult-class image classification

## _Butterfly Image Classification - 75 Species_
### DATA: https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species

## What is a tensor?
A tensor is often thought of as a generalized matrix. That is, it could be a 1-D matrix (a vector is actually such a tensor), a 3-D matrix (something like a cube of numbers), even a 0-D matrix (a single number), or a higher dimensional structure that is harder to visualize. The dimension of the tensor is called its rank.

But this description misses the most important property of a tensor!

A tensor is a mathematical entity that lives in a structure and interacts with other mathematical entities. If one transforms the other entities in the structure in a regular way, then the tensor must obey a related transformation rule.

## What do tensors have to do with images?
When we transform an image into a tensor, each of those pixels in the position they hold in the image, get transformed into tensors. Once we transform these tensors, we can train our models with this data, among other things. We can mirror images, resize, crop, and manipulate in other ways that are useful to us.

In the case of RGB images we have three channels; red, green, and blue and each of these channels contains the intensity value of pixels in the image in that particular color. This can be represented by an order three tensor and there are two major ordering conventions.

## What is a kernel?
The kernel is a matrix that moves over the input data, performs the dot product with the sub-region of input data, and gets the output as the matrix of dot products. Kernel moves on the input data by the stride value.

## What is a convolution? 
-  In most simple terms, a convolution is like sliding a filter over an image
- Convolution is a mathematical operation on two functions that produces a third function that expresses how the shape of one is modified by the other.
- In image processing, convolution is the process of transforming an image by applying a kernel over each pixel and its local neighbors across the entire image. The kernel is a matrix of values whose size and values determine the transformation effect of the convolution process.
- The Convolution Process involves these steps. (1)It places the Kernel Matrix over each pixel of the image (ensuring that the full Kernel is within the image), multiplies each value of the Kernel with the corresponding pixel it is over. (2)Then, sums the resulting multiplied values and returns the resulting value as the new value of the center pixel. (3) This process is repeated across the entire image.

![A Visual Representation of Convolution](Images/conv.png)

- As we see in the picture above, a 3x3 kernel is convoluted over a 7x7 source image. Center Element of the kernel is placed over the source pixel. The source pixel is then replaced with a weighted sum of itself and surrounding pixels. The output is placed in the destination pixel value. In this example, at the first position, we have 0 in source pixel and 4 in the kernel. 4x0 is 0, then moving to the next pixel we have 0 and 0 in both places. 0x0 is 0. Then again 0x0 is 0. Next at the center there is 1 in the source image and 0 in the corresponding position of kernel. 0x1 is 0. Then again 0x1 is 0. Then 0x0 is 0 and 0x1 is 0 and at the last position it is -4x2 which is -8. Now summing up all these results we get -8 as the answer; so the output of this convolution operation is -8. This result is updated in the Destination image.

## What is a convolutional neural network (cnn) ?
- A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain. 
- It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain. 
- It creates an adaptive system that computers use to learn from their mistakes and improve continuously.
- A cnn is a class of neural network that uses convolutional layers for analysis of visual data

> Neural Networks receive an input (a single vector), and transform it through a series of hidden layers.  
> Each hidden layer is made up of a set of neurons, where each neuron is fully connected to all neurons in 
> the previous layer, and where neurons in a single layer function completely independently and do not share 
> any connections. The last fully-connected layer is called the “output layer” and in classification settings 
> it represents the class scores.

> Regular Neural Nets don’t scale well to full images. In CIFAR-10, images are only of size 32x32x3 (32 wide, 
> 32 high, 3 color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural 
> Network would have 32*32*3 = 3072 weights. This amount still seems manageable, but clearly this 
> fully-connected structure does not scale to larger images. For example, an image of more respectable size, 
> e.g. 200x200x3, would lead to neurons that have 200*200*3 = 120,000 weights. Moreover, we would almost  
> certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full 
> connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

![An Example CNN Diagram](Images/CNN_example.jpeg)

## What are the building blocks of a cnn in PyTorch?
- **Imaging preprocessing scheme:** Preprocessing refers to all the transformations on the raw data before it is fed to the machine learning or deep learning algorithm. Images usually are not represented in a way that is easy for a network to make sense of raw. As such, it is in most all cases, a requirement for a well performing model to have inputs transformed into a way that eccentuates the features that are most important for the problem at hand.

- **Input Layer:** Holds the raw pixel values of the image.

- **Convolutional Layer(s):** will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. 

- **Rectified Linear Unit Function Layer(s):** will apply an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged.

- **Pooling Layer(s):** Performs downsampling operations along the spatial dimensions (width, height).

- **Fully Connected Layer(s):** Computes the class scores where each of the numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.

- **Loss Function:** The loss function is the criterion on which the network measures success. For example, in simple linear regression, we try to minimize mean square error. There are numerous loss funcitons that can be used for different applications. Binary cross-entropy loss is a commonly used for single-class prediction. Cross-entropy loss can be extended for use in multi-class classification.

- **Optimizer:** The optimizer is a function or algorithm that updates the network parameters at each epoch, and is integral to the learning process. The weights of the parameters are increased or decreased progressively in attempt to reduce loss on the next pass.

## Problem Description and Data
For this exercise, we will be using the "Butterfly Image Classification 75 species" available on Kaggle.com
It contains 9285 training, 375 testing,and 375 validation images all in 224 X 224 X 3 jpeg format.

Our objective is to train a convolutional neural network that has the greatest accuracy in predicting correct species/classes over the testing image set.

We will do this by training the model on the 9285 training images, using the model to make predictions on the testing data, and running a basic Sci-Kit Learn classification report.

# The Steps 
- Preprocessing and Loading
- Model Constructor and Intializing
- Selecting Optimizer and Loss Metric
- Training, Validation, and Tuning
- Making Predictions on Test Data

## *Preprocessing and Loading*
PyTorch offers a number of built-in transformation types that we can use sequentially, or in composition. 
As we see in the image below, I first specify a *semi* arbitrary mean and standard deviation normalization pointer, which we can use in the composition.

We use different compositions for the training data than for both testing and validation data.

**Composition(Training):**
- RandomResizedCrop: Crop a random portion of image and resize it to a given size.
- RandomHorizontalFlip: At a certain probability, an image will be flipped on it's horizontal axis
- ToTensor: Converts the image data into tensor form
- normalize: Changes the range of pixel intesities by the mean and standard deviation specified for each channel

**Composition(Validation and Testing):**
- Resize and CenterCrop: By increasing the image size, and center cropping, we are effectively zooming in on the center of the image by a small degree. 
- ToTensor: Converts the image data into tensor form
- normalize: Changes the range of pixel intesities by the mean and standard deviation specified for each channel

![Normalization](Images/code1.png)

To illustrate this process, we will implement a simple plotting function to show 25 random images from a random batch. By commenting out the "normalize" pointer in the compose, and running the plotting twice, we can illustrate what the transforms have done to the training data.

![Data Loaders and Visualizing a Batch](Images/code2.png)

## Visualizing 25 un-normalized images
![Visualizing a Random Batch Un-normalized](Images/un_norm_im.png)

## Visualizing 25 normalized images
![Visualizing a Random Batch Normalized](Images/norm_im.png)



## *Model Constructor and Intializing*
The model constructor is where we make our specifications for the neural network. First, we make our linear layers. The forward method can be called to make a prediction. There are three typical activation functions 'relu, tanh, sigmoid'. Relu is useful, as sigmoid and tanh are bounded <|1|, so as multiple gradients are multiplied together, the number doesn't neccesarily approach 0. We apply the relu function in the hidden layers, and use a sigmoid activation in the output layer to make predictions. Passing the linear function through the sigmoid activation will result in output being the probability of survival, analogous to a logistic regression. Note how there are two hidden layers. We can generalize this construction to an arbitrary number of layers, by adding more linear functions and activations to the __init__ and the forward pass method. 

Reminder: A neuron is a combination of a linear function and an activation
- Input dimension is the number of explanatory variables
- H1 is the number of neurons in the first hidden layer
- H2 is the number of neurons in the second hidden layer
- Output dimesion is the number of classes we are predicting (In this case, it is binary)

![Model Specification](Images/code3.png)

Note how at the end of the constructor, we initialize the model with our desired specification. The number of neurons and hidden layers needs to be carefully tuned. Having too many neurons and layers can lead to extreme overfitting. Having too few will cause our paramter estimates to underfit. It is important to simultaneously train and validate to find the sweet spot. There is no perfect answer, and there is a lot of room for intuition. The model as specified takes the following form.

![Model Initialization](Images/model.png)

## *Selecting Optimizer and Loss Metric*
![Loss Function and Optimizer](Images/code4.png)

## *Training, Validation, and Tuning*
![Training the Model](Images/code5.png)

![Training Results](Images/Fig1.png)

As we see in the graph above, loss steadily decreases as we approach 100 epochs. We then observe loss flucuating relatively extremely. This tipping point is evidence of overfitting past 100 epochs, as the network begins to chase changes that are random variation in the batches. So, we would most likely choose to re-run the model with 100 epochs to get the most reliable results on the testing data.

## *Making Predictions on Test Data*
![Making Predictions](Images/code6.png)
![Classification Report](Images/Results1.png)
