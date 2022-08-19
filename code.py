import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from pytictoc import TicToc

## --------------------- data loading and preprocessing  ---------------------##

# Assigning a standard transform scheme to a pointer
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

## Setting default batch size and number of workers. As I do not have a GPU, I set num_workers to zero.
batch_size = 32
num_workers = 0

## Setting the paths for the data sets
working_directory = 'data'
train_path = os.path.join(working_directory, 'train')
test_path = os.path.join(working_directory, 'test')
valid_path = os.path.join(working_directory, 'valid')
class_names = os.listdir('data/train')

## Loading data into dataset objects, standardizing size, transforming images to tensor, and normalizing images
## This is completed for each of the files. (Training, Validation, Testing)

train_dataset = datasets.ImageFolder(
    train_path,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
test_dataset = datasets.ImageFolder(
    test_path,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
val_dataset = datasets.ImageFolder(
    valid_path,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))


## DataLoader takes our dataset objects, and turns them into iterables. 
## Batch-size determines how many row tensors are passed to the model in each epoch.
## Setting shuffle to true, ensures that each batch is a random sample.
## NOTE: When we call our testing data, we want to pull every image at once, so batch size is set to 375.


train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=375,num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers,shuffle=True)


## This is a simple plotting function
## NOTE: To get the original un-normalized images, call this function with the transform "normalize," line commented out.

def show_images(img,label):
    plt.figure(figsize = [10,7])        ## Setting bounds on image size
    for i in range(25):                 ## Selecting 25 images
        plt.subplot(5,5,i+1)            ## This utility wrapper makes it convenient to create common layouts of subplots in a single call.
        plt.imshow(img[i].permute(1,2,0))                               ## Coercing tensor shape from "([3, 224, 224])" to "([224, 224, 3])"
        plt.title(class_names[label[i]],fontdict = {'fontsize': 6})     ## Organizing common names for images, and selecting font-size
        plt.axis('off')                                                  ## Turning off messy axis due to clipping
    plt.show()                          ## Show the plot!

## Grabbing some random training images
dataiter = iter(train_loader)       ## Making the loader into an iterable object
images, labels = dataiter.next()    ## Iterate once over the training loader to grab a batch of images
show_images(images,labels)          ## Passing the batch through the plotting function

## --------------------- constructing and intializing the model  ---------------------##
## Building our neural net
## Convolutional Neural Net constructor - Input is a batch of tensors with three channel RGB intensity for each pixel as features, and output as class predictions
## To avoid the vanishing gradient problem, we use rectified linear unit activations on the convolutional layers.

## Sizes at each step:
#   input img: (32, 3, 224, 224) -> (batch size, RGB channels, img height, img width)
#   conv1:      torch.Size([32, 6, 220, 220])
#   ReLU1:      torch.Size([32, 6, 220, 220])
#   maxpool1:   torch.Size([32, 6, 110, 110])
#   conv2:      torch.Size([32, 12, 106, 106])
#   ReLU2:      torch.Size([32, 12, 106, 106])
#   maxpool2:   torch.Size([32, 12, 53, 53])
#   flatten:    torch.Size([32, 33708])
#   output:     torch.Size([32, 75])


## We pass the two convolutional layers through ReLU activations, pooling functions, flatten the final pooling layer,...
## and pass through a linear function and the final linear output layer through no activation.
## From the output, with 75 values (one for each class), the maximum value corresponds to the most likely class for each image.

class CNN(nn.Module):
    def __init__(self,out_1,out_2):
        super(CNN, self).__init__()
            
        self.conv1 = nn.Conv2d(in_channels=3, out_channels = out_1, kernel_size = 5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=out_1, out_channels = out_2, kernel_size = 5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2*53*53,75)
    
    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

## We call the "CNN" class to initialize the model. CNN(Output1_channels, Output2_channels)
net = CNN(6,12)

## --------------------- selecting optimizer and loss function  ---------------------##

## Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

## After computing the gradients for all tensors in the model, calling optimizer. step() makes the optimizer 
## iterate over all parameters (tensors values). The optimizer is supposed to update and use their internally stored grad to update their values.
## Learning rate is a key hyperparameter that determines how fast the network moves weights to gradient minima.
## Weight decay is an optional hyperparameter which progressivly reduces |weights| each epoch, in effect penalizing overfitting.
optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

## --------------------- training the model ---------------------##

## We initialize two empty lists to append loss from each epoch to
loss_list = []              
val_loss_list = []

## Creating a timer to benchmark speed of different training choices
t = TicToc()                ## Create instance of class
start_time = t.tic()        ## Start timer

# Run the training loop
for epoch in range(500):    ## By inputing the range(x), we are choosing 'x' epochs to iterate over the training data

    print(f'Starting epoch {epoch+1}')  ## Print which epoch is begining
    
    for i, data in enumerate(train_loader):       ## Iterate over the DataLoader for training data
      inputs, targets = data        ## Obtain samples for each batch
      optimizer.zero_grad()         ## Zero out the gradient
      outputs = net(inputs)         ## Perform forward pass (Make predictions)
      #predicted_class = np.argmax(outputs.detach(),axis=-1)     ## Isolating predicted class. Useful for debugging. Commenting out for speed.
      loss = criterion(outputs, targets)        ## Calculate loss
      loss.backward()             ## Perform backward pass (differentiate loss w.r.t parameters)
      optimizer.step()            ## Perform optimization (update parameters)

    with torch.no_grad():         ## since we're validating and not training, we don't need to calculate the gradients for our outputs
        for w,z in val_loader:                 ## Obtain samples for each batch
            y_val_hat = net(w)                 ## Make a prediction    
            val_loss = criterion(y_val_hat,z)  ## Calculate loss

    ## At each epoch, we append the calculated loss to a list, so we can graph it's change over time...
    loss_list.append(loss.item())
    val_loss_list.append(val_loss.item())

## Process is complete!
print('Training process has finished.')

## Stop timer: This will output the time it took to iterate over the data for every epoch.
end_time = t.toc()
print("end time", end_time)

## A simple plotting function for showing loss changes over time as parameters are updated...
plt.plot(loss_list, linewidth=.5)
plt.plot(val_loss_list, linewidth =.5)
plt.legend(("Training Loss", "Validation Loss"))
plt.xlabel("Epoch")
plt.ylabel("CE Loss")
plt.show()

## --------------------- making predictions from our trained model ---------------------##

## We initialize two lists to place our class predictions in and the respective ground truth classes.
y_ground = []
y_pred = []

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for x,y in test_loader:     ## Obtain samples for each batch (in this case, the batch is every image in the testing set, and there is only one batch)
        pred = net(x)           ## Make a prediction 
        predicted_class = np.argmax(pred.detach(),axis=-1)      ## Isolating predicted class
        
        ## Appending the tensor values for predictions and ground truth to the lists
        for i in range(len(predicted_class)):               
            y_ground.append(y[i].item())                     
            y_pred.append(predicted_class[i].item())

## Making our final classification report for how well the model performs
print(classification_report(y_ground,y_pred, target_names = class_names, digits = 4))