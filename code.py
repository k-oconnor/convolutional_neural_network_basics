import pandas as pd
import os
from sklearn.metrics import classification_report
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.optim import Adam

## --------------------- data loading and preprocessing  ---------------------##
## Loading and normalizing images.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


batch_size = 32
num_workers = 0

working_directory = 'data'
train_path = os.path.join(working_directory, 'train')
test_path = os.path.join(working_directory, 'test')
valid_path = os.path.join(working_directory, 'valid')
class_names = os.listdir('data/train')
print(class_names)

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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,num_workers=num_workers,shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=375,num_workers=num_workers)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size,num_workers=num_workers)

## Showing some images

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images

dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


## Building our neural net
class Net(nn.Module):
    def __init__(self,out_1,out_2):
        super(Net, self).__init__()
            # input img: (32, 3, 224, 224) -> batch, color, height, width
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

net = Net(6,12)


## Define loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=1e-4)


loss_list = []                      ## We initialize two empty lists to append loss from each epoch to
val_loss_list = []
  # Run the training loop
for epoch in range(500): # 500 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(train_loader):
      
      # Get inputs
      inputs, targets = data

      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = net(inputs)
      predicted_class = np.argmax(outputs.detach(),axis=-1)

      # Compute loss
      loss = criterion(outputs, targets)
      
      # Perform backward pass
      loss.backward()
      
      # Perform optimization
      optimizer.step()

    
    for w,z in val_loader:                     ## Obtain samples for each batch
        y_val_hat = net(w)                     ## Make a prediction
        val_loss = criterion(y_val_hat,z)      ## Calculate loss

    loss_list.append(loss.item())
    val_loss_list.append(val_loss.item())
  # Process is complete.
print('Training process has finished.')

plt.plot(loss_list, linewidth=.5)
plt.plot(val_loss_list, linewidth =.5)
plt.legend(("Training Loss", "Validation Loss"))
plt.xlabel("Epoch")
plt.ylabel("CE Loss")
plt.show()

y_ground = []
y_pred = []
with torch.no_grad():
    for x,y in test_loader:
        pred = net(x)
        predicted_class = np.argmax(pred.detach(),axis=-1)
        for i in range(len(predicted_class)):
            y_ground.append(y[i].item())
            y_pred.append(predicted_class[i].item())

print(classification_report(y_ground,y_pred, target_names = class_names, digits = 4))