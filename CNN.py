#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cat, no_grad, manual_seed
from torch.utils.data import Dataset, DataLoader
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)

import torchvision.models as models
from torchvision import datasets, transforms

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images from 'yes' folder
        fire_dir = os.path.join(root_dir, 'yes')
        for img_name in os.listdir(fire_dir):
            if img_name.endswith('.png') or img_name.endswith('.jpg'): 
                self.images.append(os.path.join(fire_dir, img_name))
                self.labels.append(1)  # Label for yes

        # Load images from 'no' folder
        no_fire_dir = os.path.join(root_dir, 'no')
        for img_name in os.listdir(no_fire_dir):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                self.images.append(os.path.join(no_fire_dir, img_name))
                self.labels.append(0)  # Label for no

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize(130),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = BrainDataset(root_dir='./data/Br35H/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = BrainDataset(root_dir='./data/Br35H/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

val_dataset = BrainDataset(root_dir='./data/Br35H/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


# In[3]:


n_samples_show = 6

data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[n_samples_show - 1].imshow(images[0, 0].numpy().squeeze(), cmap="gray")
    axes[n_samples_show - 1].set_xticks([])
    axes[n_samples_show - 1].set_yticks([])
    axes[n_samples_show - 1].set_title("Labeled: {}".format(targets[0].item()))

    n_samples_show -= 1


# In[3]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d()
        
        # Adjust the input size of fc1 based on the output size of conv4
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.cat((x, 1 - x), -1)

model4 = Net()


# In[5]:


# Define model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model4.to(device)

optimizer = optim.NAdam(model4.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

# Start training
epochs = 300  # Set number of epochs
loss_list = []  # Store loss history
model4.train()  # Set model to training mode

for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = model4(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss
    loss_list.append(sum(total_loss) / len(total_loss))
           
    avg_train_loss = sum(total_loss) / len(total_loss)

    model4.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model4(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)   

    val_accuracy = 100. * correct / total

    with open("training_performance_cnn.txt", "a") as file:
        file.write(f"{epoch+1}, {avg_train_loss:.4f}, {val_accuracy:.2f}%\n")

    print(f"Epoch: {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    if epoch > 2:
        if loss_list[-1] < min(loss_list[:-1]):
            torch.save(model4.state_dict(), "cnn.pt")
            print("Saved")
    
    model4.train()
            


# In[6]:


# Plot loss convergence
plt.plot(loss_list)
plt.title("Hybrid NN Training Convergence")
plt.xlabel("Training Iterations")
plt.ylabel("Neg. Log Likelihood Loss")
plt.show()


# In[4]:


model4.load_state_dict(torch.load("cnn.pt"))

precision_list = []
recall_list = []
f1_list = []

model4.eval()  # set model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output = model4(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        all_preds.extend(pred.view(-1).tolist())
        all_targets.extend(target.view(-1).tolist())

    accuracy = 100. * correct / total


    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    print(
        "Performance on test data:\n\tAccuracy: {:.1f}%\n\tPrecision: {:.1f}%\n\tRecall: {:.1f}%\n\tF1 Score: {:.1f}%".format(
            accuracy, precision * 100, recall * 100, f1 * 100
        )
    )

with open("model_performance_cnn.txt", "w") as file:
    file.write("Performance on test data:\n")
    file.write("\tAccuracy: {:.1f}%\n".format(accuracy))
    file.write("\tPrecision: {:.1f}%\n".format(precision * 100))
    file.write("\tRecall: {:.1f}%\n".format(recall * 100))
    file.write("\tF1 Score: {:.1f}%\n".format(f1 * 100))


# In[8]:


# Plot predicted labels

n_samples_show = 8
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model4.eval()
with no_grad():
    for batch_idx, (data, target) in enumerate(val_loader):
        if count == n_samples_show:
            break
        output = model4(data[0:1])
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)

        axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title("Predicted {}".format(pred.item()))

        count += 1


# In[ ]:





# In[ ]:




