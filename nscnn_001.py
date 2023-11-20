# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Generate a simple numerical dataset from torchvision.datasets
## Set the random seed: For better reproducibility, set a random seed using np.random.seed() and torch.manual_seed():
np.random.seed(0)
torch.manual_seed(0)

## Generate the dataset: Use NumPy to generate random data points and corresponding labels.
# For example, let's generate a dataset with 100 samples, each having two features
# and a corresponding label:
num_samples = 10000
num_features = 2
X = np.random.randint(1, 100, size=(num_samples, 10), )
# y = np.random.randint(0, 2, size=(num_samples,))
y = [1 if item[0] < 50 else 0 for item in X]

## Convert the NumPy array to PyTorch tensors:
# At this point, you can convert the NumPy arrays to PyTorch tensors
# using torch.from_numpy() so that they can be used for training:
xt = torch.from_numpy(X).float() # [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ...]
yt = torch.IntTensor(y) # [1, 0, 1, ...]
## The .float() and .long() methods are used to specify the data types of the tensors.
dataset = torch.utils.data.TensorDataset(xt, yt)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-1, end_dim=1)
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(10, 10, bias=True, device=device),
            nn.Sigmoid(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.Sigmoid(),
            nn.Linear(10, 1, bias=True, device=device)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Optimizing the model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
def train(dataset, model, loss_fn, optimizer):
    size = len(dataset)
    model.train()
    for batch, (X, y) in enumerate(dataset):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# We also check the model’s performance against the test dataset to ensure we are not overfitting.
def test(dataset, model, loss_fn):
    size = len(dataset)
    num_batches = len(dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataset:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# The training process is conducted over several iterations (epochs). During each epoch, the model learns parameters to make better predictions. We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.
epochs = 2
for t in range(1, epochs + 1):
    print(f"Epoch {t}\n-------------------------------")
    train(dataset, model, loss_fn, optimizer)
    test(dataset, model, loss_fn)

print("Done!")

# Saving Models
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

