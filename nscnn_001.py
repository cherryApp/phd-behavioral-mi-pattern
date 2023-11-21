# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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
xt = torch.from_numpy(X).float().to(device) # [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ...]
yt = torch.IntTensor(y).reshape(-1, 1).to(device) # [1, 0, 1, ...]

## The .float() and .long() methods are used to specify the data types of the tensors.
dataset = torch.utils.data.TensorDataset(xt, yt)

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-1, end_dim=1)
        self.linear_stack = nn.Sequential(
            nn.Linear(8, 12, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(12, 8, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(8, 1, bias=True, device=device),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_stack(x)

model = NeuralNetwork().to(device)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
batch_size = 100

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
def train(model, loss_fn, optimizer):
    for i in range(0, len(xt), batch_size):
        Xbatch = xt[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = yt[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Latest loss {loss}')

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
    train(model, loss_fn, optimizer)
    # test(dataset, model, loss_fn)
    
# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

print("Done!")

# Saving Models
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

