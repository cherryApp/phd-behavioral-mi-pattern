import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

np.random.seed(0)
torch.manual_seed(0)

def conditionator(x):
    return (x[0] < 35
            and x[1] < 45
            and x[2] < 60
            and x[3] < 75)

def dataset_generator(num_samples=10000):
    X = np.random.randint(1, 100, size=(num_samples, 10), )
    y = [1 if conditionator(item) else 0 for item in X]

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)
    return X, y
X, y = dataset_generator()

# define the model
class BehavioralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(10, 1, bias=True, device=device),
            nn.Sigmoid()
        )
        # self.hidden1 = nn.Linear(8, 12)
        # self.relu = nn.ReLU()
        # self.hidden2 = nn.Linear(12, 8)
        # self.output = nn.Linear(8, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.relu(self.hidden1(x))
        # x = self.relu(self.hidden2(x))
        # x = self.sigmoid(self.output(x))
        x = self.linear_stack(x)
        return x

model = BehavioralNetwork().to(device)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 4
batch_size = 100

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch + 1}, latest loss {loss}')

# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# Test the model with new samples
Xt, yt = dataset_generator(num_samples=100000)
predictions = (model(Xt) > 0.5).int()
fails = 0
positives = 0
pred_positive = 0
for i in range(0, len(Xt)):
    # print('%s => %d (expected %d)' % (Xt[i].tolist(), predictions[i], yt[i]))
    if yt[i] == 1:
        positives += 1
        
    if predictions[i] == 1:
        pred_positive += 1
        
    if yt[i] != predictions[i]:
        fails += 1
        # print('%s => %d (expected %d)' % (Xt[i].tolist(), predictions[i], yt[i]))
    
print(
    'Fails: %d/%d | Positives: %d | Positive predictions: %d' 
    % (fails, len(Xt), positives, pred_positive)
)
    
print("Done!")

# Saving Models
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")