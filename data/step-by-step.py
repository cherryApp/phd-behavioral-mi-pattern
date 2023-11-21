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

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.data.txt', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)


# define the model
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(8, 12, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(12, 8, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(8, 1, bias=True, device=device),
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

model = PimaClassifier().to(device)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 70
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
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))