import torch
import torch.nn as nn

class ComparisonNetwork(nn.Module):
    def __init__(self):
        super(ComparisonNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x1, x2):
        out1 = self.fc1(x1)
        out2 = self.fc1(x2)
        out = torch.abs(out1 - out2)
        out = self.fc2(out)
        return out
    
input_size = 10  # Update with your input size
hidden_size = 30  # Update with your desired hidden size
output_size = 1  # Output size is 1 as we are comparing the arrays
net = ComparisonNetwork()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Generate example input and output data
num_samples = 1000

# Training data
x1_train = torch.randn(num_samples, input_size)
x2_train = torch.randn(num_samples, input_size)
y_train = torch.randint(0, 2, (num_samples,)).unsqueeze(1).float()

# Validation data
x1_val = torch.randn(num_samples // 4, input_size)
x2_val = torch.randn(num_samples // 4, input_size)
y_val = torch.randint(0, 2, (num_samples // 4,)).unsqueeze(1).float()

for epoch in range(20):
    # Forward pass
    outputs = net(x1_train, x2_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    outputs = net(x1_val, x2_val)
    predicted = torch.round(torch.sigmoid(outputs))
    accuracy = (predicted == y_val).sum() / y_val.size(0)
    print('Validation Accuracy: {}'.format(accuracy))