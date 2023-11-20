import torch
import torch.nn as nn

class SimilarityClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimilarityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x1, x2):
        x = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0)), dim=0)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

# Example usage
input_size = 10  # Size of input lists
hidden_size = 16  # Size of hidden layer

model = SimilarityClassifier(input_size, hidden_size)

list1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float()
list2 = torch.tensor([1, 3, 2, 4, 6, 5, 8, 7, 9, 10]).float()

pred = model(list1, list2)
print( [t.detach().numpy() for t in pred] )