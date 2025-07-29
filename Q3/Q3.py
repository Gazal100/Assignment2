import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load MNIST dataset from CSV
train_df = pd.read_csv('Assignment2\Q3\mnist_train.csv')
test_df = pd.read_csv('Assignment2\Q3\mnist_test.csv')

# Split into features and labels
# Features (X_train, X_test) are the input data (pixel values of the images) 
# that the model uses to learn patterns.
# Labels (y_train, y_test) are the target outputs (the actual digit each image represents)
# that the model tries to predict.
# This separation is essential for supervised learning, where the model learns to map inputs (features) to outputs (labels) during training, 
# and then is evaluated on how well it predicts labels for new, unseen features.
X_train = train_df.iloc[:, 1:].values / 255.0
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values / 255.0
y_test = test_df.iloc[:, 0].values

# Convert to PyTorch tensors
# DataLoaders require data in tensor format. 
# Tensors are the main data structure PyTorch uses for efficient computation 
# and automatic differentiation during training and evaluation.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Wrap in DataLoader
# A DataLoader in PyTorch is like a helper that automatically gives 
# your model small batches of data, shuffles them if needed, 
# and makes training faster and easier. 
# It handles loading the data in manageable pieces 
# so you don’t have to do it manually.
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# Define neural network
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, optimizer
model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "mnist_model.pth")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
