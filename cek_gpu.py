import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml

# Setup device
device = torch_directml.device() if torch_directml.is_available() else "cpu"
print(f"Training on: {device}")

# Model sederhana
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.fc1 = nn.Linear(16*30*30, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# Pindah model ke GPU AMD
model = SimpleCNN().to(device)

# Training loop sederhana
data = torch.randn(32, 3, 32, 32).to(device)  # Batch ke GPU
target = torch.randint(0, 10, (32,)).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

# Forward/backward di GPU AMD
output = model(data)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
