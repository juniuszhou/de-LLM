import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# try to get GPU device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # add the functions will be used in the model (nn.Sequential)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # first layer, linear transform data to 512 output
            nn.Linear(28*28, 512, bias=True),
            # second layer, nn.ReLU() is activation function. output of 0 or 1
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            # final output, 10 classes
            nn.Linear(1024, 10)
        )

# defines HOW DATA FLOWS through your neural network
    def forward(self, x):
        # flatten input tensor from (N, 1, 28, 28) to (N, 784)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        # output of network
        return logits

# put the model on the GPU
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
for param in model.parameters():
    print(param.shape)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # set the model to training mode, model also can be evaluation mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # X: (N, 1, 28, 28) include 64 images each with 1 channel, 28x28 pixels
        # y: (N) include N labels
        X, y = X.to(device), y.to(device)
        # print(X.shape, y.shape)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        # optimizer updates weights using the computed gradients once
        optimizer.step()
        # Without zero_grad(), gradients keep adding up from previous iterations!
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# save the model
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     x = x.to(device)
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')