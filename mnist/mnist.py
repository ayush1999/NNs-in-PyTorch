import torch
import torch.nn as nn
from touch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

input_size = 784        # 28 x 28 images
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.01

train_dataset = dsets.MNIST(root='/.data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(
    root='./data', transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Simple neural network (Feedforward)

class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        return out

model = net(input_size, hidden_size, num_classes)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28).cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

# Test on test data 
correct = 0
total = 0

for images,labels in test_loader:
    images = Variable(images.view(-1, 28*28).cuda())
    output = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()


print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))