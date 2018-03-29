import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

seq_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 3
learning_rate = 0.02

train_dataset = dsets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataset = dsets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.dataloader(dataset = train_dataset,
                                            batch_size = batch_size,shuffle = True)
test_loader = torch.utils.data.dataloader(dataset=test_dataset,
                                           batch_size=batch_size, shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        temp1 = self.i2h(combined)
        temp2 = self.i2o(combined)
        return self.softmax(temp2), temp1

rnn = RNN(input_size, hidden_size, num_classes)
RNN.cuda()

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr= learning_rate)

# Train : 

for epoch in range(num_epochs):
    for i, (images ,labels) in enumerate(train_loader):
        images = Variable(images.view(-1, sequence_length, input_size)).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))


