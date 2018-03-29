import torch
# return the contents of the  train file.
def open_file(filename):
    return open(filename).read()

def get_chars(data):
    l = [ele for ele in data]
    return l

train = open_file('data.txt')
n_chars = len(list(set(get_chars(open_file('data.txt')))))
print(n_chars)
print('Data loaded, total chars in train data is ',len(get_chars(train)))
chars = list(set(get_chars(open_file('data.txt'))))

def lettertoonehot(letter):
    res = torch.zeros(n_chars, 1)
    temp = list(set(get_chars(open_file('data.txt'))))
    for i in range(len(temp)):
        if letter == temp[i]:
            res[i] = 1
            return res

def onehottoletter(onehot):
    temp = onehot.numpy()
    for i in range(len(temp)):
        if temp[i] == 1:
            return chars[i]

# train size = 50 onehot vectors

def load_data(string):
    l = get_chars(string)
    X = []
    Y = []
    temp = []
    for i in range(len(l) - 20000):
        for j in range(i, i+10):
            temp.append(lettertoonehot(string[j]))
        Y.append(lettertoonehot(string[i+50]))
        X.append(temp)
        temp = []
    return X, Y


X, Y = load_data(open_file('data.txt'))
print("X and Y successfully loaded!")

import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        h = self.softmax(self.h2h(hidden) + self.i2h(input))
        o = self.softmax(self.h2o(h))
        return o

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

rnn = RNN(n_chars, 90, n_chars)
criterion = nn.NLLLoss()
learning_rate = 0.05
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
hidden = rnn.init_hidden()
epochs = 1

for epoch in range(epochs):
    for i in range(len(X)):
        for ele in X[i]:
            output, hidden = rnn(Variable(ele.t()), hidden)
        loss = criterion(output, Variable(Y[i]))

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        if (i % 1000 == 0):
            print('Current loss is: ', loss)


print(rnn(Variable(X[2][2].t()), hidden))

