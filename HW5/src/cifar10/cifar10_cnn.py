import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import numpy

# Hyper Parameters
EPOCH = 2
BATCH_SIZE = 5
LEARNING_RATE = 1e-3
MOMENTUM = 0.9



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x



if __name__ == "__main__":
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR10(
        root='./dataset',
        train=True,
        transform=transform,
        download=True
    )
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./dataset/',
        train=False,
        transform=transform
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    cnn = CNN()
    print("\n* [CNN Generated]")
    print(cnn)

    optimizer = torch.optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_function = nn.CrossEntropyLoss()

    print("\n* [Training]")
    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader, 0):
            batch_x, batch_y = data
            output = cnn(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 2000 == 0:
                print("Epoch:", epoch,
                      "| Step: %5d" % step,
                      "| train loss: %.4f" % loss)

    
    correct = 0
    total = 0
    confusion_matrix = numpy.zeros([10, 10], int)
    with torch.no_grad():
        for data in test_loader:
            batch_x, batch_y = data
            output = cnn(batch_x)
            _, predicts = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicts == batch_y).sum().item()
            for i, j in enumerate(batch_y):
                confusion_matrix[j.item(), predicts[i].item()] += 1

    accuracy = correct / total
    print("\n* [Testing]")
    print("total accuracy:", accuracy)
    print("Category".rjust(10), "- Accuracy")
    for i, j in enumerate(confusion_matrix):
        print(classes[i].rjust(10), "-", j[i] / numpy.sum(j) * 100)