import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LEARNING_RATE = 1e-3



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(              # convolutional layer
                in_channels=1,      ## height of the image
                out_channels=16,    ## number of scanning filter
                kernel_size=5,      ## size of the scanning filter
                stride=1,           ## length of scanning step
                padding=2,          ## boundary of the field
            ),
            nn.ReLU(),              # neural network
            nn.MaxPool2d(           # pooling layer
                kernel_size=2       ## 2*2 scale
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.out = nn.Linear(32 * 7 * 7, 10)    # 32 leyer, 28/2/2=7; 10 numbers

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # (32, 7, 7)
        x = x.view(x.size(0), -1)   # (32 * 7 * 7)
        output = self.out(x)
        return output



if __name__ == "__main__":
    # dataset prepare
    train_data = torchvision.datasets.MNIST(
        root='./dataset',
        train=True,
        transform=torchvision.transforms.ToTensor(), # domain (0, 1)
        download=True
    )

    print("\n* [Dataset Info]")
    print("training dataset data size:", train_data.data.size())
    print("training dataset label size:", train_data.targets.size())

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_data = torchvision.datasets.MNIST(
        root='./dataset/',
        train=False
    )
    test_x = Variable(
        torch.unsqueeze(
            test_data.data,
            dim=1
        ),
    ).type(torch.FloatTensor)[:2000]/255.
    test_y = test_data.targets[:2000]


    cnn = CNN()
    print("\n* [CNN Generated]")
    print(cnn)

    # optimize parameters
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    # train & test
    print("\n* [Training]")
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            batch_x = Variable(x)
            batch_y = Variable(y)
            output = cnn(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if step % 100 == 0: 
                    test_output = cnn(test_x)
                    predicts = torch.max(test_output, 1)[1].data.numpy()
                    accuracy = float((predicts == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                    print("Epoch:", epoch, 
                          "| Step: %4d" % step, 
                          "| train loss: %.4f" % loss, 
                          "| predict accuracy: %.4f" % accuracy)
    
    print("\n* [Testing]")
    test_output = cnn(test_x[:30])
    preditcs = torch.max(test_output, 1)[1].data.numpy()
    print("predicts:", preditcs)
    print("actually:", test_y[:30].numpy())
    print("accuracy:", sum(preditcs == (test_y[:30].numpy())) / 30.)
