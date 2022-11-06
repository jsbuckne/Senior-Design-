from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 convolutional blocks / flatten layer / linear layer / softmax
        self.conv1 = nn.Sequential(
            #use Sequential as a container to put layers
            nn.Conv2d(
                # Number of input channels:
                in_channels=1,
                # Number of output channels:
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            #use Sequential as a container to put layers
            nn.Conv2d(
                # Number of input channels:
                in_channels=16,
                # Number of output channels:
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            #use Sequential as a container to put layers
            nn.Conv2d(
                # Number of input channels:
                in_channels=32,
                # Number of output channels:
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            #use Sequential as a container to put layers
            nn.Conv2d(
                # Number of input channels:
                in_channels=64,
                # Number of output channels:
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        #flatten  multi-dimensional output of the last convolutional layer, conv4:
        self.flatten = nn.Flatten()
        #Now thqt data is flattened, can pass it to a linear layer:
        self.linear = nn.Linear(128 * 5 * 4, #shape of data from last convolutional output. 128 output channels
                                3)          #output we expect, equal to number of classes in dataset
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        #Telling Pytorch how to pass data from one layer to the next:
        x = self.conv1(input_data)
        #Passes data (x) to conv2, conv3, and conv4, flatten:
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn, (1, 64, 44))


#CNN algorithm with 4 convolutional layers, a flatten layer to make the multidimensional input one-dimesnional.


