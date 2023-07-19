import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.bn1(self.fc1(x)))
        x = self.sigmoid(self.bn2(self.fc2(x)))
        x = self.sigmoid(self.bn3(self.fc3(x)))
        x = self.sigmoid(self.bn4(self.fc4(x)))
        x = self.sigmoid(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x


if __name__ == '__main__':
    input_size = 4
    output_size = 3
    model = MLP(input_size, output_size)

    input_data = torch.randn(32, input_size)

    output = model(input_data)
    _, predicted_labels = torch.max(output, dim=1)
    print(output.size())
    print(predicted_labels)
