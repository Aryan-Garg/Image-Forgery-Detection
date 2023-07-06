#!/usr/bin/python

# import torch
import torch.nn as nn
# import torchsummary

class ShallowCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 3)
        
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU(True)

        self.init_xavier_weights()


    def init_xavier_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(self.relu(self.conv6(x))) # bs x 128 x 2 x 2 
        # print(x.shape)
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x)) 
        # print(x.shape)
        return x # bs x 3

# test_m = ShallowCNN()
# print(torchsummary.summary(test_m.to(torch.device("cuda")), (3, 256, 256)))