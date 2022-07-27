
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

DROPOUT = 0.2
out_dim = 64
class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()
        self.transformation =  transforms.Compose([transforms.Resize((105,105)), #105, 105
                                     transforms.ToTensor()
                                    ])

        #CNN Layers
        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=10),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64,128, kernel_size = 7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128,128, kernel_size = 4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128,256, kernel_size = 4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
        )

      #FCL

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
                        
            nn.Linear(2048, out_dim), # 4096, 256
            nn.ReLU(inplace=True),
            #nn.Dropout(0.25), # Koch2
            
            #nn.Linear(256,64), #Koch 2
            #nn.ReLU(inplace=True), #Koch 2
      )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def calc_distance(self, positive_path, test_path):
        positive = self.transformation(Image.fromarray(positive_path.astype('uint8'), 'RGB')) 
        test = self.transformation(Image.fromarray(test_path.astype('uint8'), 'RGB'))

        positive, test =  positive.cuda(), test.cuda()

        positive_out, test_out = self.forward(positive[None, ...], test[None, ...])

        euclidean_distance = F.pairwise_distance(positive_out, test_out, keepdim = True)

        return euclidean_distance.item()


class ConstrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

        
transformation = transforms.Compose([transforms.Resize((105,105)), #105, 105
                                     transforms.ToTensor()
                                    ])
