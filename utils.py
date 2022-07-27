import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms




transformation =  transforms.Compose([transforms.Resize((256,128)), #105, 105
                                     transforms.ToTensor()
                                    ])

def separate_in_rows(images):
    return


def transform_path(positive_path, test_path):
    positive = transformation(Image.fromarray(positive_path.astype('uint8'), 'RGB')) 
    test = transformation(Image.fromarray(test_path.astype('uint8'), 'RGB'))

    #positive, test =  positive.cuda(), test.cuda()
    positive, test =  positive, test

    return positive, test

def calc_distance(pos, neg, model):
    pos, neg = transform_path(pos, neg)
    v_pos = model(pos[None, ...])
    v_neg = model(neg[None, ...])
    euclidean_distance = F.pairwise_distance(v_pos, v_neg, keepdim = True).item()
    return euclidean_distance

def calc_distance_features(pos, neg, extractor):
    v_pos = extractor(pos)
    v_neg = extractor(neg)
    euclidean_distance = F.pairwise_distance(v_pos, v_neg, keepdim = True).item()
    return euclidean_distance
