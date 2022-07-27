import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SiameseNetworkDataset(object):
    def __init__(self, imageFolderDataset):
        self.imageFolderDataset = imageFolderDataset


    def get_tuple(self, same):
        labels = os.listdir(self.imageFolderDataset)

        img0_label = random.choice(labels)

        img_0_label_path = os.path.join(self.imageFolderDataset, img0_label)
        img0_images = os.listdir(img_0_label_path)
        img_0_path_image = random.choice(img0_images)
        img0_image_path = os.path.join(img_0_label_path, img_0_path_image)

        img0_image = Image.open(img0_image_path)

        img0 = img0_image

        if same:
            img1_label = img0_label
            img_1_label_path = os.path.join(self.imageFolderDataset, img1_label)
            img1_images = os.listdir(img_1_label_path)
            img1_images.remove(img_0_path_image)
            img1_image_path = os.path.join(img_1_label_path, random.choice(img1_images))

            img1_image = Image.open(img1_image_path)
            img1 = img1_image

        else:
            tmp_labels = labels
            tmp_labels.remove(img0_label)
            img1_label = random.choice(tmp_labels)

            img_1_label_path = os.path.join(self.imageFolderDataset, img1_label)
            img1_images = os.listdir(img_1_label_path)

            img1_image_path = os.path.join(img_1_label_path, random.choice(img1_images))
            img1_image = Image.open(img1_image_path)
            

            img1 = img1_image

        return (img0_label, img1_label), (np.array(img0),np.array(img1))

    def get_batch(self, batch_size, all_equal=False):
        if all_equal:
            labels = np.ones(batch_size)
        else:
            labels_0 = np.zeros(batch_size//2)
            labels_1 = np.ones(batch_size//2)
            labels = np.concatenate([labels_0, labels_1])
        np.random.shuffle(labels)
        
        labels_batch = []
        images_batch = []
        for label in labels:
            t_label, t_image = self.get_tuple(same=label)

            labels_batch.append(t_label)
            images_batch.append(t_image)

        return (labels_batch, images_batch)

    def display_batch(self, batch):
        
        n = len(batch[0])
        plt.figure(figsize=(n+2,n+2))
        labels, images = batch[0], batch[1]
        idx = 0
        for lb, li in zip(labels, images):

            idx += 1
            plt.subplot(n,2, idx)
            plt.title(lb[0])
            plt.imshow(li[0])
            plt.axis('off')

            idx += 1

            plt.subplot(n,2, idx)
            plt.title(lb[1])
            plt.imshow(li[1])
            plt.axis('off')

        plt.show()



"""
https://www.kaggle.com/code/bulentsiyah/plant-disease-using-siamese-network-keras
https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/
https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/
https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/siamese_contrastive.ipynb#scrollTo=GUB5YX61sYlU
https://keras.io/examples/vision/siamese_contrastive/



"""
