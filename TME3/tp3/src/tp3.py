from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Téléchargement des données MNIST
from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#  TODO: 

# Utilisez la fonction torch.min et torch.max sur l'ensemble des images MNIST
min_value = torch.min(torch.tensor(train_images))
max_value = torch.max(torch.tensor(train_images))

# Affichez les valeurs minimales et maximales
print(f"Valeur minimale : {min_value.item()}")
print(f"Valeur maximale : {max_value.item()}")


class MonDataset(Dataset):
    def __init__(self, images, labels):
        # On convertit l'ensemble des images en tensor pytorch 
        ## .unsqueeze(1) transforme le tensor d'images '(batch_size, height, width)' en '(batch_size, 1, height, width)'
        ### les valeurs des pixels dans une image MNIST sont des entiers entre 0 et 255. DIviser par 255 permet de normaliser.
        self.images = torch.tensor(images).unsqueeze(1)/255
        self.labels = torch.tensor(labels)
    
    def __getitem__(self,idx):
        return self.images[idx], self.labels[idx]
        
    def __len__(self):
        #print(len(images))
        return len(self.images) 

BATCH_SIZE = 5
mon_dataset = MonDataset(train_images,train_labels)

# Creation of dataloader
train_dataloader = DataLoader(mon_dataset, batch_size = BATCH_SIZE, shuffle = True)
# batch_size : how many samples per batch to load (default is 1)
# shuffle : set to 'True' to have the data reshuffled at every epoch (default is 'False')
# sampler : defines the strategy to fraw samples from the dataset. Can be any Iterable with __len__ implemented~
## if specified, shuffle must not be specified

