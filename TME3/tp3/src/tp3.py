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
class MonDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).unsqueeze(1).double()/255
        self.labels = torch.tensor(labels)
    
    def __getitem__(self,idx):
        return self.images[idx], self.labels[idx]
        
    def __len__(self):
        print(len(images))
        return len(self.images)
    

BATCH_SIZE = 32

train_dataloader = DataLoader(MonDataset(train_images), shuffle = True, batch_size = BATCH_SIZE)
for batch, batch_len in train_dataloader:
    print(batch, batch_len)