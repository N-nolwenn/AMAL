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
'''
QUESTION 1
Gérer les données avec Dataset et Dataloader
'''
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
data_train = MonDataset(train_images,train_labels)
data_test = MonDataset(test_images, test_labels)

# Creation of dataloader
train_dataloader = DataLoader(data_train, batch_size = BATCH_SIZE, shuffle = True)
# batch_size : how many samples per batch to load (default is 1)
test_loader = DataLoader(data_test, batch_size = BATCH_SIZE, shuffle = True)

# shuffle : set to 'True' to have the data reshuffled at every epoch (default is 'False')
# sampler : defines the strategy to fraw samples from the dataset. Can be any Iterable with __len__ implemented~
## if specified, shuffle must not be specified

'''
GPU : pas de GPU
'''


'''
Checkpointing
'''
class State:  
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch , self.iteration = 0, 0

'''
QUESTION 2
Autoencoder
'''

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, input_dim), 
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

'''
QUESTION 3:
Campagne d'expériences 
'''
epochs = 100
def NN(hidden_dim, n_epoch=epochs, epsilon = 0.001):
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if savepath.is_file():
        with savepath.open("rb") as fp
            state = torch.load(fp) 
            #on recommence depuis le modele sauvegardé
    else : 
        #create an instance of the Autoencoder within NN and move it to the device
        model = Autoencoder(train_images.shape[1], hidden_dim) 
        model = model.to(device)
        optim = torch.optim.SGD(model.parameters(), lr = epsilon)
        state = State(model, optim)
        
    
# Phase d'apprentissage

bce = torch.nn.BCELoss()

for epoch in range (state.epoch, epochs):
    for x, y in train_loader:
        
        # remise à zero des gradients des param à optim
        state.optim.zero_grad()
        
        # chargement du batch sur device
        x = x.to(device)
        
        # forward
        xhat = state.model.forward(x)
        
        # backward         
        train_loss = bce(xhat, x)
        train_loss.backward()
        
        # Màj param
        state.optim.step()
        state.iteration +=1
        
        with savepath.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)
    
    # test
    with torch.no_grad():
            
            loss_list = []
            
            for xtest, ytest in data_test:
                xtest = xtest.to(device)
                xhat_test = state.model.forward(xtest)
                loss_list.append(bce(xhat_test, xtest))
            
            test_loss = np.mean(loss_list)


    # affichage tensorboard
    writer.add_scalar('Loss/train/{}'.format(latent_dim), train_loss, epoch)
    print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
    writer.add_scalar('Loss/test/{}'.format(latent_dim), test_loss, epoch)
    

class Highway(torch.nn.Module):
    def __init__(self, arg):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # Initialisation des modules H
        self.H = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)])
        
        # Initialisation des modules T
        self.T = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)])
        
    def forward(self, x):
        """ @param x: torch.Tensor, données
        """
        # Copie du tenseur à forwarder
        x_ = torch.clone(x)
        
        # Propagation sur toutes les couches
        for layer in range(self.num_layers):
            
            # --- Calcul intermédiaire des H et T
            h_out = torch.nn.functional.relu( self.H[layer](x_) )
            t_out = torch.sigmoid( self.T[layer](x_) )
            
            # --- Mise à jour de x_
            x_ = h_out * t_out + (1 - t_out) * x
        
        return x_


def highway_neuralnet(num_layers = 10, n_epochs = N_EPOCHS, epsilon = 1e-1):
    """ Highway network sur les données MNIST.
    """
    # Sélectionner le GPU s'il est disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Chemin vers le modèle. Reprend l'apprentissage si un modèle est déjà sauvegardé.
    savepath = Path('model_highway_{}.pch'.format(num_layers))
    
    if savepath.is_file():
        with savepath.open('rb') as file:
            state = torch.load(file)
    
    else:
        # Création du modèle et de l'optimiseur, chargement sur device
        model = Highway(train_images.shape[1], num_layers)
        model = model.to(device)
        optim = torch.optim.SGD(params = model.parameters(), lr = epsilon) # lr : pas de gradient
        state = State(model, optim)
    
    # Initialisation de la loss
    bce = torch.nn.BCELoss()
    
    # --- Phase d'apprentissage
    for epoch in range(state.epoch, N_EPOCHS):
        
        for x, y in data_train:
            
            # --- Remise à zéro des gradients des paramètres à optimiser
            state.optim.zero_grad()
            
            # --- Chargement du batch sur device
            x = x.to(device)
            
            # --- Phase forward
            xhat = state.model.forward(x)
            
            # --- Phase backward
            train_loss = bce(xhat, x)
            train_loss.backward()
            
            # --- Mise à jour des paramètres
            state.optim.step()
            state.iteration += 1
            
            with savepath.open('wb') as file:
                state.epoch = epoch + 1
                torch.save(state, file)
        
        # --- Phase de test
            
        with torch.no_grad():
            
            loss_list = []
            
            for xtest, ytest in data_test:
                xtest = xtest.to(device)
                xhat_test = state.model.forward(xtest)
                loss_list.append(bce(xhat_test, xtest))
            
            test_loss = np.mean(loss_list)
            
        # --- Affichage tensorboard
            
        writer.add_scalar('Loss/train/', train_loss, epoch)
        print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
        writer.add_scalar('Loss/test/', test_loss, epoch)

    

    