
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.

    # Reshape output and target tensors
    output = output.view(-1, output.size(-1))
    target = target.view(-1)

    # Calcul du coût cross-entropique
    ce_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
    loss = ce_loss(output, target)
    
    # Créer un masque en utilisant le caractère de padding, marking non-padding positions with 1.0 and padding positions with 0.0.
    mask = torch.where(target == padcar, 0, 1)
    
    return torch.sum(loss[mask]) / torch.sum(mask)

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    """ Classe pour un réseau récurrent (RNN).
    """
    def __init__(self, input_dim, latent_dim, output_dim):
        """ @param input_dim: int, dimension de l'entrée
            @param latent_dim: int, dimension de l'état caché
            @param output_dim: int, dimension de la sortie
        """
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Initialisation des modules linéaires pour l'entrée (in), les états cachés (lat) et le décodeur (out)
        self.linear_in = nn.Linear(self.input_dim, self.latent_dim)
        self.linear_lat = nn.Linear(self.latent_dim, self.latent_dim)
        self.linear_out = nn.Linear(self.latent_dim, self.output_dim)
        
        # Initialisation du module TanH pour le calcul de l'état caché
        self.tanh = nn.Tanh()
                
    def one_step(self, x, h):
        """ Traite un pas de temps: renvoie le prochain état caché.
            @param x: torch.Tensor, batch des séquences à l'instant t de taille (batch, input)
            @param h: torch.Tensor, batch des états cachés à l'instant t-1 de taille (batch, latent)
        """
        return self.tanh( self.linear_in(x) + self.linear_lat(h) )
    
    def forward(self, x, h):
        """ Traite tout le batch de séquences passé en paramètre en appelant successivement la
            méthode forward sur tous les éléments des séquences. 
            Renvoie la séquence des états cachés calculés de taille (batch, latent)
            @param x: torch.Tensor, batch de séquences à l'instant t de taille (length, batch, dim)
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        # Initialisation de la séquence des état cachés
        hidden_states = list()
        
        # Appel de la méthode one_step sur nos séquences à chaque instant i
        for i in range(len(x)):
            h = self.one_step(x[i], h)
            hidden_states.append(h)
            print(f"Step {i + 1}: x[i] shape: {x[i].shape}, h shape: {h.shape}")
            
        return torch.stack(hidden_states)
    
    def decode(self, h):
        """ Décode le batch d'états cachés. Renvoie la sortie d'intérêt y de taille (batch, output).
            L'activation non-linéaire s'effectuera dans la boucle d'apprentissage.
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        return self.linear_out(h)
    
    def parameters(self):
        return list(self.linear_in.parameters()) + list(self.linear_lat.parameters()) + list(self.linear_out.parameters())


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    pass

class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    pass


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
