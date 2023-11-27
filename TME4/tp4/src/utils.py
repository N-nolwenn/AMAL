import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, input_size, output_size, hidden_size):

        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.linear_in = nn.Linear(self.input_size, self.hidden_size)
        self.linear_lat = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.output_size)  
        self.bh = nn.Parameter(torch.zeros(hidden_size))   
        
    
    def one_step(self, x, h):
        '''
        Traite un pas de temps
        Prend en entrée un batch x à un instant t des séquences de taille batch x dim
            et le batch des états cachés de taille batch x latent 
        Renvoie les états cachés suivants de taille batch x latent
        '''
        ht = torch.nn.Tanh(self.linear_in(x) + self.linear_lat(h) + self.bh)
        return ht
    
    def forward(self, x, h):
        '''
        Traite tout le batch de séquences passé en paramètre.
        Appelle successivement méthode one_step sur tous les élemnts des séquences
        Taille de x est length x batch x dim et Taille de h est batch x latent
        Renvoie séquence des états cachés calculés de taille length x batch x latent
        '''
        seq_size = x.size(1)

        output_seq = []

        for t in range(seq_size):
            xt = x[:,t,:]
            ht = self.one_step(xt, h)
            output_seq.append(ht)
        return torch.stack(ht, dim = 1)

    
    def decode_linear(self, h):
        '''
        Décode le batch d'états cachés
        Taille de h est batch x latent 
        Renvoie un tenseur de taille batch x output
        '''
        return self.linear_out(h)
    
    def decode_sigmoid(self, h):
        '''
        Décode le batch d'états cachés
        Taille de h est batch x latent 
        Renvoie un tenseur de taille batch x output
        '''
        return torch.nn.Sigmoide(self.linear_out(h))


class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

