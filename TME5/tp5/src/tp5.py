
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
import torch.nn.functional

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Teneur length x batch
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

class SimpleRNN(nn.Module):
    """ Classe pour un réseau récurrent (RNN).
    """
    def __init__(self, input_dim, latent_dim, output_dim):
        """ @param input_dim: int, dimension de l'entrée
            @param latent_dim: int, dimension de l'état caché
            @param output_dim: int, dimension de la sortie
        """
        super(SimpleRNN, self).__init__()
        self.device = device

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Initialisation des modules linéaires pour l'entrée (in), les états cachés (lat) et le décodeur (out)
        self.embedding = nn.Embedding(num_embeddings = self.input_dim, embedding_dim = self.latent_dim)
        self.rnn = nn.RNN(input_size = self.latent_dim, hidden_size = self.latent_dim, num_layers = 1, batch_first = True)
        self.linear_out = nn.Linear(self.latent_dim, self.output_dim)
        
        # Initialisation du module TanH pour le calcul de l'état caché
        self.tanh = nn.Tanh()
                
    def one_step(self, x, h):
        """ Traite un pas de temps: renvoie le prochain état caché.
            @param x: torch.Tensor, batch des séquences à l'instant t de taille (batch, input)
            @param h: torch.Tensor, batch des états cachés à l'instant t-1 de taille (batch, latent)
        """
        x_emb = self.embedding(x)
        output, h = self.rnn(x_emb, h)
        hidden_state = output[:, -1, :]
        return self.tanh(hidden_state)   
     
    def forward(self, x, h):
        """ Traite tout le batch de séquences passé en paramètre en appelant successivement la
            méthode forward sur tous les éléments des séquences. 
            Renvoie la séquence des états cachés calculés de taille (batch, latent)
            @param x: torch.Tensor, batch de séquences à l'instant t de taille (length, batch, dim)
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        x_emb = self.embedding(x)
        output, h = self.rnn(x_emb, h)
        hidden_states = output.squeeze(1)
        decoded_output = self.tanh(hidden_states)
        return decoded_output, h
    
    def decode(self, h):
        """ Décode le batch d'états cachés. Renvoie la sortie d'intérêt y de taille (batch, output).
            L'activation non-linéaire s'effectuera dans la boucle d'apprentissage.
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        return self.linear_out(h)
    
    def parameters(self):
        return list(self.embedding.parameters()) + list(self.rnn.parameters()) + list(self.linear_out.parameters())


class LSTM(nn.Module):
    """ Classe pour un réseau Long-Short Term Memory (LSTM).
    """
    def __init__(self, input_dim, latent_dim, output_dim):
        """ @param input_dim: int, dimension de l'entrée
            @param latent_dim: int, dimension de l'état caché
            @param output_dim: int, dimension de la sortie
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Initialisation des différentes portes d'entrée (in), de sortie (out), d'oubli (over), interne (intern) et du module de sortie
        self.gate_in = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.gate_over = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.gate_out = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.gate_intern = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.linear_out = nn.Linear(self.latent_dim, self.output_dim)
        
        # Initialisation des modules TanH et Sigmoide
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def one_step(self, x, h, C):
        """ Traite un pas de temps: renvoie les prochains états externe (ht) et interne (Ct)
            @param x: torch.Tensor, batch des séquences à l'instant t de taille (batch, input)
            @param h: torch.Tensor, batch des états cachés à l'instant t-1 de taille (batch, latent)
            @param C: torch.Tensor, batch des états internes à l'instant t-1
        """
        # Concaténation vectorielle de h_{t-1} et x_t
        if len(x.shape) == 1:
            x_concat = torch.cat((h, x.reshape(1,-1)), dim = 1)
        else:
            x_concat = torch.cat((h, x), dim = 1)
        
        # Calcul des différentes portes (oubli, entrée, sortie)
        to_keep = self.sigmoid( self.gate_in( x_concat ) )
        to_throw = self.sigmoid( self.gate_over( x_concat ) )
        output = self.sigmoid( self.gate_out( x_concat ) )
        
        # Mise à jour de la mémoire interne (Ct) et de la mémoire externe (ht)
        Ct = to_throw * C + to_keep * self.tanh( self.gate_intern( x_concat ) )
        ht = output * self.tanh(Ct)
        
        return ht, Ct
    
    def forward(self, x, h, C):
        """ Traite tout le batch de séquences passé en paramètre en appelant successivement la
            méthode forward sur tous les éléments des séquences. 
            Renvoie la séquence des états cachés calculés de taille (batch, latent) et les états internes.
            @param x: torch.Tensor, batch de séquences à l'instant t de taille (length, batch, dim)
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
            @param C: torch.Tensor, batch des états internes
        """
        # Initialisation de la séquence des état cachés
        hidden_states = list()
        
        # Appel de la méthode one_step sur nos séquences à chaque instant i
        for i in range(len(x)):
            h, C = self.one_step(x[i], h, C)
            hidden_states.append(h)
            
        return torch.stack(hidden_states), C
    
    def decode(self, h):
        """ Décode le batch d'états cachés. Renvoie la sortie d'intérêt y de taille (batch, output).
            L'activation non-linéaire s'effectuera dans la boucle d'apprentissage.
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        return self.linear_out(h)

class GRU(nn.Module):
    """ Classe pour un réseau Gated Recurrent Units (GRU).
    """
    def __init__(self, input_dim, latent_dim, output_dim):
        """ @param input_dim: int, dimension de l'entrée
            @param latent_dim: int, dimension de l'état caché
            @param output_dim: int, dimension de la sortie
        """
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Initialisation des différentes portes et modules linéaires
        self.gate_z = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.gate_r = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.gate_h = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.linear_out = nn.Linear(self.latent_dim, self.output_dim)
        
        # Initialisation des modules TanH et Sigmoide
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def one_step(self, x, h):
        """ Traite un pas de temps: renvoie les prochains états externe (ht) et interne (Ct)
            @param x: torch.Tensor, batch des séquences à l'instant t de taille (batch, input)
            @param h: torch.Tensor, batch des états cachés à l'instant t-1 de taille (batch, latent)
        """
        # Concaténation vectorielle de h_{t-1} et x_t
        if len(x.shape) == 1:
            x_concat = torch.cat((h, x.reshape(1,-1)), dim = 1)
        else:
            x_concat = torch.cat((h, x), dim = 1)
        
        # Calcul des différentes portes
        zt = self.sigmoid( self.gate_z( x_concat ) )
        rt = self.sigmoid( self.gate_r( x_concat ) )
        
        # Mise à jour des mémoires interne/externe (ht)
        ht = (1 - zt) * h + zt * self.tanh( self.gate_h( torch.cat((rt * h, x), dim = 1) ) )
        
        return ht
    
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
            
        return torch.stack(hidden_states)
    
    def decode(self, h):
        """ Décode le batch d'états cachés. Renvoie la sortie d'intérêt y de taille (batch, output).
            L'activation non-linéaire s'effectuera dans la boucle d'apprentissage.
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        return self.linear_out(h)


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
class State:
    """ Classe de sauvegarde sur l'apprentissage d'un modèle.
    """
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0,0


BATCH_SIZE = 128
N_EPOCHS = 100


# Liste des débuts de séquences
list_start = ['Than', '[appl', 'We are', 'Our country', 'Hil', 'This is', 'I am', 'The world', 'The U', 'Our gov', 'We w']


class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0

BATCH_SIZE = 128
N_EPOCHS = 100

def train_simple_rnn(model_type, embed_dim=50, latent_dim=64, output_dim=len(lettre2id) + 1, epsilon=1e-4, start=[""], k=7, maxsent=None, maxlen=None):
    file = open('../data/trump_full_speech.txt', 'r')
    lines = file.readlines()
    full_speech = ' '.join(lines).replace("\n", "")
    data_train = DataLoader(TextDataset(full_speech, maxsent=maxsent, maxlen=maxlen), shuffle=True, collate_fn=pad_collate_fn, batch_size=BATCH_SIZE, drop_last=True)

    writer = SummaryWriter("runs/runs" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    savepath = Path('{}_emb{}_lat{}.pch'.format(model_type.__name__, embed_dim, latent_dim))

    if savepath.is_file():
        with savepath.open('rb') as file:
            state = torch.load(file)
    else:
        model = model_type(input_dim=len(lettre2id), latent_dim=latent_dim, output_dim=output_dim)
        model = model.to(device)
        optim = torch.optim.Adam(params=model.parameters(), lr=epsilon)
        state = State(model, optim)

    log_softmax = torch.nn.LogSoftmax()

    for epoch in range(state.epoch, N_EPOCHS):
        for x in data_train:
            state.optim.zero_grad()
            x = x.to(device).long()

            ht = torch.zeros(BATCH_SIZE, latent_dim, requires_grad=True).to(device)

            train_loss = 0

            for t in range(len(x) - 1):
                ht = model.one_step(x[t], ht)
                train_loss += maskedCrossEntropy(log_softmax(model.decode(ht)), x[t + 1], lettre2id['<PAD>'])

            train_loss = train_loss / (len(x) - 1)
            train_loss.backward()
            state.optim.step()
            state.iteration += 1

        with savepath.open('wb') as file:
            state.epoch = epoch + 1
            torch.save(state, file)

        writer.add_scalar('Loss/train/{}/{}'.format(embed_dim, latent_dim), train_loss, epoch)

    for s in start:
        print('\n ------- GREEDY GENERATION : ------- \n')
        print(generate(model=state.model, emb=model.embedding, decoder=model.decode, latent=latent_dim, log_softmax=log_softmax, eos=0, start=s, maxlen=maxlen))
        print('\n ------- BEAM SEARCH GENERATION : ------- \n')
        print(generate_beam(model=state.model, emb=model.embedding, decoder=model.decode, latent=latent_dim, log_softmax=log_softmax, eos=0, k=k, start=s, maxlen=maxlen))

# Example: Training a SimpleRNN model
train_simple_rnn(SimpleRNN, embed_dim=50, latent_dim=64, output_dim=len(lettre2id) + 1, epsilon=1e-4, start=[""], k=7, maxsent=None, maxlen=None)