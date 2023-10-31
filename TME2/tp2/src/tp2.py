import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
#pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()

# Les données supervisées
datax = torch.tensor(datax,dtype=torch.float, requires_grad = True)
datay = torch.tensor(datay,dtype=torch.float, requires_grad = True).reshape(-1,1)


# Découpage des données en ensembles train (80%) et test (20%)
xtrain, xtest, ytrain, ytest = train_test_split(datax, datay, test_size = 0.2)

def linear_regression (xtrain, ytrain, batch_size = None, epsilon = 1e-6):

    writer = SummaryWriter("logs")

    # Intialisation fonctions Linéaire et Loss
    

    # Les paramètres à optimiser
    w = torch.randn((xtrain.shape[1], 1), requires_grad = True)
    b = torch.randn(1, requires_grad = True)

    num_samples = xtrain.shape[0]

    for n_iter in range(1000):
        if batch_size is None:
            # Scenario du Batch Gradient descent
            ## On utilise tout le jeu de données pr la màj des paramètres
            xbatch, ybatch = xtrain, ytrain 
        elif batch_size == 1:
            # Scenario du Stochastic Gradient descent
            ## On choisit un index aléatoire et seulement 1 échantillon est utilisé pr màj
            index = torch.randit(0, num_samples,())
            # fonction torch.randit(_,_,_) génère un nombre entier aléatoire.
            ## troisème paramètre spécifie la forme du tenseur de sortie ici scalaire
            xbatch, ybatch = xtrain[index], ytrain[index]
        else:
            # Scenario du Mini-Batch Gradient descent 
            ## batch_size > 1 on divise le jeu de données en pls petits ensembles
            index = torch.randperm(num_samples)[:batch_size]
            xbatch, ybatch = xtrain[index], ytrain[index]

        # Phase Forward
        yhat = torch.mm(xbatch, w) + b 
        mse = torch.nn.MSELoss()
        train_loss = mse(yhat, ybatch)

        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', train_loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {train_loss}")

        ##  TODO:  Calcul du backward (grad_w, grad_b)
        train_loss.backward(retain_graph = True)

        ##  TODO:  Mise à jour des paramètres du modèle
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad

            w.grad.zero_()
            b.grad.zero_()

linear_regression(xtest, ytest)
        