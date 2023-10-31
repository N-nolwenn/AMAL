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

def linear_regression (xtrain, ytrain, epsilon = 1e-6):

    writer = SummaryWriter("logs")

    # Intialisation fonctions Linéaire et Loss
    

    # Les paramètres à optimiser
    w = torch.randn((xtrain.shape[1], 1), requires_grad = True)
    b = torch.randn(1, requires_grad = True)

    for n_iter in range(1000):

        # Phase Forward
        yhat = torch.mm(xtrain, w) + b 
        mse = torch.nn.MSELoss()
        train_loss = mse(yhat, ytrain)

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
        