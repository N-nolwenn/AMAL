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


def linear_regression(xtrain, ytrain, xtest, ytest, batch_size = None, epsilon = 1e-6):

    writer = SummaryWriter("logs")
    test_writer = SummaryWriter("logs_test")
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
            index = torch.randint(0, num_samples,())
            # fonction torch.randit(_,_,_) génère un nombre entier aléatoire.
            ## troisème paramètre spécifie la forme du tenseur de sortie ici scalaire
            xbatch, ybatch = xtrain[index], ytrain[index]
        else:
            # Scenario du Mini-Batch Gradient descent 
            ## batch_size > 1 on divise le jeu de données en pls petits ensembles
            index = torch.randperm(num_samples)[:batch_size]
            xbatch, ybatch = xtrain[index], ytrain[index]

        # Phase Forward
        yhat = torch.matmul(xbatch, w) + b 
        mse = torch.nn.MSELoss()
        train_loss = mse(yhat, ybatch)

        # on peut visualiser avec
        # tensorboard --logdir logs/
        writer.add_scalar(f"Loss/train{batch_size}", train_loss, n_iter)

        # Sortie directe
        #print(f"Itérations {n_iter}: Training loss {train_loss}")

        ##  TODO:  Calcul du backward (grad_w, grad_b)
        train_loss.backward(retain_graph = True)

        ##  Mise à jour des paramètres du modèle
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad

            w.grad.zero_()
            b.grad.zero_()


        test_loss = mse(torch.matmul(xtest, w)+b, ytest)
        test_writer.add_scalar(f"Loss/test{batch_size}", test_loss, n_iter)
        #print(f"Iterations {n_iter} | Testing loss: {test_loss}")

    writer.close()
    test_writer.close()

#linear_regression(xtrain, ytrain, xtest, ytest, batch_size=10)
#linear_regression(xtrain, ytrain, xtest, ytest, batch_size=1)
#plinear_regression(xtrain, ytrain, xtest, ytest, batch_size=10)

NB_EPOCH = 1000

def optimiseur(xtrain, ytrain, xtest, ytest, batch_size = None, epsilon = 1e-6):

    optim_writer = SummaryWriter("logs_optim")
    optim_test_writer = SummaryWriter("logs_optim_test")

    # Initialisation de la loss
    optim = torch.optim.SGD(params=[w,b] lr= epsilon)
    optim.zero_grad()

    num_samples = xtrain.shape[0]

    for n_iter in range(NB_EPOCH):

        if batch_size is None:
            # Scenario du Batch Gradient descent
            ## On utilise tout le jeu de données pr la màj des paramètres
            xbatch, ybatch = xtrain, ytrain 
        elif batch_size == 1:
            # Scenario du Stochastic Gradient descent
            ## On choisit un index aléatoire et seulement 1 échantillon est utilisé pr màj
            index = torch.randint(0, num_samples,())
            # fonction torch.randit(_,_,_) génère un nombre entier aléatoire.
            ## troisème paramètre spécifie la forme du tenseur de sortie ici scalaire
            xbatch, ybatch = xtrain[index], ytrain[index]
        else:
            # Scenario du Mini-Batch Gradient descent 
            ## batch_size > 1 on divise le jeu de données en pls petits ensembles
            index = torch.randperm(num_samples)[:batch_size]
            xbatch, ybatch = xtrain[index], ytrain[index]

        # Forward
        optim_train_loss = torch.nn.MSELoss(torch.matmul(xbatch, w) + b, ybatch)

        optim_writer.add_scalar(f"Loss/optim_train{batch_size}", optim_train_loss, n_iter)

        # Backward
        optim_train_loss.backward(retain_graph = True)

        # Màj des param
        optim.step()
        optim.zero_grad()

        test_loss = torch.nn.MSE(torch.matmul(xtest, w)+b, ytest)
        optim_test_writer.add_scalar(f"Loss/optim_test{batch_size}", test_loss, n_iter)
    
    optim_writer.close()
    optim_test_writer.close()

#optimiseur(xtrain, ytrain, xtest, ytest, batch_size=None)
#optimiseur(xtrain, ytrain, xtest, ytest, batch_size=1)
#optimiseur(xtrain, ytrain, xtest, ytest, batch_size=10)


def Reseau1():

    w1 = torch.nn.Parameter((xtrain.shape[1], ytrain.shape[1]), requires_grad = True)
    b1 = torch.nn.Parameter(ytrain.shape[1], requires_grad = True)
    w2 = torch.nn.Parameter((xtrain.shape[1], ytrain.shape[1]), requires_grad = True)
    b2 = torch.nn.Parameter(ytrain.shape[1], requires_grad = True)
    
    optim = torch.optim.SGD(params=[w1,b1, w2, b2] lr= epsilon)
    optim.zero_grad()

    tanh = torch.nn.Tanh()
    mse = torch.nn.MSELoss()

def Reseau2():
    model = torch.nn.Sequential(
            torch.nn.Linear(xtrain.shape[1], ytrain.shape[1])
            torch.nn.Tanh()
            torch.nn.Linear(ytrain.shape[1],1))

