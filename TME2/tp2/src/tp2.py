'''
Nolwenn PIGEON 
AMAL - TME2
'''

import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
#pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import OrderedDict

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
    loss = torch.nn.MSELoss()

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
        train_loss = loss(yhat, ybatch)

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


        test_loss = loss(torch.matmul(xtest, w)+b, ytest)
        test_writer.add_scalar(f"Loss/test{batch_size}", test_loss, n_iter)
        #print(f"Iterations {n_iter} | Testing loss: {test_loss}")

    writer.close()
    test_writer.close()

linear_regression(xtrain, ytrain, xtest, ytest, batch_size=None)
linear_regression(xtrain, ytrain, xtest, ytest, batch_size=1)
linear_regression(xtrain, ytrain, xtest, ytest, batch_size=10)



def optimiseur(xtrain, ytrain, xtest, ytest, batch_size = None, epsilon = 1e-6, num_epochs = 1000):

    optim_writer = SummaryWriter("logs_optim")
    optim_test_writer = SummaryWriter("logs_optim_test")

    # Initialisation des paramètres
    w = torch.randn((xtrain.shape[1], 1), requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    optim = torch.optim.SGD(params=[w, b], lr= epsilon)
    optim.zero_grad()

    num_samples = xtrain.shape[0]
    loss = torch.nn.MSELoss()

    for n_iter in range(num_epochs):

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
        optim_train_loss = loss(torch.matmul(xbatch, w) + b, ybatch)

        optim_writer.add_scalar(f"Loss/optim_train{batch_size}", optim_train_loss, n_iter)

        # Backward
        optim_train_loss.backward(retain_graph = True)

        # Màj des param
        optim.step()
        optim.zero_grad()

        test_loss = loss(torch.matmul(xtest, w)+b, ytest)
        optim_test_writer.add_scalar(f"Loss/optim_test{batch_size}", test_loss, n_iter)
    
    optim_writer.close()
    optim_test_writer.close()

optimiseur(xtrain, ytrain, xtest, ytest, batch_size=None)
optimiseur(xtrain, ytrain, xtest, ytest, batch_size=1)
optimiseur(xtrain, ytrain, xtest, ytest, batch_size=10)

def Reseau1(xtrain, ytrain, xtest, ytest, input_size=None, hidden_size=None, output_size=None, batch_size=None, num_epochs=100):
    if input_size is None:
        input_size = xtrain.shape[1]
    if hidden_size is None:
        hidden_size = ytrain.shape[1]
    if output_size is None: 
        output_size = 1

    # Neural network architecture
    lin1 = torch.nn.Linear(input_size, hidden_size)
    tanh = torch.nn.Tanh()
    lin2 = torch.nn.Linear(hidden_size, output_size)
    mse = torch.nn.MSELoss()
    optim = torch.optim.SGD(params=list(lin1.parameters()) + list(lin2.parameters()), lr=1e-6)

    optim_writer = SummaryWriter("logs_NN1")
    optim_test_writer = SummaryWriter("logs_NN1_test")

    num_samples = xtrain.shape[0]

    for epoch in range(num_epochs):
        if batch_size is None:
            xbatch, ybatch = xtrain, ytrain 
        elif batch_size == 1:
            index = torch.randint(0, num_samples, ())
            xbatch, ybatch = xtrain[index], ytrain[index]
        else:
            index = torch.randperm(num_samples)[:batch_size]
            xbatch, ybatch = xtrain[index], ytrain[index]

        # Forward pass
        out1 = lin1(xbatch)
        out2 = tanh(out1)
        output = lin2(out2)

        # Loss calculation
        loss = mse(output, ybatch)
        optim_writer.add_scalar(f"Loss/NN1_train{batch_size}", loss.item(), epoch)

        # Backward pass
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()

        # Compute and log test loss
        test_out1 = lin1(xtest)
        test_out2 = tanh(test_out1)
        test_output = lin2(test_out2)
        test_loss = mse(test_output, ytest)
        optim_test_writer.add_scalar(f"Loss/NN1_test{batch_size}", test_loss.item(), epoch)

    optim_writer.close()
    optim_test_writer.close()

Reseau1(xtrain, ytrain, xtest, ytest, batch_size=10)
Reseau1(xtrain, ytrain, xtest, ytest, batch_size=1)
Reseau1(xtrain, ytrain, xtest, ytest, batch_size=10)

def Reseau2(xtrain, ytrain, xtest, ytest, input_size = None, hidden_size = None, output_size = None, batch_size = None, num_epochs = 1000):
    '''
    Using Sequential to create a small model. When 'model' is run,
    input will first be passed to lin1
    The output of lin1 will be used as the input to tanh
    The ouput of Tanh will become the input for lin2
    Finally the ouput of lin2 will be used as the input of mse
    '''
    if input_size is None:
        input_size = xtrain.shape[1]
    if hidden_size is None:
        hidden_size= ytrain.shape[1]
    if output_size is None: 
        output_size = 1

    model = torch.nn.Sequential(OrderedDict([
        ('lin1', torch.nn.Linear(input_size, hidden_size)),
        ('tanh',torch.nn.Tanh()),
        ('lin2', torch.nn.Linear(hidden_size, output_size)),
    ]))
    mse = torch.nn.MSELoss()
    optim =  torch.optim.SGD(params = model.parameters(), lr = 1e-6)
    

    optim_writer = SummaryWriter("logs_NN2")
    optim_test_writer = SummaryWriter("logs_NN2_test")

    num_samples = xtrain.shape[0]

    for epoch in range(num_epochs):

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
        output = model(xbatch)
        loss = mse(output, ybatch)
        optim_writer.add_scalar(f"Loss/NN2_train{batch_size}", loss.item(), epoch)

        # Backward
        loss.backward(retain_graph = True)

        # Màj des param
        optim.step()
        optim.zero_grad()


        # Test
        test_output = model(xtest)
        test_loss = mse(test_output, ytest)
        optim_test_writer.add_scalar(f"Loss/NN2_test{batch_size}", test_loss.item(), epoch)
    
    optim_writer.close()
    optim_test_writer.close()

Reseau2(xtrain, ytrain, xtest, ytest, batch_size = None)
Reseau2(xtrain, ytrain, xtest, ytest, batch_size = 1)
Reseau2(xtrain, ytrain, xtest, ytest, batch_size = 10)

