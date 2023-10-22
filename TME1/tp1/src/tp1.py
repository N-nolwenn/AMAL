
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    """Méthode forward (ctx, *inputs):  
    Calcule rés de l'application de la fonction    
    """
    """Méthode backward (ctx, *grad_outputs): 
    Calcule gradient partiel p/r à chaque entrée de forward.   
    nbr de grad_outputs = nbr de sorties de forward = nbr de inputs de forward
    """
    @staticmethod
    
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y) # ctx permet de sauvegarder un contexte lors de la passe forward

        #  TODO:  Renvoyer la valeur de la fonction
        loss = torch.mean(torch.pow(yhat - y, 2))
        return loss

    @staticmethod
    
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors # le paramètre ctx est passe lors de la backward afin de récupérer les valeurs.
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        dL_dyhat = 2 * (yhat - y)
        dL_dy = 2 * (yhat - y)
        return dL_dyhat * grad_output, dL_dy * grad_output

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    def forward(ctx, X, W, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W, b)

        # Pour une couche linéaire, on utilise l'opération de multiplication matricielle pour calculer la sortie
        output = torch.matmul(X, W) + b 
        return output
    
    def backward(ctx, grad_output):
        X, W, b = ctx.saved_tensors

        dL_dX = torch.matmul(grad_output, W.t())
        dL_dW = torch.matmul(X.t(), grad_output)
        dL_db = grad_output.sum(0)

        return dL_dX, dL_dW, dL_db

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

