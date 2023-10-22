
import torch

class LossMSE:
    def __init__(self):
        pass

    def forward(self, y_hat, y):
        self.yhat = y_hat
        self.y = y
        self.loss = torch.pow(self.yhat - self.y, 2)
        return self.loss
    
    def backward(self):
        dL_dyhat = 2 * (self.yhat - self.y)
        dL_dy = 2 * (self.yhat - self.y)
        return dL_dyhat, dL_dy