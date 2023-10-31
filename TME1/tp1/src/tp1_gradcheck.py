import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)

mse_check = torch.autograd.gradcheck(mse, (yhat, y))
print("Test du gradient pour MSE:", mse_check)


#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)

X = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5, 5, requires_grad=True, dtype=torch.float64)
b = torch.randn(5, requires_grad=True, dtype=torch.float64)

linear_check = torch.autograd.gradcheck(linear, (X, W, b))
print("Test du gradient pour Linear:", linear_check)
