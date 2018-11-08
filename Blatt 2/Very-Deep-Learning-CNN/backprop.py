# -*- coding: utf-8 -*-

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10
N, D_in, H, D_out = 1, 5, 4, 3

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)           # Von Dimension D_in x H (fully connected)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)          # Von Dimension H x D_out (fully connected)

print("x =", x)
print("y =", y)
print("w1 =", w1)
print("w2 =", w2)

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)                # Matrixmultiplikation x mit w1 wobei h ein Vektor mit H Dimensionen ist.
    h_relu = h.clamp(min=0)     # Anwenden von ReLu d.h. max(0,x)
    y_pred = h_relu.mm(w2)      # Matrixmultiplikation h_relu mit w2

    # print("h =", h)
    # print("h_relu =", h_relu)
    # print("y_pred =", y_pred)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()     # Berechne L_2 Norm
    print(t, loss)                              # Ausgabe des aktuellen Loss

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)            # Ableitung der Loss-Funktion (da quad. => Faktor 2)
    # print(grad_y_pred)
    grad_w2 = h_relu.t().mm(grad_y_pred)  # h_relu wird transponiert (=> stehender Vektor) und anschließend berechnen wir das
                                          # äußere-Produkt von dem transonierten h_relu Vektor und dem liegenden Vektor grad_y_pred
                                          # Daher ist grad_w2 wieder eine Matrix.
    # print(h_relu.t())
    # print(grad_w2)
    grad_h_relu = grad_y_pred.mm(w2.t())    # Matrixmultiplikation von grad_y_pred und der transponierten w2 Matrix.
    grad_h = grad_h_relu.clone()
    # print("grad_h =", grad_h)
    grad_h[h < 0] = 0                 # Warum werden alle Einträge aus grad_h genau dort 0 gesetzt, wo h negativ ist?
    # print("grad_h =", grad_h)
    grad_w1 = x.t().mm(grad_h)      # x wird transponiert (=> stehender Vektor) und anschließend berechnen wir das
                                    # äußere-Produkt von dem transonierten x Vektor und dem liegenden Vektor grad_h
                                    # Daher ist grad_w1 wieder eine Matrix.

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
