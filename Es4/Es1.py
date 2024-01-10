import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import lu_factor as LUdec

import numpy as np

# Exercise 1
# Generazione della matrice casuale A di dimensioni m x n
m = 100
n = 10
# A = np.random.rand(m, n)

# # Creazione di un vettore alpha con elementi costanti
# alpha = np.ones(n)

# # Calcolo del termine noto y
# y = A @ alpha

# # Calcolo della soluzione utilizzando la fattorizzazione LU
# # Risoluzione delle equazioni normali: A^T * A * x = A^T * y
# ATA = np.dot(A.T, A)
# ATy = np.dot(A.T, y)
# LU_solution = np.linalg.solve(ATA, ATy)

# # Calcolo della soluzione utilizzando la fattorizzazione Cholesky
# # La matrice A^T * A deve essere simmetrica e definita positiva
# cholesky_factor = np.linalg.cholesky(ATA)
# # cholesky_solution = np.linalg.solve(np.transpose(cholesky_factor), np.linalg.solve(cholesky_factor, ATy))

# # L = scipy.linalg.cholesky(ATA)
# x = scipy.linalg.solve_triangular(np.transpose(cholesky_factor), ATy, lower=True)
# alpha_chol = scipy.linalg.solve_triangular(cholesky_factor, x, lower=False)
# print('alpha chol', alpha_chol)

# print("Soluzione con fattorizzazione LU:", LU_solution)
# print("Soluzione con fattorizzazione Cholesky:", alpha_chol)

A = np.random.rand(m, n)  #matrice casuale mxn

alpha = np.ones(n)
y = A@alpha


ATA = A.T@A   #matrice simmetrica definita positiva
ATy = A.T@y   #termine noto

lu, piv = LUdec(ATA)
alpha_LU = scipy.linalg.lu_solve((lu,piv), ATy)
print("Soluzione con fattorizzazione LU:", alpha_LU)

#picture
L = scipy.linalg.cholesky(ATA)
x = scipy.linalg.solve_triangular(np.transpose(L), ATy, lower=True)
alpha_chol = scipy.linalg.solve_triangular(L, x, lower=False)
print("Soluzione con Cholesky:", alpha_chol)

U, s, Vh = scipy.linalg.svd(A)

print('Shape of U:', U.shape)
print('Shape of s:', s.shape)
print('Shape of V:', Vh.T.shape)

alpha_svd = np.zeros(s.shape)
for i in range(n):
  ui = U[:, i]  #accesso alla colonna i
  vi = Vh[i, :] #accesso alla riga i
  alpha_svd = alpha_svd + np.dot(ui, y) * vi / s[i] # == alpha_svd += ui@y * vi / s[i]

print("Soluzione con SVD:", alpha_svd)

# Calcolo dell'errore relativo
relative_error_svd = np.linalg.norm(alpha_svd - alpha) / np.linalg.norm(alpha)
print("Errore relativo:", relative_error_svd)

