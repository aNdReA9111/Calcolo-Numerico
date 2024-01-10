import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, solve, norm


def init_A(n):
    diagonale_principale = 9 * np.ones(n)
    sottodiagonale = -4 * np.ones(n - 1)

    matrice_tridiagonale = np.diag(diagonale_principale) + np.diag(sottodiagonale, k=1) + np.diag(sottodiagonale, k=-1)

    return matrice_tridiagonale

START = 2
END = 11
cond_numbers = []
relative_errors = []
range = range(START, END)
for n in range:
    A = init_A(n)
    cond_numbers.append(np.linalg.cond(A))
    x = np.ones((n, 1))
    L = cholesky(A, lower=True)
    b = b = A@x
    y = solve(L, b)
    x̃ = solve(L.T, y)
    relative_errors.append(norm(x-x̃)/norm(x))

# Rappresenta i numeri di condizionamento in un grafico al variare della dimensione
plt.semilogy(np.arange(START, END), cond_numbers, '-ro', marker='o', linestyle='-')
plt.xlabel('Dimensione n')
plt.ylabel('K(A)')
plt.title('Condizionamento')
plt.grid(True)
plt.show()
# Rappresenta gli errori relativi in un grafico al variare della dimensione
plt.plot(np.arange(START, END), relative_errors, '-yo', marker='o', linestyle='-')
plt.xlabel('Dimensione n')
plt.ylabel('Err')
plt.title('Errori relativi')
plt.grid(True)
plt.show()
