import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert, norm, cholesky, solve

'''
Exercise 2.2.
Si ripeta l’esercizio precedente sulla matrice di Hilbert, che si può
generare con la funzione
A = scipy.linalg.hilbert(n) per n = 5, . . . , 10.

In particolare:
1. Calcolare il numero di condizionamento di A e rappresentarlo in un grafico
    al variare di n.
2. Considerare il vettore colonna x = (1, . . . , 1)T , calcola il
    corrispondente termine noto b per il sistema lineare Ax = b e
    la relativa soluzione x̃ usando la fattorizzazione di Cholesky
    come nel caso precedente.
3. Si rappresenti l’errore relativo al variare delle dimensioni della
    matrice.

NB La decomposizione di Cholesky viene calcolata con la funzione
np.linalg.cholesky.

'''
START = 5
END = 11
# Lista per memorizzare i numeri di condizionamento
# e gli errori relativi con n da 5 a 10
cond_numbers = []
relative_errors = []
range = range(START, END)

for n in range:
    A = hilbert(n)  #è mal condizionata però è simmetrice e definita positiva
    cond_numbers.append(np.linalg.cond(A))

    # Creazione del vettore colonna x
    x = np.ones((n, 1))

    # Imposta lower=True per ottenere una matrice L triangolare inferiore con Cholesky
    L = cholesky(A, lower=True)

    b = np.dot(A, x)
    y = solve(L, b)
    x̃ = solve(L.T, y)

    relative_errors.append(norm(x-x̃)/norm(x))

# Rappresenta i numeri di condizionamento in un grafico
plt.semilogy(np.arange(START, END), cond_numbers, '-ro', marker='o', linestyle='-')
plt.xlabel('Dimensione n')
plt.ylabel('K(A)')
plt.title('Condizionamento delle matrici di Hilbert')
plt.grid(True)
plt.show()

plt.plot(np.arange(START, END), relative_errors, '-yo',  marker='o', linestyle='-')
plt.xlabel('Dimensione n')
plt.ylabel('Err')
plt.title('Errori relativi')
plt.grid(True)
plt.show()






