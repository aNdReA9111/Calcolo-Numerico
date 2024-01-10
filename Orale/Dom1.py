import numpy as np
import scipy
import scipy.linalg
from scipy.linalg import lu_factor as LUdec # pivoting
from scipy.linalg import lu as LUfull # partial pivoting

# Genera un numero casuale 'n' compreso tra 2 e 10
n = 2

# Crea una matrice quadrata di dimensione 'n' con valori casuali compresi tra 10 e 1000

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, solve, norm

n = np.random.randint(2, 11)

START = 2
END = 11
cond_numbers = []
errors = []
range = range(START, END)
for n in range:
    A = np.random.rand(n, n) * 990 + 10
    cond_numbers.append(np.linalg.cond(A))
    x = np.ones((n, 1))
    b=A@x

    lu, piv = LUdec(A)
    print('lu\n',lu,'\n')
    print('piv',piv,'\n')

    my_x= scipy.linalg.lu_solve((lu, piv), b)
    errors.append(norm(x-my_x)/norm(x))


# Rappresenta i numeri di condizionamento in un grafico al variare della dimensione
plt.semilogy(np.arange(START, END), cond_numbers, '-ro', marker='o', linestyle='-')
plt.xlabel('Dimensione n')
plt.ylabel('K(A)')
plt.title('Condizionamento')
plt.grid(True)
plt.show()
# Rappresenta gli errori relativi in un grafico al variare della dimensione
plt.plot(np.arange(START, END), errors, '-yo', marker='o', linestyle='-')
plt.xlabel('Dimensione n')
plt.ylabel('Err')
plt.title('Errori relativi')
plt.grid(True)
plt.show()
