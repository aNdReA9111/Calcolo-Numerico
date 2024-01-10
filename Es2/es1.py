"""1. matrici e norme """

import numpy as np

#help(np.linalg) # View source
#help (np.linalg.norm)
#help (np.linalg.cond)

n = 2
A = np.array([[1, 2], [0.499, 1.001]])

print ('Norme di A:')
norma1 = np.max(np.sum(A, axis=0))
#axis = 1 --> sommo riga
#axis = 0 --> sommo colonna
print("Norma uno con np.max(np.sum(A, axis=0)):", norma1)

norma_uno = np.linalg.norm(A, ord=1)
print ("Norma uno:", norma_uno)
norma_due = np.linalg.norm(A, ord=2)
print ("Norma due:", norma_due)
norma_frob = np.linalg.norm(A, ord='fro')
print ("Norma di Frobenius:", norma_frob)
norma_inf = np.linalg.norm(A, ord=np.inf)
print ("Norma infinito:", norma_inf)


cond1 = np.linalg.cond(A, 1)
print ('\nK(A)_1 = ', cond1)
cond2 = np.linalg.cond(A, 2)
print ('K(A)_2 = ', cond2)
condfro = np.linalg.cond(A, 'fro')
print ('K(A)_fro =', condfro)
condinf = np.linalg.cond(A, np.inf)
print ('K(A)_inf =', condinf, '\n')

x = np.ones((2,1))
b = A@x
print('Ax =', b)

btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde (Axtilde)
my_btilde = A@xtilde

print ('A*xtilde =\n', btilde, '\n')
print('norma =', np.linalg.norm(btilde-my_btilde,2))

deltax = np.linalg.norm(x-xtilde, ord=2)
deltab = np.linalg.norm(b-btilde, ord=2)
print ('delta x = ', deltax)
print ('delta b = ', deltab)

