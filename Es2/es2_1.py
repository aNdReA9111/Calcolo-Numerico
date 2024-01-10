"""2. fattorizzazione lu"""

import numpy as np
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
from scipy.linalg import lu_factor as LUdec # pivoting
from scipy.linalg import lu as LUfull # partial pivoting

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
x = np.ones((4,1))
b = A@x

condA = np.linalg.norm(A, 2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')


lu, piv = LUdec(A)
print('lu\n',lu,'\n')
print('piv',piv,'\n')

# risoluzione di    Ax = b   <--->  PLUx = b
my_x= scipy.linalg.lu_solve((lu, piv), b)
print('my_x = \n', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))


# IMPLEMENTAZIONE ALTERNATIVA - 1
P, L, U = LUfull(A)
print ('A = ', A)
print ('P = ', P)
print ('L = ', L)
print ('U = ', U)
print ('P*L*U = ', np.matmul(P , np.matmul(L, U)))

print ('diff = ',   np.linalg.norm(A - np.matmul(P , np.matmul(L, U)), 'fro'  ) )


# if P != np.eye(n):
# Ax = b   <--->  PLUx = b  <--->  LUx = inv(P)b  <--->  Ly=inv(P)b & Ux=y : matrici triangolari
# quindi
invP = np.linalg.inv(P)
y = scipy.linalg.solve_triangular(np.matmul(L,invP), b, lower=True, unit_diagonal=True)
my_x = scipy.linalg.solve_triangular(U, y, lower=False)

# if P == np.eye(n):
# Ax = b   <--->  PLUx = b  <--->  PLy=b & Ux=y
# y = scipy.linalg.solve_triangular(np.matmul(P,L) , b, lower=True, unit_diagonal=True)
# my_x = scipy.linalg.solve_triangular(U, y, lower=False)

print('\nSoluzione calcolata: ', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))


