import numpy as np

def create_matrix(n):
    d = 3*np.ones(n-1)
    A = -np.eye(n) + np.diag(d, k = 1) + np.diag(d, k = -1)
    return A

def sub_matrix(A, m):
    return A[:m, :m]

A = create_matrix(10)
print(A)
print()
print(sub_matrix(A, 5))


