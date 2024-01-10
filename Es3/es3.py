import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm
from skimage import data
from skimage.io import imread

#A = data.camera()

#fare con immaggini di diversa complessità e con diversi p per dimostrare che per
#immagini più complesse è necessario effettuare più passi
A = imread("notte_stellata.jpg")

#[0,1,2]
A = A[:,:,0]

print(type(A))
print(A.shape)

plt.imshow(A, cmap='gray')
plt.show()


U, s, Vh = svd(A)

print('Shape of U:', U.shape)
print('Shape of s:', s.shape)
print('Shape of V:', Vh.T.shape)

A_p = np.zeros(A.shape)
p_max = 711


for i in range(p_max):
  ui = U[:, i]  #prendo l'iesimo elemento della riga  (quindi hai una colonna)
  vi = Vh[i, :] #prendo l'iesimo elemento di ogni colonna (quindi hai una riga)

  #somma delle diadi
  A_p += s[i]*np.outer(ui,vi)

plt.imshow(A_p)
plt.show()

print('\n')
err_rel = norm(A_p-A, 'fro')/norm(A, 'fro') #deve diminuire all'aumentare di p
print('L\'errore relativo della ricostruzione di A è', err_rel)
c = (1/p_max)*np.min(A.shape)-1
print('Il fattore di compressione è c=', c)


plt.figure(figsize=(20, 10))
fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')

fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_p, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max))

plt.show()


# al variare di p
#p_max = 10#np.min(A.shape)
A_p = np.zeros(A.shape)
err_rel = np.zeros((p_max))
c = np.zeros((p_max))

for i in range(1, p_max):
  ui = U[:, i]
  vi = Vh[i, :]

  A_p += s[i]*np.outer(ui,vi)

  err_rel[i] = norm(A_p-A, 'fro')/norm(A, 'fro') #deve diminuire all'aumentare di p
  c[i] = (1/i)*np.min(A.shape)-1

plt.figure(figsize=(10, 5))
fig1 = plt.subplot(1, 2, 1)
fig1.semilogy(err_rel[1:], 'o-')
plt.title('Errore relativo')  #diminuisce

fig2 = plt.subplot(1, 2, 2)
fig2.plot(c[1:], 'o-')
plt.title('Fattore di compressione') #diminuisce (potrebbe anche diventare negativo)

plt.show()
