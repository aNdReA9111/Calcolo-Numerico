import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics, io
from scipy import signal
from numpy import fft
from utils_SR import totvar, grad_totvar
from utils import psf_fft, A, AT, gaussian_kernel
#sf = 8

# Immagine in floating point con valori tra 0 e 1
#X = data.camera().astype(np.float64) / 255.0
X = io.imread("./test_fotografico.jpg").astype(np.float64) / 255.0
m, n = X.shape

#sigma mi dice quanto velocemente si abbassa il kernel gaussiano

# Genera il filtro di blur
k = gaussian_kernel(24, 3) #restituisce una matrice 24x24
plt.imshow(k)
plt.show()

# Blur with openCV
X_blurred = cv.filter2D(X, -1, k) #gexact con convoluzione
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
#vmin e vmak = range del plot, se ci sono valori fuori dal range vengono
#approssimati all'estremo più vicino
plt.title('Original')
plt.xticks([]), plt.yticks([]) #rimozione assi x e y
plt.subplot(122).imshow(X_blurred, cmap='gray', vmin=0, vmax=1)
plt.title('Blurred (CV Filter)')
plt.xticks([]), plt.yticks([])
plt.show()

# Blur with FFT
K = psf_fft(k, 24, X.shape)
plt.imshow(np.abs(K))
plt.show()

X_blurred = A(X, K)
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
plt.xticks([]), plt.yticks([]) #rimozione assi x e y
plt.subplot(122).imshow(X_blurred, cmap='gray', vmin=0, vmax=1)
plt.title('Blurred (FFT)')
plt.xticks([]), plt.yticks([])
plt.show()

# Genera il rumore
sigma = 0.02
np.random.seed(42) #cambiando il seed cambia il rumore e quindi i grafici
noise = np.random.normal(size=X.shape) * sigma #per downsampling X_blurred
#il rumore per immagine con downsampling deve avere dimensioni dell'immagine ridotta

# Aggiungi blur e rumore
y = X_blurred + noise
plt.imshow(y)
plt.show()

#metriche = misure d'errore
#idea: basandoci sulle metriche, fare un paragone rispetto all'immagine di partenza
PSNR = metrics.peak_signal_noise_ratio(X, y)
MSE = metrics.mean_squared_error(X, y)


# Visualizziamo i risultati
plt.figure(figsize=(30, 10))
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122).imshow(y, cmap='gray', vmin=0, vmax=1)
plt.title(f'Corrupted (PSNR: {PSNR:.2f} MSE {MSE:.5f})')
plt.xticks([]), plt.yticks([])
plt.show()

# Regolarizzazione
# Funzione da minimizzare
def f(x, L):
    x = x.reshape((m, n))
    TV = totvar(x)
    Ax = A(x, K)
    return 0.5 * np.sum(np.square(Ax - y)) + L * TV

# Gradiente della funzione da minimizzare
def df(x, L):
    x = x.reshape((m, n))
    LDTV = L * grad_totvar(x)
    ATAx = AT(A(x,K),K)
    d = ATAx - y
    return d.reshape(m * n) + LDTV.reshape(m * n)

x0 = y.reshape(m*n)
lambdas = [0.045, 0.046, 0.048, 0.0504]
PSNRs = []
MSEs = []
images = []

from scipy.optimize import minimize

# Ricostruzione per diversi valori del parametro di regolarizzazione
for i, L in enumerate(lambdas):
    # Esegui la minimizzazione con al massimo 50 iterazioni
    max_iter = 50
    res = minimize(f, x0, (L), method='CG', jac=df, options={'maxiter':max_iter})

    #f = funzione
    #x0 = punto iniziale
    #L = lambda
    #method = 'CG' --> gradiente coniugato
    #jac = gradiente della funzione

    # Aggiungi la ricostruzione nella lista images
    X_curr = res.x.reshape(X.shape)
    images.append(X_curr)

    # Stampa il PSNR e MSE per il valore di lambda attuale
    PSNR = metrics.peak_signal_noise_ratio(X, X_curr)
    PSNRs.append(PSNR)

    MSE = metrics.mean_squared_error(X, X_curr)
    MSEs.append(MSE)

    # Calcolo della discrepanza tra il rumore simulato e il rumore osservato
    diff = np.sum(np.square(y-A(X_curr, K)))

    # Calcolo della tolleranza
    tolerance = 1.1 * np.sum(np.square(noise))

    print(f'PSNR: {PSNR:.3f} (\u03BB = {L:.5f})')
    print(f'MSE: {MSE:.6f} (\u03BB = {L:.5f})')
    print(f'Discrepanza: {diff:.3f} (toll: {tolerance:.3f})\n')


# Visualizziamo i risultati
plt.plot(lambdas,PSNRs)
plt.title('PSNR per $\lambda$')
plt.ylabel("PSNR")
plt.xlabel('$\lambda$')
plt.show()

plt.plot(lambdas,MSEs)
plt.title('MSE per $\lambda$')
plt.ylabel("MSE")
plt.xlabel('$\lambda$')
plt.show()

plt.figure(figsize=(30, 10))

plt.subplot(1, len(lambdas) + 2, 1).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title("Originale")
plt.xticks([]), plt.yticks([])
plt.subplot(1, len(lambdas) + 2, 2).imshow(y, cmap='gray', vmin=0, vmax=1)
plt.title("Corrotta")
plt.xticks([]), plt.yticks([])

# plotto le ricostruzioni
for i, L in enumerate(lambdas):
  plt.subplot(1, len(lambdas) + 2, i + 3).imshow(images[i], cmap='gray', vmin=0, vmax=1)
  plt.title(f"Ricostruzione ($\lambda$ = {L:.5f})")
plt.show()
