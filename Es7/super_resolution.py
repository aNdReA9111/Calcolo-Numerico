import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics, io
from scipy import signal
from numpy import fft
from utils_SR import psf_fft, A, AT, gaussian_kernel


# Immagine in floating point con valori tra 0 e 1
#X = data.camera().astype(np.float64) / 255.0
X = io.imread("./test_spazio.png").astype(np.float64) / 255.0
m, n = X.shape
sf = 2
KER_SIZE = 7

# Genera il filtro di blur
k = gaussian_kernel(KER_SIZE, 0.5)
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
plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# Blur with FFT
K = psf_fft(k, KER_SIZE, X.shape)
plt.imshow(np.abs(K))
plt.show()

X_blurred = A(X, K, sf)
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
plt.xticks([]), plt.yticks([]) #rimozione assi x e y
plt.subplot(122).imshow(X_blurred, cmap='gray', vmin=0, vmax=1)
plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# Genera il rumore
sigma = 0.02    #sigma mi dice quanto velocemente si abbassa il kernel gaussiano
np.random.seed(42) #cambiando il seed cambia il rumore e quindi i grafici
noise = np.random.normal(size=X_blurred.shape) * sigma


# Aggiungi blur e rumore
y = X_blurred + noise
#metriche = misure d'errore
#idea: basandoci sulle metriche, fare un paragone rispetto all'immagine di partenza

ATy = AT(y, K, sf)
PSNR = metrics.peak_signal_noise_ratio(X, ATy)
MSE = metrics.mean_squared_error(X, ATy)

# Visualizziamo i risultati
plt.figure(figsize=(30, 10))
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122).imshow(y, cmap='gray', vmin=0, vmax=1)
plt.title(f'Corrupted (PSNR: {PSNR:.2f} MSE {MSE:.5f})')
plt.xticks([]), plt.yticks([])
plt.show()

# Soluzione naive
from scipy.optimize import minimize
'''
# Funzione da minimizzare
def f(x):
    x = x.reshape((m, n)) #passaggio vettore-matrice
    Ax = A(x, K, sf)
    return 0.5 * np.sum(np.square(Ax - y))

# Gradiente della funzione da minimizzare
def df(x):
    x = x.reshape((m, n))
    ATAx = AT(A(x, K, sf), K, sf)
    d = ATAx - ATy
    return d.reshape(m * n) #passaggio matrice-vettore

# Minimizzazione della funzione
x0 =  ATy.reshape(m*n)
max_iter = 25
res = minimize(f, x0, method='CG', jac=df, options={'maxiter':max_iter, 'return_all':True})

#res contiene una serie di campi, flag che danno info sulla terminazione del metodo,
#numero di iterate, ...


# Per ogni iterazione calcola PSNR e MSE rispetto all'originale
PSNR = np.zeros(max_iter + 1)
MSE = np.zeros(max_iter + 1)

#allvecs = singole iterate
for k, x_k in enumerate(res.allvecs):
    PSNR[k] = metrics.peak_signal_noise_ratio(X, x_k.reshape(X.shape))
    MSE[k] = metrics.mean_squared_error(X, x_k.reshape(X.shape))

# Risultato della minimizzazione
X_res = res.x.reshape((m, n))

# PSNR e MSE dell'immagine corrotta rispetto all'oginale
starting_PSNR = np.full(PSNR.shape[0], metrics.peak_signal_noise_ratio(X, ATy))
starting_MSE = np.full(MSE.shape[0], metrics.mean_squared_error(X, ATy))

# Visualizziamo i risultati
ax2 = plt.subplot(1, 3, 1)
ax2.plot(PSNR, label="Soluzione naive")
ax2.plot(starting_PSNR, label="Immagine corrotta")
plt.legend()
plt.title('PSNR per iterazione')
plt.ylabel("PSNR")
plt.xlabel('itr')

ax2 = plt.subplot(1, 3, 2)
ax2.plot(MSE, label="Soluzione naive")
ax2.plot(starting_MSE, label="Immagine corrotta")
plt.legend()
plt.title('MSE per iterazione')
plt.ylabel("MSE")
plt.xlabel('itr')

plt.subplot(1, 3, 3).imshow(X_res, cmap='gray', vmin=0, vmax=1)
plt.title('Immagine Ricostruita')
plt.xticks([]), plt.yticks([])
plt.show()
'''
# Regolarizzazione
# Funzione da minimizzare
def f(x, L):
    nsq = np.sum(np.square(x))
    x  = x.reshape((m, n))
    Ax = A(x, K, sf)
    return 0.5 * np.sum(np.square(Ax - y)) + L*nsq

# Gradiente della funzione da minimizzare
def df(x, L):
    Lx = L * x
    x = x.reshape(m, n)
    ATAx = AT(A(x,K, sf),K,sf)
    d = ATAx - ATy
    return d.reshape(m * n) + Lx

x0 = ATy.reshape(m*n)
lambdas =  np.linspace(0.01, 0.25,20)#[0.005, 0.006,0.007 , 0.01, 0.02, 0.03, 0.04]
PSNRs = []
MSEs = []
images = []

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
    print(f'PSNR: {PSNR:.3f} (\u03BB = {L:.5f})')
    print(f'MSE: {MSE:.6f} (\u03BB = {L:.5f})')

    # Calcolo della discrepanza tra il rumore simulato e il rumore osservato
    diff = np.sum(np.square(y-A(X_curr, K, sf)))

    # Calcolo della tolleranza
    tolerance = 1.1 * np.sum(np.square(noise))
    print(f'Discrepanza: {diff:.3f} ({tolerance:.3f})\n')


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


for i, L in enumerate(lambdas):
  plt.subplot(1, len(lambdas) + 2, i + 3).imshow(images[i], cmap='gray', vmin=0, vmax=1)
  plt.title(f"Ricostruzione ($\lambda$ = {L:.5f})")
plt.show()
