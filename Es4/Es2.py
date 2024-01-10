import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import lu_factor as LUdec

case = 2
m = 10
m_plot = 1000

# Grado polinomio approssimante (≠ da interpolazione)
grado_pol_approssimante = [1, 2, 3, 5, 7]


if case==0:
    x = np.linspace(-1,1,m)     # crea un array di numeri equidistanti all'interno di un intervallo specificato
    y = np.exp(x/2)

    x1 = np.linspace(-1,1,m_plot)
    y1 = np.exp(x1/2)
elif case==1:                   # fa particolarmente schifo l'interpolazione
    x = np.linspace(-1,1,m)
    y = 1/(1+25*(x**2))

    x1 = np.linspace(-1,1,m_plot)
    y1 = 1/(1+25*(x1**2))
elif case==2:
    x = np.linspace(0,2*np.pi,m)
    y = np.sin(x)+np.cos(x)

    x1 = np.linspace(0,2*np.pi,m_plot)
    y1 = np.sin(x1)+np.cos(x1)

for n in grado_pol_approssimante:
  A = np.zeros((m, n+1))  #n+1 perchè stiamo prendendo il grado del polinomio + il termine noto

  for i in range(n+1):
    A[:, i] = x**i  #x^i

  U, s, Vh = scipy.linalg.svd(A)

  alpha_svd = np.zeros(n+1)
  for i in range(n+1):
    ui = U[:, i]
    vi = Vh[i, :]

    alpha_svd += ui@y * vi / s[i]   #alpha_svd = alpha_svd + np.dot(ui, y) * vi / s[i]

  print(alpha_svd)  #vettore dei coefficienti calcolato tramite vettore dei coefficienti

  #x sono i dati da approssimare x_plot servono per approssimare il grafico,
  # infatti ha dim m_plot (abbastanza fitto, 100 in questo caso)
  x_plot = np.linspace(x[0], x[-1], m_plot) #vettore solo per fare il grafico (per predizione)
  A_plot = np.zeros((m_plot, n+1))

  for i in range(n+1):
    A_plot[:, i] = x_plot**i

  y_plot = A_plot@alpha_svd

  plt.plot(x, y, 'o')
  plt.plot(x1, y1, '-y')  #per plottare funzione
  plt.plot(x_plot, y_plot, 'r')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(f'Regressione polinomiale di grado {n}')
  plt.grid()
  plt.show()

  # res = np.zeros(m)
  # for i in range(m):
  #   res[i] = np.linalg.norm(y[i]-y_g[i], 2)

  #distanza dal punto calcolato per approssimazione dalla retta effettiva ≈ più sono piccoli e
  # più i dati sono fittati (approssimati) meglio
  res = np.linalg.norm(A@alpha_svd-y, 2)
  print(f'Residual: {res}')




