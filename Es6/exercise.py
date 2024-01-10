import numpy as np
import matplotlib.pyplot as plt

'''
PER L'ESAME CAMBIARE IL PASSO FISSO E OSSERVARE COME DECRESCONO LE FUNZIONI
'''

#f(x,y) = 3(x − 2)^2+ (y − 1)^2
def next_step(x,grad): # backtracking procedure for the choice of the steplength
  alpha=1.1
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10

  #condizione di armijo (nel backtack non serve la conndizione di curvatura poichè se si prende una misura
  # sbagliata si torna indietro e si riprova con un altro passo)

  #se dopo jmax volte non si riesce a trovare soluzione vuol dire che bisogna trovare un altro modo
  while ((f(x+alpha*p) > f(x)+c1*alpha*np.dot(grad,p)) and j<jmax ):
    alpha= rho*alpha
    j+=1
  if (j>jmax):
    return -1
  else:
    print('alpha=',alpha)
    return alpha


def minimize(f,grad_f,x0,step,maxit,tol,xTrue,fixed=True): # funzione che implementa il metodo del gradiente
  #declare x_k and gradient_k vectors
  # x_list only for logging
  x_list=np.zeros((len(x0),maxit+1))  #noi lavoreremo sempre in R^2, quindi x0 avrà sempre dimensione 2

  norm_grad_list=np.zeros(maxit+1)
  function_eval_list=np.zeros(maxit+1)
  error_list=np.zeros(maxit+1)

  #initialize first values
  x_last = x0
  x_list[:,0] = x_last
  k=0

  function_eval_list[k]=f(x0)
  error_list[k]=np.linalg.norm(x_last-xTrue)
  norm_grad_list[k]=np.linalg.norm(grad_f(x0))

  while (np.linalg.norm(grad_f(x_last))>tol and k < maxit ):
    k=k+1
    grad = grad_f(x_last)#direction is given by gradient of the last iteration

    if fixed:
        # Fixed step
        step = step
    else:         #se non usiamo il passo fisso
        # backtracking step
        step = next_step(x_last, grad)

    if(step==-1):
      print('non convergente')
      return (k) #no convergence

    x_last=x_last-(step*grad)
    x_list[:,k] = x_last

    function_eval_list[k]=f(x_last)
    error_list[k]=np.linalg.norm(x_last-xTrue)
    norm_grad_list[k]=np.linalg.norm(grad_f(x_last))

  function_eval_list = function_eval_list[:k+1]
  error_list = error_list[:k+1]
  norm_grad_list = norm_grad_list[:k+1]

  print('iterations=',k)
  print('last guess: x=(%f,%f)'%(x_list[0,k],x_list[1,k]))

  return (x_last,norm_grad_list, function_eval_list, error_list, x_list, k)


# Es 1.2
def f(vec):
    x, y = vec
    fout = 3*((x-2)**2) +(y-1)**2
    return fout

def grad_f(vec):
    x, y = vec
    dfdx = 3*((2*x)-4)
    dfdy = 2*y-2
    return np.array([dfdx,dfdy])

x = np.linspace(-1.5, 3.5)
y = np.linspace(-1, 5, 100)

X, Y = np.meshgrid(x, y)
vec = np.array([X,Y])
Z=f(vec)

fig = plt.figure(figsize=(15, 8))

ax = plt.axes(projection='3d')
ax.set_title('$f(x)=(x-1)^2 + (y-2)^2$')
ax.view_init(elev=50., azim=30)
s = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(s)
plt.show()

fig = plt.figure(figsize=(8, 5))
contours = plt.contour(X, Y, Z, levels=1000)
plt.title('Contour plot $f(x)=(x-1)^2 + (y-2)^2$')
fig.colorbar(contours)

'''
xlist è una matrice con 1000 colonne e 2 righe, la prima e la ascissa e la seconda è l'ordinata
  :k prende le prime k colonne riempite
'''
step = 0.002
maxitS=1000
tol=1.e-5
x0 = np.array([3, 5]) #punto iniziale (a piacere)
xTrue = np.array([2, 1]) #soluzione esatta della funzione
(x_last,norm_grad_listf, function_eval_listf, error_listf, xlist, k)= minimize(f,grad_f,x0,step,maxitS,tol,xTrue,fixed=True)
plt.plot(xlist[0, :k], xlist[1, :k], '*-')
(x_last,norm_grad_list, function_eval_list, error_list, xlist, k)= minimize(f,grad_f,x0,step,maxitS,tol,xTrue,fixed=False)
plt.plot(xlist[0, :k], xlist[1, :k],'*-')
plt.legend(['fixed', 'backtracking'])
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.semilogy(norm_grad_listf)
ax1.semilogy(norm_grad_list)
ax1.set_title('$\|\\nabla f(x_k)\|$')
ax2.semilogy(function_eval_listf)
ax2.semilogy(function_eval_list)
ax2.set_title('$f(x_k)$')
ax3.semilogy(error_listf)
ax3.semilogy(error_list)
ax3.set_title('$\|x_k-x^*\|$')
fig.tight_layout()
fig.legend(['fixed', 'backtracking'], loc='lower center', ncol=4)
plt.show()

# Es 1.3
def f(vec):
    x, y = vec
    fout = 100*((y - x**2)**2) + ((1 - x)**2)
    return fout

def grad_f(vec):
    x, y = vec
    dfdx = 100*(4*(x**3)- 4*x*y) + 2*x - 2
    dfdy = 100*(2*y- 2*(x**2))
    return np.array([dfdx,dfdy])

x = np.linspace(-2, 2)
y = np.linspace(-1, 3)
X, Y = np.meshgrid(x, y)
vec = np.array([X,Y])
Z=f(vec)


fig = plt.figure(figsize=(15, 8))

ax = plt.axes(projection='3d')
ax.set_title('$f(x)=(1-x)^2+100*(y-x^2)^2$')
ax.view_init(elev=50., azim=30)
s = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(s)
plt.show()

fig = plt.figure(figsize=(8, 5))
contours = plt.contour(X, Y, Z, levels=1000)
plt.title('Contour plot $f(x)=(1-x)^2+100*(y-x^2)^2$')
fig.colorbar(contours)


step = 0.001
maxitS=1000
tol=1.e-5
x0 = np.array([-0.5, 1])
xTrue = np.array([1,1])
(x_last,norm_grad_listf, function_eval_listf, error_listf, xlist, k)= minimize(f,grad_f,x0,step,maxitS,tol,xTrue,fixed=True)
plt.plot(xlist[0, :k], xlist[1, :k],'*-')
(x_last,norm_grad_list, function_eval_list, error_list, xlist, k)= minimize(f,grad_f,x0,step,maxitS,tol,xTrue,fixed=False)
plt.plot(xlist[0, :k], xlist[1, :k],'*-')
plt.legend(['fixed', 'backtracking'])

plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.semilogy(norm_grad_listf)
ax1.semilogy(norm_grad_list)
ax1.set_title('$\|\\nabla f(x_k)\|$')
ax2.semilogy(function_eval_listf)
ax2.semilogy(function_eval_list)
ax2.set_title('$f(x_k)$')
ax3.semilogy(error_listf)
ax3.semilogy(error_list)
ax3.set_title('$\|x_k-x^*\|$')
fig.legend(['fixed', 'backtracking'], loc='lower center', ncol=4)
fig.tight_layout()
plt.show()
