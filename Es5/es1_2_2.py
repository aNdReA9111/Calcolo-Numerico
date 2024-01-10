import numpy as np
import matplotlib.pyplot as plt

# Exercise 1
# Function approssimazioni successive
def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
    i = 0
    err = np.zeros(maxit + 1, dtype=np.float64)
    err[0] = tolx + 1
    vecErrore = np.zeros(maxit + 1, dtype=np.float64)
    vecErrore[0] = np.abs(xTrue- x0)
    x = x0

    while i<maxit and ((np.abs(f(x))>tolf) or err[i]>tolx):  # scarto assoluto tra iterati
        x_new = g(x)

        #differenza in valore assoluto fra la x_nuova (appena calcolata) e la vecchia x
        err[i + 1] = np.abs(x_new-x)

        #costruiamo ad ogni iterata il vettore che dice l'errore nel calcolo della soluzione
        vecErrore[i + 1] = np.abs(xTrue-x_new)
        i = i + 1
        x = x_new

    err = err[0:i]
    vecErrore = vecErrore[0:i]
    return (x, i, err, vecErrore)

#df è la derivata prima di f
def newton(f, df, tolf, tolx, maxit, xTrue, x0=0):
    g = lambda x: x- f(x)/df(x)
    (x, i, err, vecErrore) = succ_app(f, g, tolf, tolx, maxit, xTrue, x0)
    return (x, i, err, vecErrore)


f = lambda x: x - x**(1/3) - 2
df = lambda x: 1 - (1/(3*np.cbrt(x**2)))
g = lambda x: x**(1/3) + 2

xTrue = 3.5213
fTrue = f(xTrue)
print("fTrue = ", fTrue)

xplot = np.linspace(3, 5)
fplot = f(xplot)

plt.plot(xplot, fplot)
plt.plot(xTrue, fTrue, "or", label="True")

tolx = 10 ** (-10)
tolf = 10 ** (-6)
maxit = 100
x0 = 3

[sol_g, iter_g, err_g, vecErrore_g] = succ_app(f, g, tolf, tolx, maxit, xTrue, x0)
print("Metodo approssimazioni successive g \n x =", sol_g, "\n iter_new=", iter_g)

plt.plot(sol_g, f(sol_g), "o", label="g")

[sol_newton, iter_newton, err_newton, vecErrore_newton] = newton(
    f, df, tolf, tolx, maxit, xTrue, x0
)
print("Metodo Newton \n x =", sol_newton, "\n iter_new=", iter_newton)

plt.plot(sol_newton, f(sol_newton), "ob", label="Newton")
plt.legend()
plt.grid()
plt.show()

# GRAFICO Errore vs Iterazioni

# g
plt.plot(vecErrore_g, ".-", color="blue")

# Newton
plt.plot(vecErrore_newton, ".-", color="red")

plt.legend(("g", "newton"))
plt.xlabel("iter")
plt.ylabel("errore")
plt.title("Errore vs Iterazioni")
plt.grid()
plt.show()
