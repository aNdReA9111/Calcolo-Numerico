# Recap function


def upper_text(text):
    return text.upper()


upper_text2 = lambda text: text.upper()

stringa = "Hello world"

# Function call
print("upper_text: ", upper_text(stringa))
print("upper_text2: ", upper_text2(stringa))


# storing the function in a variable
def hello(func):
    greeting = func("Greetings!")
    print(greeting)


hello(upper_text2)


def hello2(func, text="Hello world!"):
    greeting = func(text)
    print(greeting)


hello2(upper_text2, stringa)
hello2(upper_text2)

import numpy as np
import matplotlib.pyplot as plt


# Esercizio 2.1
f = lambda x: x**3 + 4 * x * np.cos(x) - 2
df = lambda x: ...
g1 = lambda x: (2 - x**3) / (4 * np.cos(x))

xTrue = 0.536839
fTrue = f(xTrue)
print("fTrue = ", fTrue)

xplot = np.linspace(0, 2)
fplot = f(xplot)

plt.plot(xplot, fplot)
plt.plot(xTrue, fTrue, "or", label="True")

tolx = 10 ** (-10)
tolf = 10 ** (-6)
maxit = 100
x0 = ...

[sol_g1, iter_g1, err_g1, vecErrore_g1] = succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
print("Metodo approssimazioni successive g1 \n x =", sol_g1, "\n iter_new=", iter_g1)

plt.plot(sol_g1, f(sol_g1), "*", label="g1")

[sol_newton, iter_newton, err_newton, vecErrore_newton] = newton(
    f, df, tolf, tolx, maxit, xTrue, x0
)
print("Metodo Newton \n x =", sol_newton, "\n iter_new=", iter_newton)

plt.plot(sol_newton, f(sol_newton), "+b", label="Newton")
plt.grid()
plt.legend()
plt.show()

# GRAFICO Errore vs Iterazioni

# g1
plt.plot(vecErrore_g1, ".-", color="blue")
# Newton
plt.plot(vecErrore_newton, ".-", color="red")

plt.legend(("g1", "newton"))
plt.xlabel("iter")
plt.ylabel("errore")
plt.title("Errore vs Iterazioni")
plt.grid()
plt.show()

# Esercizio 2.2
f = lambda x: x - x ** (1 / 3) - 2
df = lambda x: ...
g1 = lambda x: x ** (1 / 3) + 2

xTrue = 3.5213
fTrue = f(xTrue)
print("fTrue = ", fTrue)

xplot = np.linspace(3, 5)
fplot = f(xplot)

plt.plot(xplot, fplot)
plt.plot(xTrue, fTrue, "^r", label="True")

tolx = 10 ** (-10)
tolf = 10 ** (-6)
maxit = 100
x0 = 3

[sol_g1, iter_g1, err_g1, vecErrore_g1] = succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
print("Metodo approssimazioni successive g1 \n x =", sol_g1, "\n iter_new=", iter_g1)

plt.plot(sol_g1, f(sol_g1), "o", label="g1")

[sol_newton, iter_newton, err_newton, vecErrore_newton] = newton(
    f, df, tolf, tolx, maxit, xTrue, x0
)
print("Metodo Newton \n x =", sol_newton, "\n iter_new=", iter_newton)

plt.plot(sol_newton, f(sol_newton), "+b", label="Newton")
plt.grid()
plt.legend()
plt.show()

# GRAFICO Errore vs Iterazioni

# g1
plt.plot(vecErrore_g1, ".-", color="blue")
# Newton
plt.plot(vecErrore_newton, ".-", color="red")

plt.legend(("g1", "newton"))
plt.xlabel("iter")
plt.ylabel("errore")
plt.title("Errore vs Iterazioni")
plt.grid()
plt.show()
