import matplotlib.pyplot as plt
import numpy as np

def fibonacci(n):
    if(n == 0 or n == 1):
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def golden_ratio(k):
    if(k>0):
        return fibonacci(k+1)/fibonacci(k)


x = np.arange(1, 14)
y = [golden_ratio(k) for k in x]

sigma = (1+np.sqrt(5))/2

plt.plot(x, y)
plt.plot(x, sigma*np.ones(np.shape(x)))

plt.legend(['ratio', r'$\phi$'])
plt.show()
