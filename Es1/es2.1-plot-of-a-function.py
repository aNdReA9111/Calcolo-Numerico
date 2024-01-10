import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-0, 10, 1000)
y1= np.sin(x)
y2= np.cos(x)

plt.plot(x, y1, 'r')
plt.plot(x, y2, 'y')

plt.xlabel("x")
plt.ylabel("y")
plt.legend(['sin(x)', 'cos(x)'])
plt.title("sin(x) and cos(x)")

plt.show()
