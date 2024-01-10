import matplotlib.pyplot as plt
import numpy as np



x = np.linspace(-1,15, 100)
y = np.exp(x)

plt.plot(x,y, 'r')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['e^x'])
plt.title("e^x")

plt.show()
