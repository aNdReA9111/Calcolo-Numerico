import matplotlib.pyplot as plt
import numpy as np

x = 2*np.linspace(-5, 5)
y = np.sin(x)
y1= x**2
y2= np.cos(x)

plt.plot(y, '--om')


fig, ax = plt.subplot(rows=2, ncols=2)
ax[0][0] = plot(x, y1)
ax[0][0] = plot(x, y2)
ax[0][1] = plot(x, y1)
ax[1][0] = plot(x, y2)
ax[1][0] = plot(x, y1)
ax[1][0] = plot(x, y2)
plt.show()
