import matplotlib.pyplot as plt
import numpy as np

def fibonacci(n):
    if(n == 0 or n == 1):
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Function to compute the nth Fibonacci number
# def fibonacci(n):
#     if n <= 0:
#         return 0
#     elif n == 1:
#         return 1
#     else:
#         a, b = 0, 1
#         for _ in range(2, n + 1):
#             a, b = b, a + b
#         return b

# Function to compute the ratio of Fibonacci numbers
def fibonacci_ratio(k):
    ratios = []
    for i in range(1, k):
        ratio = fibonacci(i + 1) / fibonacci(i)
        ratios.append(ratio)
    return ratios

# Calculate Fibonacci ratios for a range of k values
k_values = 5
ratios = fibonacci_ratio(k_values)

# Compute the error with respect to the golden ratio (φ)
phi = (1 + np.sqrt(5)) / 2
errors = [abs(phi - ratio) for ratio in ratios]


x = np.arange(1, k_values)
plt.plot(x, ratios)
plt.plot(x, phi*np.ones(np.shape(x)))

plt.grid(True)
plt.legend(['ratio', r'$\phi$'])
plt.show()

# Create a plot of the error
plt.plot(errors, label='Error with respect to φ')
plt.xlabel('k')
plt.ylabel('Error')
plt.title('Error with respect to the golden ratio (φ)')
plt.legend()
plt.grid(True)
plt.show()
