import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def g(x):
    return x**3

def f(x):
    return 0.6 * x
def p(x):
    return 0.1 * x
def r(x):
    return 0.9 * x

# Generate x values
x = np.linspace(-1, 1, 1000)

# Calculate y values for each function
y_g = g(x)
y_f = f(x)
y_p = p(x)
y_r = r(x)

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(x, y_g, label="g(x) = x^3", color='red', linewidth=3)
plt.plot(x, y_f, label="f(x) = 0.6x", color='purple' )
plt.plot(x, y_p, label="f(x) = 0.1x", color='green', linestyle=':')
plt.plot(x, y_r, label="f(x) = 0.9x", color='black',linestyle=':')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of g(x) = x^3 and f(x) = 0.6x')

# Add a legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
