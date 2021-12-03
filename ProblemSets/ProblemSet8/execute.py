# %%
import numpy as np
import scipy.optimize as opt
import random
import matplotlib.pyplot as plt
from function5 import optimal

# %%
# Set parameter values
E = 15
F = 0.001
alpha = 0.8
K = 100
T = 10
pi = np.array([[0.7, 0.3],[0.2, 0.8]])

# Store optimal prices and corresponding value functions 
Opt_prices, value_G, value_B = optimal(E, F, alpha, K, pi)

Ex_weather = list(range(1,2))

# Plot the optimal prices 
plt.figure()
plt.scatter(Ex_weather[0:], Opt_prices[::-1])
plt.xlabel('Expected Weather condition')
plt.ylabel('Price')
plt.title('Optimal prices for the two period dynamic programming problem')
plt.savefig('Prices.png')

# Plot the value functions 
plt.figure()
fig, ax = plt.subplots()
ax.scatter(Ex_weather[0:], value_G[::-1], label = "Expected Good weather")
ax.scatter(Ex_weather[0:], value_B[::-1], label = "Expected Bad weather")
legend = ax.legend(loc = "upper left", shadow = False)
plt.ylim([1e-127,1e-121])
plt.xlabel('Expected Weather condition')
plt.ylabel('Value function')
plt.title('Value functions for the two perioddynamic programming problem')
plt.savefig('VFs.png')
# %%
