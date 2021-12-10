# %%
import SS
import numpy as np

# %%
# Set parameters
alpha = 0.3
delta = 0.1
A = 1.0
sigma = 1.5
beta = 0.8
b = 0.501
v = 1.554
l = 1.0
P = 80
chi = np.ones(P)

# Make initial guesses
r_guess = 0.1
b_guesses = np.array([0.01 for i in range(P-1)])
n_guesses = np.array([0.8 for i in range(P)])
bn_guesses = np.array([b_guesses]+[n_guesses])

r ,bn, success, euler_error_bn = SS.SS_solver(r_guess, bn_guesses, 
                                           alpha, delta, A, sigma, chi, l, v, b,  beta)


# %%
