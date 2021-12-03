# %%
# Import necessary packages
import numpy as np
from numpy.core.fromnumeric import argmin
import scipy.optimize as opt
import random
import matplotlib.pyplot as plt

# %%
def pro_q(p, E, F, alpha):
    '''
    Computes the production quantity at t = 0
    
    Args
    p - price at which the probability is computed
    E, F - parameters of the logistic function
    alpha - parameter that depends on the expected state of the system (w1)
    (w1, expected weather condition)
   
    
    Returns:
    Production quantity for a given p, E, F, and alpha
    '''
    Q0 = (E-F*p) * alpha
    return Q0

def beta(alpha, piG, piB):
    '''
    Computes the probability of an item being sold in the actual state of the system at t=1 and 2
    
    A function of alpha and the elements of the transition probability matrix
    
    piG - The probability that the actual weather is good
    piB - The probability that the actual weather is bad

    '''
    betas = alpha * piG + (1-alpha) * piB
    return betas
    

def sales(p, E, F, alpha, piG, piB):
    '''
    Computes the realized sales quantity at t = 1
    
    Args
    beta - parameter that depends on the actual state of the system (w2) 
    resulting from a function of alpha and the elements of the transition probability matrix
    
    Returns:
    Realized sales quantity at t = T
    '''
    
    Q_T =  (E-F*p)* beta(alpha, piG, piB)
    return Q_T




def price_2(p, E, F, alpha, piG, piB, K):
    ''' 
    Computes the discounted price at t = 2 given the remaining inventory from t = 1
    
    Arg
    K - parameter that penalizes the remaining inventory
    
    Returns:
    Price at t = 2 
    '''
    P2 = p - K * (pro_q(p, E, F, alpha)-sales(p, E, F, alpha, piG, piB)) * (1-beta(alpha, piG, piB))
    return P2

def boundary_con(p, E, F, alpha, piG, piB):
    '''
    Returns the boundary condition for the case in which the product is sold out at t=1
    '''
    
    bound_con = sales(p, E, F, alpha, piG, piB)+ p * (-F * beta(alpha, piG, piB))
    
    return bound_con




def FOC(p, E, F, alpha, piG, piB, K):
    '''
    Returns the first order condition
    
    Also pass the relevant row of the transition probability matrix (pi)
    and the value computed for both states from the future period
    
    
    
    '''
    
    FOC_comp = sales(p, E, F, beta(alpha, piG, piB),piG, piB) 
    + p * (-F * beta(alpha, piG, piB)) 
    + (E-F(p-K(1-beta(alpha, piG, piB))(alpha-beta(alpha, piG, piB))(E-F*p))) * (1+K*F*(alpha-beta(alpha, piG, piB)-alpha*beta(alpha, piG, piB)+((beta(alpha, piG, piB))**2))) 
    + (p - K * (pro_q(p, E, F, alpha)-sales(p, E, F, beta(alpha, piG, piB),piG, piB)) * (1-beta(alpha, piG, piB))) * (-F*beta(alpha, piG, piB) - K*(F**2)*(alpha-beta(alpha, piG, piB)-alpha*beta(alpha, piG, piB)+((beta(alpha, piG, piB))**2)))
    
    return FOC_comp 




def optimal(E, F, alpha, K, pi):
    '''
    Computes the optimum price and value functions for each state
    
    Args:
    curr_state - expected weather condition at t = 0 / the actual weather condition at t = 1 and 2
    (w1, if it is 0, the weather is expected to be good and the actual weather is either good or bad.
    If it is 1, the weather is expected to be bad and the actual weather is always bad.)   
       
    '''    
    price_list = []
    value_func_G = []
    value_func_B = []
    pi_0 = pi[0]
    pi_1 = pi[1]
    
    ##Boundary value
    # Optimal price for the case in which the product is sold out at t=1
    # Expected weather condition is good
    # Use appropriate row of transition probability matrix
    piG = pi_0[0]
    piB = pi_0[1]
    min_price = opt.root(boundary_con, 0, args = (E, F, alpha, piG, piB))
    p = min_price.x[0]
    val_G = p * sales(p, E, F, beta(alpha, piG, piB), piG, piB)
    price_list.append(p)    
    value_func_G.append(val_G)
    
    # Optimal price for the case in which the product is sold out at t=1
    # Expected weather condition is bad
    # Use appropriate row of transition probability matrix
    piG = pi_1[0]
    piB = pi_1[1]
    min_price = opt.root(boundary_con, 0, args = (E, F, alpha, piG, piB))
    p = min_price.x[0]
    val_B = p * sales(p, E, F, beta(alpha, piG, piB), piG, piB)
    price_list.append(p) 
    value_func_G.append(val_B)
    
               
      
    # Expected weather condition is good
    # Use appropriate row of transition probability matrix
    piG = pi_0[0]
    piB = pi_0[1]
    
    # Compute optimal price for the selling period
    root_result = opt.root(FOC, p, args = (E, F, alpha, piG, piB, K))
    p = root_result.x[0]
        
    # Compute value functions for the selling period
    val_G = p * sales(p, E, F, beta(alpha, piG, piB), piG, piB) + (p - K * (pro_q(p, E, F, alpha)-sales(p, E, F, beta(alpha, piG, piB), piG, piB)) * (1-beta(alpha, piG, piB)))*(E-F(p-K(1-beta(alpha, piG, piB))(alpha-beta(alpha, piG, piB))(E-F*p)))                  
    val_B = p * sales(p, E, F, (1-beta(alpha, piG, piB))) + (p - K * (pro_q(p, E, F, beta(alpha, piG, piB))-sales(p, E, F, (1-beta(alpha, piG, piB), piG, piB))) * (1-(1-beta(alpha, piG, piB))))*(E-F(p-K(1-(1-beta(alpha, piG, piB)))(alpha-(1-beta(alpha, piG, piB)))(E-F*p)))
    value_func_G.append(val_G)
    value_func_B.append(val_B)
    price_list.append(p)                
        
                    
    # Expected weather condition is bad
    piG = pi_1[0]
    piB = pi_1[1]
    # Compute optimal price for period i
    root_result = opt.root(FOC, p, args = (E, F, alpha, piG, piB, K))
    p = root_result.x[0]

    # Compute value functions for period i
    val_G = p * sales(p, E, F, beta(alpha, piG, piB), piG, piB) + (p - K * (pro_q(p, E, F, alpha)-sales(p, E, F, beta(alpha, piG, piB), piG, piB)) * (1-beta(alpha, piG, piB)))*(E-F(p-K(1-beta(alpha, piG, piB))(alpha-beta(alpha, piG, piB))(E-F*p)))                  
    val_B = p * sales(p, E, F, (1-beta(alpha, piG, piB))) + (p - K * (pro_q(p, E, F, beta(alpha, piG, piB))-sales(p, E, F, (1-beta(alpha, piG, piB), piG, piB))) * (1-(1-beta(alpha, piG, piB))))*(E-F(p-K(1-(1-beta(alpha, piG, piB)))(alpha-(1-beta(alpha, piG, piB)))(E-F*p)))
    value_func_G.append(val_G)
    value_func_B.append(val_B)
    price_list.append(p)
    return price_list, value_func_G, value_func_B
# %%
