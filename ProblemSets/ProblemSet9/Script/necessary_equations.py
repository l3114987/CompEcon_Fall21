# %%
import numpy as np

def get_L(n):
    '''
    Function to compute aggregate
    labor supplied
    '''
    L = n.sum()
    return L


def get_K(b):
    '''
    Function to compute aggregate
    capital supplied
    '''
    K = b.sum()
    return K

def get_r(K, L, params):
    '''
    Compute the interest rate from
    the firm's FOC
    '''
    alpha, delta, A = params
    r = alpha * A * (L / K) ** (1 - alpha) - delta
    return r


def get_w(r, params):
    '''
    Solve for the w that is consistent
    with r from the firm's FOC
    '''
    alpha, delta, A = params
    w = (1 - alpha) * A * ((alpha * A) / (r + delta)) ** (alpha / (1 - alpha))
    return w



def get_c(r, w, b_s, b_sp1, n):
    '''
    Find consumption using the budget constraint
    and the choice of savings (b_sp1)
    '''
    c = (1 + r) * b_s + w * n - b_sp1
    return c

def mu_c_func(c, sigma):
    '''
    Marginal utility of consumption
    '''
    mu_c = c ** -sigma
    
    
    return mu_c


def mu_n_func(n, chi, b, l, v):
    '''
    Marginal disutility of labor supply
    '''
    mu_n = chi * (b / l) * ((n / l) ** (v - 1)) * (1 - ((n/l) ** v)) ** ((1 - v) / v)
    return mu_n



def get_c(r, w, b_s, b_sp1, n):
    '''
    Find consumption using the budget constraint (equation 4.6)
    and the choice of savings (b_sp1)
    '''
    c = (1 + r) * b_s + w * n - b_sp1
    return c



def hh_foc(bn_list, r, w, params):
    '''
    Define the household first order conditions
    '''
    sigma, chi, l, v, b, beta = params
    
    P = int((len(bn_list)+1)/2)
   
    b_s = []
    b_sp1 = []
    n_s = []
   
    b_s.append([0.0, bn_list[:P-2]])
    b_sp1.append([bn_list[:P-2], 0.0])
    n_s.append(bn_list[-P:])
    
    euler_error_bn = np.zeros(2*P-1)
   
    for i in range(P-1):
        c_s = get_c(r, w, b_s[i], b_sp1[i+1], n_s[i])
        mu_c = mu_c_func(c_s, sigma)
        c_s1 =get_c(r, w, b_s[i+1], b_sp1[i+2], n_s[i+1])
        mu_c1 = mu_c_func(c_s1, sigma)
        euler_error_bn[i] = mu_c - beta * (1 + r) * mu_c1
    
    for j in range(P):
        c_s = get_c(r, w, b_s[j], b_sp1[j+1], n_s[j])
        mu_c = mu_c_func(c_s, sigma)
        mu_n = mu_n_func(n_s[j],chi[j], b, l, v)   
        euler_error_bn[P-1+j] = w * mu_c - mu_n 
       
    return euler_error_bn


# %%
