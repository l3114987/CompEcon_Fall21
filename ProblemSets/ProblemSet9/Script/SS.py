# %%
import necessary_equations as ne
import scipy.optimize as opt

# then, let's write a function for the SS algorithm
    # guess r -> w
    # solve for b2 and b3 from hh_foc
    # use MC with b2, b3, n -> K, L
    # use K, L in get_r -> r'
    # check if r' = guess of r
    # loop again if not


def SS_solver(r_guess, bn_guesses, alpha, delta, A, sigma, chi, l, v, b, beta):
    '''
    Solves for the SS of the economy
    '''
    
    xi = 0.8
    tol = 1e-8
    max_iter = 500
    dist = 7
    iter = 0
    r = r_guess
    bn = bn_guesses
    S = int((len(bn)+1)/2)
    
    
    while (dist > tol) & (iter < max_iter):
        w = ne.get_w(r, (alpha, delta, A)) # i- (a)
        sol = opt.root(ne.hh_foc, bn, 
        args=(r, w,(sigma, chi, l, v, b, beta))) # ii- (a)
        bn = sol.x
        euler_error_bn = sol.fun
        
        b_sp1 = bn[:S-1]
        n_s = bn[-S:]
        K = ne.get_K(b_sp1)
        L = ne.get_L(n_s)
        r_prime = ne.get_r(K, L, (alpha, delta, beta))
        dist = (r - r_prime) ** 2
        iter += 1
        r = xi * r + (1 - xi) * r_prime     
    success = iter < max_iter

    return r, bn, success, euler_error_bn


# %%
