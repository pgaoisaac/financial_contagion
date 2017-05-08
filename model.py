"""Contagion model from Elliott, et al. Priya Veeraraghavan 2017"""

import numpy as np

def make_A_from_data(G, cross_holding_frac):
    """From cross holdings C generate A matrix as in Eliott et al
    
    Arguments:
        C: n x n matrix where the ijth entry is the ammount that country j owes country i. 
        cross_holding_frac: n x 1 matrix, for each country, what fraction of its debt is cross-held
        """
    n = G.shape[0]
    c = cross_holding_frac
    G_hat = np.multiply(np.matmul(np.ones((n, 1)), np.expand_dims(np.sum(G.T, axis=0), axis=0))*(1-c)/c, np.eye(n))
    G = G.T + G_hat
    C = np.divide(G, np.matmul(np.ones((n,1)), np.expand_dims(np.sum(G, axis=0), axis=0)))
    C_hat = np.multiply(C, np.eye(n))
    A = np.matmul(C_hat, np.linalg.inv(np.eye(n)-C+C_hat))
    return np.around(A, decimals=2)
    
# Code to run the cascade  
def run_cascade(A, D, p, theta, v_0=[]):
    # if no comparator value, construct v_0 from A and p
    p = np.around(p, decimals=5)
    if len(v_0) == 0:
        v_0 = np.matmul(A, np.matmul(D, p))

    v_1 = np.matmul(A, np.matmul(D, p))
    n = A.shape[0]
    b = np.zeros(n)
    failed = set()
    failure_list = []
    t = 1
    v_min = theta * v_0
    
    
    while len(failed) < n:

        # when A*p < v_min, fail
        thresh = np.reshape(np.matmul(A,p), n) - v_min
        current_failures = set(np.where(thresh < 0)[0].tolist())
        new_failures = list(current_failures - failed)
        
        if len(new_failures) > 0:
            
            # failed countries lose v_min/2
            b = np.array([v_min[i]/2 if i in new_failures else 0 for i in range(n)])
            p = p - b
            
            # report failed countries in timestep
            failure_list.append(new_failures)
            failed = failed.union(current_failures)
            t += 1
            
        else:
            # when steady state occurs, no new failures
            break
        if t > 200:
            break
            
    return failure_list