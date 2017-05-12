"""Verification of datasets provided by Elliott, et al. 
   Priya Veeraraghavan 2017"""

import numpy as np
from model import make_A_from_data
from model import run_cascade

IMB_2011=np.array([[0,174862,1960,5058,40311,6679,27015,292181],
          [198304,0,2663,2762,227813,2271,54178,187771],
          [39458,32977,0,150,2302,8077,1001,10939],
          [33689,95329,488,0,17429,17528,8218,141275],
          [329550,133954,444,1293,0,2108,29938,60099],
          [21817,30208,51,517,3188,0,78005,21214],
          [115162,146096,292,4696,26939,21620,0,86298],
          [214982,458789,14730,139291,54194,5971,394007,0]])

IMB_used = np.delete(
    np.delete(
        np.delete(
            np.delete(IMB_2011, 3, 0),
                6, 0),
        3, 1),
    6,1)

URB_2011=np.array([[0,174862,2017,5505,43013,6676,27535,298229],
                   [209996,0,3024,3136,234434,2555,54489,187755],
                   [44353,13355,0,192,2186,8121,969,10537],
                   [27462,95329,474,0,15457,4340,7844,127380],
                   [332345,133954,444,1306,0,2209,30970,59356],
                   [21760,30208,51,568,3183,0,75951,20940],
                   [114702,146096,291,4944,27726,23116,0,83129],
                   [206946,458789,11711,137342,50284,5760,391167,0]])

URB_used = np.delete(
    np.delete(
        np.delete(
            np.delete(URB_2011, 3, 0),
                6, 0),
        3, 1),
    6,1)
cross_holding_frac = np.array([1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3])


# run the elliott algorithm
def reproduce_elliott(A):
    theta_list = [.9, .93, .935, .94]
    country_list = ['France', 'Germany', 'Greece', 'Italy', 'Portugal', 'Spain']

    p_GDP = np.array([11.61506276, 14.88284519, 1.267782427, 9.20083682, 1, 6.251046025]) # corresponds to p1
    p_08 = np.array([11.98745, 15.27615, 1.468619, 9.65272, 1.058577, 6.698745])
    v_0 = np.matmul(A, p_08)
    
    
    for theta in theta_list:
        failure_list = run_cascade(A, np.eye(A.shape[0]), p_GDP, theta, v_0=v_0)
        
        # transform into countries
        countries_failed = map(lambda lst: [country_list[i] for i in lst], failure_list)
        print theta, countries_failed
        
# run incorrect code
A_incorrect = make_A_from_data(IMB_used, cross_holding_frac)
A_incorrect.itemset((5, 3), .2)
print "Incorrect Reproduction"
reproduce_elliott(A_incorrect)

# run correct code
print "Correct Reproduction"
A = make_A_from_data(IMB_used, cross_holding_frac)
reproduce_elliott(A)