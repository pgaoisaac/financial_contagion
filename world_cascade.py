"""Script for determining failure rankings for set of world countries, BIS statistics 2015
Priya Veeraraghavan 2017"""

import numpy as np
from model import run_cascade
from model import make_A_from_data

claims_9B = np.loadtxt('claims_2015_9B.csv', delimiter=',', skiprows=1)
claims_9D = np.loadtxt('claims_2015_9D.csv', delimiter=',', skiprows=1)

n = claims_9B.shape[0]
cross_holding_frac = np.tile([1.0/3], n)

# us billions 2015
GDP_2015 = {'Australia' : 1333,
           'Austria': 376.95,
           'Belgium': 455.086,
           'Canada': 1553,
           'Chile': 240.796,
           'France': 2419,
            'Germany': 3363,
            'Greece': 194.851,
            'Ireland': 283.703,
            'Italy': 1821,
            'Japan': 4383,
            'Korea': 1378,
            'Netherlands': 750.284,
            'Portugal': 199.113,
            'Spain': 1193,
            'Switzerland': 670.79,
            'Turkey': 717.88,
            'UK': 2861,
            'US': 18037
           }

min_gdp = min(GDP_2015.values())
for key, value in GDP_2015.items():
    GDP_2015[key] = value/min_gdp

p_GDP = np.array([x[1] for x in sorted(GDP_2015.items(), key=lambda x: x[0])])
countries = [x[0] for x in sorted(GDP_2015.items(), key=lambda x: x[0])]
A_9B = make_A_from_data(claims_9B, cross_holding_frac)
A_9D = make_A_from_data(claims_9D, cross_holding_frac)

def run_claims_list(A, p_GDP, countries, theta):
    """
    Run the cascade algorithm for a certain holdings sensitivity matrix A
    
    Arguments: 
        A: sensitivity matrix
        p_GDP: prices proportional to GDP
        countries: list of countries, same order as p_GDP
    """
    failures = {}
    for i in range(len(p_GDP)):
        D = np.eye(A.shape[0])
        p = np.copy(p_GDP)
        p.itemset(i, np.matmul(D, p_GDP)[i]/2)
        
        failure_list = run_cascade(A, D, p, theta, v_0=p_GDP)
        
        # transform into countries
        countries_failed = map(lambda lst: ",".join([countries[i] for i in lst]), failure_list)
        
        # resize each of the lists to make it a useable array
        max_timesteps = len(p_GDP)

        countries_failed_array = np.concatenate([countries_failed] + np.tile(["."], (max_timesteps-len(countries_failed), 1)).tolist(), axis=0)
                                                
        #countries_failed_array = np.concatenate([np.concatenate([x, np.tile(".", max_timesteps-len(x))], axis=0)], axis=0)
        assert countries_failed_array.shape[0] == max_timesteps
        
        #countries_failed_by_timestep = dict(zip(range(len(countries_failed)), countries_failed))
        failures[countries[i]] = countries_failed_array
    return failures


def zip_two_failure_dicts(dict1, dict2):
    """Assumed the two failure dicts from a constant theta
    
    Args:
        dict1, dict2 of the form {countryname: countries_failed_array}}
                                                
    Returns:
        dict {countryname: {0: [[countryfromdict1, countryfromdict2]
                                [countryfromdict1, .]]}}"""
    zipped = dict()
    
    for country in dict1.keys():
        country_dict = np.concatenate([np.expand_dims(dict1[country], axis=1), np.expand_dims(dict2[country], axis=1)], axis=1)
        zipped[country] = country_dict
        
    return zipped

def find_discrepancies(dict1, dict2):
    """Given dicts from two analyses of cross holdings with the SAME countries, find discrepancies in failure order"""
    for country in dict1.keys():
        cd1 = dict1[country]
        cd2 = dict2[country]
        for i in range(len(cd1)):
            if set(cd1[i].split(",")) != set(cd2[i].split(",")):
                print "Discrepancy between models with %s failed at timestep %d: %s vs %s\n" % (country, i, cd1[i], cd2[i])

theta_list = [.6, .9, .935, .95, .975]
for theta in theta_list:
    failures_9B = run_claims_list(A_9B, p_GDP, countries, theta)
    failures_9D = run_claims_list(A_9D, p_GDP, countries, theta) 
    zipped_failures = zip_two_failure_dicts(failures_9B, failures_9D)
    

    failure_array = np.concatenate(map(lambda country: np.concatenate([[["Failed:", country]], 
                                                       zipped_failures[country]], axis=0), zipped_failures.keys()), axis=1)
    
    find_discrepancies(failures_9B, failures_9D)
    outfile = "failures_%d.tsv" % int(theta*1000)
    np.savetxt(outfile, failure_array, fmt="%s", delimiter="\t")
