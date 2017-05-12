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
        
    Return: {countryname: ['countriesfailedt1,countriesfailedt1', 'countrysfailedt2,contriesfailedt2', '.']}
    """
    failures = {}
    for i in range(len(p_GDP)):
        D = np.eye(A.shape[0])
        p = np.copy(p_GDP)
        p.itemset(i, np.matmul(D, p_GDP)[i]/2)
        
        failure_list = run_cascade(A, D, p, theta, v_0=np.matmul(A, p_GDP))
        
        # transform into countries
        countries_failed = map(lambda lst: ",".join([countries[i] for i in lst]), failure_list)
        
        # resize each of the lists to make it a useable array
        max_timesteps = len(p_GDP)

        countries_failed_array = np.concatenate([countries_failed] + np.tile([" "], (max_timesteps-len(countries_failed), 1)).tolist(), axis=0)                                        
        assert countries_failed_array.shape[0] == max_timesteps
        
        failures[countries[i]] = countries_failed_array
    return failures



# Vary theta and record size of cascade
# Some countries like Greece had sensitive thresholds so intermediate amounts of failure not found
def vary_theta_failures_by_country(A, p_GDP, countries):
    """Runs the claims cascade by failing each country to half of v_0 in turn. 
    
    Arguments:
        A: Dependency matrix described in Elliott et al.
        p_GDP: GDP for each country (in the same order as countries), normalized such that the LOWEST p_GDP == 1.0
        countries: List of strings of country names.
        
    Returns:
        DataFrame with columns 'country', 'theta', 'failures' denoting the number of failures that country caused at varying values of theta.
    """
    
    all_results = []
    thetas = np.linspace(.7, 1.0, 100)
    
    for i in thetas: 
        failures_9D = run_claims_list(A, p_GDP, countries, i)
        theta_results = []
        
        for j in range(len(countries)):
            def count_failures(x):
                c = x.split(",")
                if c[0] != ' ':
                    return len(c)
                else:
                    return 0
    
            country_failed = sum(map(count_failures, failures_9D[countries[j]]))
            theta_results.append([countries[j], float(i), int(country_failed)])
                           
        all_results.append(theta_results)

    tidy_df = np.asarray(np.concatenate(all_results, axis=0))                         
    tidy_df = pd.DataFrame(tidy_df,
                       columns=['country', 'theta', 'failures'])
    
    return tidy_df


def plot_individual_country_cascades(tidy_df):
    """Line plot for each country varying theta. 
    
    Arguments:
        tidy_df: must have columns 'failures', 'theta', 'country'
        
    Plots to failures_vary_theta/{countryname}.png
    """
    
    # plot each country individually
    prefix = "failures_vary_theta/"
    countries = list(set(tidy_df['country']))
    for i in range(len(countries)):
        country = countries[i]
        values = map(float, tidy_df.groupby('country').get_group(country)['failures'])
        thetas = map(float, tidy_df.groupby('country').get_group(country)['theta'])

        seaborn.set_style("darkgrid")
        seaborn.tsplot(values, time=thetas, err_style=None)
        plt.title("Number of Failures Following %s Given Varying Theta" % country)
        plt.ylabel("Total Number of Failures")
        plt.xlabel("Theta")
        
        # save
        outfile = prefix+"%s.png" % country
        plt.savefig(outfile)
        plt.close()
        


def plot_theta_first_failures(tidy_df):
    """Barplot for theta to first failure for each country
    
    Arguments:
        tidy_df: must have columns 'failures', 'theta', 'country' 
        
    Plots to first_failures_barplot.png"""
    
    tidy_df['failures'] = map(int, tidy_df['failures'])  
    tidy_df['theta'] = map(float, tidy_df['theta'])
    tidy_df['first_failure'] = map(lambda x: 1 if x==1 else 0, tidy_df['failures'])
    
    # now plot
    NUM_COLORS = 4
    cmap =  plt.cm.get_cmap(name='jet')
    color = [cmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    
    plt.figure(figsize=(20,15))
    m = plot_df.sort(columns='theta', ascending=False).groupby('first_failure').get_group(1)
    max_theta = plot_df.groupby('first_failure').get_group(1).groupby('country').agg({'theta': np.max})
    max_theta.sort('theta', inplace=True)

    # plot
    ax = seaborn.barplot(x = max_theta.index, y = max_theta.theta, color = color[1], ci=None)

    # tweak plot parameters
    plt.ylim([.65, 1.0])
    plt.ylabel("Theta", fontsize=28)
    plt.xlabel(' ')
    plt.title("Theta Threshold for First Failure", fontsize=36)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=55, fontsize=28)
    ax.set_yticklabels(['0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.0'], fontsize=28)

    # save
    outfile = "first_failures_barplot.png"
    plt.savefig(outfile)
    plt.close()

    
failures_df = vary_theta_failures_by_country(A_9D, p_GDP, countries)   
plot_individual_country_cascades(failures_df)
plot_theta_first_failures(failures_df)