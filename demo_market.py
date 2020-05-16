from function_file import *
from pathlib import Path
from netCDF4 import Dataset
import gc

import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.stats import rv_continuous

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def unit_pv_production(filename):
    """creates data array of time (hours) and global radiation (W/m2) for PV production, minute interval"""
    a = Dataset(filename)

    # shortwave downwards radiation (global radiation)
    gh = a.variables['DSGL2'][:]
    gh = np.ma.getdata(gh)[np.newaxis]
    d = []
    for i in gh[0, :]:
        d.append(i)
    gh = np.array(d).transpose().reshape((len(d), 1))
    irradiance = gh * 0.15 * 1 * 60 / 3600 / 1000  # kWh
    return irradiance


def create_pv_generation_from_capacity(capacity):
    """create nd array generation data by system capacity input (e.g. 5kW), location: Delft"""
    # generation is computed hourly for the entire month of May
    solar_production = unit_pv_production(
        'C:/Users/Lawrence/Documents/GitHub/Thesis/weather data/cesar_bsrn_irradiancedownward_la1_t1_v1.0_201805.nc')
    solar_production[solar_production < 0] = 0
    return np.squeeze(convert_minutes_to_hour(solar_production, 1) * capacity)


def compile_scenario_microgrid_loads(n):
    """randomly selects and compiles n number of load profiles from folder into single array, load profiles are created on minute interval"""
    dir = Path(
        "C:/Users/Lawrence/Documents/GitHub/Thesis/load_profiles")  # goes into folder dir and choses a random file
    load_array = np.empty([44640, 0])  # must know the length of the array beforehand, but what if its not?
    # print(load_array.shape)
    for i in range(n):  # runs n amount of times to get n columns
        filename = random.choice(os.listdir(dir))
        print('random chosen file: ' + filename)  # check
        file_to_open = dir / filename  # appends chosen file name to file path
        with open(file_to_open, encoding="utf-8") as csvDataFile:
            csvReader = np.genfromtxt(csvDataFile, delimiter=';')
            new_load = csvReader[1:, 2].reshape(len(csvReader[1:, 2]), 1)  # reshaped to have a dimension in axis=1
            # print(new_load.shape)
            load_array = np.append(load_array, new_load, axis=1)  # this append is creating 0 dimension
            # print(load_array.shape)
    # print('end of for loop')
    # print(load_array)
    return load_array


def convert_minutes_to_hour(minute_array, n):
    '''creates hourly load from minute load'''
    hour_index = int(len(minute_array) / 60)
    hour_array = np.zeros(shape=(n, hour_index))
    for i in range(n):
        for j in range(hour_index):
            hour_array[i, j] = sum(minute_array[60 * j: (60 * j + 59), i])
    return hour_array


def kde_bandwidth_estimator(x):
    """inputs an array of data, cross validates the best bandwidth by fitting over part of data"""
    """estimation of best bandwidth for kde_scipy: 
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/ """
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.1, 1.0, 30)},
                        cv=20)  # 20-fold cross-validation
    grid.fit(x)
    return print(grid.best_params_)

"""code for kde custom class within function: """


def getDistribution(data):
    bandwidth = 0.002
    kernel = gaussian_kde(data, bw_method=bandwidth / data.std(ddof=1))

    # data input cannot be of Dataframe type, but rather a Series

    class rv(rv_continuous):
        def rvs_custom(self, x):
            # created custom class because
            return kernel.resample(x)

        def cdf_custom(self, x):
            return kernel.integrate_box_1d(-np.Inf, x)

        def pdf_custom(self, x):
            return kernel.evaluate(x)

    return rv(name='kdedist')


"""compile distributions from df_loads dataframe"""


def create_list_of_kde(dataframe):
    """returns a list of objects by calls getDistribution() to create kde's for every column in dataframe"""
    distr_list = []
    for column in dataframe:
        distr_list.append(getDistribution(dataframe.iloc[:, column]))
    return distr_list


def sampleDistribution(distr, n):
    """returns a list of samples drawn from distr from population size n"""
    return np.squeeze(distr.rvs_custom(n))


def plotDistribution(distr):
    return sns.distplot(distr.rvs_custom(5000))


def create_loads_and_save_df(n):
    """samples n load profiles from given subdirectory and returns dataframe (columns = Hours, rows = nth sample)"""
    loads = convert_minutes_to_hour(compile_scenario_microgrid_loads(n), n).T  # dataframes transposed because of
    # downstream processes
    df_loads = pd.DataFrame(loads)
    df_loads.to_csv('C:/Users/Lawrence/Documents/GitHub/Thesis/dataframes/sampling_loads/df_load_samples' + str(n))
    return df_loads


"""obtain dataframe of loads from file"""
# distributions are created for each hour in month of May
df_loads = pd.read_csv('C:/Users/Lawrence/Documents/GitHub/Thesis/dataframes/sampling_loads/df_load_samples5000',
                       index_col=0).T
distr_list = create_list_of_kde(df_loads)

# parameters for model: constant through each iteration
population = 50
hh_pv = 25
hh = population - hh_pv
capacity = 5
hour_counter = 12     # This sets the current market hour

#define parameters for all distribution functions (for now these are constants)
WTP = 0.23       #euro/kWh Willingness to Pay, always 23 in NL due to electricity rate for retail electricity
LCOE = 0.117     #euro/kWh LCOE for 5 kWp system in NL

# create pv generation data: constant throughout instances
solar_generation_month = create_pv_generation_from_capacity(capacity)

param_b_v = [WTP,    0.1,     0,      WTP]
param_s_v = [LCOE,   0.1,  LCOE,      WTP]

"""create necessary dataframes"""
# dataframes used for each iteration
MCP_df = df_MCP()  # dataframe (df) for saving the MCP
utility_profits_df = df_utility_profits()  # create df that keep tracks of Buyers and Sellers utilty and profits
# over iterations
quantity_traded_df = df_quantity_traded()  # create df that keep tracks of quantity traded
market_size_df = df_market_size()

"""BEGINNING OF AUCTION MARKET ALGO"""
# create consumption data for the market hour: hour_counter keeps track of which hour it is:
# this will have to be samples for each hour until hour_counter hits max
consumption_data = sampleDistribution(distr_list[hour_counter], population)

# random sampling of population: output list of indexes for  hh_pv number of households to get pv generation
index_of_hh_pv = random.sample(range(0, population - 1), hh_pv)

# for the hh that have pv, subtract pv generation from their net balance
for i in index_of_hh_pv:
    consumption_data[i] -= solar_generation_month[hour_counter]

# buyers and sellers quantities
s_q = [-x for x in consumption_data if x < 0]
b_q = [x for x in consumption_data if x > 0]

#keeps track of the population
s = len(s_q)  # number of sellers for this instance
b = len(b_q)  # number of buyers for this instance

b_v = create_buyers_v_truncnorm(b, param_b_v)
s_v = create_sellers_v_truncnorm(s, param_s_v)

# create buyer and seller dataframes where prices and allocations of each mechanism result will be saved
buyers_df = df_buyers(b_v, b_q)
sellers_df = df_sellers(s_v, s_q)

"""USER APP: Assume user is a pure consumer and allow them to enter trades into the existing script"""
#goal is to allow interaction of a curious user through an interface to participate in the market set up in this thesis project
user_load = convert_minutes_to_hour(compile_scenario_microgrid_loads(1), 1)
user_true_valuation = create_buyers_v_truncnorm(user_load.size, param_b_v)

#provide to the front-end the following information:

print(user_load[0,hour_counter])            #units: kWh
print(user_true_valuation[hour_counter])    #units: euro-cents

#ask for user input:
user_bid_price = 0.22

#compile bid into the buyers dataframe and index its position
#first compare the prices and find the place where the bid should be on the merit order

user_index = 0
for i in buyers_df.index:
    if buyers_df['price'][i] < user_bid_price:
        user_index = i
        lower_half_df = buyers_df.tail(len(buyers_df)-user_index)
        upper_half_df = buyers_df.head(user_index).append({'price': user_bid_price, 'quantity': user_true_valuation[hour_counter]},
                                          ignore_index=True)
        buyers_df = pd.concat([upper_half_df, lower_half_df]).reset_index(drop=True)
        break

"""start of Average Auction"""
# run auction and get result from SW maximization
buyers_allocation, sellers_allocation, result = run_normal_double_auction(buyers_df, sellers_df)

# updates new column to buyers and sellers df with the social welfare maximizing allocations
buyers_df, sellers_df = update_deafult_mechanism(buyers_allocation, sellers_allocation,
                                                     buyers_df, sellers_df)

# check if any matches are made
if sum(buyers_df['Average allocation']) == 0:
    """need to add in some type of log for the """
    print('there are no possible trades in this auction round')

else:
    # find buyers and sellers in merit order, get the MCP of the auction
    buyers_in_merit_default, sellers_in_merit_default, MCP_default = get_average_mechanism_MCP(buyers_df,
                                                                                                   sellers_df)
    # update MCP of Average auction to dfs
    buyers_df, sellers_df = update_average_MCP_to_df(buyers_allocation, sellers_allocation,
                                                     buyers_df, sellers_df,
                                                     MCP_default)

    '''start of VCG'''
    # VCG: Clarke Pivcot Rule
    VCG_p_buyers, VCG_p_sellers = VCG_clarke_pivot_rule(buyers_df, sellers_df, result)

    MCP_buyers_VCG, MCP_sellers_VCG = get_VCG_mechanism_MCP(VCG_p_buyers, VCG_p_sellers)

    # VCG: update prices and allocations to dataframes
    buyers_df, sellers_df = update_VCG_prices_and_allocation(VCG_p_buyers, VCG_p_sellers,
                                                                 buyers_df, sellers_df)

    '''start of Huang'''
    # Huang: market clearing price rule
    buyers_in_merit_Huang, sellers_in_merit_Huang, MCP_buyers_Huang, MCP_sellers_Huang = get_Huang_mechanism_MCP(
        buyers_df,
        sellers_df)  # result of this is what we need
    # Huang: allocation rule
    q_vector, d_vector = allocation_Huang_mechanism_Q_a(buyers_in_merit_Huang, sellers_in_merit_Huang)

    # Huang: prices and allocations to dataframe
    buyers_df, sellers_df = update_Huang_to_df(q_vector, d_vector,
                                               buyers_df, sellers_df,
                                               MCP_buyers_Huang, MCP_sellers_Huang)

    '''save results of iteration to dataframes'''
    # update MCP of this iteration
    MCP_vector_iteration = [MCP_default, MCP_buyers_VCG, MCP_sellers_VCG, MCP_buyers_Huang, MCP_sellers_Huang]

    MCP_df = update_iteration_MCP(MCP_vector_iteration, MCP_df)

    # find and save profits and utilities of each mechanism for this iteration
    utility_profits_df = calculate_profits_utility(buyers_df, sellers_df, utility_profits_df)

    # find quantity traded for each mechanism for this iteration
    quantity_traded_df = calculate_market_liquidity(sellers_df, quantity_traded_df)

    # find aggregate supply and demand of market for this iteration and append to df
    market_size_df = calculate_market_size(sellers_df, buyers_df, market_size_df)


"""Return Results back to UserInterface"""
#print the prices received for each auction:
user_results = buyers_df.loc[user_index]
user_savings = (user_true_valuation[hour_counter]-user_results['VCG price'])*user_results['VCG allocation']

print(user_results[1])
print(user_results[2])
print(user_results[3])
print(user_results[4])
print(user_results[5])
print(user_results[6])
print(user_results[7])

print(user_savings)

#print the market clearing prices for each mechanism
print(MCP_default)
print(MCP_buyers_VCG)
print(MCP_sellers_VCG)
print(MCP_buyers_Huang)
print(MCP_sellers_Huang)