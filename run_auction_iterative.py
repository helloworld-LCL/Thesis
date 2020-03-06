from __future__ import division  # converts to float before division , useful and recommended for pyomo
import os
import sys
import random
import numpy as np
import pandas as pd
from function_file import *
from gurobipy import *
from pyomo.environ import *
import matplotlib.pyplot as plt
import time
import pickle
'''
#buyers_df, sellers_df = test_case_2()
'''

'''
#annotation of variables
n:          number of household agents participating in market; n = b + s

b:          number of buyers
mu_v_b:     mean valuation of buyer
sig_v_b:    valuation standard deviation of buyer
min_v_b:    lower bound support
max_v_b:    upper bound support

mu_q_b:     mean quantity of buyer
sig_q_b:    quantity standard deviation of buyer
min_q_b:
max_q_b:

s:          number of sellers
mu_v_s:     mean valuation of seller
sig_v_s:    valuation standard deviation of seller
min_v_s
max_v_s

mu_q_s:     mean quantity of seller
sig_q_s:    quantity standard deviation of seller
min_q_s:    
max_q_s:    
'''

'''set number of households'''
b = 50          #buyers
s = 50          #sellers
n = b+s         #number of households

#define parameters for all distribution functions
FiT = 0.08      #euro/kWh Feed in Tariff
ToU = 0.5       #euro/kWh (peak rate) Time of Use

#param =    [mean, standard dev, lower bound, upper bound] (for now these are all for truncated uniform distributions)

'''
#buyers
param_b_v = [0.4,   0.1,  0,      ToU]
param_b_q = [2,     1,    1,      3]

#sellers
param_s_v = [0.2,   1,    FiT,    1]  #the 1 euro/kWh is for upper bound only, for a realistic estimate find LCOE of residential
param_s_q = [0.8,   1,    0.2,    2]

'''

#case of unit normal distribution
#buyers
param_b_v = [0.5,   1,    0,    1]
param_b_q = [0.5,   1,    0,    1]

#sellers
param_s_v = [0.5,   1,    0,    1]
param_s_q = [0.5,   1,    0,    1]


pairs_to_explore = [[10, 90], [20, 80], [30, 70], [40, 60], [60, 40], [70, 30], [80, 20], [90, 10]]
total_iterations = 500

for x in pairs_to_explore:
    b = x[0]
    s = x[1]

    'create necessary dataframes'
    # dataframes used for each iteration
    MCP_df = df_MCP()  # dataframe (df) for saving the MCP
    utility_profits_df = df_utility_profits()  # create df that keep tracks of Buyers and Sellers utilty and profits over iterations
    quantity_traded_df = df_quantity_traded()  # create df that keep tracks of quantity traded
    # eff_df = df_allo_eff()                          #dataframe for calculations for Huang mechanism allocative efficiency (2/14 not important for now)

    # dataframes used for analysis over all iterations
    # MCP_df_mean = df_MCP()                          #dataframe for saving the average MCP's over all iterations run
    # eff_df_mean = df_allo_eff()                     #dataframe for saving the average efficiency over all iterations run (useful only when comparing different number of agents)

    'set iterations'

    for index in range(total_iterations):
        # index in dataframe starts from 0 while instance starts from
        b_v = create_buyers_v_truncnorm(b, param_b_v)
        b_q = create_buyers_q_truncnorm(b, param_b_q)

        s_v = create_sellers_v_truncnorm(s, param_s_v)
        s_q = create_sellers_q_truncnorm(s, param_s_q)

        # create buyer and seller dataframes where prices and allocations of each mechanism result will be saved
        buyers_df = df_buyers(b_v, b_q)
        sellers_df = df_sellers(s_v, s_q)

        '''start of Average Auction (this is the default Social Welfare maximizing mechanism acting as a benchmark)'''
        # run auction and get result from SW maximization
        buyers_allocation, sellers_allocation, result = run_normal_double_auction(buyers_df, sellers_df)

        # updates new column to buyers and sellers df with the social welfare maximizing allocations
        buyers_df, sellers_df = update_deafult_mechanism(buyers_allocation, sellers_allocation,
                                                         buyers_df, sellers_df)

        # check if any matches are made
        if sum(buyers_df['Average allocation']) == 0:
            continue

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

            '''
            # find allocative efficiency of Huang mechanism
            eff_Huang = allocative_efficiency_Huang(buyers_df, sellers_df,
                                                    MCP_buyers_Huang, MCP_sellers_Huang,
                                                    result)

            eff_df.loc[index] = [eff_Huang]
            '''

    # find averaged MCP of all iterations ran with same households and same distributions
    # MCP_df_mean = calculate_mean_MCP(MCP_df, MCP_df_mean)

    # save MCP_df, utility_profits_df, quantity_traded_df to a csv file
    df_to_csv_auto(MCP_df, utility_profits_df, quantity_traded_df, b, s, total_iterations)

'''
#prints df in full
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(sellers_df)
index = eff_df_mean.index.values

fig, ax = plt.subplots()
ax.plot(index, eff_df_mean)
ax.set(xlabel='households (n)', ylabel='allocative efficiency',
           title='Allocative efficiency vs number of households: buyers = ' + str(
               mean_valuation_buyers) + ' cents, sellers = 8 cents')
plt.savefig('sample_buyers_mean_' + str(mean_valuation_buyers) + '.png')
'''

