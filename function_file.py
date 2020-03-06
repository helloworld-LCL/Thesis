from __future__ import division  # converts to float before division , useful and recommended for pyomo
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from netCDF4 import Dataset
import csv
from collections import OrderedDict

# optimization packages
from scipy.optimize import linprog
import cvxpy as cp
from gurobipy import *
from pyomo.environ import *

# statistical sampling packages
from scipy.stats import norm
from scipy.stats import truncnorm

'''I/O Dataframes'''


def df_buyers(b_v, b_q):
    return pd.DataFrame({'price': sorted(b_v, reverse=1), 'quantity': b_q})


def df_sellers(s_v, s_q):
    return pd.DataFrame({'price': sorted(s_v), 'quantity': s_q})


def df_MCP():
    '''create dataframe for MCP's'''
    # columns: may have to accomodate for multiple MCP's, e.g. two price system
    return pd.DataFrame(columns=['MCP default',
                                 'MCP VCG buyers', 'MCP VCG sellers',
                                 'MCP Huang buyers', 'MCP Huang sellers'])

def df_allo_eff():
    '''create dataframe for model MCP'''
    return pd.DataFrame(columns=['allocative efficiency'])


def df_utility_profits():
    # need to update
    return pd.DataFrame(columns=['Average Utility', 'Average Profit',
                                 'VCG Utility', 'VCG Profit',
                                 'Huang Utility', 'Huang Profit'])

def df_quantity_traded():
    return pd.DataFrame(columns=['Average quantity',
                                 'VCG quantity',
                                 'Huang quantity'])

'''Auction Preparation Dataframes'''

#probability distribution functions
def visualization_samples(sample):
    '''takes in random sampling created by distribution function, visualize samples by value and sorted values'''
    fig, ax = plt.subplots()
    ax.plot(range(len(sample)), sample, label='scatter')
    sample.sort()
    ax.plot(sample, label='sorted')
    ax.legend()
    return ax


def trunc_visualization(parameters):
    [mu, sig, min, max] = parameters
    a, b, = (min - mu) / sig, (max - mu) / sig
    x_range = np.linspace(min - 1, max + 1, 1000)
    fig, ax = plt.subplots()
    sns.lineplot(x_range, truncnorm.pdf(x_range, a, b, loc=mu, scale=sig), label='pdf')
    sns.lineplot(x_range, truncnorm.cdf(x_range, a, b, loc=mu, scale=sig), label='cdf')
    ax.legend()
    return ax


def create_buyers_v_truncnorm(b, parameters):
    '''takes in number of buyers, and parameters to create list of valuations based on truncated normal distribution'''
    [mu_v_b, sig_v_b, min_v, max_v] = parameters
    a_para, b_para = (min_v - mu_v_b) / sig_v_b, (max_v - mu_v_b) / sig_v_b
    v_b = truncnorm.rvs(a_para, b_para, size=b, loc=mu_v_b, scale=sig_v_b)
    return v_b


def create_buyers_q_truncnorm(b, parameters):
    '''takes in number of buyers, and parameters to create list of quantities based on truncated normal distribution'''
    [mu_q_b, sig_q_b, min_q, max_q] = parameters
    a_para, b_para = (min_q - mu_q_b) / sig_q_b, (max_q - mu_q_b) / sig_q_b
    q_b = truncnorm.rvs(a_para, b_para, size=b, loc=mu_q_b, scale=sig_q_b)
    return q_b


def create_sellers_v_truncnorm(s, parameters):
    '''takes in number of buyers, and parameters to create list of valuations based on truncated normal distribution'''
    [mu_v_b, sig_v_b, min_v, max_v] = parameters
    a_para, b_para = (min_v - mu_v_b) / sig_v_b, (max_v - mu_v_b) / sig_v_b
    v_b = truncnorm.rvs(a_para, b_para, size=s, loc=mu_v_b, scale=sig_v_b)
    return v_b


def create_sellers_q_truncnorm(s, parameters):
    '''takes in number of buyers, and parameters to create list of quantities based on truncated normal distribution'''
    [mu_q_s, sig_q_s, min_q, max_q] = parameters
    a_para, b_para = (min_q - mu_q_s) / sig_q_s, (max_q - mu_q_s) / sig_q_s
    q_b = truncnorm.rvs(a_para, b_para, size=s, loc=mu_q_s, scale=sig_q_s)
    return q_b


"""Default Market Ooperations"""

def run_normal_double_auction(buyers_df, sellers_df):
    # define sets and parameter values in indexed tuples
    n = len(buyers_df)
    m = len(sellers_df)

    a = OrderedDict()
    b = OrderedDict()
    c = OrderedDict()
    d = OrderedDict()

    sellers_counter = 1
    for i in range(len(sellers_df)):
        c[sellers_counter] = sellers_df.iloc[i, 0]
        d[sellers_counter] = sellers_df.iloc[i, 1]
        sellers_counter += 1

    buyers_counter = 1
    for i in range(len(buyers_df)):
        a[buyers_counter] = buyers_df.iloc[i, 0]
        b[buyers_counter] = buyers_df.iloc[i, 1]
        buyers_counter += 1

    # create model

    model = AbstractModel()

    model.n = Param(within=NonNegativeIntegers, default=n)
    model.m = Param(within=NonNegativeIntegers, default=m)

    # number of buyers
    model.I = RangeSet(1, model.n)
    # number of sellers
    model.J = RangeSet(1, model.m)

    # prices for buyers bids
    model.a = Param(model.I, initialize=a)
    # upper bounds for quantity demanded
    model.b = Param(model.I, initialize=b)

    # prices for sellers bids
    model.c = Param(model.J, initialize=c)
    # upper bounds for quantity supplied
    model.d = Param(model.J, initialize=d)

    # quantities for buyers bids
    model.x = Var(model.I, domain=NonNegativeReals)
    # quantities for sellers bids
    model.y = Var(model.J, domain=NonNegativeReals)

    def obj_expression(model):
        return summation(model.a, model.x) - summation(model.c, model.y)

    model.OBJ = Objective(rule=obj_expression, sense=maximize)

    def x_constraint_rule(model, i):
        return model.x[i] <= model.b[i]

    def y_constraint_rule(model, j):
        return model.y[j] <= model.d[j]

    def xy_contraint_rule(model):
        return sum(model.x[:]) - sum(model.y[:]) == 0

    model.Constraint1 = Constraint(model.I, rule=x_constraint_rule)
    model.Constraint2 = Constraint(model.J, rule=y_constraint_rule)
    model.Constraint3 = Constraint(rule=xy_contraint_rule)

    '''solution space'''
    instance = model.create_instance()  # this is needed if it is an abstract model
    instance.dual = Suffix(direction=Suffix.IMPORT)  # needed to store dual values

    opt = SolverFactory('gurobi')
    result = opt.solve(instance)
    instance.solutions.store_to(result)

    '''
    #call duals from instance 
    print ("Duals")
    for c in instance.component_objects(Constraint, active=True):
        print ("   Constraint",c)
        for index in c:
            print ("      ", index, instance.dual[c[index]])
    '''

    '''update amount allocated to each seller and buyer by making new list'''
    buyers_allocation = []
    sellers_allocation = []
    for i in range(n):
        buyers_allocation.append(result.solution.variable['x[' + str(i + 1) + ']']['Value'])

    for i in range(m):
        sellers_allocation.append(result.solution.variable['y[' + str(i + 1) + ']']['Value'])

    return buyers_allocation, sellers_allocation, result


'''Normal Market Mechanism MCP'''

def get_average_mechanism_MCP(buyers_df, sellers_df):
    ''' find the highest supply bid and lowest demand bid, MCP is the average price between those two bid prices '''
    buyers_in_merit = buyers_df[['price', 'Average allocation']].values.tolist()
    sellers_in_merit = sellers_df[['price', 'Average allocation']].values.tolist()

    buyers_in_merit2 = [x for x in buyers_in_merit if x[1] != 0]
    sellers_in_merit2 = [x for x in sellers_in_merit if x[1] != 0]

    if len(buyers_in_merit2) and len(sellers_in_merit2) != 0:
        b = min(x[0] for x in buyers_in_merit2)
        s = max(x[0] for x in sellers_in_merit2)
        MCP = (b + s) / 2

    else:
        MCP = 0

    return buyers_in_merit2, sellers_in_merit2, MCP


def update_deafult_mechanism(buyers_allocation, sellers_allocation, buyers_df, sellers_df):
    '''adds columns to sellers and buyers dataframes with the Average allocated values and MCP'''
    # prices_buyers = [MCP if x != 0 else 0 for x in buyers_allocation]
    # prices_sellers = [MCP if x != 0 else 0 for x in sellers_allocation]

    # buyers_df['Average price'] = prices_buyers
    buyers_df['Average allocation'] = buyers_allocation

    # sellers_df['SW price'] = prices_sellers
    sellers_df['Average allocation'] = sellers_allocation
    return buyers_df, sellers_df


def update_average_MCP_to_df(buyers_allocation, sellers_allocation, buyers_df, sellers_df, MCP):
    prices_buyers = [MCP if x != 0 else 0 for x in buyers_allocation]
    prices_sellers = [MCP if x != 0 else 0 for x in sellers_allocation]
    buyers_df['Average price'] = prices_buyers
    sellers_df['Average price'] = prices_sellers
    return buyers_df, sellers_df


"""Huang mechanism: M-1 and L-1 players trade for quantities, bid prices are static"""


def get_Huang_mechanism_MCP(buyers_df, sellers_df):
    '''sorts out M-1 buyers, L-1 sellers and adapts equation (5) of paper: A Game-Theoretic Approach to Energy Trading in the Smart Grid'''
    buyers_in_merit = buyers_df[['price', 'Average allocation']].values.tolist()
    sellers_in_merit = sellers_df[['price', 'Average allocation']].values.tolist()
    buyers_in_merit2 = [x for x in buyers_in_merit if x[1] != 0]
    sellers_in_merit2 = [x for x in sellers_in_merit if x[1] != 0]
    if len(buyers_in_merit2) and len(sellers_in_merit2) != 1:
        MCP_buyers = min(x[0] for x in buyers_in_merit2)
        MCP_sellers = max(x[0] for x in sellers_in_merit2)
        # removes the last feasible pair of bids from the list
        del buyers_in_merit2[-1]  # pops the last item out so there are M-1 buyers
        del sellers_in_merit2[-1]  # pops the last item out so there are L-1 buyers
    else:
        MCP_buyers = 0
        MCP_sellers = 0
        buyers_in_merit2, sellers_in_merit2 = [], []
    return buyers_in_merit2, sellers_in_merit2, MCP_buyers, MCP_sellers




def allocation_Huang_mechanism_Q_a(buyers_in_merit, sellers_in_merit):
    '''takes in the agents in merit from function get_Huang_mechanism_MCP and outputs q and d allocation vectors'''

    a_vector = []  # takes sellers list and extracts all available quantities they are willing to sell
    for i in sellers_in_merit:
        a_vector.append(i[1])

    x_vector = []  # takes sellers list and extracts all available quantities they are willing to sell
    for i in buyers_in_merit:
        x_vector.append(i[1])

    if len(a_vector) == 0:
        q_vector = [0]
        d_vector = [0]

    elif len(x_vector) == 0:
        q_vector = [0]
        d_vector = [0]

    else:
        supply = sum(a_vector)
        demand = sum(x_vector)

        if supply > demand:  # case where we need to apply equation (7)
            oversupply = supply - demand
            N = len(a_vector)  # number of agents initially within merit
            B_i = oversupply / N
            q_vector = a_vector  # this will keep storing updated allocation until condition is met
            def find_q_vector(q_vector, B_i, N, demand):
                # check if Bi is greater than any value in a_vector, if it is, take out the a_vector and redistribute, do this until all items in a_vector are 0, be careful on the indexing of the vector
                min_q = min(x for x in q_vector if x != 0)
                if B_i > min_q and min_q != 0:  # this means that agents that result in 0 allocation after sharing oversupply burden gets treated as 'in merit'
                    # B_i = B_i + (B_i - min_q) / (N - 1) #i = 1 by default, otherwise it accounts for identical vector values and adjusts B_i accordingly.
                    q_vector = [0 if x == min_q else 0 if x == 0 else x for x in q_vector]
                    '''make code still work if there are 2 vector items that have same value, rn line above will replace same quantity. Bi needs to change accordingly'''
                    N = sum(1 for i in q_vector if i != 0)
                    B_i = (sum(q_vector) - demand) / N
                    print(N, B_i)
                    return find_q_vector(q_vector, B_i, N, demand)
                else:
                    q_vector = [0 if x == 0 else x - B_i for x in q_vector]
                    return q_vector

            q_vector = find_q_vector(q_vector, B_i, N, demand)  # final allocation of a vector
            d_vector = x_vector  # allocation of demand vector stays the same

        elif demand >= supply:
            overdemand = demand - supply
            N = len(x_vector)  # number of agents initially within merit
            B_j = overdemand / N
            d_vector = x_vector  # this will keep storing updated allocation until condition is met
            def find_d_vector(d_vector, B_j, N, supply):
                # check if Bi is greater than any value in a_vector, if it is, take out the a_vector and redistribute, do this until all items in a_vector are 0, be careful on the indexing of the vector
                min_d = min(x for x in d_vector if x != 0)
                if B_j > min_d and min_d != 0:  # this means that agents that result in 0 allocation after sharing oversupply burden gets treated as 'in merit'
                    # B_i = B_i + (B_i - min_q) / (N - 1) #i = 1 by default, otherwise it accounts for identical vector values and adjusts B_i accordingly.
                    d_vector = [0 if x == min_d else 0 if x == 0 else x for x in d_vector]
                    '''make code still work if there are 2 vector items that have same value, rn line above will replace same quantity. Bi needs to change accordingly'''
                    N = sum(1 for i in d_vector if i != 0)
                    B_j = (sum(d_vector) - supply) / N
                    return find_d_vector(d_vector, B_j, N, supply)
                else:
                    d_vector = [0 if x == 0 else x - B_j for x in d_vector]
                    return d_vector

            q_vector = a_vector  # allocation of supply vector stays the same
            d_vector = find_d_vector(d_vector, B_j, N, supply)  # final allocation of x vector

    return q_vector, d_vector


def update_Huang_to_df(q_vector, d_vector, buyers_df, sellers_df, MCP_buyers, MCP_sellers):
    '''adds column to sellers and buyers dataframes with the Average allocated quantities'''
    d_vector = d_vector + [0] * (len(buyers_df) - len(d_vector))
    q_vector = q_vector + [0] * (len(sellers_df) - len(q_vector))

    prices_buyers = [MCP_buyers if x != 0 else 0 for x in d_vector]
    prices_sellers = [MCP_sellers if x != 0 else 0 for x in q_vector]

    buyers_df['Huang price'] = prices_buyers
    buyers_df['Huang allocation'] = d_vector

    sellers_df['Huang price'] = prices_sellers
    sellers_df['Huang allocation'] = q_vector

    return buyers_df, sellers_df


def allocative_efficiency_Huang(buyers_df, sellers_df, MCP_buyers, MCP_sellers, result):
    # max allocation utility
    profit_sellers = 0
    utility_buyers = 0

    for i in range(len(sellers_df['price'])):
        profit_sellers = profit_sellers + (MCP_sellers - sellers_df.loc[i, 'price']) * sellers_df.loc[
            i, 'Huang allocation']

    for i in range(len(buyers_df['price'])):
        utility_buyers = utility_buyers + (buyers_df.loc[i, 'price'] - MCP_buyers) * buyers_df.loc[
            i, 'Huang allocation']

    eff = (profit_sellers + utility_buyers) / result.solution.objective['OBJ']['Value']
    return eff


def VCG_clarke_pivot_rule(buyers_df, sellers_df, result):
    prices_sellers = []
    prices_buyers = []

    SW = result.solution.objective['OBJ']['Value']

    for i in range(len(sellers_df)):
        index_sellers = i
        # find SW without ith seller's welfare
        sellers_wf_neg = SW + sellers_df.iloc[index_sellers]['Average allocation'] * sellers_df.iloc[index_sellers]['price']

        # run DA with ith seller absent
        VCG_sellers_df = sellers_df.drop(index_sellers).reset_index(drop=True)
        b, sellers_allocation_VCG, r_sellers = run_normal_double_auction(buyers_df, VCG_sellers_df)
        sellers_wf_wo = r_sellers.solution.objective['OBJ']['Value']

        # price seller gets = externality/quantity allocated

        if sellers_df.iloc[index_sellers]['Average allocation'] == 0:
            seller_price = 0
        else:
            seller_price = abs((sellers_wf_neg - sellers_wf_wo) / sellers_df.iloc[index_sellers]['Average allocation'])

        prices_sellers.insert(len(prices_sellers), seller_price)

    for i in range(len(buyers_df)):
        index_buyers = i
        # find SW without ith buyer's welfare
        buyers_wf_neg = SW - buyers_df.iloc[index_buyers]['Average allocation'] * buyers_df.iloc[index_buyers]['price']

        # run DA with ith seller absent
        VCG_buyers_df = buyers_df.drop(index_buyers).reset_index(drop=True)
        buyers_allocation_VCG, s, r_buyers = run_normal_double_auction(VCG_buyers_df, sellers_df)
        buyers_wf_wo = r_buyers.solution.objective['OBJ']['Value']

        # price buyer pays = externality/quantity allocated
        if buyers_df.iloc[index_buyers]['Average allocation'] == 0:
            buyer_price = 0
        else:
            buyer_price = abs((buyers_wf_neg - buyers_wf_wo)) / buyers_df.iloc[index_buyers]['Average allocation']

        prices_buyers.insert(len(prices_buyers), buyer_price)

    return prices_buyers, prices_sellers


def get_VCG_mechanism_MCP(prices_buyers, prices_sellers):
    filtered_buyers = list(filter(lambda num: num != 0, prices_buyers))
    filtered_sellers = list(filter(lambda num: num != 0, prices_sellers))
    MCP_buyers_VCG = sum(filtered_buyers) / len(filtered_buyers)
    MCP_sellers_VCG = sum(filtered_sellers) / len(filtered_sellers)
    return MCP_buyers_VCG, MCP_sellers_VCG


def update_VCG_prices_and_allocation(prices_buyers, prices_sellers, buyers_df, sellers_df):
    '''adds column to sellers and buyers dataframes with the VCG allocated prices'''
    buyers_df['VCG price'] = prices_buyers
    sellers_df['VCG price'] = prices_sellers
    buyers_df['VCG allocation'] = buyers_df['Average allocation']
    sellers_df['VCG allocation'] = sellers_df['Average allocation']

    return buyers_df, sellers_df

def update_iteration_MCP(MCP_vector_iteration, MCP_df):
    MCP_vector =    [{'MCP default': MCP_vector_iteration[0],
                    'MCP VCG buyers': MCP_vector_iteration[1],
                    'MCP VCG sellers': MCP_vector_iteration[2],
                    'MCP Huang buyers': MCP_vector_iteration[3],
                    'MCP Huang sellers': MCP_vector_iteration[4]}]
    # append the averaged MCP of iterations to dataframe
    MCP_df = MCP_df.append(MCP_vector, ignore_index=True, sort=False)
    return MCP_df

def calculate_BSI():
    
    return


def calculate_profits_utility(buyers_df, sellers_df, utility_profits_df):
    cent_ut = sum((buyers_df['price'] - buyers_df['Average price']) * buyers_df['Average allocation'])
    cent_prof = sum((sellers_df['Average price'] - sellers_df['price']) * sellers_df['Average allocation'])
    VCG_ut = sum((buyers_df['price'] - buyers_df['VCG price']) * buyers_df['VCG allocation'])
    VCG_prof = sum((sellers_df['VCG price'] - sellers_df['price']) * sellers_df['VCG allocation'])
    Huang_ut = sum((buyers_df['price'] - buyers_df['Huang price']) * buyers_df['Huang allocation'])
    Huang_prof = sum((sellers_df['Huang price'] - sellers_df['price']) * sellers_df['Huang allocation'])

    utility_profits_df = utility_profits_df.append({'Average Utility': cent_ut,
                                                    'Average Profit': cent_prof,
                                                    'VCG Utility': VCG_ut,
                                                    'VCG Profit': VCG_prof,
                                                    'Huang Utility': Huang_ut,
                                                    'Huang Profit': Huang_prof}, ignore_index=True)
    return utility_profits_df

def calculate_market_liquidity(sellers_df, quantity_traded_df):
    averae_q    = sum(sellers_df['Average allocation'])
    VCG_q       = sum(sellers_df['VCG allocation'])
    Huang_q     = sum(sellers_df['Huang allocation'])

    quantity_traded_df = quantity_traded_df.append({'Average quantity': averae_q,
                                        'VCG quantity': VCG_q,
                                        'Huang quantity': Huang_q}, ignore_index=True)
    return quantity_traded_df

def calculate_mean_MCP(MCP_df, MCP_df_mean):
    MCP_mean_data = [{  'MCP default': MCP_df.mean().values[0],
                        'MCP VCG buyers': MCP_df.mean().values[1],
                        'MCP VCG sellers': MCP_df.mean().values[2],
                        'MCP Huang buyers': MCP_df.mean().values[3],
                        'MCP Huang sellers': MCP_df.mean().values[4]}]
        # append the averaged MCP of iterations to dataframe
    MCP_df_mean = MCP_df_mean.append(MCP_mean_data, ignore_index=True, sort=False)
    return MCP_df_mean

def df_to_csv(MCP_df, utility_profits_df, quantity_traded_df):
    results_df = pd.concat([MCP_df, utility_profits_df, quantity_traded_df], axis=1)
    input_title = input('enter title for csv(use underscores):  ')
    results_df.to_csv(str(input_title)+'.csv')
    return

def df_to_csv_auto(MCP_df, utility_profits_df, quantity_traded_df, b, s, total_iterations):
    results_df = pd.concat([MCP_df, utility_profits_df, quantity_traded_df], axis=1)
    input_title = 'iterations_'+ str(total_iterations) +'_default_b_'+str(b)+'_s_'+str(s)
    results_df.to_csv(str(input_title)+'.csv')
    return

def df_to_csv_RW(MCP_df, utility_profits_df, quantity_traded_df, b, s, total_iterations, hour_counter):
    results_df = pd.concat([MCP_df, utility_profits_df, quantity_traded_df], axis=1)
    input_title = 'HR_' + str(hour_counter) + '_iterations_'+ str(total_iterations) +'_default_b_'+str(b)+'_s_'+str(s)
    results_df.to_csv('C:/Users/Lawrence/Documents/GitHub/Thesis/dataframes/real_world_model/' + str(input_title)+'.csv')
    return

'''GAME THEORY FUNCTIONS'''

'''Test Cases'''

def test_case():
    #test 1 with tuple format
    tuple_list = [(1001, 13.8, 80),
                  (1002, 13.5, 90),
                  (1003, 12.7, 100),
                  (1004, 12, 100),
                  (1005, 12.8, 50),
                  (1006, 12.9, 40)]
    print('   Test Case Tuple list:   ')
    print(tuple_list)

    return tuple_list

def test_case_2():
    # test 2
    sellers_prices = [1, 2, 2.9, 3.9, 5, 6]
    sellers_quantities = [1, 1, 1, 1, 1, 1]

    buyers_prices = [6, 5, 4.1, 3.1, 2, 1]
    buyers_quantities = [1, 1, 1, 1, 1, 1]

    buyers_df = pd.DataFrame({'price': buyers_prices, 'quantity': buyers_quantities})
    sellers_df = pd.DataFrame({'price': sellers_prices, 'quantity': sellers_quantities})

    return buyers_df, sellers_df