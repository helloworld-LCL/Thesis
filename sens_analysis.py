from function_file import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


def df_read_csv(foldername, filename):
    path = str('C:/Users/Lawrence/Documents/GitHub/Thesis/dataframes/') + foldername + str('/') + filename + str('.csv')
    df = pd.read_csv(path, index_col=0)
    return df

def calculate_results_mean_std(df):
    utility_mean = [df['Average Utility'].mean(), df['VCG Utility'].mean(), df['Huang Utility'].mean()]
    profit_mean = [df['Average Profit'].mean(), df['VCG Profit'].mean(), df['Huang Profit'].mean()]

    utility_std = [df['Average Utility'].std(), df['VCG Utility'].std(), df['Huang Utility'].std()]
    profit_std = [df['Average Profit'].std(), df['VCG Profit'].std(), df['Huang Profit'].std()]

    MCP_mean = [df['MCP default'].mean(), df['MCP VCG buyers'].mean(), df['MCP VCG sellers'].mean()], [
        df['MCP Huang buyers'].mean(), df['MCP Huang sellers'].mean()]
    MCP_std = [df['MCP default'].std(), df['MCP VCG buyers'].std(), df['MCP VCG sellers'].std()], [
        df['MCP Huang buyers'].std(), df['MCP Huang sellers'].std()]

    quantity_mean = [df['Average quantity'].mean(), df['VCG quantity'].mean(), df['Huang quantity'].mean()]
    quantity_std = [df['Average quantity'].std(), df['VCG quantity'].std(), df['Huang quantity'].std()]
    return utility_mean, utility_std, profit_mean, profit_std, MCP_mean, MCP_std, quantity_mean, quantity_std

def create_dfs_by_cat(list, names):
    list_of_df = []
    for i in range(len(list[0].columns)):
        list_of_df.append(concat_columns_from_dfs(list, i, names))

    df_MCP = pd.concat([x.mean() for x in list_of_df[:5]], axis=1).T
    cat_MCP = [x.columns.name for x in list_of_df[:5]]
    df_MCP = df_MCP.rename(index=lambda s: cat_MCP[
        s])  # lambda goes thru each index position and changes the content based on info from another list

    df_prof = pd.concat([x.mean() for x in list_of_df[6:11:2]], axis=1).T
    cat_prof = [x.columns.name for x in list_of_df[6:11:2]]
    df_prof = df_prof.rename(index=lambda s: cat_prof[s])

    df_ut = pd.concat([x.mean() for x in list_of_df[5:11:2]], axis=1).T
    cat_ut = [x.columns.name for x in list_of_df[5:11:2]]
    df_ut = df_ut.rename(index=lambda s: cat_ut[s])

    df_quant = pd.concat([x.mean() for x in list_of_df[11:]], axis=1).T
    cat_quant = [x.columns.name for x in list_of_df[11:]]
    df_quant = df_quant.rename(index=lambda s: cat_quant[s])

    df_market_revenue = pd.concat([pd.DataFrame([((x.iloc[:, 0].mul(x.iloc[:, 11])).sub(x.iloc[:, 0].mul(x.iloc[:, 11]))).mean(),
                                                 ((x.iloc[:, 1].mul(x.iloc[:, 12])).sub(x.iloc[:, 2].mul(x.iloc[:, 12]))).mean(),
                                                 ((x.iloc[:, 3].mul(x.iloc[:, 13])).sub(x.iloc[:, 4].mul(x.iloc[:, 13]))).mean()]) for x in list], axis=1)
    cat_rev = ['Average Budget Balance', 'VCG Budget Balance', 'Huang Budget Balance']
    df_market_revenue.columns = names
    df_market_revenue = df_market_revenue.rename(index=lambda s: cat_rev[s])

    return df_MCP, df_prof, df_ut, df_quant, df_market_revenue

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() * 1.02
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def seaborn_barplot(df):
    fig, ax = plt.subplots()
    sns.set()
    ax = sns.barplot(data=df)
    show_values_on_bars(ax)  # calls function setting the value of bar at certain position
    plt.title(df.columns.name)  # uses the name of the columns as the title of the barplot

    return ax

def seaborn_heatmap(df, cent):
    """input dataframe to plot and center of the colorbar"""
    fig, ax = plt.subplots()
    sns.set()
    ax = sns.heatmap(data=df, fmt=".4f",  annot=True, linewidths=0.5, linecolor= 'black', cmap = "RdBu", center=cent)
    plt.xticks(rotation=0)
    #colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('kWh', rotation = 0, labelpad=20)
    #black borders
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    return ax

def concat_columns_from_dfs(list_of_df, column_index, column_names):
    """takes in particular row of columns from identical sized dfs and merge them, column names are separately
    defined """
    concat_df = pd.concat([df.iloc[:, column_index] for df in list_of_df], axis=1)
    col_name = concat_df.columns[0]
    concat_df.columns = column_names
    concat_df.columns.name = col_name
    return concat_df

def find_pu_values(df, column_name):
    """df values of scenario results converted to p.u. based on the default values in first column"""
    df_pu = df.iloc[:, 0:].div(df.iloc[0][str(column_name)], axis=0)
    return df_pu

def find_budget_balance_pct(df_default, df_budget_balance):
    """values reflect the percentage of total market liquidity in the default case"""
    df_bb_new = df_budget_balance.div(df_default.iloc[:, 0].mul(df_default.iloc[:, 11]).mean())
    return df_bb_new

"""Part 1 of SA"""
"""goal is to determine which parameters create the most variability in the outputs"""
# Base case: standard normal truncated distribution functions with values [0-1], mean of 0.5 and std of 1
# SA One-at-a-time: decrease std to 0.5 for b_v, b_q, s_v and s_q

df_default = df_read_csv('SA1_parameters', 'iterations_500_default_b_50_s_50')
df_b_v_std_05 = df_read_csv('SA1_parameters', 'iterations_500_b_v_std_0_5')
df_b_q_std_05 = df_read_csv('SA1_parameters', 'iterations_500_b_q_std_0_5')
df_s_v_std_05 = df_read_csv('SA1_parameters', 'iterations_500_s_v_std_0_5')
df_s_q_std_05 = df_read_csv('SA1_parameters', 'iterations_500_s_q_std_0_5')
df_b_v_std_15 = df_read_csv('SA1_parameters', 'iterations_500_b_v_std_1_5')
df_b_q_std_15 = df_read_csv('SA1_parameters', 'iterations_500_b_q_std_1_5')
df_s_v_std_15 = df_read_csv('SA1_parameters', 'iterations_500_s_v_std_1_5')
df_s_q_std_15 = df_read_csv('SA1_parameters', 'iterations_500_s_q_std_1_5')


df_b_v_mu_0 = df_read_csv('SA1_parameters', 'iterations_500_b_v_mu_0')
df_b_q_mu_0 = df_read_csv('SA1_parameters', 'iterations_500_b_q_mu_0')
df_s_v_mu_0 = df_read_csv('SA1_parameters', 'iterations_500_s_v_mu_0')
df_s_q_mu_0 = df_read_csv('SA1_parameters', 'iterations_500_s_q_mu_0')
df_b_v_mu_1 = df_read_csv('SA1_parameters', 'iterations_500_b_v_mu_1')
df_b_q_mu_1 = df_read_csv('SA1_parameters', 'iterations_500_b_q_mu_1')
df_s_v_mu_1 = df_read_csv('SA1_parameters', 'iterations_500_s_v_mu_1')
df_s_q_mu_1 = df_read_csv('SA1_parameters', 'iterations_500_s_q_mu_1')

#vary std
list_std_df = [df_default, df_b_v_std_05, df_b_q_std_05, df_s_v_std_05, df_s_q_std_05,
              df_b_v_std_15, df_b_q_std_15, df_s_v_std_15, df_s_q_std_15]

#vary mean
list_mu_df = [df_default, df_b_v_mu_0, df_b_q_mu_0, df_s_v_mu_0, df_s_q_mu_0,
              df_b_v_mu_1, df_b_q_mu_1, df_s_v_mu_1, df_s_q_mu_1]

# define label names
column_names_std = ['default std\n1', 'buyers v\n0.5', 'buyers q:\n0.5', 'sellers v:\n0.5', 'sellers q:\n0.5', 'buyers v:\n1.5',
                'buyers q:\n1.5', 'sellers v:\n1.5', 'sellers q:\n1.5']

column_names_mu = ['default mean\n0.5', 'buyers v:\n0', 'buyers q:\n0', 'sellers v:\n0', 'sellers q:\n0',
                   'buyers v:\n1', 'buyers q:\n1', 'sellers v:\n1', 'sellers q:\n1']

# sorts the dataframes and concat by category and compares with adjusted parameters (e.g. df of 'Average Utility'
# over all scenarios)


df_MCP_SA1_std, df_prof_SA1_std, df_ut_SA1_std, df_quant_SA1_std, df_market_revenue_std = create_dfs_by_cat(list_std_df, column_names_std)
df_MCP_SA1_mu, df_prof_SA1_mu, df_ut_SA1_mu, df_quant_SA1_mu, df_market_revenue_mu = create_dfs_by_cat(list_mu_df, column_names_mu)

#std values convert to p.u.
df_MCP_SA1_std_pu = find_pu_values(df_MCP_SA1_std, 'default std\n1')
df_prof_SA1_std_pu = find_pu_values(df_prof_SA1_std, 'default std\n1')
df_ut_SA1_std_pu = find_pu_values(df_ut_SA1_std, 'default std\n1')
df_quant_SA1_std_pu = find_pu_values(df_quant_SA1_std, 'default std\n1')
df_bb_SA1_std_pct = find_budget_balance_pct(df_default, df_market_revenue_std)

seaborn_heatmap(df_MCP_SA1_std_pu, 1)
plt.title('Standard Deviation Adjustment Effect on MCP (p.u.)')
seaborn_heatmap(df_prof_SA1_std_pu, 1)
plt.title('Standard Deviation Adjustment Effect on Sellers Profit (p.u.)')
seaborn_heatmap(df_ut_SA1_std_pu, 1)
plt.title('Standard Deviation Adjustment Effect on Buyers Utility (p.u.)')
seaborn_heatmap(df_quant_SA1_std_pu, 1)
plt.title('Standard Deviation Adjustment Effect on Quantity Traded (p.u.)')
seaborn_heatmap(df_bb_SA1_std_pct, 0)
plt.title('Standard Deviation Adjustment Effect on Budget Balance (percentage change)')

#mean values convert to p.u.
df_MCP_SA1_mu_pu = find_pu_values(df_MCP_SA1_mu, 'default mean\n0.5')
df_prof_SA1_mu_pu = find_pu_values(df_prof_SA1_mu, 'default mean\n0.5')
df_ut_SA1_mu_pu = find_pu_values(df_ut_SA1_mu, 'default mean\n0.5')
df_quant_SA1_mu_pu = find_pu_values(df_quant_SA1_mu, 'default mean\n0.5')
df_bb_SA1_mu_pct = find_budget_balance_pct(df_default, df_market_revenue_mu)

seaborn_heatmap(df_MCP_SA1_mu_pu, 1)
plt.title('Mean Adjustment Effect on MCP (p.u.)')

seaborn_heatmap(df_prof_SA1_mu_pu, 1)
plt.title('Mean Adjustment Effect on Sellers Profit (p.u.)')

seaborn_heatmap(df_ut_SA1_mu_pu, 1)
plt.title('Mean Adjustment Effect on Buyers Utility (p.u.)')

seaborn_heatmap(df_quant_SA1_mu_pu, 1)
plt.title('Mean Adjustment Effect on Quantity Traded (p.u.)')

seaborn_heatmap(df_bb_SA1_mu_pct, 0)
plt.title('Mean Adjustment Effect on Budget Balance (percentage change)')

"""Part 2 of SA"""
# goal is to change the number of buyers and sellers and see how it changes the dynamics

df_b10_s90 = df_read_csv('SA2_population', 'iterations_500_default_b_10_s_90')
df_b20_s80 = df_read_csv('SA2_population', 'iterations_500_default_b_20_s_80')
df_b30_s70 = df_read_csv('SA2_population', 'iterations_500_default_b_30_s_70')
df_b40_s60 = df_read_csv('SA2_population', 'iterations_500_default_b_40_s_60')
df_default = df_read_csv('SA2_population', 'iterations_500_default_b_50_s_50')
df_b60_s40 = df_read_csv('SA2_population', 'iterations_500_default_b_60_s_40')
df_b70_s30 = df_read_csv('SA2_population', 'iterations_500_default_b_70_s_30')
df_b80_s20 = df_read_csv('SA2_population', 'iterations_500_default_b_80_s_20')
df_b90_s10 = df_read_csv('SA2_population', 'iterations_500_default_b_90_s_10')

list_of_df2 = [df_b10_s90, df_b20_s80, df_b30_s70, df_b40_s60, df_default, df_b60_s40, df_b70_s30, df_b80_s20,
               df_b90_s10]

column_names2 = ['b: 10\n s: 90', 'b: 20\n s: 80', 'b: 30\n s: 70', 'b: 40\n s: 60', 'b: 50\n s: 50', 'b: 60\n s: 40',
                 'b: 70\n s: 30', 'b: 80\n s: 20', 'b: 90\n s: 10']

df_MCP_SA2, df_prof_SA2, df_ut_SA2, df_quant_SA2,  df_market_revenue_SA2 = create_dfs_by_cat(list_of_df2, column_names2)

df_MCP_SA2_pu = find_pu_values(df_MCP_SA2, 'b: 50\n s: 50')
df_prof_SA2_pu = find_pu_values(df_prof_SA2, 'b: 50\n s: 50')
df_ut_SA2_pu = find_pu_values(df_ut_SA2, 'b: 50\n s: 50')
df_quant_SA2_pu = find_pu_values(df_quant_SA2, 'b: 50\n s: 50')
df_market_revenue_SA2_pct = find_budget_balance_pct(df_default, df_market_revenue_SA2)

seaborn_heatmap(df_MCP_SA2_pu, 1)
plt.title('Population Size Effect on MCP (p.u.)')

seaborn_heatmap(df_prof_SA2_pu, 1)
plt.title('Population Size Effect on Sellers Profit (p.u.)')

seaborn_heatmap(df_ut_SA2_pu, 1)
plt.title('Population Size Effect on Buyers Utility (p.u.)')

seaborn_heatmap(df_quant_SA2_pu, 1)
plt.title('Population Size Effect on Quantity Traded (p.u.)')

seaborn_heatmap(df_market_revenue_SA2_pct, 0)
plt.title('Population Size Effect on Budget Balance (percentage change)')


"""Part 3 of SA"""
# compare microgrid prices variation in std of valuations

df_mprices_default = df_read_csv('SA3_microgrid_prices', 'microgrid_b50_s50_price_support_b_v_std_0_1')
df_mprices_sv_bv_both_1 = df_read_csv('SA3_microgrid_prices', 'microgrid_b50_s50_price_support_b_v_s_v_std_1')
df_mprices_shrink_sv = df_read_csv('SA3_microgrid_prices', 'microgrid_b50_s50_price_support_s_v_std_0_1')
df_mprices_default_shrink_sv = df_read_csv('SA3_microgrid_prices', 'microgrid_b50_s50_price_support_b_v_s_v_std_0_1')

list_of_df3 = [df_mprices_default, df_mprices_shrink_sv, df_mprices_default_shrink_sv, df_mprices_sv_bv_both_1]
column_names3 = ['b_v std: 0.1\ns_v std: 1', 'b_v std: 1\ns_v std: 0.1', 'b_v std: 0.1\ns_v std: 0.1', 'b_v std: 1\ns_v std: 1']

df_MCP_SA3, df_prof_SA3, df_ut_SA3, df_quant_SA3, df_market_revenue_SA3 = create_dfs_by_cat(list_of_df3, column_names3)

seaborn_heatmap(df_MCP_SA3, 0.43)
plt.title('Microgrid Market Clearing Price')

seaborn_heatmap(df_prof_SA3, 3.7)
plt.title('Microgrid Sellers Profit')

seaborn_heatmap(df_ut_SA3, 0.75)
plt.title('Microgrid Buyers Utility')

seaborn_heatmap(df_quant_SA3, 22)
plt.title('Microgrid Quantity Traded')

seaborn_heatmap(df_market_revenue_SA3, 0)
plt.title('Microgrid Budget Balance')
