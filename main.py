# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:27:08 2018

@author: eric.benhamou, david sabbagh, valentin melot
"""

from data_processor import Data_loader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import technical_analysis as ta
import hmm as hmm
import utils


def transform_data(data, obs_states):
    rsi = ta.rsi(data, period=4, smooth=3)
    result = np.zeros((rsi.values.shape[0], obs_states))
    for i in range(rsi.values.shape[0]):
        index = np.where(obs_range > rsi.values[i])[0][0] - 1
        result[i, index] = 1
    return result


def load_data():
    # load data
    folder = 'data\\'
    CHFDataFile = "6S ##-##.Last.csv"  # quoted in CHFUSD so need to invert it
    data_CHF_obj = Data_loader(
        CHFDataFile, folder, True, delimiter=',', date_format='%Y%m%d %H%M%S')
    GoldDataFile = "GC ##-##.Last.csv"
    data_Gold_obj = Data_loader(
        GoldDataFile, folder, True, delimiter=',', date_format='%Y%m%d %H%M%S')

    # create combined data
    CHF_dict = {
        'Date': data_CHF_obj.get_field('Date'),
        'CHF': 1 / data_CHF_obj.get_field('Close')}  # inversion to get USDCHF from CHFUSD
    Gold_dict = {
        'Date': data_Gold_obj.get_field('Date'),
        'Gold': data_Gold_obj.get_field('Close')}
    df_CHF = pd.DataFrame.from_dict(CHF_dict).set_index('Date')
    df_Gold = pd.DataFrame.from_dict(Gold_dict).set_index('Date')
    df_joined = df_CHF.join(df_Gold, how='inner')
    df_joined['nb'] = np.arange(df_joined.shape[0])

    # takes every 10 minutes so drop all indexes
    period = 10
    df = df_joined[df_joined.nb % period == 0]
    return df

# load data
df = load_data()

# plot data
plt.figure(figsize=(12, 5))
plt.xlabel('Dates')
ax1 = df.CHF.plot(color='blue', grid=True, label='USDCHF')
ax2 = df.Gold.plot(color='red', grid=True, secondary_y=True, label='Gold')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
lgd = plt.legend(h1 + h2, l1 + l2, loc=4)
plt.title('Price of USDCHF and Gold (01/02/2013 to 31/05/2013)')
utils.save_figure(plt, 'Main', '1', lgd)

# compute data
obs_states = 8
obs_range = np.linspace(0, 100, obs_states + 1)  # 0., 0.125, 0.25, ...
chf_data = transform_data(df.CHF, obs_states)
gold_data = transform_data(df.Gold, obs_states)
chmm_data = np.array([ np.ravel(np.outer(chf_data[t, :], gold_data[t, :])) for t in range(chf_data.shape[0])])
hidden_states = 5

# show some results
def get_some_results(data, name, hidden_states):
    obs_states = data.shape[1]
    hmm_obj = hmm.Hmm(data, 'rescaled', hidden_states, obs_states)
    hmm_obj.compute_proba()
    hmm_obj.plot_proba(200, 'Conditional proba ({})'.format(
        name), '{}'.format(name), '1')
    hmm_obj.EM(True)
    hmm_obj.print_parameters()
    hmm_obj.plot_proba(200, 'Conditional proba ({})'.format(
        name), '{}'.format(name), '2')
    hmm_obj.plot_likelihood('{}'.format(name), '3')
    hmm_obj.compute_viterbi_path()
    hmm_obj.plot_most_likely_state(
        200, 'Viterbi ({})'.format(name), '{}'.format(name), '4')
    return hmm_obj

# individual hmms
chf_hmm = get_some_results(chf_data, 'CHF', hidden_states)
gold_hmm = get_some_results(gold_data, 'Gold', hidden_states)

# joint hmm
joined_hmm = get_some_results(chmm_data, 'CHF-Gold', hidden_states * hidden_states)
