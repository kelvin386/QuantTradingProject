import numpy as np
import pandas as pd
def get_intended_impact(alphas_series, impact_summary, stock, model_type):
    if model_type == 'linear': c = 1
    else: c = 0.5

    half_life = impact_summary.loc[stock]['half_life']
    beta = np.log(2) / half_life
    time_unit = 10
    
    intended_impacts = 1 / (1 + c) * (alphas_series - alphas_series.diff(1).shift(-1).fillna(0) / beta / time_unit)
    intended_impacts.iloc[-1] = alphas_series.iloc[-1] # I_T^* = \alpha_T
    
    return intended_impacts

def get_optimal_trades(monthly_scaling_factor, alphas_series, impact_summary,
                       stock, date, model_type):
    '''
    Get optimal trades for a stock at a given date;
    OW and AFS models have different intended impacts and thus different optimal trades;
    To get pct optimal trades, divide by ADV
    '''
    px_vol, ADV = monthly_scaling_factor.loc[stock, date]

    impact_coef = impact_summary.loc[stock]['beta_estimate']
    half_life = impact_summary.loc[stock]['half_life']
    beta = np.log(2) / half_life
    time_unit = 10
    decay_factor = np.exp(-beta * time_unit)

    # optimal trade
    intended_impacts = get_intended_impact(alphas_series, impact_summary, stock, model_type)

    if model_type == 'linear':
        # optimal_trades = beta * (alphas_series - alphas_series.diff(1).diff(1).shift(-1).fillna(0) / (beta ** 2 * time_unit ** 2)) \
        #                   / ((px_vol / ADV * impact_coef) * 2) * time_unit 
        optimal_trades = (beta * intended_impacts * time_unit + intended_impacts.diff(1).shift(-1).fillna(0)) \
                          / (px_vol / ADV * impact_coef)
    elif model_type == 'sqrt':
        optimal_trades = (2 * intended_impacts * intended_impacts.diff(1).shift(-1).fillna(0) + beta * intended_impacts ** 2) \
                          / (px_vol / ADV * impact_coef) * time_unit

    optimal_trades.iloc[0] = intended_impacts.iloc[0] / (px_vol / ADV * impact_coef) # I_0^* / lambda
    optimal_trades.iloc[-1] += (alphas_series.iloc[-1] - intended_impacts.iloc[-2] * decay_factor) / (px_vol / ADV * impact_coef)
    
    # pct_synthetic_alpha_optimal_trades = optimal_trades / ADV
    # total_trade_sizes = pct_synthetic_alpha_optimal_trades.abs().sum()
    
    return optimal_trades
