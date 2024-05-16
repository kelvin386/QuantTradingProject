import numpy as np
import pandas as pd

def get_optimal_trades(monthly_scaling_factor, alphas_series, impact_summary,
                       stock, date):
    px_vol, ADV = monthly_scaling_factor.loc[stock, date]

    impact_coef = impact_summary.loc[stock]['beta_estimate']
    half_life = impact_summary.loc[stock]['half_life']
    beta = np.log(2) / half_life
    time_unit = 10
    decay_factor = np.exp(-beta * time_unit)


    # optimal trade
    intended_impacts = 1/2 * (alphas_series - alphas_series.diff(1).shift(-1).fillna(0) / beta / time_unit)
    intended_impacts.iloc[-1] = alphas_series.iloc[-1] # I_T^* = \alpha_T

    
    optimal_trades = (beta * (alphas_series - alphas_series.diff(1).diff(1).shift(-1).fillna(0) / (beta ** 2 * time_unit ** 2)
                    ) / (px_vol / ADV * impact_coef) / 2 * time_unit)

    optimal_trades.iloc[0] = intended_impacts.iloc[0] / (px_vol / ADV * impact_coef) # I_0^* / lambda
    optimal_trades.iloc[-1] += (alphas_series.iloc[-1] - intended_impacts.iloc[-2] * decay_factor) / (px_vol / ADV * impact_coef)
    pct_synthetic_alpha_optimal_trades = optimal_trades / ADV
    total_trade_sizes = pct_synthetic_alpha_optimal_trades.abs().sum()
    
    return pct_synthetic_alpha_optimal_trades, total_trade_sizes

