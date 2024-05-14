import numpy as np
import pandas as pd

def get_synthetic_alpha(corr, px_df, stock, alpha_horizon=6, smooth=True, diagnosis=False):
    returns_df = synthetic_alpha(corr, px_df, stock, alpha_horizon)
    if diagnosis:
        print("Diagnosis on <unsmoothed> return series; only for diagnosis purposes, \n",
              "correlation, actual variance, synthetic variance: \n",
              diagnosis(returns_df))
    
    synthetic_alpha_diffs = returns_df.drop("actual", axis="columns").unstack("time")["synthetic"]
    synthetic_alphas = synthetic_alpha_diffs.iloc[:, ::-1].cumsum(axis="columns").iloc[:, ::-1].shift(-1, axis="columns").fillna(0)
    
    if smooth:
        return synthetic_alphas.ewm(halflife=200, axis="columns").mean()
    else:
        return synthetic_alphas


# Helper functions ##############################################################
def synthetic_alpha_coef(corr, px_df_stock, ret_stock, alpha_horizon=6):
    ret_var = ret_stock.values.var()
    px_minus2 = (px_df_stock ** -2).values.mean()
    
    x = corr ** 2
    y = corr * np.sqrt(1 - corr ** 2) * np.sqrt(ret_var / alpha_horizon / px_minus2)
    return x, y

def diagnosis(returns_df):
    correlation = returns_df.corr().iloc[0,1]
    actual_variance = returns_df["actual"].var()
    synthetic_variance = returns_df["synthetic"].var()
    
    return correlation, actual_variance, synthetic_variance

def synthetic_alpha(corr, px_df, stock, alpha_horizon):
    px_df = px_df.loc[stock]
    ret = px_df.pct_change(alpha_horizon, axis="columns").dropna(axis=1)
    
    x, y = synthetic_alpha_coef(corr, px_df, ret, alpha_horizon)

    np.random.seed(42)
    
    returns = px_df.T.pct_change(alpha_horizon, axis=0).iloc[alpha_horizon:]
    returns.index.name = "time"

    px_changes = px_df.T.diff(alpha_horizon, axis=0).iloc[alpha_horizon:]
    W_diffs = np.random.normal(loc=0, scale=1.0, size=(px_df.shape[0], px_df.shape[1]-1))
    Ws = np.concatenate((np.zeros((W_diffs.shape[0], 1)), W_diffs.cumsum(axis=1)), axis=1).T
    W_h_diffs = Ws[alpha_horizon::] - Ws[:-alpha_horizon]
    px_changes = px_changes * x + W_h_diffs * y
    synthetic_returns = px_changes / (px_df.T.shift(1, axis=0))
    synthetic_returns.index.name = "time"

    returns_df = pd.DataFrame({
        "actual": returns.unstack(),
        "synthetic": synthetic_returns.unstack(),
    })
    
    print(diagnosis(returns_df))

    return returns_df
# End of Helper functions #######################################################
