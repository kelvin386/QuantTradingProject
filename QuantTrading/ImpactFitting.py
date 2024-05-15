import numpy as np
import pandas as pd

space_kernels = {
    "linear": lambda x: x,
    "sqrt": lambda x: np.sign(x) * np.sqrt(np.abs(x)),
}


def get_impact_state(traded_volume_df, monthly_scaling_factor, half_life, model_type):
    """
    Calculate the impact state of the stock.
    half_life: half life of the impact
        - beta = log(2) / h -- as seen in the Obizhaeva-Wang model
        - decay_factor = exp(-beta * dt)
    model_type: type of model to use
    """
    beta = np.log(2) / half_life
    time_unit = 10  # 10 seconds
    decay_factor = np.exp(-beta * time_unit)
    pre_ewm = traded_volume_df.copy()

    pre_ewm = pre_ewm.divide(monthly_scaling_factor["volume"], axis="rows")
    pre_ewm = space_kernels[model_type](pre_ewm)
    pre_ewm = pre_ewm.multiply(monthly_scaling_factor["px_vol"], axis="rows")

    pre_ewm.iloc[:, 1:] /= (1 - decay_factor)
    cum_impact = pre_ewm.ewm(alpha=1-decay_factor, adjust=False, axis="columns").mean()  # Across columns
    return cum_impact

def get_regression_results(cum_impact, px_df, in_sample_month, explanation_horizon_periods, all=False):
    """
    Calculate the regression results for the given data.
    """
    # Get the regression statistics for the given data
    req_stat_df = impact_regression_statistics(cum_impact, explanation_horizon_periods, px_df)
    req_stat_df = req_stat_df.loc[req_stat_df["y"] >= 1e-4].copy()
    req_stat_df["date"] = pd.to_datetime(req_stat_df["date"])
    # Get the regression results for the given data
    daily_stock_reg_info_df = regression_result_by_stock(req_stat_df, in_sample_month, all=all)
    return daily_stock_reg_info_df

# For generalised model #########################################################
def get_index_impact_coef(traded_volume_df, px_df, monthly_scaling_factor,
                          half_life, model_type, in_sample_month):
    """
    Get index-level impact coef (a generalised impact model)
    """
    # Convert stock level data to 'index' data by taking means
    traded_volume_df = traded_volume_df.groupby('date').mean()
    px_df = px_df.groupby('date').mean()
    monthly_scaling_factor = monthly_scaling_factor.groupby('date').mean()
    
    # plug in impact calculation and coef calculation functions
    impact_px_df = get_impact_state(traded_volume_df, monthly_scaling_factor, 
                                    half_life, model_type)
    reg_summary = get_regression_results(impact_px_df, px_df, 
                                            in_sample_month, explanation_horizon_periods=6, all=True)
    
    reg_summary['half_life'] = half_life
    return reg_summary[['beta_estimate', 'alpha_estimate', 'is_rsq', 'half_life']]

def train_validation_split(req_stat_df, valid_fraction=0.2):
    total_data = req_stat_df[['x', 'y']]
    total_size = len(total_data)
    
    indices = np.array(total_data.index)
    np.random.shuffle(indices)
    valid_indices = indices[:int(valid_fraction * total_size)]
    train_indices = indices[int(valid_fraction * total_size):]
    
    x, y = total_data['x'].iloc[train_indices], total_data['y'].iloc[train_indices]
    x_valid, y_valid = total_data['x'].iloc[valid_indices], total_data['y'].iloc[valid_indices]
    
    return x, y, x_valid, y_valid




# Helper functions; very fucking nasty ##########################################

def impact_regression_statistics(cum_impact, explanation_horizon_periods, px_df):
    # Diff for non-cumulative impact for given bin (column)
    impact_changes = cum_impact.diff(explanation_horizon_periods, axis="columns").T.unstack()
    req_stat_df = impact_changes.reset_index().rename({0: "x"}, axis="columns")
    # Not completely sure why we remove before 10:00
    req_stat_df = req_stat_df.loc[req_stat_df["time"] >= "10:00"].dropna(axis=0).copy()

    # pct_change for percentage returns between bins
    returns = px_df.pct_change(explanation_horizon_periods, axis="columns").T.unstack()
    returns = returns.reset_index().rename({0: "y"}, axis="columns")

    # Adding the y for price returns to the same req_stat_df
    req_stat_df["y"] = returns["y"]  # joins on index even though req_stat_df shorter
    req_stat_df["xy"] = req_stat_df["x"] * req_stat_df["y"]
    req_stat_df["xx"] = req_stat_df["x"] * req_stat_df["x"]
    req_stat_df["yy"] = req_stat_df["y"] * req_stat_df["y"]
    req_stat_df["count"] = 1

    return req_stat_df

def regression_result_by_stock(daily_stock_reg_info_df, in_sample_month, all=False):
    in_sample_df = daily_stock_reg_info_df.loc[daily_stock_reg_info_df["date"].dt.month == in_sample_month]
    out_sample_df = daily_stock_reg_info_df.loc[daily_stock_reg_info_df["date"].dt.month == in_sample_month + 1]

    if not all:
        # Sum products over all dates (in month) for each stock
        in_sample_summary_df = in_sample_df.groupby("stock")[["xy", "xx", "yy", "x", "y", "count"]].sum()
        out_sample_summary_df = out_sample_df.groupby("stock")[["xy", "xx", "yy", "x", "y", "count"]].sum()
    else:
        in_sample_summary_df = in_sample_df[["xy", "xx", "yy", "x", "y", "count"]].sum(axis=0).to_frame().T
        out_sample_summary_df = out_sample_df[["xy", "xx", "yy", "x", "y", "count"]].sum(axis=0).to_frame().T

    in_sample_summary_df.columns = "is_" + in_sample_summary_df.columns
    out_sample_summary_df.columns = "oos_" + out_sample_summary_df.columns

    # Already summed over dates for each stock, IS and OOS now two sets of columns to combine
    summary_df = pd.merge(in_sample_summary_df, out_sample_summary_df, left_index=True, right_index=True, how="inner")

    # Slope_hat = S_xy / S_xx
    # I think the error was using is_x and is_y in computing expressions of beta
    summary_df["beta_estimate"] = (summary_df["is_xy"] - summary_df["is_x"] * summary_df["is_y"] / summary_df["is_count"]) / \
                                  (summary_df["is_xx"] - summary_df["is_x"]**2 / summary_df["is_count"])
    # Intercept_hat = y_bar - slope_hat * x_bar
    summary_df["alpha_estimate"] = (summary_df["is_y"] / summary_df["is_count"]) - \
                                   (summary_df["beta_estimate"] * summary_df["is_x"] / summary_df["is_count"])

    # SSE = Sum(y - yBar)^2 = S_yy - S_y^2/n
    summary_df["is_sse"] = (summary_df["is_yy"] - summary_df["is_y"]**2 / summary_df["is_count"])  # typo here??
    summary_df["is_mse"] = (summary_df["is_yy"]
                           - 2 * summary_df["alpha_estimate"] * summary_df["is_y"]
                           - 2 * summary_df["beta_estimate"] * summary_df["is_xy"]
                           + 2 * summary_df["beta_estimate"] * summary_df["is_x"] * summary_df["alpha_estimate"]
                           + summary_df["beta_estimate"]**2 * summary_df["is_xx"]
                           + summary_df["alpha_estimate"]**2 * summary_df["is_count"])

    # R^2 = 1 - MSE/SSE
    summary_df["is_rsq"] = 1 - (summary_df["is_mse"] / summary_df["is_sse"])

    summary_df["oos_sse"] = (summary_df["oos_yy"] - summary_df["oos_y"]**2 / summary_df["oos_count"])
    summary_df["oos_mse"] = (summary_df["oos_yy"]
                            - 2 * summary_df["alpha_estimate"] * summary_df["oos_y"]
                            - 2 * summary_df["beta_estimate"] * summary_df["oos_xy"]
                            + 2 * summary_df["beta_estimate"] * summary_df["oos_x"] * summary_df["alpha_estimate"]
                            + summary_df["beta_estimate"]**2 * summary_df["oos_xx"]
                            + summary_df["alpha_estimate"]**2 * summary_df["oos_count"])

    summary_df["oos_rsq"] = 1 - (summary_df["oos_mse"] / summary_df["oos_sse"])

    return summary_df

def regression_result_all(daily_stock_reg_info_df, in_sample_month):
    in_sample_summary_df = daily_stock_reg_info_df.loc[daily_stock_reg_info_df["date"].dt.month == in_sample_month].sum(axis=0)
    out_sample_summary_df = daily_stock_reg_info_df.loc[daily_stock_reg_info_df["date"].dt.month == in_sample_month + 1].sum()

    in_sample_summary_df.columns = "is_" + in_sample_summary_df.columns
    out_sample_summary_df.columns = "oos_" + out_sample_summary_df.columns

    # Already summed over dates for each stock, IS and OOS now two sets of columns to combine
    summary_df = pd.merge(in_sample_summary_df, out_sample_summary_df, left_index=True, right_index=True, how="inner")

    # Slope_hat = S_xy / S_xx
    # I think the error was using is_x and is_y in computing expressions of beta
    summary_df["beta_estimate"] = (summary_df["is_xy"] - summary_df["is_x"] * summary_df["is_y"] / summary_df["is_count"]) / \
                                  (summary_df["is_xx"] - summary_df["is_x"]**2 / summary_df["is_count"])
    # Intercept_hat = y_bar - slope_hat * x_bar
    summary_df["alpha_estimate"] = (summary_df["is_y"] / summary_df["is_count"]) - \
                                   (summary_df["beta_estimate"] * summary_df["is_x"] / summary_df["is_count"])

    # SSE = Sum(y - yBar)^2 = S_yy - S_y^2/n
    summary_df["is_sse"] = (summary_df["is_yy"] - summary_df["is_y"]**2 / summary_df["is_count"])  # typo here??
    summary_df["is_mse"] = (summary_df["is_yy"]
                           - 2 * summary_df["alpha_estimate"] * summary_df["is_y"]
                           - 2 * summary_df["beta_estimate"] * summary_df["is_xy"]
                           + 2 * summary_df["beta_estimate"] * summary_df["is_x"] * summary_df["alpha_estimate"]
                           + summary_df["beta_estimate"]**2 * summary_df["is_xx"]
                           + summary_df["alpha_estimate"]**2 * summary_df["is_count"])

    # R^2 = 1 - MSE/SSE
    summary_df["is_rsq"] = 1 - (summary_df["is_mse"] / summary_df["is_sse"])

    summary_df["oos_sse"] = (summary_df["oos_yy"] - summary_df["oos_y"]**2 / summary_df["oos_count"])
    summary_df["oos_mse"] = (summary_df["oos_yy"]
                            - 2 * summary_df["alpha_estimate"] * summary_df["oos_y"]
                            - 2 * summary_df["beta_estimate"] * summary_df["oos_xy"]
                            + 2 * summary_df["beta_estimate"] * summary_df["oos_x"] * summary_df["alpha_estimate"]
                            + summary_df["beta_estimate"]**2 * summary_df["oos_xx"]
                            + summary_df["alpha_estimate"]**2 * summary_df["oos_count"])

    summary_df["oos_rsq"] = 1 - (summary_df["oos_mse"] / summary_df["oos_sse"])

    return summary_df

# End of helper functions #######################################################
