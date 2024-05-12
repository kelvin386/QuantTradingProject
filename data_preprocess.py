import pandas as pd
import pickle


year = "2019"
data_dir = u"../Data/"
bin_sample_path = f"{data_dir}binSamples/"
fill_sample_path = f"{data_dir}fillSamples/"
result_path = f"{data_dir}"

def data_initilise(data_dir=u"../Data/", year="2019"):
    """All indices are tuples of (stock, date)"""

    bin_sample_path = f"{data_dir}binSamples/"
    fill_sample_path = f"{data_dir}fillSamples/"
    result_path = f"{data_dir}"
    
    filename = f'inter_results_pre_{year}all.csv'
    stock_info_df = pd.read_csv(result_path+filename)
    
    traded_volume_df = stock_info_df[["stock", "date", "trade", "time"]].pivot(index=["stock", "date"], columns=["time"])["trade"].fillna(0).astype(int)
    px_df = stock_info_df[["stock", "date", "midEnd", "time"]].pivot(index=["stock", "date"], columns=["time"])["midEnd"].\
        fillna(method="ffill", axis="columns").fillna(method="bfill", axis="columns")

    daily_stock_info_df = pd.DataFrame({
        "px_vol": px_df.pct_change(1, axis="columns").std(axis="columns"),
        "volume": traded_volume_df.abs().sum(axis="columns"),
    })

    return traded_volume_df, px_df, daily_stock_info_df

def monthly_stock_info(daily_stock_info_df, num_days_precompute=20):
    stacked_info = daily_stock_info_df.reset_index().pivot(index="date", columns="stock", values=["px_vol", "volume"])\
                                    .rolling(num_days_precompute).mean().shift(0)

    monthly_stock_info_df = pd.DataFrame({
        "px_vol": stacked_info.px_vol.unstack(),
        "volume": stacked_info.volume.unstack(),
    }).reset_index()
    
    return monthly_stock_info_df

def monthly_scaling_factor(monthly_stock_info_df, idx):
    return monthly_stock_info_df.set_index(["stock", "date"]).loc[idx]


traded_volume_df, px_df, daily_stock_info_df = data_initilise(data_dir, year)
monthly_stock_info_df = monthly_stock_info(daily_stock_info_df)
monthly_scaling_factor = monthly_scaling_factor(monthly_stock_info_df, traded_volume_df.index)

# Save each dataframe to a pickle file
def save_to_pickle(data, filename):
    path = "../pkl_dump/"
    with open(path + filename, 'wb') as f:
        pickle.dump(data, f)


# Save traded_volume_df
save_to_pickle(traded_volume_df, 'traded_volume_df.pkl')

# Save px_df
save_to_pickle(px_df, 'px_df.pkl')

# Save daily_stock_info_df
save_to_pickle(daily_stock_info_df, 'daily_stock_info_df.pkl')

# Save monthly_scaling_factor
save_to_pickle(monthly_scaling_factor, 'monthly_scaling_factor.pkl')
