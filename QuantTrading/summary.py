# Main functions
def get_impact_state(traded_volume_df, monthly_scaling_factor, half_life, model_type):
'''Calculate impact state using df.ewm()'''

def get_regression_results(impact_state, px_df, in_sample_month, 
                           explanation_horizon_periods, all=False):
'''The main function to get fitted impact coefs;
   all=True: return index impact coef.'''
   
# Helper functions for naive regression
def impact_regression_statistics(impact_state, explanation_horizon_periods, px_df):
'''Get regression statistics for impact state and px_df so that r2 and fitting can
   be calculated quickly and using one-pass of all stocks.'''

def regression_result_by_stock(daily_stock_reg_info_df, in_sample_month, all=False):
'''Get regression results indexed by stock.'''

############################################################################################################
############################################################################################################
############################################################################################################

# Main functions for ridge-style regression
def get_index_impact_coef(traded_volume_df, px_df, monthly_scaling_factor,
                          half_life, model_type, in_sample_month):
'''Pack the traded volume and price into a stock index, 
and calculate the index-level coefficients'''

def eta_info(x, y, x_valid, y_valid, eta_list, 
             initial_params, global_coef, loss_function):
'''Get the loss function for different eta values'''

# Helper functions for ridge-style regression
def train_validation_split(req_stat_df, valid_fraction=0.2):
'''Split the data into training and validation set for ridge regression'''

def optimize_ridge(x, y, eta, initial_param, global_coef, loss_function):
'''Fit the ridge regression model using the given data and loss function.'''