# Quantitative Trading and Price Impact

## Contributors
- Kelvin Wu
- Ethan Cui
- Ahmed Reda Seghrouchni

## Project Overview
This project integrates various quantitative trading components into a cohesive framework, including data handling, model fitting, backtesting, optimal trading strategies, performance metrics, and sensitivity analysis. Examples and outputs illustrate both standard and advanced procedures.

## Data Scope
### Data Pre-processing
- **traded_volume_df**: Sum of traded volume in each bin, double-indexed by stock and date.
- **px_df**: Mid-price at the end of each bin window, double-indexed by stock and date.
- **monthly_scaling_factor**: Rolling average daily price volatility and daily volume over 20 days, double-indexed by stock and date.

### In-sample and Out-of-sample Data
- **In-sample**: May 2019
- **Out-of-sample**: June 2019

## Impact Model Fitting
### Models
- **Obizhaeva-Wang (OW) Model**: 
    \[
    dI_t = -\beta I_t dt + \lambda \frac{\sigma}{\text{ADV}} dQ_t
    \]
- **Alfonsi-Fruth-Schied (AFS) Model**:
    \[
    dJ_t = -\beta J_t \, dt + \lambda \frac{\sigma}{\text{ADV}} \, dQ_t \\
    I_t = J_t^{0.5}
    \]

### Performance Metrics
- **In-sample and Out-of-sample $R^2$**

## Backtest Engine
### Waelbroeck's Backtest Algorithm
1. **Initialize Parameters**: Historical strategy \(Q^r\), historical prices \(P^r\), new strategy \(Q\), price impact model \(I\).
2. **Compute Fundamental Price**:
    \[
    S_t = P^r_t - I_t(Q^r)
    \]
3. **Simulate New Prices**:
    \[
    P_t(Q) = P^r_t - I_t(Q^r) + I_t(Q)
    \]

### Backtest Procedure
1. **Generate synthetic alpha series**
2. **Obtain optimal impact**
3. **Generate optimal trading strategy**
4. **Calculate PnL**
5. **Obtain performance metrics**

## Sensitivity Analysis and Stress Testing
### Scenarios
- **Varying Correlations**: Low (0.1), Medium (0.6), High (0.9)
- **Different Alpha Horizons**: Short (6), Medium (12), Long (30)
- **Forced Liquidation**: Liquidate all positions mid-day

### Observations
- **Low Correlation**: Minimal trading activity, flat trade positions.
- **High Correlation**: Significant oscillations in trades, strong alpha signal.
- **AFS Model**: More pronounced impact compared to OW model.

### Performance Metrics
- **Expected Daily PnL**
- **Sharpe Ratio**
- **Transaction Costs**
- **Max Daily Drawdown**
- **Max Impact Dislocation**

## Plots
- **Trade/ADV under different correlations and models**
- **Synthetic Alphas and Impacts under different correlations and models**
- **Trade/ADV under different alpha horizons and models**
- **Forced Liquidation Prices and Impact**
