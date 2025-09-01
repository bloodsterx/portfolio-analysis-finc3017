import polars as pl
import statsmodels.api as sm
import csv

lf = pl.LazyFrame(pl.read_csv("Part_B_Data_530521305.csv",
                              skip_rows=1))

lf_market = pl.LazyFrame(pl.read_csv("CAPM_Data.csv"))
lf_market = lf_market.with_columns(
    (pl.col("Mkt-RF") - pl.col("RF")).alias("premium"))

tickers = lf.collect_schema().names()

metrics = ['Ticker', 'alpha', 'beta', 'se_alpha', 'se_beta', 't_alpha', 't_beta', 'p_alpha', 'p_beta']

data = {'Metric': metrics}

for ticker in tickers[1:]:
    mean_return_s = lf.select(ticker).mean()

    dat = lf.select(["Date", ticker]).join(
        other=lf_market.select(
            ["Date", "premium", "RF"]), on="Date", how="inner")

    dat = dat.with_columns(
        (pl.col(ticker) - pl.col("RF")).alias("RI-RF"))

    # Collect to pandas and build design matrix with intercept
    y = dat.select("RI-RF").collect().to_pandas().squeeze()
    X = dat.select("premium").collect().to_pandas()
    X = sm.add_constant(X)  # adds intercept so alpha is estimated

    results = sm.OLS(endog=y, exog=X).fit()

    # Print full OLS summary (includes coefficients, SE, t, p)
    # print(f"\n===== {ticker} CAPM (excess returns) =====")
    print(results.summary())

    values = [str(val) for val in [

        ticker,

        results.params.get("const"),

        results.params.get("premium"),

        results.bse.get("const"),

        results.bse.get("premium"),

        results.tvalues.get("const"),

        results.tvalues.get("premium"),

        results.pvalues.get("const"),

        results.pvalues.get("premium"),

    ]]

    data[ticker] = values

out = pl.DataFrame(data)
print(out)
out.write_csv("output.csv")

import numpy as np
from scipy.optimize import minimize

# Assume your data: replace with your actual 20x20 cov matrix and 20x1 mean vector
Sigma = np.array([

])  # Your 20x20 covariance matrix
r = np.array([...])      # Your 20x1 sample means vector

def utility(x, r, Sigma, tol):
    if tol == 0:  # Min variance: minimize x^T Sigma x / 2 (ignore returns)
        return 0.5 * x.T @ Sigma @ x
    else:
        risk_aversion = 0.5 * (1 / tol)
        return - (x.T @ r - risk_aversion * x.T @ Sigma @ x)  # Minimize -U to maximize U

# Constraints
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum x = 1
bounds = [(0.025, None)] * 20  # x_i >= 0.025, no upper bound

# For each Tol
for tol in [0, 1, 2]:
    if tol == 0:
        # Min variance: minimize quadratic variance
        res = minimize(lambda x: utility(x, r, Sigma, tol=0), x0=np.ones(20)/20,
                       method='trust-constr', bounds=bounds, constraints=constraints)
    else:
        res = minimize(lambda x: utility(x, r, Sigma, tol), x0=np.ones(20)/20,
                       method='SLSQP', bounds=bounds, constraints=constraints)
    print(f"Tol={tol}: Optimal x = {res.x}, Utility = {-res.fun if tol > 0 else -res.fun}")

# Verify: Compare res.x to your Solver x; if close (e.g., within 1E-4), confirmed.





# tickers = lf.slice(1)

# print(tickers.collect())
