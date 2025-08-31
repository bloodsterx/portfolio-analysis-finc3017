import polars as pl
import statsmodels.api as sm

lf = pl.LazyFrame(pl.read_csv("Part_B_Data_530521305.csv",
                              skip_rows=1))

lf_market = pl.LazyFrame(pl.read_csv("CAPM_Data.csv"))
lf_market = lf_market.with_columns(
    (pl.col("Mkt-RF") - pl.col("RF")).alias("premium"))

tickers = lf.collect_schema().names()

for ticker in tickers[1:]:
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
    print(f"\n===== {ticker} CAPM (excess returns) =====")
    print(results.summary())

    # Also print a compact coefficient table for easy extraction
    coef = {
        "alpha": results.params.get("const"),
        "beta": results.params.get("premium"),
        "se_alpha": results.bse.get("const"),
        "se_beta": results.bse.get("premium"),
        "t_alpha": results.tvalues.get("const"),
        "t_beta": results.tvalues.get("premium"),
        "p_alpha": results.pvalues.get("const"),
        "p_beta": results.pvalues.get("premium"),
        "r2": results.rsquared,
        "adj_r2": results.rsquared_adj,
    }
    print(coef)

# tickers = lf.slice(1)

# print(tickers.collect())
