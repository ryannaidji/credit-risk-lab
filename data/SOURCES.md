# Data sources

All data here is real and public. Nothing is synthetic. (The monitoring code separately applies a
*simulated* corruption to the real holdout; see README.)

| File | Source | License |
|---|---|---|
| `credit_default.csv` | Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H | CC BY 4.0 |
| `cc_delinquency_sa.csv` | Board of Governors of the Federal Reserve System, *Charge-Off and Delinquency Rates on Loans and Leases at Commercial Banks*, delinquency rates, seasonally adjusted, credit cards, all banks. https://www.federalreserve.gov/releases/chargeoff/delallsa.htm | Public domain per the Board's website notice; cite the Board as source |
| `split.csv` | Derived: stratified 70/30 split by ID, `python -m credit_risk.data` | Same as above |

`credit_default.csv` is converted from the original XLS, with the label column renamed `default`.
Snapshots were downloaded on 2026-11-08 with `scripts/fetch_data.py`.
