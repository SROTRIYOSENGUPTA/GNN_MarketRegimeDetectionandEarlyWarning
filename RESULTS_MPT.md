# MPT Backtest Results

Annualized Sharpe ratio, n=5 seeds (s7, s42, s101, s123, s2025).

| Config | s7 | s42 | s101 | s123 | s2025 | mean | std |
|---|---|---|---|---|---|---|---|
| A_bloomberg_gnn_cov_mv | 0.0280 | 1.4388 | -0.2816 | -1.3938 | 1.7798 | 0.3142 | 1.3006 |
| B_bloomberg_sample_cov_mv | -0.0411 | 1.3878 | -0.2251 | -1.4849 | 1.7566 | 0.2786 | 1.3115 |
| C_none_gnn_cov_mv | 0.8247 | 1.4637 | 1.7988 | 0.9738 | 1.0765 | 1.2275 | 0.3974 |
| D_none_sample_cov_mv | 0.9203 | 1.5541 | 1.8736 | 1.1133 | 1.0773 | 1.3077 | 0.3943 |
| E_bloomberg_equal_weight_picks | 0.6354 | 1.0055 | 0.4844 | 1.0727 | 0.5907 | 0.7577 | 0.2637 |
| F_equal_weight_universe | 1.0878 | 1.0878 | 1.0878 | 1.0878 | 1.0878 | 1.0878 | 0.0000 |
