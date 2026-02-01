import pandas as pd
print(pd.read_csv("q3_weekly_norm_fixed.csv").columns.tolist())
print(pd.read_csv("q3_do_well_defs.csv").columns.tolist())
print(pd.read_csv("latent_estimates.csv").columns.tolist())
print('---------------------------------------------')

import pandas as pd
lat = pd.read_csv("latent_estimates.csv")
print(lat["rule_type"].value_counts(dropna=False))
print(lat[["rule_type","judge_metric","fan_vote_mean"]].groupby("rule_type").agg(["min","max"]))
