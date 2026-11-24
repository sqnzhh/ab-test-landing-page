"""A/B test analysis: new vs old landing page (conversion).
Steps: data quality -> SRM check -> conversion & z-test -> CI -> power/MDE -> daily dynamics -> country segments."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

ROOT = Path(__file__).parent
CHARTS = ROOT / "charts"; CHARTS.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data/ab_data.csv", parse_dates=["timestamp"])
countries = pd.read_csv(ROOT / "data/countries.csv")

# 1. Data quality
raw = len(df)
mismatch = ((df.group == "treatment") != (df.landing_page == "new_page")).sum()
df = df[(df.group == "treatment") == (df.landing_page == "new_page")]
dups = df.user_id.duplicated().sum()
df = df.drop_duplicates("user_id", keep="first")
print(f"Rows: {raw}; group/page mismatches removed: {mismatch} ({mismatch/raw:.2%}); duplicate users removed: {dups}; clean users: {len(df)}")
print(f"Test period: {df.timestamp.min():%Y-%m-%d} to {df.timestamp.max():%Y-%m-%d} ({df.timestamp.dt.date.nunique()} days)")

# 2. Sample ratio mismatch (expected 50/50)
counts = df.group.value_counts()
chi = stats.chisquare(counts.values)
print(f"\nSRM check: control={counts['control']}, treatment={counts['treatment']}, chi2 p-value={chi.pvalue:.3f}")

# 3. Conversion & two-proportion z-test
g = df.groupby("group").converted.agg(["sum", "count", "mean"])
print("\n", g.assign(mean=lambda x: (x["mean"] * 100).round(3)).rename(columns={"mean": "cr_%"}))
conv = [g.loc["treatment", "sum"], g.loc["control", "sum"]]
n = [g.loc["treatment", "count"], g.loc["control", "count"]]
z, p_two = proportions_ztest(conv, n, alternative="two-sided")
lo, hi = confint_proportions_2indep(conv[0], n[0], conv[1], n[1], method="wald")
diff = g.loc["treatment", "mean"] - g.loc["control", "mean"]
print(f"\nDiff (treatment - control): {diff*100:.3f} pp; relative: {diff/g.loc['control','mean']*100:.2f}%")
print(f"z = {z:.3f}, p-value (two-sided) = {p_two:.3f}; 95% CI for diff: [{lo*100:.3f}, {hi*100:.3f}] pp")

# 4. Power / MDE: what effect could this test detect?
base = g.loc["control", "mean"]
analysis = NormalIndPower()
for mde_rel in [0.01, 0.02, 0.05]:
    es = proportion_effectsize(base * (1 + mde_rel), base)
    need = analysis.solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1)
    print(f"Needed users per group for +{mde_rel:.0%} relative uplift (80% power): {need:,.0f}")
es_obs = proportion_effectsize(base, base - 0.0001)
mde = None
for rel in np.arange(0.001, 0.2, 0.001):
    es = proportion_effectsize(base * (1 + rel), base)
    if analysis.power(es, nobs1=n[1], alpha=0.05, ratio=n[0]/n[1]) >= 0.8:
        mde = rel; break
print(f"MDE with current sample (80% power): ~{mde:.1%} relative ({base*mde*100:.2f} pp)")

# 5. Daily dynamics (why peeking is dangerous)
df["date"] = df.timestamp.dt.date
daily = df.groupby(["date", "group"]).converted.agg(["sum", "count"]).unstack()
cum = daily.cumsum()
cum_cr = cum["sum"] / cum["count"] * 100
pvals = []
for d in cum.index:
    _, pv = proportions_ztest([cum.loc[d, ("sum", "treatment")], cum.loc[d, ("sum", "control")]],
                              [cum.loc[d, ("count", "treatment")], cum.loc[d, ("count", "control")]])
    pvals.append(pv)
print(f"\nDays on which the cumulative p-value dropped below 0.05: {sum(p < 0.05 for p in pvals)} of {len(pvals)}")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
cum_cr.plot(ax=a1, marker="o"); a1.set_title("Cumulative conversion rate by group (%)"); a1.set_ylabel("CR, %")
a2.plot(cum.index, pvals, marker="o", color="tab:red"); a2.axhline(0.05, ls="--", color="gray")
a2.set_title("Cumulative p-value over time (dashed = 0.05)"); a2.set_ylabel("p-value")
fig.tight_layout(); fig.savefig(CHARTS / "01_cumulative_cr_pvalue.png", dpi=120); plt.close(fig)

# 6. Segments by country
seg = df.merge(countries, on="user_id")
rows = []
for c, d in seg.groupby("country"):
    gg = d.groupby("group").converted.agg(["sum", "count", "mean"])
    _, pv = proportions_ztest([gg.loc["treatment", "sum"], gg.loc["control", "sum"]],
                              [gg.loc["treatment", "count"], gg.loc["control", "count"]])
    rows.append({"country": c, "users": len(d), "cr_control_%": gg.loc["control", "mean"] * 100,
                 "cr_treatment_%": gg.loc["treatment", "mean"] * 100,
                 "diff_pp": (gg.loc["treatment", "mean"] - gg.loc["control", "mean"]) * 100, "p_value": pv})
seg_df = pd.DataFrame(rows).round(3)
seg_df["p_bonferroni"] = (seg_df.p_value * len(seg_df)).clip(upper=1).round(3)
print("\n=== By country ===\n", seg_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(seg_df)); w = .38
ax.bar(x - w/2, seg_df["cr_control_%"], w, label="control (old page)")
ax.bar(x + w/2, seg_df["cr_treatment_%"], w, label="treatment (new page)")
ax.set_xticks(x, seg_df.country); ax.set_ylim(11, 13); ax.set_ylabel("CR, %")
ax.set_title("Conversion by country and group"); ax.legend()
fig.tight_layout(); fig.savefig(CHARTS / "02_conversion_by_country.png", dpi=120); plt.close(fig)
