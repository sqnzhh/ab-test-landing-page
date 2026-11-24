# A/B Test Analysis: New Landing Page vs Old

Should an e-commerce company launch a new landing page? Full A/B test analysis on ~290K users (public Udacity dataset, Jan 2017, 23 days).

**Stack:** Python (Pandas, SciPy, statsmodels), Matplotlib.

## Steps
1. **Data quality.** Removed 3,893 rows (1.3%) where the group and the page did not match, plus 1 duplicate user → 290,584 users.
2. **Sample Ratio Mismatch check.** Control 145,274 vs treatment 145,310; chi-square p = 0.95 → the split is fine.
3. **Main metric: conversion rate.** Control **12.04%**, treatment **11.88%** (−0.16 pp, −1.3% relative).
   Two-proportion z-test: z = −1.31, **p = 0.19**; 95% CI for the difference: [−0.39; +0.08] pp.
4. **Power / MDE.** With this sample the test could detect an effect of about **±2.9% relative** (0.35 pp) at 80% power. Detecting a +1% uplift would need ~1.15M users per group.
5. **Peeking check.** I recalculated the p-value every day: it never dropped below 0.05, so stopping early would not have changed the decision.
6. **Segments.** By country (US, UK, CA) there is no significant difference either (all p > 0.13, Bonferroni-corrected p > 0.39).

## Conclusion
No evidence that the new page improves conversion; the confidence interval even leans negative. **Recommendation: do not launch the new page.** If the team expects a small effect (<3%), the test would need several times more traffic; in that case it's better to test a stronger change.

Run: `bash download_data.sh && pip install -r requirements.txt && python ab_test_analysis.py`

Charts: `charts/01_cumulative_cr_pvalue.png`, `charts/02_conversion_by_country.png`
