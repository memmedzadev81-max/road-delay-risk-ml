# Road Construction Delay Risk Prediction

> Intermediate ML project on a portfolio of **820 road projects**.  
> Goal: flag the jobs that are most likely to finish late *before* they start, and turn the model output into a simple risk band that a project office can actually use.

This builds on the same domain as my earlier [Oracle SQL delay analysis](https://github.com/memmedzadev81-max/road-project-delay-analysis).  
Here the focus is supervised learning: clean the data, test a few models, look at feature importance, and produce actionable risk bands.

---

## Questions

1. What share of projects finish late, and how large is the cost impact?
2. Which drivers matter most — terrain, contractor size, season, design complexity, utilities?
3. Can a simple model (logistic + tree ensembles) separate high-risk from low-risk jobs?
4. How should limited supervision capacity be allocated?

---

## Dataset

| Item | Detail |
|------|--------|
| Source | Synthetic portfolio (patterns inspired by real construction risk) |
| Rows | 820 projects |
| File | `data/road_projects.csv` |
| Target | `delayed` (0/1) |
| Key features | terrain, project type, contractor size, start season, design complexity, utility conflicts, material price index, rainfall, freeze-thaw, previous delay rate of similar contractors |

**Data note**  
The data is simulated. Dirty records (region spelling variants, a few missing rainfall values) are intentional so the cleaning step is visible. Results show the *method*, not real market statistics. The same pipeline works on a real export once the column names are mapped.

---

## Pipeline

1. Light cleaning (strip region names, fill rainfall with region median)
2. EDA + χ² tests and point-biserial correlations
3. One-hot + scaling → Logistic Regression / Random Forest / Gradient Boosting
4. 5-fold stratified CV + hold-out metrics (ROC-AUC, PR-AUC, classification report)
5. Feature importance and 4-band risk score on the full portfolio

---

## Main findings

| Finding | Number |
|---------|--------|
| Overall delay rate | **~39%** |
| Small vs large contractor | **53%** vs **28%** |
| Mountainous vs flat | **50%** vs **27%** |
| Winter vs summer starts | **49%** vs **31%** |
| High vs low design complexity | **53%** vs **32%** |
| Best model ROC-AUC (hold-out) | **~0.66** (Logistic / RF) |
| Critical risk band actual delay rate | **~65%** |
| Low risk band actual delay rate | **~12%** |
| Avg cost overrun on delayed jobs | **~5 mln AZN** |

Top drivers that keep appearing:

- Previous delay rate of the contractor style
- Small contractors
- Mountainous / urban terrain
- Winter starts
- High design complexity
- Utility conflicts and freeze-thaw cycles

---

## Risk bands (portfolio)

| Band | Projects | Actual delay rate | Total budget (mln AZN) |
|------|----------|-------------------|------------------------|
| Low | ~145 | ~12% | ~2 900 |
| Medium | ~230 | ~27% | ~4 500 |
| High | ~250 | ~45% | ~4 300 |
| Critical | ~195 | ~65% | ~3 400 |

Critical + High together are roughly half the portfolio but carry the large majority of actual delays. That is where extra supervision, better contractors, or schedule buffers should go first.

---

## Recommendations

1. On mountain + winter packages prefer larger or proven contractors; the delay gap is large.
2. Front-load utility relocation on urban jobs — conflicts show up consistently.
3. Avoid pure winter starts for high-complexity designs when the programme allows.
4. Track the bidder’s recent delay rate more carefully at tender stage.
5. Use the 4-band score as a weekly filter for the project control team (Critical first).

---

## How to run

```bash
git clone https://github.com/memmedzadev81-max/road-delay-risk-ml.git
cd road-delay-risk-ml
pip install -r requirements.txt
python analysis.py
```

Outputs land in `outputs/`:

- `eda_delay_rates.png`
- `feature_importance.png`
- `roc_curves.png`
- `scored_projects.csv` (project_id + delay probability + risk band)

To regenerate the synthetic data:

```bash
python src/generate_data.py
```

---

## Limitations

- Synthetic data → patterns are illustrative.
- No time-series leakage check beyond a simple stratified split (real projects would need temporal or regional blocking).
- Cost overrun is a simple derived field, not a full EVM model.
- Class imbalance is moderate; `class_weight="balanced"` and a slightly lowered threshold (0.45) are used.

---

## Author

**Vusal Mammadzade** — Data Analyst · background in transportation construction  
GitHub: [memmedzadev81-max](https://github.com/memmedzadev81-max)
