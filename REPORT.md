# REPORT — arXiv Researcher Churn: Predict, Explain, Act

*Final project, Introduction to Data Science (UPTP). Oscar Barrios.*

---

## 1. Problem and framing

Streaming platforms predict which users will stop listening; we predict which
**researchers will stop publishing**. The entity is an author active in arXiv's
`cs.CL` category; churn is defined with a **temporal cutoff** (2024-06-30):
features observe only pre-cutoff behaviour, the label only post-cutoff
activity. This is deliberately stricter than the suggested
`days_inactive > 180` rule: a same-clock threshold would leak recency into
both label and features, inflating every metric. Our design reproduces the
production setting — "given the record up to today, who goes silent next?"

The dataset: **295 researchers** discovered via a quarterly-stratified corpus
sample (2014–2025, 4 800 papers), each with their **complete** submission
history fetched individually (20 333 author-paper rows). Churn rate: **18.6%**
(55 churned / 240 active) — above the ~10% danger zone, and
`class_weight="balanced"` is used everywhere as a second guard.

## 2. Which features matter most — four methods compared

Fourteen engineered features (no raw API fields) covering recency, frequency
and magnitude were ranked by ANOVA F-test (filter), RFE over logistic
regression (wrapper), and decision-tree / random-forest importances
(embedded). Consolidated table in the notebook; consensus rule: a feature
survives with ≥ 2 of 4 votes after correlation pruning (|r| > 0.9).

**Result (8 features):** `recency_days_at_cutoff`, `recent_share_2y`,
`avg_gap_days`, `n_categories`, `papers_per_year`, `career_years`,
`categories_per_paper`, `solo_ratio`. (The variance filter removed
`is_solo_researcher` — constant in this cohort: prolific `cs.CL` authors are
never majority-solo. No pair exceeded the 0.9 correlation threshold.)

The methods agree on the headline: **temporal features dominate**.
`recency_days_at_cutoff` is ranked #1 by all four methods (ANOVA F = 186 vs
74 for the runner-up) — researchers do not vanish abruptly; their cadence
decays first. The disagreements are textbook cases:

* `papers_per_year` — ANOVA ranks it 10th of 13 (its marginal distribution
  barely separates the classes), yet RFE selects it and both trees rank it
  top-5: its signal only appears **in combination** with recency. Filters
  score features in isolation and miss exactly this.
* `is_multidisciplinary` — the mirror image: ANOVA ranks it 4th, but RF puts
  it last; given `n_categories`, the binary flag is redundant — the filter
  cannot see redundancy, embedded methods can.
* The single decision tree concentrates 65% importance on recency and zeroes
  out six features; the forest distributes credit more smoothly. Where DT and
  RF disagree, we trust the average of 100 trees over one greedy tree.

This is the unit's thesis in practice: **better features beat better
algorithms**. A plain RandomForest over well-built temporal features performs
at a level no algorithm could reach over raw fields.

## 3. Model comparison — three setups (5-fold stratified CV)

| setup | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| A — selected original features | 0.885 | 0.746 | 0.582 | 0.652 |
| B — PCA components (3) | 0.858 | 0.671 | 0.473 | 0.537 |
| C — original + network features | **0.892** | **0.759** | **0.618** | **0.678** |

**PCA vs. originals:** PCA compresses for *variance*, not for *churn
discrimination*; the 2-D projection separates churners well enough to be a
useful diagnostic plot, but as model inputs the 3 components cost **0.115 F1**
(0.537 vs 0.652) and, worse, drop recall from 0.582 to 0.473 — the model
misses a quarter more of the researchers actually leaving. Dimensionality
reduction earns its keep here as **explanation** (the elbow plot and the
churn-colored scatter make feature quality visible), not as the production
representation. (The scaler and PCA run *inside* the CV pipeline, so setup B
is not advantaged by leakage.)

**Network features:** the co-authorship graph (295 nodes, 417 edges) yields
degree, betweenness and PageRank centralities. Added to setup A they lift F1
from 0.652 to **0.678** and recall from 0.582 to **0.618**. An honest nuance
the updated selection table reveals: none of the three centralities survives
the consensus vote *on its own* — their signal emerges jointly with the
temporal features, the same filter-vs-interaction lesson as `papers_per_year`.
The sociological reading is the valuable part: low-degree, low-PageRank
researchers — the isolated ones — churn more. Collaboration *is* retention.

**What the API serves:** setup A. Every input can be computed from a single
researcher's public record; serving setup C would require recomputing graph
centralities for each request, coupling inference to a full dataset refresh.
This portability/accuracy trade-off is documented rather than hidden.

## 4. The recommendation engine

The researcher × category interaction matrix (how many of an author's papers
touch each arXiv category) is factorized with truncated SVD
(`scipy.sparse.linalg.svds`, k = 12, 84.3% of variance). An at-risk
researcher and every candidate paper live in the same latent "interest
space"; `/recommend` returns the most cosine-similar **recent papers by
still-active authors**, never the researcher's own work.

The gating rule is business logic, not decoration: recommendations are
returned **only when churn probability ≥ 0.5**. Re-engagement nudges sent to
healthy users train them to ignore you; the budget belongs on the at-risk
segment.

## 5. Retention interventions tied to the top features

* `recency_days_at_cutoff` / `recent_share_2y` high → **early-warning digest**:
  personalized "new in your subfield" alerts (exactly what `/recommend`
  produces) the moment cadence decays, not after two silent years.
* low `degree_centrality` / `pagerank` → **collaboration matchmaking**:
  suggest co-authors and workshops; isolated researchers lack the social pull
  that keeps people publishing.
* `has_long_break` = 1 → **soft re-onboarding** after known gaps (parental
  leave, industry stints): curated "what you missed" summaries lower the
  re-entry cost.
* high `first_author_ratio` + short `career_years` → the PhD-attrition
  profile; institutional mentoring matters more than content nudges.

## 6. Ethics — acting on predicted, not confirmed, churn

A churn score is a **probability, not a fact**, and every intervention should
be designed to be harmless when the prediction is wrong.

* **False positives:** labeling an active researcher "at risk" is harmless if
  the action is a relevant-papers digest; it would be harmful if used for
  hiring, funding or evaluation decisions. This model must never feed such
  decisions: leaving arXiv ≠ leaving science (people move to journals,
  industry labs, or simply rest).
* **Label honesty:** our ground truth is *arXiv silence*, a proxy. The report
  and API say so explicitly (`churn_definition` is returned by `/features`).
* **Feedback loops:** recommending only to at-risk users means the engaged
  majority never benefits; in production we would A/B-test the gate threshold
  rather than hard-coding 0.5 forever.
* **Name ambiguity:** author identity is string-matched; homonyms can merge
  records. Acceptable for a course-scale study, disclosed as a limitation,
  and a reason this system should not make individual-level claims publicly.

## 7. PM reflection — if this were Netflix/Spotify

As a PM I would treat this pipeline as one **predict → explain → act** loop:

1. **Predict** runs nightly, scoring the whole user base; the output is not a
   list of "doomed users" but a ranked re-engagement budget.
2. **Explain** decides the *channel*: recency-driven risk gets content nudges
   (our `/recommend`); isolation-driven risk (network features) gets social
   features — collaborative playlists at Spotify, co-author matchmaking here.
   The feature-selection table is literally the intervention router.
3. **Act** is an experiment, never a fiat: hold out a control group, measure
   *incremental* retention (did the nudge cause the return?), and feed results
   back into the threshold. Shipping the model is the cheapest part; the
   organizational loop — score → intervene → measure → retrain — is the
   product. The 0.5 gate in `/recommend` is the first, deliberately simple
   version of that policy.

The deepest lesson transfers across domains: churn is rarely sudden. Cadence
decays before users leave — at Spotify (sessions per week), on arXiv (papers
per year). Systems that watch the *derivative* of engagement act while there
is still something to save.

## 8. Limitations

* arXiv silence is a proxy for leaving research; venue migration looks like churn.
* Author names are not disambiguated (no ORCID join at this scale).
* One category (`cs.CL`) — generalization to other fields is untested.
* The co-authorship graph only contains edges *within* the sampled cohort;
  centralities are community-relative, not global.
