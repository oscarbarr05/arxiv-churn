"""One-shot data collection (run by the project author, NOT by the grader —
the repo ships with the resulting CSVs committed in data/raw/).

Stage 1: quarterly corpus sample of cs.CL papers 2014-2025 (sampling frame).
Stage 2: full submission history for every discovered researcher, so the
churn label reflects real activity rather than a sampling artifact.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import scraper  # noqa: E402

TARGET_AUTHORS = 320
SEED = 42

print("=== Stage 1: corpus sample (cs.CL, 2014-2025, 100/quarter) ===")
corpus = scraper.fetch_corpus(category="cs.CL", year_from=2014, year_to=2025,
                              per_quarter=100)
path = scraper.save_raw(corpus, "papers.csv")
print(f"saved {len(corpus)} unique papers -> {path}")

# discover researchers: appear on >= 2 sampled papers (filters one-offs and
# reduces homonym noise), then random-sample down to the target size
authors = (
    corpus["authors"].str.split("|").explode().value_counts()
)
candidates = authors[authors >= 2]
print(f"{len(candidates)} authors appear >= 2 times in the corpus sample")
if len(candidates) > TARGET_AUTHORS:
    candidates = candidates.sample(TARGET_AUTHORS, random_state=SEED)
names = sorted(candidates.index.tolist())
print(f"fetching full histories for {len(names)} researchers "
      f"(~{len(names) * 4 / 60:.0f} min at 3 s/request)")

print("=== Stage 2: full per-author histories ===")
histories = scraper.fetch_author_histories(names)
path = scraper.save_raw(histories, "author_papers.csv")
print(f"saved {len(histories)} author-paper rows -> {path}")
