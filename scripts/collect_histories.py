"""Resumable Stage 2: fetch full per-author histories.

Reads data/raw/papers.csv (Stage 1 output), derives the same candidate list
(seed 42), skips authors already present in author_papers.csv, appends new
rows and saves incrementally. Re-run until "DONE" is printed.
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import scraper  # noqa: E402

TARGET_AUTHORS = 320
SEED = 42
MAX_SECONDS = 8.5 * 60          # stop cleanly before the runner's timeout
OUT = ROOT / "data" / "raw" / "author_papers.csv"

corpus = pd.read_csv(ROOT / "data" / "raw" / "papers.csv")
counts = corpus["authors"].str.split("|").explode().value_counts()
candidates = counts[counts >= 2]
if len(candidates) > TARGET_AUTHORS:
    candidates = candidates.sample(TARGET_AUTHORS, random_state=SEED)
names = sorted(candidates.index.tolist())

done: set[str] = set()
existing = None
if OUT.exists():
    existing = pd.read_csv(OUT)
    done = set(existing["queried_author"].unique())
todo = [n for n in names if n not in done]
print(f"{len(done)} done, {len(todo)} to go")

start = time.time()
batch: list[pd.DataFrame] = [] if existing is None else [existing]
fetched = 0
for name in todo:
    if time.time() - start > MAX_SECONDS:
        print("time budget reached — saving and exiting (re-run to resume)")
        break
    df = scraper.fetch_author_histories([name], verbose=False)
    if not df.empty:
        batch.append(df)
    fetched += 1
    if fetched % 20 == 0:
        pd.concat(batch, ignore_index=True).to_csv(OUT, index=False)
        print(f"  progress: {len(done) + fetched}/{len(names)} authors")

if batch:
    pd.concat(batch, ignore_index=True).to_csv(OUT, index=False)
remaining = len(todo) - fetched
print("DONE" if remaining <= 0 else f"{remaining} authors remaining")
