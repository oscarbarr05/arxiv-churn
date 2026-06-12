"""arXiv corpus scraper.

Fetches paper metadata from the public arXiv API (no key, no login) and saves
one row per paper to data/raw/papers.csv. Knows NOTHING about churn or
modeling — it only collects raw data (single responsibility).

Design notes
------------
* The arXiv API asks clients to wait 3 seconds between requests
  (https://info.arxiv.org/help/api/tou.html). We honour that with time.sleep.
* We sample papers in QUARTERLY date slices across many years instead of one
  big "most recent" query. A recent-only sample would contain almost no
  inactive (churned) researchers — a textbook sampling-bias mistake.
* Every field is read with .get(...) / defensive parsing because Atom entries
  occasionally miss fields.
* Pagination inside a slice is supported via the `start` parameter.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

API_URL = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
SLEEP_SECONDS = 3.0          # arXiv API terms of use
PAGE_SIZE = 100              # results per request inside a slice

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _parse_entry(entry: ET.Element) -> dict | None:
    """Turn one Atom <entry> into a flat dict; return None if malformed."""
    def text(tag: str, default: str = "") -> str:
        el = entry.find(ATOM + tag)
        return el.text.strip() if el is not None and el.text else default

    arxiv_id = text("id")
    published = text("published")
    title = " ".join(text("title").split())
    if not arxiv_id or not published:
        return None

    authors = [
        a.findtext(ATOM + "name", default="").strip()
        for a in entry.findall(ATOM + "author")
    ]
    authors = [a for a in authors if a]
    if not authors:
        return None

    categories = [
        c.get("term", "") for c in entry.findall(ATOM + "category") if c.get("term")
    ]
    primary = entry.find(ARXIV + "primary_category")
    primary_cat = primary.get("term", "") if primary is not None else (
        categories[0] if categories else ""
    )
    abstract = " ".join(text("summary").split())

    return {
        "arxiv_id": arxiv_id.rsplit("/", 1)[-1],
        "title": title,
        "published": published[:10],          # YYYY-MM-DD
        "authors": "|".join(authors),         # pipe-joined, split later
        "n_authors": len(authors),
        "primary_category": primary_cat,
        "categories": "|".join(categories),
        "abstract_chars": len(abstract),
    }


def _fetch_page(query: str, start: int, max_results: int) -> list[dict]:
    """One API request -> list of paper dicts (handles missing fields)."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    req = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={"User-Agent": "uptp-ids-final-project/1.0 (academic use)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    papers = []
    for entry in root.findall(ATOM + "entry"):
        parsed = _parse_entry(entry)
        if parsed:
            papers.append(parsed)
    return papers


def quarter_slices(year_from: int, year_to: int) -> list[tuple[str, str]]:
    """[(start, end)] quarterly ranges in arXiv's YYYYMMDDHHMM format."""
    slices = []
    for year in range(year_from, year_to + 1):
        for q_start, q_end in [
            ("0101", "0331"), ("0401", "0630"),
            ("0701", "0930"), ("1001", "1231"),
        ]:
            slices.append((f"{year}{q_start}0000", f"{year}{q_end}2359"))
    return slices


def fetch_corpus(
    category: str = "cs.CL",
    year_from: int = 2014,
    year_to: int = 2025,
    per_quarter: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch up to `per_quarter` papers per quarter for one arXiv category.

    Reusable & parameterized: pass any category (cs.CL, cs.CV, stat.ML, ...)
    and any year window. Returns a papers DataFrame, one row per paper.
    """
    all_papers: list[dict] = []
    for start_date, end_date in quarter_slices(year_from, year_to):
        query = (
            f"cat:{category} AND submittedDate:[{start_date} TO {end_date}]"
        )
        got_in_slice = 0
        start = 0
        while got_in_slice < per_quarter:
            want = min(PAGE_SIZE, per_quarter - got_in_slice)
            try:
                page = _fetch_page(query, start=start, max_results=want)
            except Exception as exc:                      # network hiccup
                if verbose:
                    print(f"  retrying {start_date[:6]} after error: {exc}")
                time.sleep(SLEEP_SECONDS * 2)
                try:
                    page = _fetch_page(query, start=start, max_results=want)
                except Exception:
                    page = []
            all_papers.extend(page)
            got_in_slice += len(page)
            start += want
            time.sleep(SLEEP_SECONDS)                     # rate limit
            if len(page) < want:                          # slice exhausted
                break
        if verbose:
            print(f"{start_date[:4]} Q{(int(start_date[4:6]) - 1) // 3 + 1}: "
                  f"+{got_in_slice} papers (total {len(all_papers)})")

    df = pd.DataFrame(all_papers).drop_duplicates(subset="arxiv_id")
    return df.reset_index(drop=True)


def fetch_author_histories(
    names: list[str],
    max_papers_per_author: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Stage 2: full submission history for each discovered author.

    The corpus (stage 1) is only a *sampling frame* to discover researchers;
    one query per author (au:"Name") retrieves their complete arXiv record so
    that recency and the churn label reflect REAL activity, not a sampling
    artifact. Only papers where the name matches an author exactly are kept
    (cheap guard against partial-name matches; full disambiguation of
    homonyms is out of scope and noted as a limitation in the report).
    """
    rows: list[dict] = []
    for i, name in enumerate(names):
        query = f'au:"{name}"'
        try:
            page = _fetch_page(query, start=0, max_results=max_papers_per_author)
        except Exception as exc:
            if verbose:
                print(f"  retry {name!r} after error: {exc}")
            time.sleep(SLEEP_SECONDS * 2)
            try:
                page = _fetch_page(query, start=0, max_results=max_papers_per_author)
            except Exception:
                page = []
        kept = 0
        for paper in page:
            authors = paper["authors"].split("|")
            if name in authors:
                rows.append({**paper, "queried_author": name})
                kept += 1
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(names)} authors fetched "
                  f"({len(rows)} author-paper rows)")
        time.sleep(SLEEP_SECONDS)
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def save_raw(df: pd.DataFrame, filename: str = "papers.csv") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / filename
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    corpus = fetch_corpus()
    path = save_raw(corpus)
    print(f"saved {len(corpus)} papers -> {path}")
