import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import scraper

page = scraper._fetch_page(
    "cat:cs.CL AND submittedDate:[201401010000 TO 201403312359]", 0, 5
)
print(len(page), "papers")
print(page[0])
