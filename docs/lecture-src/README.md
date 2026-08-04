# Lecture note sources

HTML sources for the study PDFs in the repo root. Committed so the PDFs can be
edited and re-rendered instead of rewritten.

| source | rendered to |
| --- | --- |
| `week1_day1-4_condensed.html` | `WEEK1_DAY1-4_condensed.pdf` (10 pages) |
| `week1_day5-8_condensed.html` | `WEEK1_DAY5-8_condensed.pdf` (9 pages) |

Measured numbers in these documents come from `results/*.csv`. Where a CSV and a
prose doc in `docs/` disagree, the CSV wins.

Render with headless Chrome, from the repo root:

```
chrome.exe --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="WEEK1_DAY1-4_condensed.pdf" \
  "file:///<abs-path>/docs/lecture-src/week1_day1-4_condensed.html"
```

Page geometry (A4, margins, font sizes) lives in the `@page` and `body` rules at
the top of each source file. Day sections carry `class="day"`, which forces a page
break before each one.
