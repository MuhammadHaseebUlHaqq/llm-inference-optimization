# Lecture note sources

HTML sources for the study PDFs in the repo root. Committed so the PDFs can be
edited and re-rendered instead of rewritten.

| source | rendered to |
| --- | --- |
| `week1_day1-4_condensed.html` | `WEEK1_DAY1-4_condensed.pdf` (10 pages) |
| `week1_day5-8_condensed.html` | `WEEK1_DAY5-8_condensed.pdf` (9 pages) |
| `week1_day9-12_condensed.html` | `DAY9-12_condensed.pdf` (9 pages) |

Day numbering is continuous across the 3-week roadmap, so it does not match the
per-week numbering in `ROADMAP_3week_interview_prep.pdf`. Days 1-7 are Week 1,
Day 8 onward is Week 2. That is why the Days 9-12 PDF has no week prefix.

Measured numbers in these documents come from `results/*.csv`. Where a CSV and a
prose doc in `docs/` disagree, the CSV wins.

Render with `scripts/render_lecture_pdfs.sh`, from the repo root:

```
bash scripts/render_lecture_pdfs.sh              # all three
bash scripts/render_lecture_pdfs.sh day9-12      # one, matched on the source stem
```

The script finds a Chrome or Chromium binary, converts the source path to the
absolute `file://` URL Chrome requires, and writes each PDF to the repo root
under the name in the table above. Set `CHROME=/path/to/chrome` to override the
lookup. It is a thin wrapper over the invocation that used to be typed by hand:

```
chrome --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="WEEK1_DAY1-4_condensed.pdf" \
  "file:///<abs-path>/docs/lecture-src/week1_day1-4_condensed.html"
```

Check the page counts in the table after rendering. Chrome repaginates silently
when a source changes, and those counts are part of the record.

Page geometry (A4, margins, font sizes) lives in the `@page` and `body` rules at
the top of each source file. Day sections carry `class="day"`, which forces a page
break before each one.
