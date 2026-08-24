#!/usr/bin/env bash
# Re-render the condensed lecture PDFs from their HTML sources.
#
# The sources in docs/lecture-src/ are the editable form; the PDFs in the repo
# root are build output. Rendering by hand meant retyping a long headless-Chrome
# invocation with an absolute file:// URL, which is exactly the step that gets
# skipped, so the PDFs drift from the sources. This does it in one command.
#
# Usage, from the repo root:
#   bash scripts/render_lecture_pdfs.sh            # render all three
#   bash scripts/render_lecture_pdfs.sh day9-12    # render one, by source stem
#
# Page geometry (A4, margins, font sizes) lives in the @page and body rules at
# the top of each source file, not here. Day sections carry class="day", which
# forces a page break before each one.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="$repo_root/docs/lecture-src"

# source stem : output PDF, in the repo root. Day numbering is continuous across
# the 3-week roadmap, which is why the days 9-12 file carries no week prefix.
targets=(
  "week1_day1-4_condensed:WEEK1_DAY1-4_condensed.pdf"
  "week1_day5-8_condensed:WEEK1_DAY5-8_condensed.pdf"
  "week1_day9-12_condensed:DAY9-12_condensed.pdf"
)

find_chrome() {
  # Explicit override wins, then PATH, then the usual install locations. Any
  # Chromium-family binary works; the flags below are Chrome's.
  if [ -n "${CHROME:-}" ]; then echo "$CHROME"; return; fi
  local c
  for c in google-chrome chrome chromium chromium-browser msedge; do
    if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return; fi
  done
  for c in \
    "/c/Program Files/Google/Chrome/Application/chrome.exe" \
    "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/usr/bin/chromium"
  do
    if [ -x "$c" ]; then echo "$c"; return; fi
  done
  echo "no Chrome or Chromium binary found; set CHROME=/path/to/chrome" >&2
  exit 1
}

to_file_url() {
  # Chrome wants an absolute file:// URL. On Git Bash the path is /c/... but
  # Chrome is a native Windows binary, so hand it a Windows path instead.
  local p="$1"
  if command -v cygpath >/dev/null 2>&1; then
    echo "file:///$(cygpath -m "$p")"
  else
    echo "file://$p"
  fi
}

chrome="$(find_chrome)"
echo "chrome: $chrome"

only="${1:-}"
rendered=0

for entry in "${targets[@]}"; do
  stem="${entry%%:*}"
  out="${entry#*:}"
  if [ -n "$only" ] && [[ "$stem" != *"$only"* ]]; then continue; fi

  src="$src_dir/$stem.html"
  if [ ! -f "$src" ]; then
    echo "missing source: $src" >&2
    exit 1
  fi

  echo "rendering $stem.html -> $out"
  # --no-pdf-header-footer drops the default URL/date furniture, which otherwise
  # prints on every page and makes the output look like a webpage dump.
  "$chrome" --headless --disable-gpu --no-pdf-header-footer --no-sandbox --print-to-pdf="$repo_root/$out" "$(to_file_url "$src")"
  rendered=$((rendered + 1))
done

if [ "$rendered" -eq 0 ]; then
  echo "no source matched '$only'" >&2
  exit 1
fi

echo "rendered $rendered file(s). Check the page count before committing: headless"
echo "Chrome silently repaginates when a source changes, and the counts recorded"
echo "in docs/lecture-src/README.md are part of the record."
