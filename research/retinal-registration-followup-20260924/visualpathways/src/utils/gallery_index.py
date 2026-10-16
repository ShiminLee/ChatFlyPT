"""Build a searchable index page for a per-instance HTML gallery.

The gallery notebook (`gallery_vcbn.ipynb`, `html_per_row` mode) writes one
self-contained HTML page per neuron instance into a flat folder. With thousands
of files there is no way to browse them. This module relocates those pages into
a `pages/` subfolder and writes an `index.html` at the gallery root: a left
sidebar with a search box and instance list, and a right-hand iframe that shows
the selected gallery page. The sidebar stays visible the whole time, so the user
can return to the index without any change to the gallery pages themselves.

Run as a script against an existing folder, or call `build_gallery_index` from
the notebook after writing the pages.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path

INDEX_NAME = "index.html"


def natural_sort_key(s: str) -> list[object]:
    """Return a sort key that orders embedded numbers numerically.

    So ``aIPg2`` sorts before ``aIPg10`` (mirrors the notebook's ordering).

    Args:
        s: The string to build a key for.

    Returns:
        A list of lowercased string fragments and integers.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", str(s))
    ]


_INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { font-family: arial, sans-serif; display: flex; }
  nav {
    width: 180px; flex: 0 0 180px; height: 100vh; display: flex;
    flex-direction: column; border-right: 1px solid #ccc; background: #fafafa;
  }
  .head { padding: 10px 12px 6px; border-bottom: 1px solid #eee; }
  .head h1 { font-size: 16px; margin: 0 0 8px; color: #333; }
  #search {
    width: 100%; padding: 6px 8px; font-size: 14px;
    border: 1px solid #bbb; border-radius: 4px;
  }
  #count { font-size: 12px; color: #777; margin-top: 6px; }
  ul { list-style: none; margin: 0; padding: 0; overflow-y: auto; flex: 1; }
  li { display: flex; align-items: center; border-bottom: 1px solid #f0f0f0; }
  li.hidden { display: none; }
  li a.name {
    flex: 1; padding: 7px 10px; font-size: 13px; color: #1a5; cursor: pointer;
    text-decoration: none; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }
  li a.name:hover { background: #eef7f0; }
  li.active a.name { background: #d6ecdb; font-weight: 600; color: #084; }
  li a.ext {
    padding: 7px 8px; font-size: 13px; color: #aaa; text-decoration: none;
  }
  li a.ext:hover { color: #08a; }
  main { flex: 1; height: 100vh; }
  #viewer { width: 100%; height: 100%; border: 0; }
  #placeholder {
    display: flex; align-items: center; justify-content: center;
    height: 100%; color: #999; font-size: 15px;
  }
</style>
</head>
<body>
<nav>
  <div class="head">
    <h1>__TITLE__</h1>
    <input id="search" type="text" placeholder="Search instance…"
           autocomplete="off" autofocus>
    <div id="count"></div>
  </div>
  <ul id="list"></ul>
</nav>
<main>
  <iframe id="viewer" name="viewer" style="display:none"></iframe>
  <div id="placeholder">Select an instance to view its gallery.</div>
</main>
<script>
const INSTANCES = __INSTANCES_JSON__;
const PAGES = __PAGES_JSON__;

const list = document.getElementById("list");
const search = document.getElementById("search");
const count = document.getElementById("count");
const viewer = document.getElementById("viewer");
const placeholder = document.getElementById("placeholder");
const items = {};

function pageUrl(name) {
  return PAGES + "/" + encodeURIComponent(name) + ".html";
}

for (const name of INSTANCES) {
  const li = document.createElement("li");
  const a = document.createElement("a");
  a.className = "name";
  a.textContent = name;
  a.href = "#" + encodeURIComponent(name);
  const ext = document.createElement("a");
  ext.className = "ext";
  ext.textContent = "↗";
  ext.title = "Open in new tab";
  ext.target = "_blank";
  ext.href = pageUrl(name);
  li.appendChild(a);
  li.appendChild(ext);
  list.appendChild(li);
  items[name] = li;
}

function select(name) {
  if (!(name in items)) return;
  for (const li of Object.values(items)) li.classList.remove("active");
  items[name].classList.add("active");
  viewer.src = pageUrl(name);
  viewer.style.display = "";
  placeholder.style.display = "none";
  items[name].scrollIntoView({ block: "nearest" });
}

function applyFilter() {
  const q = search.value.trim().toLowerCase();
  let shown = 0;
  for (const name of INSTANCES) {
    const match = name.toLowerCase().includes(q);
    items[name].classList.toggle("hidden", !match);
    if (match) shown++;
  }
  count.textContent = shown + " / " + INSTANCES.length + " instances";
}

function fromHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (h) select(h);
}

search.addEventListener("input", applyFilter);
window.addEventListener("hashchange", fromHash);

applyFilter();
fromHash();
</script>
</body>
</html>
"""


def _move_pages(gallery_dir: Path, pages_dir: Path) -> int:
    """Move top-level ``*.html`` (except the index) into ``pages_dir``.

    Args:
        gallery_dir: Folder holding the flat gallery HTML files.
        pages_dir: Destination subfolder (created if absent).

    Returns:
        The number of files moved.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for fp in gallery_dir.glob("*.html"):
        if fp.name == INDEX_NAME:
            continue
        dest = pages_dir / fp.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(fp), str(dest))
        moved += 1
    return moved


def build_gallery_index(
    gallery_dir: str | Path,
    pages_subdir: str = "pages",
    title: str | None = None,
    move: bool = True,
) -> Path:
    """Move gallery pages into a subfolder and write a searchable ``index.html``.

    Idempotent: re-running rescans the pages subfolder and rewrites the index;
    files already in the subfolder stay put.

    Args:
        gallery_dir: The gallery folder (where ``index.html`` is written).
        pages_subdir: Name of the subfolder holding the per-instance pages.
        title: Page title; defaults to the gallery folder name.
        move: If True, relocate any top-level ``*.html`` into the pages subfolder
            before building the index. Set False when pages are already written
            straight into the subfolder.

    Returns:
        Path to the written ``index.html``.
    """
    gallery_dir = Path(gallery_dir)
    pages_dir = gallery_dir / pages_subdir
    if move:
        _move_pages(gallery_dir, pages_dir)

    instances = sorted(
        (fp.stem for fp in pages_dir.glob("*.html")), key=natural_sort_key
    )
    page_title = title if title is not None else gallery_dir.name

    htmltext = (
        _INDEX_TEMPLATE.replace("__TITLE__", html.escape(page_title))
        .replace("__INSTANCES_JSON__", json.dumps(instances))
        .replace("__PAGES_JSON__", json.dumps(pages_subdir))
    )
    index_path = gallery_dir / INDEX_NAME
    index_path.write_text(htmltext, encoding="utf-8")
    print(f"Indexed {len(instances)} page(s) -> {index_path}")
    return index_path


def _main() -> None:
    """Parse CLI arguments and build the index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gallery_dir", help="Gallery folder containing the HTML pages")
    parser.add_argument(
        "--pages-subdir", default="pages", help="Subfolder for per-instance pages"
    )
    parser.add_argument(
        "--no-move",
        dest="move",
        action="store_false",
        help="Pages are already in the subfolder; only (re)write index.html",
    )
    parser.add_argument("--title", default=None, help="Index page title")
    args = parser.parse_args()
    build_gallery_index(
        args.gallery_dir,
        pages_subdir=args.pages_subdir,
        title=args.title,
        move=args.move,
    )


if __name__ == "__main__":
    _main()
