"""
Generic figure-export, PDF-tiling and image-repair helpers for galleries.

Pure utilities shared by the per-instance gallery builders
(``gallery_vpn`` / ``gallery_vcbn``): matplotlib/plotly -> PDF/SVG byte
exporters, PyMuPDF page-tiling primitives (``place`` / ``place_image`` /
``split_rect_lr`` / ``insert_title_italic_f``), small color and label helpers
(``_text_color_for_bg`` / ``add_segment_labels`` / ``spread_path_labels``), and
the /Indexed-colour image-shrink + integrity-repair routines
(``index_pdf_images`` and friends). None of these depend on the gallery data
frames; they take figures / paths / rects as arguments.
"""

import base64
import io
import os
import re
import zlib

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# ---- fonts for PyMuPDF (builtin PDF base-14 fonts) ------------------------
FONT_TEXT = fitz.Font("helv")   # Helvetica
FONT_ITALIC = fitz.Font("tiit")  # Times-Italic


# --------------------------------------------------------------------------
# sorting / naming / formatting
# --------------------------------------------------------------------------
def natural_sort_key(s):
    """Convert string to list of strings and integers for natural sorting"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def safe_name(s):
    """Filesystem-safe stem: replace chars invalid in Windows filenames (e.g. the '/'
    in instances like 'pC1_2a/2b_R') so per-row writes don't try to make a subdir."""
    return str(s).translate(str.maketrans('<>:"/\\|?*', '---------'))


def _fmt(v, fmt="{:.2f}"):
    """Format a possibly-NaN value for the stats table."""
    if pd.isna(v):
        return "n/a"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _text_color_for_bg(hex_color):
    """Return black or white, whichever has better contrast on hex_color."""
    def _lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _hex_to_rgb(hex_color)
    L = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)   # relative luminance
    return "#000000" if L > 0.4 else "#ffffff"


# --------------------------------------------------------------------------
# matplotlib / plotly -> bytes exporters
# --------------------------------------------------------------------------
def mpl_to_pdf_bytes(fig, tight=True):
    """Matplotlib Figure -> vector PDF bytes (imshow rasterised), and close it.

    tight=True  -> bbox_inches='tight': crops to drawn content, so the exported
                   page size varies with content (fine for the heatmaps).
    tight=False -> keeps the full figsize canvas, so every export has identical
                   dimensions -> the panel renders at a fixed size across pages
                   (use for the pathway plot)."""
    for ax in fig.axes:
        for im in ax.get_images():        # AxesImages = the heatmap imshow(s)
            im.set_rasterized(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight' if tight else None)
    plt.close(fig)
    return buf.getvalue()


def mpl_to_rasterized_pdf_bytes(fig):
    """Matplotlib Figure -> vector PDF bytes (imshow rasterised), and close it."""
    for ax in fig.axes:
        for im in ax.get_images():        # AxesImages = the heatmap imshow(s)
            im.set_rasterized(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


def plotly_to_pdf_bytes(fig):
    """Plotly Figure -> vector PDF bytes (needs kaleido; same dep your PNG path used)."""
    return fig.to_image(format='pdf')


def mpl_to_svg_b64(fig):
    """Matplotlib fig -> base64 SVG <img> src (vector, crisp at any zoom), and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------
# PyMuPDF page-tiling primitives
# --------------------------------------------------------------------------
def place(page, pdf_bytes, rect, title=None, subtext=None):
    """Tile a single-page PDF into `rect` (keeps vectors, preserves aspect)."""
    src = fitz.open("pdf", pdf_bytes)
    page.show_pdf_page(rect, src, 0)          # keep_proportion=True by default
    src.close()
    if title:
        page.insert_text((rect.x0 + 4, rect.y0 + 11), title, fontsize=9)
    if subtext:
        lines = subtext.split('\n') if isinstance(subtext, str) else list(subtext)
        fs, line_h = 6, 6 * 1.4
        # draw bottom-up: last line sits ~4 pt above rect.y1, earlier lines stack above
        for k, line in enumerate(reversed(lines)):
            page.insert_text((rect.x0 + 4, rect.y1 - 50 - k * line_h), line, fontsize=fs)


def place_image(page, img_path, rect, title=None):
    """Place a raster image (PNG) into `rect`, preserving aspect ratio."""
    page.insert_image(rect, filename=str(img_path), keep_proportion=True)
    if title:
        page.insert_text((rect.x0 + 4, rect.y0 + 11), title, fontsize=9)


def place_plotly_raster(page, fig, rect, title=None, target_dpi=150):
    """Rasterise a plotly fig at `target_dpi` and place it into `rect`."""
    w_px = max(1, int(rect.width / 72 * target_dpi))
    h_px = max(1, int(rect.height / 72 * target_dpi))
    png = fig.to_image(format='png', width=w_px, height=h_px)
    page.insert_image(rect, stream=png, keep_proportion=True)
    if title:
        page.insert_text((rect.x0 + 4, rect.y0 + 11), title, fontsize=9)


def split_rect_lr(rect, left_frac=0.4, gap=6.0):
    """Split a fitz.Rect into (left, right) sub-rects with a vertical gap between."""
    xmid = rect.x0 + (rect.width - gap) * left_frac
    left = fitz.Rect(rect.x0,        rect.y0, xmid,       rect.y1)
    right = fitz.Rect(xmid + gap,     rect.y0, rect.x1,    rect.y1)
    return left, right


def insert_title_italic_f(page, origin, prefix, suffix, fontsize=9):
    """One title line with the 'f' italic (PyMuPDF = one font per span)."""
    tw = fitz.TextWriter(page.rect)
    pos = fitz.Point(*origin)
    _, pos = tw.append(pos, prefix, font=FONT_TEXT,   fontsize=fontsize)
    _, pos = tw.append(pos, "f",    font=FONT_ITALIC, fontsize=fontsize)
    _, pos = tw.append(pos, suffix, font=FONT_TEXT,   fontsize=fontsize)
    tw.write_text(page)


# --------------------------------------------------------------------------
# figure label helpers
# --------------------------------------------------------------------------
def add_segment_labels(fig, cmap, font_size=26, pad=1):
    """Label each stacked segment with its VI category, centered inside it.

    Also zeros the figure margins and widens the single bar so it fills and
    centers in the panel. Fixed `font_size`; segments too short to fit it are
    hidden (uniformtext). Text color is black/white per segment, by luminance.
    """
    for tr in fig.data:
        name = tr.name
        tr.text = [name] * len(tr.x)
        tr.textposition = "inside"
        tr.insidetextanchor = "middle"      # vertical centering within segment
        tr.textangle = 0
        tr.insidetextfont = dict(
            color=_text_color_for_bg(cmap[name]), family="arial", size=font_size,
        )
    fig.update_layout(
        margin=dict(l=pad, r=pad, t=pad, b=pad),  # recover wasted space + center
        bargap=0.04,                              # bar fills ~96% of width
        uniformtext_minsize=font_size, uniformtext_mode="hide",
    )
    return fig


def spread_path_labels(fig, *, pad_px=2.0, iters=400, leader=True):
    """De-overlap the pathway node labels in-place (handles horizontal collisions).

    plot_pathway centers each cell-type label on its node. The worst overlaps are
    HORIZONTAL: nodes in a left-to-right chain (consecutive flow layers) sit close in x,
    so their labels run together. Two labels can only collide if their x-extents overlap,
    so we find such pairs and push them apart in y (keeping each label's x, so it stays
    over its own node), iterating until clear. Optionally draws a thin leader line from
    each moved label back to its node. Text has clip_on=False, so shifted labels still
    draw if they leave the axes.
    """
    if not fig.axes:
        return
    ax = fig.axes[0]
    texts = [t for t in ax.texts if t.get_text().strip()]
    n = len(texts)
    if n < 2:
        return
    fig.canvas.draw()                       # need a renderer to measure label boxes
    r = fig.canvas.get_renderer()
    bbs = [t.get_window_extent(renderer=r) for t in texts]   # display-space boxes
    dy = [0.0] * n                          # accumulated vertical offset (px) per label

    def span(i):                            # current y-extent of label i (px)
        return bbs[i].y0 + dy[i], bbs[i].y1 + dy[i]

    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                if min(bbs[i].x1, bbs[j].x1) - max(bbs[i].x0, bbs[j].x0) <= 0:
                    continue                # no x overlap -> labels can't collide
                a0, a1 = span(i)
                b0, b1 = span(j)
                ov = min(a1, b1) - max(a0, b0)
                if ov + pad_px > 0:         # overlapping (or within pad) in y -> separate
                    s = (ov + pad_px) / 2 + 0.5
                    if a0 + a1 >= b0 + b1:  # push the higher one up, the lower one down
                        dy[i] += s; dy[j] -= s
                    else:
                        dy[i] -= s; dy[j] += s
                    moved = True
        if not moved:
            break

    if not any(abs(d) > 0.5 for d in dy):
        return
    inv = ax.transData.inverted()
    unit = inv.transform((0, 1))[1] - inv.transform((0, 0))[1]   # data units per px (y)
    for t, off in zip(texts, dy):
        x, y = t.get_position()             # original position == node center
        ny = y + off * unit
        if leader and abs(off) > 3:
            ax.plot([x, x], [y, ny], color="0.6", lw=0.5, zorder=1)
        t.set_position((x, ny))


# --------------------------------------------------------------------------
# PDF image shrink (/Indexed colour) + integrity repair
# --------------------------------------------------------------------------
def _expected_len(d, x):
    """Bytes a strict viewer expects in the decoded image stream, from its dims."""
    cs = d.xref_get_key(x, 'ColorSpace')[1]
    w = int(d.xref_get_key(x, 'Width')[1]); h = int(d.xref_get_key(x, 'Height')[1])
    bpc = int(d.xref_get_key(x, 'BitsPerComponent')[1] or 8)
    ncomp = 1 if ('Indexed' in cs or 'Gray' in cs) else (3 if 'RGB' in cs else 4)
    return (((w * ncomp * bpc + 7) // 8) * h), cs


def _image_issues(d):
    """List (xref, kind) for /Image objects strict viewers (Acrobat/Ghostscript)
    reject: 'palette' = lookup length != (hival+1)*3; 'data' = decoded stream length
    != Width*Height. Both have been real bugs here (hival mismatch; and passing
    pre-compressed bytes to update_stream, which double-compresses -> short data)."""
    issues = []
    for x in range(1, d.xref_length()):
        if d.xref_get_key(x, 'Subtype')[1] != '/Image':
            continue
        expect, cs = _expected_len(d, x)
        m = re.search(r'/Indexed\s*/\w+\s*(\d+)\s*<([0-9A-Fa-f]*)>', cs)
        if m and len(m.group(2)) // 2 != (int(m.group(1)) + 1) * 3:
            issues.append((x, 'palette')); continue
        try:
            got = len(d.xref_stream(x))
        except Exception:
            got = -1
        if got != expect:
            issues.append((x, 'data'))
    return issues


def _repair_images(d):
    """Auto-fix the two known defect classes in place. Returns count fixed."""
    fixed = 0
    for x, kind in _image_issues(d):
        if kind == 'palette':                       # hival declared too high
            cs = d.xref_get_key(x, 'ColorSpace')[1]
            m = re.search(r'(/Indexed\s*/\w+)\s*\d+\s*<([0-9A-Fa-f]*)>', cs)
            d.xref_set_key(x, 'ColorSpace', f'[{m.group(1)} {len(m.group(2))//2//3 - 1} <{m.group(2)}>]')
            fixed += 1
        elif kind == 'data':                        # stream was double-compressed
            expect, _ = _expected_len(d, x)
            try:
                raw = zlib.decompress(d.xref_stream(x))
            except Exception:
                raw = b''
            if len(raw) == expect:
                d.update_stream(x, raw)             # re-store with a SINGLE Flate pass
                fixed += 1
    return fixed


def index_pdf_images(pdf_file, colors=256):
    """Re-encode opaque RGB images in `pdf_file` to /Indexed colour, in place."""
    d = fitz.open(str(pdf_file))
    n = 0
    for x in range(1, d.xref_length()):
        if d.xref_get_key(x, 'Subtype')[1] != '/Image':
            continue
        try:
            pix = fitz.Pixmap(d, x)
        except Exception:
            continue
        if pix.alpha or (pix.n - pix.alpha) < 3:        # only opaque RGB
            continue
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        p = im.quantize(colors=colors, method=Image.MEDIANCUT)
        # Pillow >=10 returns ONLY the used palette entries (the near-2-tone anterior
        # needs ~30); hival MUST match the bytes we write or strict viewers reject it.
        pal = bytes(p.getpalette('RGB'))
        ncol = len(pal) // 3
        # Pass RAW bytes: update_stream defaults to compress=True and Flate-encodes ONCE.
        # Passing pre-zlib'd bytes here double-compresses -> "Insufficient data for an image".
        d.update_stream(x, p.tobytes())
        d.xref_set_key(x, 'ColorSpace', f'[/Indexed /DeviceRGB {ncol - 1} <{pal.hex()}>]')
        d.xref_set_key(x, 'BitsPerComponent', '8')
        for k in ('DecodeParms', 'SMask', 'Decode'):
            d.xref_set_key(x, k, 'null')
        n += 1

    # Self-heal any palette/data defects (e.g. from a stale kernel running older code).
    nfix = _repair_images(d)
    if nfix:
        print(f"  self-check repaired {nfix} image(s)")

    tmp = str(pdf_file) + '.tmp'
    d.save(tmp, garbage=4, deflate=True, clean=True)   # fitz can't overwrite the open file
    d.close()
    os.replace(tmp, str(pdf_file))

    # Hard gate: re-open and confirm no image would be rejected by a strict viewer.
    chk = fitz.open(str(pdf_file))
    left = _image_issues(chk)
    chk.close()
    if left:
        raise RuntimeError(f"{len(left)} image(s) still malformed (strict viewers will "
                           f"error/blank): {left[:5]}")
    return n
