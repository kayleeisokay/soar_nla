"""Monospace text panels for the qualitative paper figures.

Text is laid out on a fixed character grid (DejaVu Sans Mono at `BODY_PT`), so
token positions are computed in columns and drawn in data coordinates of an
axes whose units are inches (`ax.set_xlim(0, width)`, `ax.set_ylim(height, 0)`).
Devanagari tokens are shaped with HarfBuzz and drawn as glyph outlines from the
bundled Noto Sans Devanagari (SIL OFL, `assets/fonts/`); that path needs the
`figures` extra (`uharfbuzz`, `fonttools`) and is only loaded when such a token
appears.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from soar_nla.figure_style import INK, MUTED, NAVY

BLUE = NAVY
BLUE_WASH = "#DCE6F2"
RED = "#B64342"
RED_WASH = "#F6CFCB"
GREEN = "#2F6B3F"
GREEN_WASH = "#DDF3DE"
LINE = "#CFCECE"
BOX_FILL = "#FAFAFA"

BODY_PT = 6.2
MONO_ADVANCE = 0.602
CHAR_W = BODY_PT * MONO_ADVANCE / 72
LINE_H = BODY_PT * 1.62 / 72
MONO = "DejaVu Sans Mono"

DEVANAGARI_FONT = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSansDevanagari.ttf"
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

NO_SPACE_BEFORE = set(",.;:!?)]}’")
NO_SPACE_AFTER = set("([{“")


@lru_cache(maxsize=1)
def _devanagari_font():
    import uharfbuzz as hb
    from fontTools.ttLib import TTFont

    tt = TTFont(DEVANAGARI_FONT)
    hb_font = hb.Font(hb.Face(DEVANAGARI_FONT.read_bytes()))
    hb_font.scale = (tt["head"].unitsPerEm,) * 2
    return tt, tt.getGlyphSet(), hb_font


def _glyph_path(glyph_set, glyph_name: str) -> MplPath:
    from fontTools.pens.basePen import BasePen

    class MatplotlibPen(BasePen):
        """Convert fontTools outlines into a Matplotlib path."""

        def __init__(self, glyphs):
            super().__init__(glyphs)
            self.vertices: list = []
            self.codes: list = []

        def _moveTo(self, point):
            self.vertices.append(point)
            self.codes.append(MplPath.MOVETO)

        def _lineTo(self, point):
            self.vertices.append(point)
            self.codes.append(MplPath.LINETO)

        def _curveToOne(self, point1, point2, point3):
            self.vertices.extend((point1, point2, point3))
            self.codes.extend((MplPath.CURVE4,) * 3)

        def _qCurveToOne(self, point1, point2):
            self.vertices.extend((point1, point2))
            self.codes.extend((MplPath.CURVE3,) * 2)

        def _closePath(self):
            self.vertices.append((0, 0))
            self.codes.append(MplPath.CLOSEPOLY)

    pen = MatplotlibPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    return MplPath(pen.vertices, pen.codes)


def draw_devanagari(ax: plt.Axes, text: str, x: float, line_center: float, color: str) -> None:
    """Shape Devanagari with HarfBuzz and draw its glyph outlines as vectors."""
    import uharfbuzz as hb

    tt, glyph_set, hb_font = _devanagari_font()
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    hb.shape(hb_font, buffer)
    scale = BODY_PT / 72 / tt["head"].unitsPerEm
    cursor = 0
    for info, position in zip(buffer.glyph_infos, buffer.glyph_positions):
        path = _glyph_path(glyph_set, tt.getGlyphName(info.codepoint))
        transform = (
            Affine2D()
            .scale(scale, -scale)
            .translate(
                x + (cursor + position.x_offset) * scale,
                line_center + 0.036 - position.y_offset * scale,
            )
            + ax.transData
        )
        ax.add_patch(PathPatch(path, transform=transform, facecolor=color, edgecolor="none", zorder=3))
        cursor += position.x_advance


def wrap_tokens(tokens: list[tuple[str, object]], columns: int) -> list[list[tuple[int, str, object]]]:
    """Greedy-wrap `(token, style)` pairs into lines of `(column, token, style)`."""
    lines, line, col = [], [], 0
    previous = ""
    for token, style in tokens:
        lead = "" if not line or token in NO_SPACE_BEFORE or previous in NO_SPACE_AFTER else " "
        width = len(lead) + len(token)
        if line and col + width > columns:
            lines.append(line)
            line, col, lead = [], 0, ""
        line.append((col + len(lead), token, style))
        col += len(lead) + len(token)
        previous = token
    if line:
        lines.append(line)
    return lines


def mono_text(ax: plt.Axes, x: float, y: float, text: str, color: str = INK) -> None:
    ax.text(x, y, text, family=MONO, fontsize=BODY_PT, color=color, va="center", zorder=3)


def draw_token_lines(ax: plt.Axes, lines, x: float, y: float, changed_color: str, changed_wash: str) -> None:
    """Draw wrapped `(column, token, changed)` lines; changed tokens get a colored wash."""
    for row, line in enumerate(lines):
        baseline = y + (row + 0.5) * LINE_H
        for col, token, changed in line:
            tx = x + col * CHAR_W
            if changed:
                ax.add_patch(Rectangle(
                    (tx, baseline - 0.40 * LINE_H), len(token) * CHAR_W, 0.80 * LINE_H,
                    facecolor=changed_wash, edgecolor="none", zorder=2,
                ))
            color = changed_color if changed else INK
            if DEVANAGARI_RE.search(token):
                draw_devanagari(ax, token, tx, baseline, color)
            else:
                mono_text(ax, tx, baseline, token, color)


def draw_marked_text(ax: plt.Axes, text: str, x: float, center_y: float, marked: str) -> None:
    """Draw pre-wrapped text, vertically centred, with the one `[marked]` token reversed out in blue."""
    lines = text.splitlines()
    top = center_y - len(lines) * LINE_H / 2
    marker = f"[{marked}]"
    for row, line in enumerate(lines):
        baseline = top + (row + 0.5) * LINE_H
        if marker not in line:
            mono_text(ax, x, baseline, line)
            continue
        prefix, suffix = line.split(marker, 1)
        token_x = x + len(prefix) * CHAR_W
        mono_text(ax, x, baseline, prefix)
        ax.add_patch(Rectangle(
            (token_x, baseline - 0.40 * LINE_H), len(marked) * CHAR_W, 0.80 * LINE_H,
            facecolor=BLUE, edgecolor="none", zorder=2,
        ))
        mono_text(ax, token_x, baseline, marked, "white")
        mono_text(ax, token_x + len(marked) * CHAR_W, baseline, suffix)


def add_box(ax: plt.Axes, x: float, y: float, width: float, height: float, accent: str) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0,rounding_size=0.035",
        facecolor=BOX_FILL, edgecolor=LINE, linewidth=0.8, zorder=1,
    ))
    ax.add_patch(Rectangle((x, y), 0.03, height, color=accent, linewidth=0, zorder=2))


def caption(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """Small uppercase label above a box (INSTRUCTION, PASSAGE, ...)."""
    ax.text(x, y, text, fontsize=5.4, family=MONO, color=MUTED)


def heading(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(x, y, text, fontsize=8.6, fontweight="bold", color=INK)
