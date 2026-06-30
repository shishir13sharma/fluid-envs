"""Minimal PIL drawing helpers used by the Predator-Prey renderer.

These are intentionally tiny and dependency-light (Pillow + numpy only) so the
environment can be rendered without pulling in any project-internal utilities.
All cell positions are given as ``(row, col)`` and the canvas is laid out with
``x = col * cell_size`` (horizontal) and ``y = row * cell_size`` (vertical).
"""

from typing import Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

Color = Union[str, Tuple[int, int, int]]


def draw_grid(rows: int, cols: int, cell_size: int = 35, fill: Color = "white",
              line_color: Color = "black") -> Image.Image:
    """Create a blank ``rows x cols`` grid image with cell borders drawn."""
    width, height = cols * cell_size, rows * cell_size
    img = Image.new("RGB", (width, height), color=fill)
    draw = ImageDraw.Draw(img)
    for r in range(rows + 1):
        draw.line([(0, r * cell_size), (width, r * cell_size)], fill=line_color)
    for c in range(cols + 1):
        draw.line([(c * cell_size, 0), (c * cell_size, height)], fill=line_color)
    return img


def fill_cell(img: Image.Image, pos: Sequence[int], cell_size: int = 35,
              fill: Color = "black", margin: float = 0.0) -> None:
    """Fill the cell at ``pos = (row, col)`` with a solid colour."""
    row, col = int(pos[0]), int(pos[1])
    m = int(margin * cell_size)
    x0, y0 = col * cell_size + m, row * cell_size + m
    x1, y1 = (col + 1) * cell_size - m, (row + 1) * cell_size - m
    ImageDraw.Draw(img).rectangle([(x0, y0), (x1, y1)], fill=fill)


def draw_circle(img: Image.Image, pos: Sequence[int], cell_size: int = 35,
                fill: Color = "red", radius_frac: float = 0.3) -> None:
    """Draw a filled circle centred in the cell at ``pos = (row, col)``."""
    row, col = int(pos[0]), int(pos[1])
    cx = col * cell_size + cell_size // 2
    cy = row * cell_size + cell_size // 2
    rad = int(radius_frac * cell_size)
    ImageDraw.Draw(img).ellipse(
        [(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=fill)


def write_cell_text(img: Image.Image, text: str, pos: Sequence[int],
                    cell_size: int = 35, fill: Color = "white",
                    margin: float = 0.0) -> None:
    """Write ``text`` inside the cell at ``pos = (row, col)``."""
    row, col = int(pos[0]), int(pos[1])
    x = col * cell_size + int(margin * cell_size)
    y = row * cell_size + int(margin * cell_size)
    ImageDraw.Draw(img).text((x, y), text, fill=fill, font=ImageFont.load_default())
