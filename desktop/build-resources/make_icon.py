"""Generates the Windows .ico for the packaged app.

Dev-time only tool (needs Pillow: pip install pillow - not a runtime
dependency of MusicTagger itself). Run once to produce icon.ico, then check
that file in:

    python desktop/build-resources/make_icon.py

Matches the accent purple and the note-glyph already used in the web UI's
header mark.
"""

from pathlib import Path

from PIL import Image, ImageDraw

ACCENT = (91, 91, 214)         # --accent from styles.css
WHITE = (255, 255, 255)

SIZES = (16, 24, 32, 48, 64, 128, 256)


def draw_note(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded-square background, matching the app's radius-heavy visual language.
    radius = size * 0.22
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=ACCENT)

    # A simple musical note glyph: a stem plus two note-heads, scaled to the
    # canvas. Proportions tuned by eye against the brand mark in the header.
    s = size
    stem_w = max(1, round(s * 0.075))
    stem_x = round(s * 0.62)
    stem_top = round(s * 0.16)
    stem_bottom = round(s * 0.66)
    draw.rectangle([stem_x, stem_top, stem_x + stem_w, stem_bottom], fill=WHITE)

    # Flag at the top of the stem.
    draw.polygon(
        [
            (stem_x + stem_w, stem_top),
            (stem_x + stem_w + s * 0.20, stem_top + s * 0.10),
            (stem_x + stem_w + s * 0.16, stem_top + s * 0.22),
            (stem_x + stem_w, stem_top + s * 0.16),
        ],
        fill=WHITE,
    )

    head_r = s * 0.135
    draw.ellipse(
        [stem_x - head_r * 1.6, stem_bottom - head_r, stem_x + head_r * 0.4, stem_bottom + head_r],
        fill=WHITE,
    )
    second_head_y = stem_bottom - s * 0.10
    second_head_x = stem_x - s * 0.30
    draw.ellipse(
        [second_head_x - head_r * 1.6, second_head_y - head_r,
         second_head_x + head_r * 0.4, second_head_y + head_r],
        fill=WHITE,
    )
    return img


def main() -> None:
    out_dir = Path(__file__).parent
    images = [draw_note(size) for size in SIZES]
    images[-1].save(out_dir / "icon.ico", format="ICO",
                    sizes=[(s, s) for s in SIZES])
    print(f"wrote {out_dir / 'icon.ico'}")


if __name__ == "__main__":
    main()
