"""
renderer.py — Text-to-Image renderer for ITIEC synthetic data generation.

Renders Indonesian text strings into realistic-looking images with
varied fonts, sizes, colors, backgrounds, and layouts.
"""

import os
import random
import textwrap
from PIL import Image, ImageDraw, ImageFont

# ── Font catalogue ────────────────────────────────────────────────────────────
# Covers sans-serif, serif, mono, and display — ordered by visual diversity.
FONT_PATHS = [
    # Sans-serif (most common in social media / digital text)
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Light.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Light.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # Serif (formal documents, print)
    "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    # Mono (code, receipts, formal typed docs)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
]

# Only keep fonts that actually exist on the current system
FONT_PATHS = [f for f in FONT_PATHS if os.path.exists(f)]

# ── Colour palettes ───────────────────────────────────────────────────────────
# Each palette: (background_rgb, text_rgb)
# Mimics real-world text image types: screenshots, photos, signs, handwritten notes.
COLOUR_PALETTES = [
    # Screenshot / white background (most common)
    ((255, 255, 255), (0, 0, 0)),
    ((255, 255, 255), (30, 30, 30)),
    ((255, 255, 255), (20, 20, 100)),       # dark blue text
    ((255, 255, 255), (100, 0, 0)),         # dark red text
    # Off-white / cream (paper, note)
    ((253, 245, 230), (30, 20, 10)),
    ((255, 253, 240), (40, 30, 10)),
    ((240, 240, 235), (20, 20, 20)),
    # Light grey background
    ((240, 240, 240), (30, 30, 30)),
    ((220, 220, 220), (0, 0, 0)),
    # Light yellow (sticky note)
    ((255, 255, 153), (30, 30, 0)),
    ((255, 250, 130), (50, 30, 0)),
    # WhatsApp / chat bubble colours
    ((220, 248, 198), (0, 0, 0)),           # sent (green bubble)
    ((255, 255, 255), (0, 0, 0)),           # received (white bubble)
    # Dark background (OLED screenshot, dark mode)
    ((30, 30, 30), (220, 220, 220)),
    ((18, 18, 18), (240, 240, 240)),
    ((40, 40, 60), (210, 210, 255)),
    # Instagram story gradient-like tints
    ((255, 240, 245), (60, 0, 40)),
    ((240, 248, 255), (0, 30, 80)),
]

# ── Canvas size presets ───────────────────────────────────────────────────────
# (width, height) — loosely matching real device screenshots / photos
CANVAS_PRESETS = [
    (640, 120),   # single line, wide
    (480, 160),
    (400, 200),
    (600, 200),
    (640, 240),
    (480, 300),
    (640, 360),
    (400, 300),
    (320, 160),
]


def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font, falling back to PIL default on failure."""
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def render_text_image(
    text: str,
    font_path: str | None = None,
    font_size: int | None = None,
    canvas_size: tuple[int, int] | None = None,
    bg_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] | None = None,
    padding: int | None = None,
    max_width_chars: int = 60,
    seed: int | None = None,
) -> Image.Image:
    """
    Render *text* onto a PIL Image and return it.

    Parameters
    ----------
    text : str
        The text string to render (may be wrapped automatically).
    font_path : str, optional
        Path to .ttf file. Chosen randomly if None.
    font_size : int, optional
        Font size in pixels. Randomly chosen in [16, 48] if None.
    canvas_size : (w, h), optional
        Canvas dimensions. Randomly chosen if None.
    bg_color : (R, G, B), optional
        Background colour. Randomly chosen if None.
    text_color : (R, G, B), optional
        Text colour. Randomly chosen if None.
    padding : int, optional
        Pixel padding around text. Randomly chosen in [10, 30] if None.
    max_width_chars : int
        Approximate max chars per line before wrapping (default 60).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PIL.Image.Image
    """
    rng = random.Random(seed)

    # ── Resolve randomisable parameters ──────────────────────────────────────
    if font_path is None:
        font_path = rng.choice(FONT_PATHS) if FONT_PATHS else None
    if font_size is None:
        font_size = rng.randint(16, 48)
    if canvas_size is None:
        canvas_size = rng.choice(CANVAS_PRESETS)
    if bg_color is None or text_color is None:
        palette = rng.choice(COLOUR_PALETTES)
        bg_color = bg_color or palette[0]
        text_color = text_color or palette[1]
    if padding is None:
        padding = rng.randint(10, 30)

    # ── Wrap text ─────────────────────────────────────────────────────────────
    wrapped = textwrap.fill(text, width=max_width_chars)

    # ── Load font ─────────────────────────────────────────────────────────────
    font = _load_font(font_path, font_size) if font_path else ImageFont.load_default()

    # ── Measure wrapped text to fit canvas ───────────────────────────────────
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    try:
        bbox = dummy_draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow fallback
        text_w, text_h = dummy_draw.multiline_textsize(wrapped, font=font, spacing=6)

    # Resize canvas to fit text + padding if needed
    min_w = text_w + 2 * padding
    min_h = text_h + 2 * padding
    w = max(canvas_size[0], min_w)
    h = max(canvas_size[1], min_h)

    # ── Draw ──────────────────────────────────────────────────────────────────
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Randomly position text (top-left, centred, or slight offset)
    placement = rng.choice(["topleft", "center", "offset"])
    if placement == "center":
        x = (w - text_w) // 2
        y = (h - text_h) // 2
    elif placement == "topleft":
        x = padding
        y = padding
    else:
        x = rng.randint(padding, max(padding, w - text_w - padding))
        y = rng.randint(padding, max(padding, h - text_h - padding))

    try:
        draw.multiline_text((x, y), wrapped, font=font, fill=text_color, spacing=6)
    except Exception:
        draw.text((x, y), text, font=font, fill=text_color)

    return img


def batch_render(
    texts: list[str],
    output_dir: str,
    base_filename: str = "img",
    seed_offset: int = 0,
    **kwargs,
) -> list[str]:
    """
    Render a list of texts to images and save them to *output_dir*.

    Returns a list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, text in enumerate(texts):
        img = render_text_image(text, seed=seed_offset + i, **kwargs)
        filename = f"{base_filename}_{i:06d}.png"
        path = os.path.join(output_dir, filename)
        img.save(path, format="PNG")
        paths.append(path)
    return paths


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "Ketika menunjuk titik dan diatas tanah milik negear tidak dimanfaatkan.",
        "saya mw pergi ke rmh km skrng, bisa gk?",
        "gue bgt suka sama lagu itu, bngt2 keren banget deh",
        "tmn gue blg klo dia udh gabisa dtg ke acara hr ini",
    ]
    out_dir = "/tmp/itiec_renderer_test"
    saved = batch_render(sample_texts, out_dir, base_filename="test")
    print(f"Saved {len(saved)} test images to {out_dir}")
    for p in saved:
        print(f"  {p}")
