"""Marketplace export presets.

`generative` is False for any slot where the platform expects a photograph of
the real product. The pipeline still renders those, but flags them on export so
nobody uploads a generated hero image into a slot that forbids one.
"""

PRESETS = {
    "amazon_main": {
        "label": "Amazon main image",
        "size": (2000, 2000),
        "format": "jpeg",
        "background": "pure white (RGB 255,255,255)",
        "fill": 0.85,
        "generative": False,
        "note": "Amazon requires a real photograph of the actual product on pure white. "
                "Use this slot for a colour-corrected photo, not a generated scene.",
    },
    "amazon_secondary": {
        "label": "Amazon secondary / lifestyle",
        "size": (2000, 2000),
        "format": "jpeg",
        "background": "any",
        "fill": 0.75,
        "generative": True,
        "note": "Lifestyle and infographic slots allow rendered and composed imagery.",
    },
    "etsy": {
        "label": "Etsy listing",
        "size": (2000, 1600),
        "format": "jpeg",
        "background": "any",
        "fill": 0.72,
        "generative": True,
        "note": "Etsy recommends 2000px on the shortest side; 5:4 crops well in the grid.",
    },
    "shopify": {
        "label": "Shopify product",
        "size": (2048, 2048),
        "format": "webp",
        "background": "any",
        "fill": 0.80,
        "generative": True,
        "note": "Square, zoomable. WebP keeps the file small without visible loss.",
    },
    "ebay": {
        "label": "eBay gallery",
        "size": (1600, 1600),
        "format": "jpeg",
        "background": "any",
        "fill": 0.88,
        "generative": True,
        "note": "1600px triggers eBay's zoom view.",
    },
    "hero_4k": {
        "label": "Site hero (4K)",
        "size": (3840, 2160),
        "format": "jpeg",
        "background": "any",
        "fill": 0.55,
        "generative": True,
        "note": "For your own storefront banner. Wide crop, product off-centre.",
    },
    "transparent_png": {
        "label": "Transparent cutout",
        "size": (2048, 2048),
        "format": "png",
        "background": "transparent",
        "fill": 0.90,
        "generative": True,
        "note": "Alpha-channel cutout for compositing into your own layouts.",
    },
}


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Options: {', '.join(PRESETS)}")
    return PRESETS[name]


# Kamal's standard shot list. These are the four jobs that recur, written as
# instructions the pipeline can act on directly. They exist here rather than in
# someone's notes file so the wording stays identical run to run - which is the
# only way two images of different products look like they belong in the same
# catalogue.
SHOT_TEMPLATES = [
    {
        "id": "apply_to_bed",
        "label": "Apply design to a dressed bed",
        "needs_upload": True,
        "text": ("Apply the duvet design in the attached image exactly as it is to a "
                 "dressed bed in a UK-style bedroom. Match the colour, motif, pattern "
                 "scale, repeat and the reversible side precisely to the reference. "
                 "Every detail of the duvet must match the reference image. "
                 "Full bed front view. 1:1 aspect ratio."),
    },
    {
        "id": "uk_roomset",
        "label": "UK roomset from a reference",
        "needs_upload": True,
        "text": ("Make a UK-style bedding roomset using the attached duvet as the "
                 "reference. Use the exact same colour, design, pattern and repeat, "
                 "and the exact reversible design and colour. One angle: full bed "
                 "front view. Keep the bed frame colour consistent. 1:1 aspect ratio."),
    },
    {
        "id": "white_mockup",
        "label": "White background mock-up",
        "needs_upload": True,
        "text": ("Present this duvet cover set on a plain white background as a "
                 "product mock-up. Everything else stays unchanged, including the "
                 "duvet design, colours, motifs and repeat. 1:1 aspect ratio."),
    },
    {
        "id": "folded",
        "label": "Folded set",
        "needs_upload": True,
        "text": ("Show this duvet cover set folded, as a folded duvet cover set "
                 "product image. Everything else stays unchanged, including the "
                 "design, colours, motifs and repeat. 1:1 aspect ratio."),
    },
]
