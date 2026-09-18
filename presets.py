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
