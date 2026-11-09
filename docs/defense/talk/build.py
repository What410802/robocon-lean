#!/usr/bin/env python3
"""Subset fonts for talk/deck.html and render per-slide previews.

Run with the repo's train env (has fontTools) and any env with Playwright:
  /home/bis/robocon-lean/.pixi/envs/train/bin/python docs/defense/talk/build.py
The deck itself is a hand-written single HTML file; nothing is generated except
fonts/ and previews/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "deck.html"
SCRIPT = HERE / "script.md"
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"


def subset_fonts() -> None:
    from fontTools import subset
    from fontTools.ttLib import TTFont

    (HERE / "fonts").mkdir(exist_ok=True)
    text = DECK.read_text()
    if SCRIPT.exists():
        text += SCRIPT.read_text()
    text += "0123456789 ←→·—…°×−±≤≥/%()[]{}"
    for weight in ("Regular", "Bold"):
        font = TTFont(NOTO.format(weight), fontNumber=2)
        opts = subset.Options()
        opts.flavor = "woff2"
        opts.recalc_timestamp = False
        sub = subset.Subsetter(options=opts)
        sub.populate(text=text)
        sub.subset(font)
        for rec in font["name"].names:
            if rec.nameID in (1, 4, 6, 16):
                name = "BlackWTalk" + ("-Bold" if weight == "Bold" else "")
                rec.string = name.encode(rec.getEncoding())
        font.flavor = "woff2"
        font.save(HERE / "fonts" / f"blackw-{weight.lower()}.woff2")
    print("fonts subset from deck.html + script.md")


def render_previews() -> None:
    from playwright.sync_api import sync_playwright

    out = HERE / "previews"
    out.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 934}, device_scale_factor=1)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(DECK.as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(300)
        count = page.evaluate("slides.length")
        overflow = []
        for i in range(count):
            page.evaluate(f"go({i})")
            page.wait_for_timeout(60)
            overflow += page.evaluate(
                """() => [...document.querySelectorAll('.slide.on .stat, .slide.on .take, .slide.on h1, .slide.on .node, .slide.on .annot, .slide.on table, .slide.on .plate')]
                     .filter(e => {
                       // horizontal overflow is always a real bug
                       if (e.scrollWidth > e.clientWidth + 2) return true;
                       // vertical only for elements whose box is fixed by CSS/layout
                       const fixed = e.classList.contains('node') || e.classList.contains('plate') ||
                                     e.tagName === 'TABLE' ||
                                     getComputedStyle(e).position === 'absolute' && e.style.height;
                       return fixed && e.scrollHeight > e.clientHeight + 2;
                     })
                     .map(e => e.className + ' :: ' + (e.textContent || '').slice(0, 60))"""
            )
            # hide video chrome for stills
            page.evaluate("document.querySelectorAll('video').forEach(v=>v.pause())")
            page.screenshot(path=str(out / f"{i + 1:02d}.png"), clip={"x": 0, "y": 0, "width": 1600, "height": 900})
        browser.close()
    print(f"previews: {count} slides")
    if overflow:
        print("OVERFLOW:")
        for line in overflow:
            print("  ", line)
        sys.exit(1)
    if errors:
        print("JS ERRORS:", errors)
        sys.exit(1)
    print("no overflow, no JS errors")


if __name__ == "__main__":
    subset_fonts()
    render_previews()
