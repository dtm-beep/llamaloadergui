# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 DTM-beep — https://github.com/DTM-beep
"""Markup balance test for templates/gui.html — no dependencies.

Why this exists: commit 0aef54a added the "Remote / SSH" section by *replacing*
the CPU section's opening `<div class="section">` + header lines instead of
inserting before them. One missing opening tag puts every later `</div>` one
level too high, so .settings-panel / .content / .main close early and the
pinned launch bar (whose one-line command preview is a long unbroken
"sentence") becomes a flex child of <body> and squishes every other box.
FastAPI renders the template happily either way — only the browser notices,
which is why this check exists: catch it in CI, not on screen.

    .venv/bin/python tests/test_markup.py
"""
import os
import sys
from html.parser import HTMLParser

VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
    # SVG children used inline
    "path", "circle", "rect", "line", "polyline", "polygon", "use", "stop",
}

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "templates", "gui.html")

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if not cond:
        FAIL += 1
    print(("  ok   " if cond else " FAIL  ") + name + (f"  <- {detail}" if detail and not cond else ""))


class Checker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []      # (tag, line, col)
        self.errors = []     # human-readable strings
        self.classes = {}    # class attribute -> count of elements

    def handle_starttag(self, tag, attrs):
        if tag in ("div", "section", "span", "select", "button", "label", "textarea"):
            cls = dict(attrs).get("class", "")
            for c in cls.split():
                self.classes[c] = self.classes.get(c, 0) + 1
        if tag in VOID:
            return
        self.stack.append((tag, *self.getpos()))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            line, col = self.getpos()
            self.errors.append(f"</{tag}> at line {line} closes nothing")
            return
        if self.stack[-1][0] != tag:
            line, col = self.getpos()
            open_tag, ol, oc = self.stack[-1]
            self.errors.append(
                f"</{tag}> at line {line} closes {open_tag} opened at line {ol} "
                f"(implicit close — nesting is off by one)")
            # unwind so the rest of the report stays meaningful
            names = [t for t, *_ in self.stack]
            if tag in names:
                while self.stack and self.stack[-1][0] != tag:
                    self.stack.pop()
            if self.stack:
                self.stack.pop()
            return
        self.stack.pop()


def main():
    src = open(TEMPLATE, encoding="utf-8").read()
    c = Checker()
    c.feed(src)

    check("template parses", bool(src.strip()))
    check("every tag closed", not c.stack,
          ", ".join(f"<{t}> opened line {ln}" for t, ln, _ in c.stack[:8]))
    check("no stray closers / off-by-one nesting", not c.errors, "; ".join(c.errors[:6]))

    # Layout-critical structure (the squish bug broke exactly these).
    check("launch bar exists", c.classes.get("launch-bar") == 1)
    check("settings panel exists", c.classes.get("settings-panel") == 1)
    check("logs panel exists", c.classes.get("logs-panel") == 1)
    sections = c.classes.get("section", 0)
    headers = c.classes.get("section-header", 0)
    bodies = c.classes.get("section-body", 0)
    check("every section has a header", headers == sections,
          f"{sections} .section vs {headers} .section-header")
    check("every section has a body", bodies == sections,
          f"{sections} .section vs {bodies} .section-body")

    # Every interactive control the backend reads must be unique — a dropped
    # wrapper historically duplicated or orphaned fields.
    ids = [line for line in _ids(src)]
    dupes = {i for i in ids if ids.count(i) > 1}
    check("no duplicate element ids", not dupes, ", ".join(sorted(dupes)[:8]))

    print("\n" + ("ALL PASS" if FAIL == 0 else f"{FAIL} FAILURE(S)"))
    return 1 if FAIL else 0


def _ids(src):
    import re
    for m in re.finditer(r'\bid="([^"]+)"', src):
        yield m.group(1)


sys.exit(main())
