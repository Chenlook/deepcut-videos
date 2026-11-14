"""Deep Cut video engine.

Usage:  python3 engine.py story.json out.mp4        (full render with sound)
        python3 engine.py story.json out_dir --stills   (one preview frame per scene)

A story is a JSON file: see stories/example_illit.json and RUNBOOK.md.
The look (palette, type, chrome, sound kit) is fixed; layout details, colours of the
glow, background motif, transitions, entrance order and the music (key, progression,
groove, melody) are drawn from the story's seed, so no two videos come out the same.
"""
import sys, os, math, json, random, subprocess, tempfile, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "fonts") + os.sep
UNB, CH_B, CH_S, MONO, HAN = FD + "Unbounded.ttf", FD + "ChakraPetch-Bold.ttf", FD + "ChakraPetch-SemiBold.ttf", FD + "SpaceMono-Bold.ttf", FD + "BlackHanSans-Regular.ttf"
W, H, FPS = 1080, 1920, 30
INK, WHITE, GREY = (6, 7, 9), (246, 246, 242), (150, 154, 162)
NEO, CYAN, VIOLET, PINK = (190, 255, 50), (60, 215, 255), (160, 120, 255), (255, 70, 150)
LEFT, MAXW = 70, 880
BEAT = 0.4
DEFAULT_DUR = dict(hook=2.4, ranking=3.2, number=2.4, grid=3.2, items=2.4, flow=3.2, streak=2.4, versus=3.2, timeline=3.6, sides=4.0, statement=2.8, outro=2.4)
_fc, _cache = {}, {}


def F(path, size, wght=None):
    k = (path, int(size), wght)
    if k not in _fc:
        f = ImageFont.truetype(path, int(size))
        if wght:
            f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]


def HF(size):
    return F(UNB, size, 850)


def fit(texts, maxw, maxsize, font=None, minsize=24):
    font = font or HF
    s = maxsize
    while s > minsize and max(font(s).getlength(x) for x in texts) > maxw:
        s -= 2
    return s


def ease(p):
    p = max(0.0, min(1.0, p))
    return 1 - (1 - p) ** 3


def backout(p, c=2.0):
    p = max(0.0, min(1.0, p))
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


def wrap(text, f, maxw):
    lines, cur = [], ""
    for w in text.split(" "):
        t = (cur + " " + w).strip()
        if f.getlength(t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def put(canvas, s, cx, cy, scale=1.0, alpha=1.0):
    if scale != 1.0:
        s = s.resize((max(2, int(s.width * scale)), max(2, int(s.height * scale))), Image.BILINEAR)
    if alpha < 0.999:
        s = s.copy()
        s.putalpha(s.getchannel("A").point(lambda v: int(v * alpha)))
    x, y = int(cx - s.width / 2), int(cy - s.height / 2)
    x0, y0 = max(0, -x), max(0, -y)
    x1, y1 = min(s.width, W - x), min(s.height, H - y)
    if x1 > x0 and y1 > y0:
        canvas.alpha_composite(s.crop((x0, y0, x1, y1)), (max(0, x), max(0, y)))


def chrome_grad(w, h):
    t = np.linspace(0, 1, max(1, h), dtype=np.float32)
    v = np.where(t < 0.5, 0.45 + 1.1 * t, 0.12 + 0.9 * (t - 0.5))
    a = np.clip(v[:, None, None] * np.array([240, 244, 250], np.float32), 0, 255)
    return Image.fromarray(np.broadcast_to(a, (max(1, h), w, 3)).astype(np.uint8), "RGB")


def star_mask(size):
    S = 4
    n = size * S
    m = Image.new("L", (n, n), 0)
    c = n / 2
    pts = [(c + c * math.copysign(abs(math.cos(a)) ** 3, math.cos(a)), c + c * math.copysign(abs(math.sin(a)) ** 3, math.sin(a))) for a in np.linspace(0, 2 * math.pi, 240, endpoint=False)]
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m.resize((size, size), Image.LANCZOS)


def chrome_text(txt, size, shadow=NEO, sw=10):
    key = ("ct", txt, size, shadow, sw)
    if key in _cache:
        return _cache[key]
    f = F(UNB, size, 900)
    w, h = int(f.getlength(txt)) + 6 * sw + 30, int(size * 1.25) + 6 * sw
    inner, outer = Image.new("L", (w, h), 0), Image.new("L", (w, h), 0)
    ImageDraw.Draw(inner).text((3 * sw, 3 * sw + size), txt, font=f, fill=255, anchor="ls")
    ImageDraw.Draw(outer).text((3 * sw, 3 * sw + size), txt, font=f, fill=255, anchor="ls", stroke_width=sw, stroke_fill=255)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("L", (w, h), 0)
    sh.paste(outer, (int(sw * 1.4), int(sw * 1.6)))
    img.paste(shadow + (255,), None, sh)
    img.paste(INK + (255,), None, outer)
    bb = inner.getbbox() or (0, 0, w, h)
    g = Image.new("RGB", (w, h), (0, 0, 0))
    g.paste(chrome_grad(w, bb[3] - bb[1]), (0, bb[1]))
    img.paste(g, None, inner)
    _cache[key] = img.crop(img.getbbox())
    return _cache[key]


def token_sprite(txt, maxw=880, maxsize=470):
    size = maxsize
    while size > 80 and F(UNB, size, 900).getlength(txt) > maxw - 80:
        size -= 10
    return chrome_text(txt, size)


def rrect(w, h, r, fill, border=None, bw=4):
    S = 2
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=r * S, fill=fill + (255,), outline=(border + (255,)) if border else None, width=bw * S)
    return img.resize((w, h), Image.LANCZOS)


_stars = None


def stars():
    global _stars
    if _stars is None:
        m, g = star_mask(84), chrome_grad(84, 84)
        _stars = []
        for a in range(0, 90, 3):
            im = Image.new("RGBA", (84, 84), (0, 0, 0, 0))
            im.paste(g, (0, 0), m.rotate(a, resample=Image.BICUBIC))
            _stars.append(im)
    return _stars


# ---------------- background ----------------
gy, gx = np.mgrid[0:96, 0:54].astype(np.float32)
gx /= 54
gy /= 96
CS = 440
cy_, cx_ = np.mgrid[0:CS, 0:CS].astype(np.float32)
cxn, cyn = (cx_ - CS / 2) / (CS / 2), (cy_ - CS / 2) / (CS / 2)
C_TH, C_R = np.arctan2(cyn, cxn), np.hypot(cxn, cyn)
yy_ = np.arange(H, dtype=np.float32)[:, None]
xx_ = np.arange(W, dtype=np.float32)[None, :]
VIG = (1 - 0.26 * (((xx_ - W / 2) / (W / 2)) ** 2 * 0.5 + ((yy_ - H / 2) / (H / 2)) ** 2 * 0.5) ** 1.3).astype(np.float32)[..., None]
GRAIN = np.random.default_rng(9).normal(0, 2.5, (H, W, 1)).astype(np.float32)


def chrome_blob(ph, tint, shape):
    a, b, c = shape
    p1, p2, p3, p4, p5 = ph
    R = 0.74 + 0.085 * np.sin(a * C_TH + p1) + 0.055 * np.sin(b * C_TH - p2) + 0.04 * np.sin(c * C_TH + p3)
    q = np.clip(C_R / R, 0, 1)
    z = np.sqrt(1 - q * q)
    nx, ny = cxn / R, cyn / R
    yy = ny + 0.16 * np.sin(3.2 * nx + p4) + 0.08 * np.sin(7 * nx - p5)
    v = np.where(yy < 0, np.clip(0.30 + 0.85 * np.clip(yy + 1, 0, None) ** 1.6, 0, 1.05), np.clip(0.06 + 0.55 * np.clip(yy, 0, None) ** 0.8, 0, 1))
    v = v + 0.5 * np.exp(-((nx + 0.35) ** 2 + (ny + 0.45) ** 2) / 0.02)
    col = v[..., None] * np.array([235, 240, 248], np.float32) + ((1 - z) ** 2.2)[..., None] * np.array(tint, np.float32) * 0.9
    alpha = np.clip((R - C_R) * CS / 2 / 1.5, 0, 1)
    return Image.fromarray(np.nan_to_num(np.dstack([np.clip(col, 0, 255), alpha * 255])).astype(np.uint8), "RGBA")


def dark_bg(T, col, V, blob=False):
    L = V["total"]

    def w(f):
        return 2 * math.pi * max(1, round(f * L / (2 * math.pi))) / L * T

    acc = np.array(col, np.float32)
    img = np.zeros((96, 54, 3), np.float32) + np.array(INK, np.float32)
    gxo, gyo = V["glow"]
    for cx, cy, r, k in ((gxo + 0.12 * math.sin(w(1.1)), gyo + 0.06 * math.cos(w(0.9)), 0.46, 0.30), (1 - gxo + 0.15 * math.cos(w(0.8)), 0.95, 0.42, 0.22)):
        dd = ((gx - cx) ** 2 + ((gy - cy) * 1.78) ** 2) / (r * r)
        img += np.exp(-dd)[..., None] * acc * k
    canvas = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB").resize((W, H), Image.BICUBIC).convert("RGBA")
    if blob:
        b = chrome_blob((w(1.5), w(1.1), w(2.1), w(1.3), w(0.8)), col, V["blob_shape"]).resize((760, 760), Image.BICUBIC)
        if V["blob_side"] == "right":
            canvas.alpha_composite(b.crop((0, 0, 330, 760)), (750, 980 + int(20 * math.sin(w(1.0)))))
        else:
            canvas.alpha_composite(b.crop((430, 0, 760, 760)), (0, 1500 + int(20 * math.sin(w(1.0)))))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    hz = 1540
    if V["motif"] == "grid":
        for k in range(-9, 10):
            d.line([(540 + k * 46, hz), (540 + k * 520, H)], fill=col + (70,), width=2)
        ph = (T * max(1, round(1.4 * L)) / L) % 1
        for i in range(9):
            fch = ((i + ph) / 9) ** 2.3
            d.line([(0, hz + (H - hz) * fch), (W, hz + (H - hz) * fch)], fill=col + (int(30 + 110 * fch),), width=2)
        d.line([(0, hz), (W, hz)], fill=col + (150,), width=3)
    elif V["motif"] == "rings":
        for i in range(7):
            r = ((i + (T * max(1, round(0.5 * L)) / L) % 1) / 7) ** 1.4 * 900
            d.ellipse([540 - r, 1900 - r * 0.55, 540 + r, 1900 + r * 0.55], outline=col + (int(150 * (1 - r / 900)) + 20,), width=3)
    else:  # bars
        for i in range(14):
            x = (i * 97 + T * (W + 80) * max(1, round(60 * L / (W + 80))) / L) % (W + 80) - 40
            hgt = 120 + 180 * (0.5 + 0.5 * math.sin(w(2.0) + i * 1.7))
            d.rectangle([x, H - hgt, x + 26, H], fill=col + (70,))
        d.line([(0, hz + 100), (W, hz + 100)], fill=col + (90,), width=2)
    for x, y, sp, sz in V["parts"]:
        ypos = (y - T * 1500 * max(1, round(sp * L / 1500)) / L) % 1500
        d.rectangle([x, ypos, x + sz, ypos + sz], fill=col + (120,))
    canvas.alpha_composite(ov)
    return canvas


def pulse(T):
    return math.exp(-(T % (2 * BEAT)) * 6)


def caption(canvas, text, t, t0=0.6, dark=True, y=1270):
    if not text or t < t0:
        return
    p = ease((t - t0) / 0.28)
    parts = text.split("*")
    size = 50
    while size > 30 and sum(F(CH_B, size).getlength(x) for x in parts) > MAXW - 72:
        size -= 2
    f = F(CH_B, size)
    tw = sum(f.getlength(x) for x in parts)
    bar = rrect(int(tw) + 72, 96, 22, (22, 24, 30) if dark else INK, border=NEO if dark else None, bw=3)
    d = ImageDraw.Draw(bar)
    x = 36
    for k, seg in enumerate(parts):
        d.text((x, 50), seg, font=f, fill=NEO if k % 2 else WHITE, anchor="lm")
        x += f.getlength(seg)
    put(canvas, bar, LEFT + bar.width / 2, y + 48 + (1 - p) * 50, alpha=p)


def hud(canvas, story, T, total, dark=True):
    d = ImageDraw.Draw(canvas)
    fg, acc = (WHITE, NEO) if dark else (INK, INK)
    d.rectangle([LEFT, 178, LEFT + 30, 208], fill=acc)
    d.text((LEFT + 46, 208), "DEEP CUT", font=F(UNB, 34, 800), fill=fg, anchor="ls")
    d.text((LEFT + 300, 206), "케이팝 뉴스", font=F(HAN, 30), fill=GREY if dark else INK, anchor="ls")
    d.text((950, 208), story.get("section", "K-POP").upper()[:12], font=F(UNB, 34, 800), fill=fg, anchor="rs")
    d.rounded_rectangle([LEFT, 240, 950, 250], radius=5, fill=(58, 60, 66) if dark else (150, 200, 40))
    d.rounded_rectangle([LEFT, 240, LEFT + max(12, (950 - LEFT) * T / total), 250], radius=5, fill=acc)
    src = "SOURCE  " + story.get("source", "")
    sz = 23
    while sz > 15 and F(MONO, sz).getlength(src) > MAXW:
        sz -= 1
    d.text((LEFT + 4, 1462), src, font=F(MONO, sz), fill=GREY if dark else INK, anchor="lm")


# ---------------- scenes: fn(c, t, T, d, X) ; X = per-scene context ----------------
PAL = [NEO, (225, 230, 238), CYAN, VIOLET, PINK, WHITE]


def s_hook(c, t, T, d, X):
    tok = token_sprite(d.get("token", "?"))
    sc = 0.92 * (1 + 0.22 * (1 - ease(t / 0.28)) + 0.03 * pulse(T))
    put(c, tok.rotate(X["tilt"], resample=Image.BICUBIC, expand=True) if X["tilt"] else tok, 520, 800, scale=sc)
    dr = ImageDraw.Draw(c)
    top = d.get("top", [])[:2]
    if top:
        s = fit(top, 900, 86)
        for i, l in enumerate(top):
            dr.text((LEFT, 400 + i * int(s * 1.16)), l, font=HF(s), fill=WHITE, anchor="ls", stroke_width=6, stroke_fill=INK)
    if d.get("bottom"):
        s = fit([d["bottom"]], 900, 100)
        dr.text((LEFT, 1120), d["bottom"], font=HF(s), fill=NEO, anchor="ls", stroke_width=6, stroke_fill=INK)
    if d.get("pill") and t >= 1.25:
        p = backout((t - 1.25) / 0.3)
        fs = fit([d["pill"]], 700, 46, lambda z: F(CH_B, z))
        pill = rrect(760, 100, 50, NEO)
        ImageDraw.Draw(pill).text((380, 52), d["pill"], font=F(CH_B, fs), fill=INK, anchor="mm")
        put(c, pill, LEFT + 380, 1240 + (1 - p) * 80)


def s_ranking(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    dr.text((LEFT, 372), d.get("title", ""), font=F(CH_B, fit([d.get("title", "")], 640, 42, lambda z: F(CH_B, z))), fill=WHITE, anchor="ls")
    dr.text((950, 372), d.get("tag", ""), font=F(CH_B, 42), fill=NEO, anchor="rs")
    rows = d["rows"][:6]
    hl = d.get("highlight", 0)
    small = 118 if len(rows) <= 5 else 100
    y = 420
    for k, row in enumerate(rows):
        rank, name = str(row[0]), row[1]
        sub = row[2] if len(row) > 2 else None
        h = 190 if k == hl else small
        t0 = 0.05 + k * 0.14
        if t >= t0:
            p = ease((t - t0) / 0.3)
            x = LEFT + X["side"] * (1 - p) * 1000
            if k == hl:
                g = int(8 * pulse(T))
                dr.rounded_rectangle([x - g, y - g, x + MAXW + g, y + h + g], radius=30, fill=NEO)
                dr.text((x + 36, y + 122), rank, font=HF(fit([rank], 130, 120)), fill=INK, anchor="ls")
                ns = fit([name], MAXW - 230, 84)
                dr.text((x + 190, y + (92 if sub else 118)), name, font=HF(ns), fill=INK, anchor="ls")
                if sub:
                    dr.text((x + 194, y + 152), sub, font=F(CH_B, fit([sub], MAXW - 230, 44, lambda z: F(CH_B, z))), fill=INK, anchor="ls")
            else:
                dim = 0.45 if t > 1.9 else 1.0
                dr.rounded_rectangle([x, y, x + MAXW, y + h], radius=22, fill=(24, 26, 32), outline=(70, 74, 82), width=2)
                dr.text((x + 40, y + h * 0.69), rank, font=HF(fit([rank], 90, 62)), fill=tuple(int(v * dim) for v in GREY), anchor="ls")
                dr.text((x + 150, y + h * 0.65), name, font=F(CH_B, fit([name], MAXW - 190, 44, lambda z: F(CH_B, z))), fill=tuple(int(v * dim) for v in WHITE), anchor="ls")
        y += h + 16
    caption(c, d.get("caption"), t, 1.0, y=max(1270, min(1330, y + 30)))


def fmt_value(d, v):
    if isinstance(d["value"], (int, float)):
        body = f"{int(round(v)):,}" if float(d["value"]).is_integer() else f"{v:,.1f}"
    else:
        body = str(d["value"])
    return f"{d.get('prefix', '')}{body}{d.get('suffix', '')}"


def s_number(c, t, T, d, X):
    dark = X["mode"] == "dark"
    fg = WHITE if dark else INK
    hi = NEO if dark else INK
    dr = ImageDraw.Draw(c)
    dr.text((LEFT, 430), d.get("label", ""), font=F(CH_B, fit([d.get("label", "")], 760, 62, lambda z: F(CH_B, z))), fill=fg, anchor="ls")
    num = isinstance(d["value"], (int, float))
    final = fmt_value(d, d["value"] if num else 0)
    size = fit([final], MAXW, 230)
    v = d["value"] * ease(t / 0.9) if num and t < 0.9 else (d["value"] if num else 0)
    if num and t < 0.9 and float(d["value"]).is_integer() and d["value"] >= 1000:
        v = round(v / 100.0) * 100
    kw = dict(stroke_width=6, stroke_fill=INK) if dark else {}
    dr.text((LEFT - 6, 700 - int(8 * pulse(T))), fmt_value(d, v), font=HF(size), fill=hi, anchor="ls", **kw)
    if d.get("under"):
        dr.text((LEFT, 830), d["under"], font=HF(fit([d["under"]], MAXW, 96)), fill=fg, anchor="ls")
    dr.rounded_rectangle([LEFT, 910, LEFT + MAXW, 980], radius=35, outline=fg, width=6)
    p = ease(t / 1.1)
    dr.rounded_rectangle([LEFT + 10, 920, LEFT + 10 + (MAXW - 20) * max(0.06, p), 970], radius=25, fill=hi)
    st = stars()
    put(c, st[int(T * 30) % len(st)], 880, 420, scale=1.6)
    caption(c, d.get("caption"), t, 1.0, dark=dark)


def cd_sprite(k, size):
    key = ("cd", k, size)
    if key not in _cache:
        col = PAL[k % len(PAL)]
        s = rrect(size, size, int(size * 0.17), col, border=INK, bw=4)
        dd = ImageDraw.Draw(s)
        a, b = size * 0.22, size * 0.78
        dd.ellipse([a, a, b, b], fill=INK)
        dd.ellipse([size * 0.32, size * 0.32, size * 0.68, size * 0.68], outline=col, width=3)
        dd.ellipse([size * 0.44, size * 0.44, size * 0.56, size * 0.56], fill=col)
        _cache[key] = s
    return _cache[key]


def grid_geom(n):
    cols = 6 if n > 20 else 5 if n > 12 else 4 if n > 6 else 3
    rows = math.ceil(n / cols)
    cell = int(min(126, (846 - (cols - 1) * 18) / cols * 1.0, (560 - (rows - 1) * 18) / rows))
    return cols, rows, cell, min(0.1, 2.2 / n)


def s_grid(c, t, T, d, X):
    n = max(1, min(30, int(d["count"])))
    cols, rows, cell, step = grid_geom(n)
    dr = ImageDraw.Draw(c)
    shown = max(0, min(n, int((t - 0.3) / step) + 1)) if t >= 0.3 else 0
    done = t >= 0.3 + n * step + 0.05
    txt = (f"{n}+" if d.get("plus") else f"{n}") if done else f"{shown:02d}"
    dr.text((LEFT - 4, 520), txt, font=HF(170), fill=NEO, anchor="ls", stroke_width=6, stroke_fill=INK)
    lab = d.get("label", [])[:2]
    if lab:
        ls = fit(lab, 440, 70)
        for i, l in enumerate(lab):
            dr.text((LEFT + 440, 440 + i * 80), l, font=HF(ls), fill=WHITE, anchor="ls")
    gw = cols * cell + (cols - 1) * 18
    x0, y0 = LEFT + (MAXW - gw) // 2, 580
    order = X["order"][:n]
    for i, k in enumerate(order):
        t0 = 0.3 + i * step
        if t < t0:
            break
        p = backout((t - t0) / 0.22, 3.0)
        cx, cy = x0 + (k % cols) * (cell + 18) + cell / 2, y0 + (k // cols) * (cell + 18) + cell / 2
        put(c, cd_sprite(k, cell), cx, cy, scale=max(0.05, p))
    caption(c, d.get("caption"), t, 0.3 + n * step + 0.1, y=1190)


def item_card(kind):
    key = ("item", kind)
    if key in _cache:
        return _cache[key]
    if kind == 0:
        s = rrect(230, 320, 24, NEO, border=WHITE, bw=8)
        dd = ImageDraw.Draw(s)
        dd.ellipse([75, 70, 155, 150], fill=INK)
        dd.rounded_rectangle([50, 165, 180, 290], radius=50, fill=INK)
    elif kind == 1:
        s = rrect(300, 400, 10, WHITE, border=INK, bw=6)
        dd = ImageDraw.Draw(s)
        dd.rectangle([24, 24, 276, 250], fill=VIOLET)
        s.paste(NEO + (255,), (75, 62), star_mask(150))
        for i in range(3):
            dd.rectangle([24, 280 + i * 36, 276 - i * 60, 300 + i * 36], fill=INK)
    else:
        s = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        dd = ImageDraw.Draw(s)
        dd.ellipse([10, 10, 190, 190], fill=PINK, outline=WHITE, width=8)
        s.paste(WHITE + (255,), (50, 50), star_mask(100))
        dd.ellipse([150, 150, 290, 290], fill=CYAN, outline=WHITE, width=8)
        s.paste(INK + (255,), (180, 180), star_mask(80))
    _cache[key] = s
    return s


def s_items(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    dr.text((LEFT, 420), d.get("title", ""), font=HF(fit([d.get("title", "")], MAXW, 92)), fill=WHITE, anchor="ls")
    pos = [(270, 770), (545, 720), (820, 790)]
    for k, lab in enumerate(d.get("labels", [])[:3]):
        t0 = 0.1 + k * 0.22
        if t < t0:
            continue
        p = backout((t - t0) / 0.3, 2.6)
        key = ("itemrot", X["kinds"][k], X["rots"][k])
        if key not in _cache:
            _cache[key] = item_card(X["kinds"][k]).rotate(X["rots"][k], resample=Image.BICUBIC, expand=True)
        put(c, _cache[key], pos[k][0], pos[k][1] + 10 * math.sin(T * 5 + k * 2), scale=max(0.05, p))
        if p > 0.9:
            ImageDraw.Draw(c).text((pos[k][0], 1060), lab, font=F(CH_B, fit([lab], 270, 38, lambda z: F(CH_B, z))), fill=NEO, anchor="ms")
    caption(c, d.get("caption"), t, 1.0)


def s_flow(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    dr.text((LEFT, 410), d.get("title", ""), font=HF(fit([d.get("title", "")], MAXW, 84)), fill=WHITE, anchor="ls")
    nodes = d["nodes"][:4]
    gap = 230 if len(nodes) <= 3 else 175
    bh = 140 if len(nodes) <= 3 else 110
    for k, txt in enumerate(nodes):
        hot = k == len(nodes) - 1
        t0 = 0.15 + k * (2.2 / len(nodes))
        y = 480 + k * gap
        if t < t0:
            continue
        p = backout((t - t0) / 0.3)
        box = rrect(MAXW, bh, 30, NEO if hot else (24, 26, 32), border=None if hot else NEO, bw=4)
        ImageDraw.Draw(box).text((MAXW / 2, bh / 2 + 2), txt, font=HF(fit([txt], MAXW - 80, 60)), fill=INK if hot else WHITE, anchor="mm")
        put(c, box, LEFT + MAXW / 2, y + bh / 2, scale=0.7 + 0.3 * p)
        if not hot and t >= t0 + 0.35:
            q = ease((t - t0 - 0.35) / 0.3)
            dr = ImageDraw.Draw(c)
            a = y + bh + 10
            dr.line([(540, a), (540, a + (gap - bh - 40) * q)], fill=NEO, width=8)
            if q > 0.9:
                dr.polygon([(516, a + gap - bh - 44), (564, a + gap - bh - 44), (540, a + gap - bh - 16)], fill=NEO)
    caption(c, d.get("caption"), t, 2.3, y=1200)


def s_streak(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    big = str(d.get("big", ""))
    bw = HF(170).getlength(big)
    dr.text((LEFT, 440), big, font=HF(170), fill=NEO, anchor="ls", stroke_width=6, stroke_fill=INK)
    title = d.get("title", "")
    dr.text((LEFT + bw + 24, 440), title, font=HF(fit([title], MAXW - bw - 30, 110)), fill=WHITE, anchor="ls")
    n = max(2, min(6, int(d.get("count", 5))))
    r0 = min(70, int((MAXW - (n - 1) * 30) / n / 2))
    stepx = (MAXW - 2 * r0) / (n - 1)
    for k in range(n):
        cx, cy = LEFT + r0 + k * stepx, 700
        t0 = 0.25 + k * (1.3 / n)
        dr.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], outline=(80, 84, 92), width=6)
        last = k == n - 1 and d.get("last")
        if t >= t0:
            p = backout((t - t0) / 0.25, 3.0)
            r = r0 * max(0.05, p)
            if not last:
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NEO)
                if p > 0.8:
                    u = r0 / 70
                    dr.line([(cx - 30 * u, cy + 2 * u), (cx - 8 * u, cy + 26 * u), (cx + 32 * u, cy - 24 * u)], fill=INK, width=max(6, int(12 * u)), joint="curve")
            else:
                dr.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=WHITE)
                tok = chrome_text(str(d["last"]), max(40, int(86 * r0 / 70)), sw=5)
                put(c, tok, cx + 4, cy, scale=max(0.05, p) * (1 + 0.05 * pulse(T)) * min(1.0, (2 * r0 - 10) / tok.width))
                dr = ImageDraw.Draw(c)
        lab = d.get("last_label", "") if last else d.get("labels", "")
        dr.text((cx, cy + r0 + 42), lab, font=F(CH_B, fit([lab or "X"], stepx - 8 if n > 1 else 200, 30, lambda z: F(CH_B, z), 14)), fill=NEO if last else WHITE, anchor="ms")
    for i, l in enumerate(d.get("lines", [])[:2]):
        dr.text((LEFT, 960 + i * 64), l, font=F(CH_B, fit([l], MAXW, 48, lambda z: F(CH_B, z))), fill=WHITE if i == 0 else GREY, anchor="ls")
    caption(c, d.get("caption"), t, 1.7, y=1200)


def s_versus(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    p = ease(t / 0.3)
    off = (1 - p) * W
    dr.polygon([(-off, 300), (W - off, 300), (-off, 1260)], fill=NEO)
    dr.line([(W + off, 300), (off, 1260)], fill=WHITE, width=6)
    left, right = d.get("left", [])[:2], d.get("right", [])[:2]
    if p > 0.6 and left:
        s = fit(left, 640, 104)
        for i, l in enumerate(left):
            dr.text((LEFT, 470 + i * int(s * 1.08)), l, font=HF(s), fill=INK, anchor="ls")
    if t >= 0.5 and right:
        q = ease((t - 0.5) / 0.3)
        s = fit(right, 620, 90)
        for i, l in enumerate(right):
            dr.text((950 + (1 - q) * 700, 1090 + i * int(s * 1.1)), l, font=HF(s), fill=WHITE, anchor="rs")
    if t >= 0.8:
        q = backout((t - 0.8) / 0.3, 3.0)
        dr.ellipse([540 - 120 * q, 780 - 120 * q, 540 + 120 * q, 780 + 120 * q], fill=INK, outline=WHITE, width=6)
        put(c, chrome_text("VS", 110, sw=6), 546, 780, scale=max(0.05, q) * (1 + 0.04 * pulse(T)))
    if d.get("pill") and t >= 1.5:
        q = backout((t - 1.5) / 0.3)
        fs = fit([d["pill"]], 660, 50, lambda z: F(CH_B, z))
        pill = rrect(720, 104, 52, WHITE)
        ImageDraw.Draw(pill).text((360, 54), d["pill"], font=F(CH_B, fs), fill=INK, anchor="mm")
        put(c, pill, LEFT + 360, 1350 + (1 - q) * 80 + 8 * math.sin(T * 9))


def s_timeline(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    dr.text((LEFT, 410), d.get("title", ""), font=HF(fit([d.get("title", "")], MAXW, 84)), fill=WHITE, anchor="ls")
    evs = d["events"][:4]
    gap = min(210, 760 // len(evs))
    y0 = 520
    span = (len(evs) - 1) * gap
    dr.line([(LEFT + 20, y0), (LEFT + 20, y0 + span * ease(t / 2.2))], fill=(80, 84, 92), width=6)
    tf = F(CH_S, 44)
    for k, (date, label) in enumerate(evs):
        t0 = 0.2 + k * (2.0 / len(evs))
        if t < t0:
            continue
        p = backout((t - t0) / 0.25, 3.0)
        y = y0 + k * gap
        r = 20 * max(0.05, p)
        last = k == len(evs) - 1
        dr.ellipse([LEFT + 20 - r, y - r, LEFT + 20 + r, y + r], fill=NEO if last else WHITE)
        q = ease((t - t0) / 0.3)
        x = LEFT + 70 + (1 - q) * 80
        dr.text((x, y + 18), str(date), font=HF(fit([str(date)], MAXW - 80, 58)), fill=NEO, anchor="ls")
        for i, l in enumerate(wrap(label, tf, MAXW - 90)[:2]):
            dr.text((x, y + 72 + i * 52), l, font=tf, fill=WHITE, anchor="ls")
    caption(c, d.get("caption"), t, 2.4, y=max(1270, min(1340, y0 + span + 190)))


def side_card(who, text, col):
    key = ("side", who, text, col)
    if key not in _cache:
        f = F(CH_S, 48)
        lines = wrap(text, f, MAXW - 88)[:4]
        h = 40 + 34 + 20 + len(lines) * 62 + 30
        img = rrect(MAXW, h, 30, (18, 20, 26), border=col, bw=4)
        dd = ImageDraw.Draw(img)
        dd.rectangle([44, 44, 58, 68], fill=col)
        dd.text((72, 57), who.upper(), font=F(MONO, 28), fill=col, anchor="lm")
        for i, l in enumerate(lines):
            dd.text((44, 40 + 34 + 20 + 46 + i * 62), l, font=f, fill=WHITE, anchor="ls")
        _cache[key] = img
    return _cache[key]


def s_sides(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    dr.text((LEFT, 410), d.get("title", ""), font=HF(fit([d.get("title", "")], MAXW, 84)), fill=WHITE, anchor="ls")
    a = side_card(d["a"]["who"], d["a"]["text"], NEO)
    b = side_card(d["b"]["who"], d["b"]["text"], CYAN)
    ya = 470
    yb = ya + a.height + 90
    if t >= 0.2:
        p = ease((t - 0.2) / 0.35)
        put(c, a, LEFT + MAXW / 2 - X["side"] * (1 - p) * 1000, ya + a.height / 2)
    if t >= 1.3:
        q = backout((t - 1.3) / 0.3, 3.0)
        put(c, chrome_text("VS", 60, sw=4), 540, ya + a.height + 45, scale=max(0.05, q))
    if t >= 1.6:
        p = ease((t - 1.6) / 0.35)
        put(c, b, LEFT + MAXW / 2 + X["side"] * (1 - p) * 1000, yb + b.height / 2)
    caption(c, d.get("caption"), t, 2.6, y=max(1270, min(1350, yb + b.height + 30)))


def s_statement(c, t, T, d, X):
    dark = X["mode"] == "dark"
    fg = WHITE if dark else INK
    hi = NEO if dark else INK
    dr = ImageDraw.Draw(c)
    if d.get("kicker"):
        f = F(CH_B, 36)
        p = ease(t / 0.25)
        tw = f.getlength(d["kicker"])
        dr.rounded_rectangle([LEFT, 400, LEFT + (tw + 44) * p, 460], radius=10, fill=hi)
        if p > 0.7:
            dr.text((LEFT + 22, 431), d["kicker"], font=f, fill=INK if dark else NEO, anchor="lm")
    lines = d.get("lines", [])[:4]
    texts = [l.lstrip("*") for l in lines]
    size = fit(texts, MAXW - 20, 150)
    lh = int(size * 1.12)
    y = 520
    for k, raw in enumerate(lines):
        t0 = 0.10 + k * 0.13
        if t >= t0:
            p = ease((t - t0) / 0.32)
            off = int((1 - p) * lh)
            col = hi if raw.startswith("*") else fg
            kw = dict(stroke_width=6, stroke_fill=INK) if dark else {}
            layer = Image.new("RGBA", (W, lh + 20), (0, 0, 0, 0))
            ImageDraw.Draw(layer).text((LEFT, int(size * 0.98) + 6 + off), raw.lstrip("*"), font=HF(size), fill=col, anchor="ls", **kw)
            c.alpha_composite(layer, (0, y - 6))
            if not dark and raw.startswith("*") and p >= 1:
                ImageDraw.Draw(c).rectangle([LEFT, y + lh - 14, LEFT + HF(size).getlength(raw.lstrip("*")), y + lh - 6], fill=INK)
        y += lh
    t_rule = 0.10 + len(lines) * 0.13 + 0.05
    if t >= t_rule:
        ImageDraw.Draw(c).rectangle([LEFT, y + 26, LEFT + MAXW * ease((t - t_rule) / 0.35), y + 31], fill=hi)
    caption(c, d.get("caption"), t, t_rule + 0.2, dark=dark, y=max(y + 70, 1000) if y + 70 < 1300 else 1300)


def s_outro(c, t, T, d, X):
    dr = ImageDraw.Draw(c)
    lines = d.get("lines", ["FOLLOW", "*DEEP CUT"])[:2]
    if True:
        s = fit([l.lstrip("*") for l in lines], MAXW, 150)
        for i, l in enumerate(lines):
            if t >= i * 0.2:
                dr.text((LEFT, 520 + i * int(s * 1.14)), l.lstrip("*"), font=HF(s), fill=NEO if l.startswith("*") else WHITE, anchor="ls")
        if t >= 0.4 and d.get("sub"):
            dr.text((LEFT, 520 + 2 * int(s * 1.14) - 40), d["sub"], font=F(CH_B, fit([d["sub"]], MAXW, 54, lambda z: F(CH_B, z))), fill=WHITE, anchor="ls")


SCENES = dict(hook=s_hook, ranking=s_ranking, number=s_number, grid=s_grid, items=s_items, flow=s_flow, streak=s_streak, versus=s_versus, timeline=s_timeline, sides=s_sides, statement=s_statement, outro=s_outro)


# ---------------- planning: variation from the seed ----------------

def plan(story):
    seed = story.get("seed")
    if seed is None:
        seed = int(hashlib.sha256(story.get("slug", "x").encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    glows = [NEO, CYAN, VIOLET, PINK]
    rng.shuffle(glows)
    V = dict(motif=rng.choice(["grid", "rings", "bars"]), wipe=rng.choice(["down", "up", "left", "right"]), blob_side=rng.choice(["right", "left"]),
             blob_shape=rng.choice([(3, 5, 2), (4, 3, 2), (2, 5, 3), (3, 4, 5)]), glow=(rng.uniform(0.7, 0.95), rng.uniform(0.2, 0.36)),
             parts=[(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(40, 130), rng.randint(2, 5)) for _ in range(46)])
    scenes, t = [], 0.0
    token = next((s.get("token") for s in story["scenes"] if s["type"] == "hook"), None)
    for i, s in enumerate(story["scenes"]):
        typ = s["type"]
        dur = round(round(float(s.get("dur", DEFAULT_DUR[typ])) / BEAT) * BEAT, 4)
        X = dict(glow=NEO if typ in ("hook", "outro") else glows[i % 4], side=rng.choice([-1, 1]), tilt=rng.choice([0, 0, -4, 4]),
                 mode="dark", token=token, rots=[rng.choice([-14, -10, -6]), rng.choice([-3, 3, 5]), rng.choice([8, 12, 15])])
        kinds = [0, 1, 2]
        rng.shuffle(kinds)
        X["kinds"] = kinds
        if typ == "number" and rng.random() < 0.6:
            X["mode"] = "flat"
        if typ == "statement" and rng.random() < 0.3:
            X["mode"] = "flat"
        if typ == "grid":
            n = max(1, min(30, int(s["count"])))
            cols = grid_geom(n)[0]
            order = list(range(n))
            style = rng.choice(["seq", "snake", "random", "reverse"])
            if style == "random":
                rng.shuffle(order)
            elif style == "reverse":
                order.reverse()
            elif style == "snake":
                order = [k if (k // cols) % 2 == 0 else (k // cols) * cols + min(n - 1, (k // cols) * cols + cols - 1) - k for k in order]
                order = [k for k in order if 0 <= k < n]
                order = list(dict.fromkeys(order + list(range(n))))
            X["order"] = order
        scenes.append(dict(type=typ, data=s, dur=dur, start=round(t, 4), X=X))
        t += dur
    rem = round(t % 1.6, 4)
    if rem > 1e-3 and rem < 1.6 - 1e-3:  # pad the last scene so picture and music loop on a bar line
        scenes[-1]["dur"] = round(scenes[-1]["dur"] + 1.6 - rem, 4)
        t += 1.6 - rem
    V["total"] = round(t, 4)
    return dict(V=V, scenes=scenes, total=round(t, 4), seed=seed, rng=rng)


def frame(story, P, i, t):
    sc = P["scenes"][i]
    T = sc["start"] + t
    X = sc["X"]
    dark = X["mode"] == "dark"
    canvas = dark_bg(T, X["glow"], P["V"], blob=sc["type"] in ("hook", "outro")) if dark else Image.new("RGBA", (W, H), NEO + (255,))
    LOOP = 0.7
    if i == len(P["scenes"]) - 1 and i > 0 and P["scenes"][0]["type"] == "hook" and t > sc["dur"] - LOOP:
        p = ease((t - (sc["dur"] - LOOP)) / (LOOP - 0.1))
        a_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        SCENES[sc["type"]](a_layer, t, T, sc["data"], X)
        b_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        h0 = P["scenes"][0]
        SCENES["hook"](b_layer, 0.0, 0.0, h0["data"], h0["X"])
        for layer, al in ((a_layer, 1 - p), (b_layer, p)):
            if al > 0.004:
                layer.putalpha(layer.getchannel("A").point(lambda v, al=al: int(v * al)))
                canvas.alpha_composite(layer)
    else:
        SCENES[sc["type"]](canvas, t, T, sc["data"], X)
    hud(canvas, story, T if i < len(P["scenes"]) - 1 or t < sc["dur"] - 0.05 else P["total"], P["total"], dark)
    n = int(round(t * FPS))
    d = ImageDraw.Draw(canvas)
    if n < 5 and i > 0:
        col = NEO if dark else INK
        f = n / 5
        wipe = P["V"]["wipe"]
        if wipe == "down":
            y0 = -320 + f * (H + 320)
            d.rectangle([0, y0, W, y0 + 320], fill=col)
        elif wipe == "up":
            y0 = H - f * (H + 320)
            d.rectangle([0, y0, W, y0 + 320], fill=col)
        elif wipe == "right":
            x0 = -260 + f * (W + 260)
            d.rectangle([x0, 0, x0 + 260, H], fill=col)
        else:
            x0 = W - f * (W + 260)
            d.rectangle([x0, 0, x0 + 260, H], fill=col)
    if n < 5 and i > 0:
        z = 1 + 0.06 * (1 - n / 5) ** 2
        zw, zh = int(W * z), int(H * z)
        canvas = canvas.resize((zw, zh), Image.BILINEAR).crop(((zw - W) // 2, (zh - H) // 2, (zw - W) // 2 + W, (zh - H) // 2 + H))
    a = np.asarray(canvas.convert("RGB")).copy()
    if n < 3 and i > 0:
        k = (10, 5, 2)[n]
        a[..., 0] = np.roll(a[..., 0], k, 1)
        a[..., 2] = np.roll(a[..., 2], -k, 1)
    f = a.astype(np.float32) * VIG + np.roll(GRAIN, (n * 131) % H, 0)
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------- sound: soft, in key, generated from the same seed ----------------
SR = 44100
CHORDS = {"Fm": (41, [60, 63, 65, 68], [72, 75, 77, 80, 82]), "Db": (37, [61, 63, 65, 68], [73, 75, 77, 80, 82]),
          "Ab": (44, [60, 63, 68, 70], [72, 75, 77, 80, 82]), "Eb": (39, [58, 63, 67, 70], [70, 72, 75, 77, 79])}
PROGS = [["Fm", "Db", "Ab", "Eb"], ["Ab", "Fm", "Db", "Eb"], ["Db", "Eb", "Fm", "Ab"], ["Fm", "Ab", "Eb", "Db"]]
GROOVES = [((0, 10), (8,)), ((0, 6, 10), (8,)), ((0, 11), (8,)), ((0, 10, 14), (8,))]


def soundtrack(P, path):
    rng = random.Random(P["seed"] + 7)
    nrng = np.random.default_rng(P["seed"] % (2 ** 31))
    total = P["total"]
    N = int(total * SR)
    drums, music, fx = (np.zeros((N, 2), np.float32) for _ in range(3))
    prog, (kicks, rims), tr = rng.choice(PROGS), rng.choice(GROOVES), rng.choice([-3, -2, 0, 0, 1, 2])
    STEP = 0.1

    def tt(d):
        return np.arange(int(d * SR), dtype=np.float32) / SR

    def hz(n):
        return 440.0 * 2 ** ((n + tr - 69) / 12)

    def add(buf, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR)) % N
        sig = sig[:N]
        gl, gr = gain * (1 - max(0, pan)), gain * (1 + min(0, pan))
        k = min(len(sig), N - i)
        buf[i:i + k, 0] += sig[:k] * gl
        buf[i:i + k, 1] += sig[:k] * gr
        if k < len(sig):  # the tail continues at the start, so the loop point is seamless
            buf[: len(sig) - k, 0] += sig[k:] * gl
            buf[: len(sig) - k, 1] += sig[k:] * gr

    def filt(x, kind, fc):
        return signal.sosfilt(signal.butter(2, fc, kind, fs=SR, output="sos"), x).astype(np.float32)

    def noise(d):
        return nrng.standard_normal(int(d * SR)).astype(np.float32)

    def ep(note, d):
        t = tt(d)
        f = hz(note)
        x = np.sin(2 * np.pi * f * t) + 0.28 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 3) + 0.10 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t * 6)
        return x * (1 + 0.06 * np.sin(2 * np.pi * 4.5 * t)) * np.minimum(1, t * 60) * np.exp(-t * 1.6) * np.minimum(1, (d - t) * 6)

    def bell(note, d=0.5):
        t = tt(d)
        f = hz(note)
        return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 12)) * np.exp(-t * 11) * np.minimum(1, t * 600)

    def sub(note, d):
        t = tt(d)
        return np.tanh(np.sin(2 * np.pi * hz(note) * t) * 1.5) * np.minimum(1, t * 80) * np.minimum(1, (d - t) * 30) * np.exp(-t * 1.2)

    tk = tt(0.3)
    SK = np.sin(2 * np.pi * np.cumsum(44 + 50 * np.exp(-tk * 30)) / SR) * np.exp(-tk * 10)
    RM = filt(noise(0.12), "band", [1400, 3200]) * np.exp(-tt(0.12) * 38)
    SH = filt(noise(0.035), "high", 8000) * np.exp(-tt(0.035) * 130)
    CK = filt(noise(0.006), "high", 3500) * np.exp(-tt(0.006) * 500)

    def whoosh(d=0.28):
        t = tt(d)
        p = t / d
        n = noise(d)
        return (filt(n, "band", [400, 1800]) * (1 - p) + filt(n, "band", [1800, 8000]) * p) * p ** 2

    def mel_at(t, k):
        return CHORDS[prog[int(t / 1.6 + 1e-6) % 4]][2][k % 5]

    first = P["scenes"][0]["dur"]
    last_start = P["scenes"][-1]["start"]
    quiet = {round(s["start"], 3) for s in P["scenes"] if s["type"] in ("flow", "sides")}
    kick_times, t0, bar = [], 0.0, 0
    while t0 < total - 1e-6:
        d = min(1.6, total - t0)
        root, notes, mel = CHORDS[prog[bar % 4]]
        for i, n in enumerate(notes):
            add(music, ep(n, d), t0 + i * 0.012, 0.20, pan=-0.3 + 0.2 * i)
        if t0 >= first - 1e-6 and round(t0, 3) not in quiet and t0 + 1.6 <= last_start + 1e-6:
            for s in kicks:
                add(drums, SK, t0 + s * STEP, 0.75)
                kick_times.append(t0 + s * STEP)
            for s in rims:
                add(drums, RM, t0 + s * STEP, 0.30)
            for s in range(0, 16, 2):
                add(drums, SH, t0 + s * STEP, 0.07 if s % 4 else 0.04, pan=0.2)
            add(music, sub(root, 0.9), t0, 0.42)
            add(music, sub(root, 0.5), t0 + kicks[-1] * STEP, 0.36)
        if t0 >= first - 1e-6:
            for k, s in enumerate(rng.sample([2, 4, 6, 12, 14], 3)):
                n = mel[rng.randrange(5)]
                add(music, bell(n), t0 + s * STEP, 0.10, pan=0.4 if k % 2 else -0.4)
                add(music, bell(n), t0 + s * STEP + 0.3, 0.04, pan=-0.4 if k % 2 else 0.4)
        t0 += 1.6
        bar += 1
    env = np.ones(N, np.float32)
    duck = 1 - 0.25 * np.exp(-tt(0.25) * 12)
    for kt in kick_times:
        i = int(kt * SR)
        seg = duck[: N - i]
        env[i:i + len(seg)] = np.minimum(env[i:i + len(seg)], seg)
    # scene-synced notes, all taken from the chord sounding at that moment
    for s in P["scenes"][1:]:
        add(fx, whoosh(), s["start"] - 0.28, 0.07)
    for s in P["scenes"]:
        st, d, typ = s["start"], s["data"], s["type"]
        times = []
        if typ == "grid":
            n = max(1, min(30, int(d["count"])))
            step = grid_geom(n)[3]
            times = [st + 0.3 + k * step for k in range(n)]
        elif typ == "ranking":
            times = [st + 0.05 + k * 0.14 for k in range(min(6, len(d["rows"])))]
        elif typ == "streak":
            n = max(2, min(6, int(d.get("count", 5))))
            times = [st + 0.25 + k * (1.3 / n) for k in range(n)]
        elif typ == "timeline":
            n = min(4, len(d["events"]))
            times = [st + 0.2 + k * (2.0 / n) for k in range(n)]
        elif typ == "flow":
            n = min(4, len(d["nodes"]))
            times = [st + 0.15 + k * (2.2 / n) for k in range(n)]
        elif typ == "items":
            times = [st + 0.1 + k * 0.22 for k in range(3)]
        elif typ == "sides":
            times = [st + 0.2, st + 1.6]
        elif typ == "versus":
            times = [st + 0.8]
        for k, tm in enumerate(times):
            add(fx, bell(mel_at(tm, k) + (12 if k >= 10 else 0), 0.4), tm, 0.12 if len(times) > 8 else 0.18, pan=0.3 if k % 2 else -0.3)
        if typ == "number" and isinstance(d.get("value"), (int, float)):
            for k in range(14):
                add(fx, CK, st + k * 0.065, 0.10)
    music *= env[:, None]
    mix = np.tanh((drums * 0.8 + music * 0.9 + fx * 0.9) * 0.7)
    mix = mix / max(1e-6, np.abs(mix).max()) * 0.55
    wavfile.write(path, SR, (mix * 32767).astype(np.int16))


# ---------------- validation + render ----------------

def validate(story):
    errs = []
    if not story.get("source"):
        errs.append("missing 'source'")
    sc = story.get("scenes", [])
    if not sc or sc[0].get("type") != "hook":
        errs.append("first scene must be a 'hook'")
    if not sc or sc[-1].get("type") != "outro":
        errs.append("last scene must be an 'outro'")
    for i, s in enumerate(sc):
        if s.get("type") not in SCENES:
            errs.append(f"scene {i}: unknown type {s.get('type')!r}")
    total = sum(float(s.get("dur", DEFAULT_DUR.get(s.get("type"), 0))) for s in sc)
    if not 12 <= total <= 40:
        errs.append(f"total duration {total:.1f}s outside 12-40s")
    if errs:
        raise SystemExit("Invalid story:\n- " + "\n- ".join(errs))


def render(story, out):
    validate(story)
    P = plan(story)
    tmp = tempfile.mkdtemp(prefix="deepcut_")
    parts = []
    for i, sc in enumerate(P["scenes"]):
        part, final = os.path.join(tmp, f"{i:02d}.tmp.mp4"), os.path.join(tmp, f"{i:02d}.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                              "-c:v", "libx264", "-crf", "22", "-maxrate", "6M", "-bufsize", "12M", "-preset", "fast", "-pix_fmt", "yuv420p", part], stdin=subprocess.PIPE)
        for n in range(round(sc["dur"] * FPS)):
            p.stdin.write(frame(story, P, i, n / FPS).tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"encode failed on scene {i}")
        os.rename(part, final)
        parts.append(final)
        print(f"scene {i} {sc['type']} ok", flush=True)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    joined, wav = os.path.join(tmp, "joined.mp4"), os.path.join(tmp, "sound.wav")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])
    soundtrack(P, wav)
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", joined, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out])
    print(f"done {out} {P['total']}s seed {P['seed']} motif {P['V']['motif']}")


def stills(story, outdir):
    validate(story)
    P = plan(story)
    os.makedirs(outdir, exist_ok=True)
    ims = [Image.fromarray(frame(story, P, i, max(0.0, sc["dur"] - 0.35))) for i, sc in enumerate(P["scenes"])]
    for i, im in enumerate(ims):
        im.save(os.path.join(outdir, f"{i:02d}_{P['scenes'][i]['type']}.png"))
    sheet = Image.new("RGB", (270 * len(ims), 480))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((270, 480)), (270 * i, 0))
    sheet.save(os.path.join(outdir, "sheet.jpg"), quality=88)
    print("stills ok", len(ims), "total", P["total"])


if __name__ == "__main__":
    story = json.load(open(sys.argv[1]))
    if len(sys.argv) > 3 and sys.argv[3] == "--stills":
        stills(story, sys.argv[2])
    else:
        render(story, sys.argv[2])
