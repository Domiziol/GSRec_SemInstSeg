from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # linux; na win podaj ścieżkę do .ttf
FONT_SIZE = 22
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
FIXED_COL_WIDTH = 260  # px – dobierz raz i koniec


# ====== PASTE PATHS HERE ======
NPZ_DIR = Path("./data/replica/scan1/2Dclassification_tests/test2/results/")
IMG_DIR = Path("./data/replica/scan1/images/")
OUT_DIR = Path("./data/replica/scan1/2Dclassification_tests/test2/visual_results2")
INFO_JSON = Path("./data/replica/scan1//info_semantic.json")
IMG_DIGITS = 6        # 00001.png
ALPHA = 1
MASK_KEY = "masks"
LABEL_KEY = "labels"
# ==============================

# colors = (
#     (242,73,73),
#     (73,242,118),
#     (242,221,73),
#     (73,155,242),
#     (242,142,73),
#     (155,73,242),
#     (73,242,242),
#     (242,73,203),
#     (203,242,73),
#     (242,160,203),
#     (73,191,191),
#     (203,191,242),
#     (191,142,73),
#     (242,236,191),
#     (128,38,38),
#     (191,242,203),
#     (142,142,38),
#     (242,209,191),
#     (38,38,128),
#     (128,128,128),
#     (242,38,38),
#     (38,242,38),
#     (38,38,242),
#     (242,242,38),
#     (242,38,242),
#     (38,242,242),
#     (191,191,191),
#     (128,38,128),
#     (128,83,38),
#     (83,128,38),
#     (38,128,83),
#     (38,83,128),
#     (83,38,128),
#     (128,38,83),
#     (191,38,38),
#     (38,191,38),
#     (38,38,191),
#     (191,191,38),
#     (191,38,191),
#     (38,191,191),
#     (102,38,38),
#     (38,102,38),
#     (38,38,102),
#     (102,102,38),
#     (102,38,102),
#     (38,102,102),
#     (166,83,83),
#     (83,166,83),
#     (83,83,166),
#     (166,166,83),
#     (166,83,166),
#     (83,166,166),
#     (204,121,38),
#     (38,204,121),
#     (121,38,204),
#     (204,38,121),
#     (121,204,38),
#     (38,121,204),
#     (153,96,38),
#     (38,153,96),
#     (96,38,153),
#     (153,38,96),
#     (96,153,38),
#     (38,96,153),
#     (242,162,38),
#     (162,242,38),
#     (38,242,162),
#     (38,162,242),
#     (162,38,242),
#     (242,38,162),
#     (204,142,83),
#     (83,204,142),
#     (142,83,204),
#     (204,83,142),
#     (142,204,83),
#     (83,142,204),
#     (242,121,121),
#     (121,242,121),
#     (121,121,242),
#     (242,242,121),
#     (242,121,242),
#     (121,242,242),
#     (64,64,140),
#     (140,64,64),
#     (64,140,64),
#     (191,96,96),
#     (96,191,96),
#     (96,96,191),
#     (191,191,96),
#     (242,74,190),
#     (96,191,191),
#     (77,38,38),
#     (38,77,38),
#     (38,38,77),
#     (77,77,38),
#     (77,38,77)
# )

# colors = (
#     (242,73,73),(109,127,248),(242,221,73),(73,155,242),(242,142,73),
#     (155,73,242),(73,242,242),(242,73,203),(203,242,73),(242,160,203),
#     (244,108,129),(203,191,242),(191,142,73),(242,236,191),(128,38,38),
#     (191,242,203),(142,142,38),(242,209,191),(38,38,128),(210,137,91),
#     (242,38,38),(38,242,38),(38,38,242),(242,242,38),(242,38,242),
#     (38,242,242),(191,191,191),(128,38,128),(202,191,155),(83,128,38),
#     (38,128,83),(38,83,128),(83,38,128),(128,38,83),(191,38,38),
#     (38,191,38),(38,38,191),(191,191,38),(191,38,191),(38,191,191),
#     (102,38,38),(38,102,38),(38,38,102),(102,102,38),(102,38,102),
#     (38,102,102),(157,17,16),(83,166,83),(83,83,166),(166,166,83),
#     (166,83,166),(83,166,166),(204,121,38),(38,204,121),(121,38,204),
#     (204,38,121),(121,204,38),(38,121,204),(153,96,38),(38,153,96),
#     (96,38,153),(153,38,96),(96,153,38),(38,96,153),(242,162,38),
#     (162,242,38),(38,242,162),(38,162,242),(162,38,242),(242,38,162),
#     (204,142,83),(83,204,142),(142,83,204),(204,83,142),(203,235,193),
#     (83,142,204),(242,121,121),(121,242,121),(69,247,172),(242,242,121),
#     (242,121,242),(121,242,242),(64,64,140), (140,64,64),(64,140,64),
#     (191,96,96),(96,191,96),(96,96,191),(191,191,96),(191,96,191),
#     (3,97,104),(77,38,38),(38,77,38),(38,38,77),(77,77,38),
#     (77,38,77), (242,74,190), (182,145,212), (199,206,110), (245,134,71),
#     (27,253,70)
# )
colors = (
    (242,73,73),(109,127,248),(106, 45, 110),(73,155,242),(242,142,73),
    (155,73,242),(73,242,242),(242,73,203),(203,242,73),(242,160,203),
    (244,108,129),(203,191,242),(191,142,73),(242,236,191),(128,38,38),
    (191,242,203),(142,142,38),(59, 232, 245),(38,38,128),(210,137,91),
    (242,38,38),(38,242,38),(38,38,242),(242,242,38),(242,38,242),
    (38,242,242),(191,191,191),(128,38,128),(202,191,155),(83,128,38),
    (38,128,83),(38,83,128),(83,38,128),(128,38,83),(191,38,38),
    (38,191,38),(38,38,191),(191,191,38),(191,38,191),(38,191,191),
    (102,38,38),(38,102,38),(38,38,102),(102,102,38),(102,38,102),
    (38,102,102),(157,17,16),(83,166,83),(83,83,166),(166,166,83),
    (166,83,166),(83,166,166),(204,121,38),(38,204,121),(121,38,204),
    (204,38,121),(121,204,38),(38,121,204),(84, 46, 12),(38,153,96),
    (96,38,153),(153,38,96),(96,153,38),(38,96,153),(100, 76, 209),
    (162,242,38),(38,242,162),(38,162,242),(162,38,242),(242,38,162),
    (204,142,83),(83,204,142),(142,83,204),(204,83,142),(203,235,193),
    (83,142,204),(242,121,121),(121,242,121),(69,247,172),(242,242,121),
    (242,121,242),(121,242,242),(64,64,140), (140,64,64),(64,140,64),
    (191,96,96),(96,191,96),(96,96,191),(191,191,96),(191,96,191),
    (3,97,104),(77,38,38),(38,77,38),(38,38,77),(77,77,38),
    (77,38,77), (242,74,190), (182,145,212), (199,206,110), (245,134,71),
    (27,253,70)
)
assert len(colors) == 101

def render_legend_panel(
    legend_items,                 # list[(lab, name)]
    colors,
    font,
    max_panel_height,             # wysokość obrazka, do której legenda ma się zmieścić
    pad=18,
    box=20,
    row=30,
    col_gap=28,
    min_cols=1,
    max_cols=6,
    bg=(255, 255, 255),
    outline=(0, 0, 0),
):
    """
    Zwraca obraz PIL (panel legendy) dopasowany kolumnami do max_panel_height.
    """

    dummy = Image.new("RGB", (10, 10), bg)
    d0 = ImageDraw.Draw(dummy)

    # helper: policz szerokość tekstu
    def tlen(s: str) -> int:
        return int(d0.textlength(str(s), font=font))

    n = len(legend_items)
    if n == 0:
        return Image.new("RGB", (1, max_panel_height), bg)

    # Dobierz liczbę kolumn tak, by zmieścić w pionie
    # (więcej kolumn => mniej wierszy)
    best = None
    for ncols in range(min_cols, min(max_cols, n) + 1):
        nrows = math.ceil(n / ncols)
        panel_h = pad * 2 + row * nrows
        if panel_h <= max_panel_height:
            best = ncols
            break
    if best is None:
        best = min(max_cols, n)  # nie mieści się nawet przy max_cols, trudno — damy max_cols

    ncols = best
    nrows = math.ceil(n / ncols)

    # Rozdzielamy kolumnami “od góry” (czytelne)
    cols = []
    for c in range(ncols):
        cols.append(legend_items[c * nrows:(c + 1) * nrows])

    # Szerokość każdej kolumny (max szerokość tekstu w kolumnie)
    col_text_max = []
    for col in cols:
        if not col:
            col_text_max.append(0)
        else:
            col_text_max.append(max(tlen(name) for _, name in col))

    # Szerokość kolumny: pad + box + gap + text + pad
    box_text_gap = 12
    # col_w = [pad + box + box_text_gap + tw + pad for tw in col_text_max]
    # panel_w = sum(col_w) + col_gap * (ncols - 1)
    col_w = [FIXED_COL_WIDTH for _ in range(ncols)]
    panel_w = ncols * FIXED_COL_WIDTH + col_gap * (ncols - 1)
    panel_h = pad * 2 + row * nrows

    panel = Image.new("RGB", (panel_w, panel_h), bg)
    d = ImageDraw.Draw(panel)

    x = 0
    for c in range(ncols):
        y = pad
        for lab, name in cols[c]:
            c_rgb = colors[lab % len(colors)]
            d.rectangle([x + pad, y, x + pad + box, y + box], fill=c_rgb, outline=outline, width=1)
            d.text((x + pad + box + box_text_gap, y - 2), str(name), fill=(0, 0, 0), font=font)
            y += row
        x += col_w[c] + col_gap

    return panel

def save_legend_only(path, legend_items, pad=12, box=16, row=22, bg=(255,255,255)):
    # legend_items: list of (lab:int, name:str)
    dummy = Image.new("RGB", (10, 10), bg)
    d0 = ImageDraw.Draw(dummy)

    max_text_w, max_rows = 0, len(legend_items)
    for lab, name in legend_items:
        max_text_w = max(max_text_w, int(d0.textlength(str(name), font=font)))

    w = pad*3 + box + max_text_w
    h = pad*2 + row*max_rows
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)

    y = pad
    for lab, name in legend_items:
        c = colors[lab % len(colors)]
        d.rectangle([pad, y, pad + box, y + box], fill=c, outline=(0,0,0))
        d.text((pad*2 + box, y-1), str(name), fill=(0,0,0), font=font)
        y += row

    img.save(path)

from PIL import Image, ImageDraw, ImageFont
import math

def save_legend_grid(path, legend_items, ncols=4,
                     pad=20, box=22, row=30, col_gap=28,
                     font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     font_size=18, bg=(255,255,255)):
    """
    legend_items: list of (lab:int, name:str)
    ncols: ile kolumn (im więcej, tym bardziej poziomo)
    """

    font = ImageFont.truetype(font_path, font_size)

    dummy = Image.new("RGB", (10, 10), bg)
    d0 = ImageDraw.Draw(dummy)

    # szerokość tekstu per element
    text_ws = [int(d0.textlength(str(name), font=font)) for _, name in legend_items]
    # szerokość kolumny = box + przerwy + max tekst w tej kolumnie
    n = len(legend_items)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    # rozdzielamy elementy kolumnami (kolumny wypełniane od góry)
    cols = []
    for c in range(ncols):
        col_items = legend_items[c*nrows:(c+1)*nrows]
        cols.append(col_items)

    # max szerokość tekstu w każdej kolumnie
    col_text_max = []
    for col in cols:
        if not col:
            col_text_max.append(0)
            continue
        widths = [int(d0.textlength(str(name), font=font)) for _, name in col]
        col_text_max.append(max(widths))

    col_w = [pad + box + 12 + tw + pad for tw in col_text_max]  # 12 = odstęp box->tekst
    W = sum(col_w) + col_gap*(ncols-1)
    H = pad*2 + row*nrows

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    x = 0
    for c in range(ncols):
        y = pad
        for lab, name in cols[c]:
            c_rgb = colors[lab % len(colors)]
            d.rectangle([x + pad, y, x + pad + box, y + box], fill=c_rgb, outline=(0,0,0), width=1)
            d.text((x + pad + box + 12, y - 2), str(name), fill=(0,0,0), font=font)
            y += row
        x += col_w[c] + col_gap

    img.save(path, format="PNG")


OUT_DIR.mkdir(exist_ok=True)

classes = json.loads(INFO_JSON.read_text(encoding="utf-8"))["classes"]
# font = ImageFont.load_default()

def add_legend_pro(
    overlay_img,
    legend_items,
    colors,
    font,
    pad_outer=2,
    legend_bg=(255, 255, 255),
):
    """
    Dokleja legendę po prawej stronie w wysokiej jakości.
    Automatycznie dobiera liczbę kolumn żeby zmieścić się w wysokości obrazka.
    """
    w, h = overlay_img.size

    legend_panel = render_legend_panel(
        legend_items=legend_items,
        colors=colors,
        font=font,
        max_panel_height=h - 2 * pad_outer,
        pad=18,
        box=20,
        row=30,
        col_gap=28,
        min_cols=1,
        max_cols=6,
        bg=legend_bg,
    )

    lw, lh = legend_panel.size
    canvas = Image.new("RGB", (w + pad_outer + lw + pad_outer, h), legend_bg)
    canvas.paste(overlay_img, (0, 0))

    # wyśrodkuj w pionie
    y0 = max(pad_outer, (h - lh) // 2)
    canvas.paste(legend_panel, (w + pad_outer, y0))
    return canvas

def add_legend(overlay_img, legend_items, pad=12, box=16, row=22):
    # legend_items: list of (label:int, name:str) in desired order (unique labels)
    draw_tmp = ImageDraw.Draw(overlay_img)
    max_text_w = 0
    for lab, name in legend_items:
        w = draw_tmp.textlength(f"{lab}: {name}", font=font)
        max_text_w = max(max_text_w, int(w))
    legend_w = pad*3 + box + max_text_w
    w, h = overlay_img.size
    canvas = Image.new("RGB", (w + legend_w, h), (255, 255, 255))
    canvas.paste(overlay_img, (0, 0))
    d = ImageDraw.Draw(canvas)

    y = pad
    for lab, name in legend_items:
        c = colors[lab % len(colors)]
        d.rectangle([w + pad, y, w + pad + box, y + box], fill=c, outline=(0,0,0))
        d.text((w + pad*2 + box, y-1), str(name), fill=(0,0,0), font=font)
        y += row
        if y + row > h - pad:  # stop if legend exceeds height
            d.text((w + pad*2 + box, y-1), name, fill=(0,0,0), font=font)
            break
    return canvas

for npz_path in sorted(NPZ_DIR.glob("*.npz")):
    img_id = str(int(npz_path.stem)).zfill(IMG_DIGITS)
    img_path = next((IMG_DIR / f"{img_id}{e}" for e in (".png",".jpg",".jpeg") if (IMG_DIR / f"{img_id}{e}").exists()), None)
    if not img_path: continue

    base = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    data = np.load(npz_path)
    masks, labels = data[MASK_KEY], data[LABEL_KEY].astype(int)
    # u = np.unique(labels)
    # print(npz_path.name, "unique labels:", u[:50], "..." if len(u) > 50 else "")
    # print("contains 96?", 96 in set(u.tolist()))

    # overlay
    img = base.copy()
    for m, lab in zip(masks, labels):
        mm = m.astype(bool)
        img[mm] = (1 - ALPHA) * img[mm] + ALPHA * np.array(colors[lab % len(colors)], dtype=np.float32)
    overlay = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # legend (unique labels in first-seen order)
    seen, legend = set(), []
    for lab in labels.tolist():
        if lab in seen: continue
        seen.add(lab)
        c = classes[lab] if 0 <= lab < len(classes) else {"name": f"UNKNOWN_{lab}"}
        name = c["name"] if isinstance(c, dict) else str(c)
        legend.append((lab, name))

    # out = add_legend(overlay, legend)
    out = add_legend_pro(overlay, legend, colors=colors, font=font)
    out.save(OUT_DIR / f"{img_id}_overlay_legend.png")

legend_all = []
for lab, c in enumerate(classes):
    name = c["name"] if isinstance(c, dict) and "name" in c else str(c)
    legend_all.append((lab, name))

# save_legend_only(OUT_DIR / "global_legend.png", legend_all)

# GT for replica/scan1
VALID_IDS = [3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59,
             60, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98]

VALID_IDS = [i - 1 for i in VALID_IDS]   # apply your indexing shift

legend_valid = []
for lab in VALID_IDS:
    if 0 <= lab < len(classes):
        c = classes[lab]
        name = c["name"] if isinstance(c, dict) and "name" in c else str(c)
        legend_valid.append((lab, name))

# save_legend_only(OUT_DIR / "scene_legend.png", legend_valid)
save_legend_grid(OUT_DIR / "scene_legend_horizontal.png", legend_valid, ncols=4)