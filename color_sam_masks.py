## owner: Dominika Ziolkiewicz

## THESIS
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import math


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 22
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

masksPath = Path("./data/replica/scan1/2Dclassification_tests/test2/results/")
imagesPath = Path("./data/replica/scan1/images/")
outputPath = Path("./data/replica/scan1/2Dclassification_tests/test2/visual_results2")
infoSemanticJson = Path("./data/replica/scan1//info_semantic.json")
nameSpace = 6

outputPath.mkdir(exist_ok=True)
classes = json.loads(infoSemanticJson.read_text(encoding="utf-8"))["classes"]

#OLD
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


def save_legend_only(path, legend, pad=12, box=16, row=22, bkgColor=(255,255,255)):
    
    dummy = Image.new("RGB", (10, 10), bkgColor)
    d0 = ImageDraw.Draw(dummy)

    maxRows = 0
    for _, name in legend:
        maxTextWidth = max(maxTextWidth, int(d0.textlength(str(name), font=font)))

    w = pad*3 + box + maxTextWidth
    h = pad*2 + row*maxRows
    img = Image.new("RGB", (w, h), bkgColor)
    d = ImageDraw.Draw(img)

    y = pad
    for i, name in legend:
        c = colors[i % len(colors)]
        d.rectangle([pad, y, pad + box, y + box], fill=c, outline=(0,0,0))
        d.text((pad*2 + box, y-1), str(name), fill=(0,0,0), font=font)
        y += row

    img.save(path)



def only_legend(path, legend, nColumns):
                     
    pad=20
    box=22
    row=30
    col_gap=28
                              

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    dummy = Image.new("RGB", (10, 10), (255,255,255))
    d0 = ImageDraw.Draw(dummy)

    n = len(legend)
    nColumns = max(1, min(nColumns, n))
    nRows = math.ceil(n / nColumns)

    
    cols = []
    for c in range(nColumns):
        col_items = legend[c * nRows : (c+1) * nRows]
        cols.append(col_items)

    
    maxWidth = []
    for col in cols:
        if not col:
            maxWidth.append(0)
            continue

        widths = [int(d0.textlength(str(name), font=font)) for _, name in col]
        maxWidth.append(max(widths))

    col_w = [pad + box + 12 + tw + pad for tw in maxWidth]
    w = sum(col_w) + col_gap*(nColumns-1)
    h = pad*2 + row*nRows

    img = Image.new("RGB", (w, h), (255,255,255))
    d = ImageDraw.Draw(img)

    x = 0
    for c in range(nColumns):
        y = pad


        for i, name in cols[c]:
            c_rgb= colors[i % len(colors)]

            d.rectangle([x + pad, y, x + pad + box, y + box], fill=c_rgb, outline=(0,0,0), width=1)
            d.text((x +pad + box+ 12, y - 2),str(name), fill=(0,0,0),font=font)
            y+= row

        x += col_w[c] + col_gap

    img.save(path, format="PNG")


def add_legend(img, legend, pad_out=2, bkgColor=(255, 255, 255)):
    
    w, h = img.size
    pad=18
    max_panel_height=h - 2 * pad_out
    pad=18
    box=20
    row=30
    col_gap=28
    min_cols=1
    max_cols=6

    dummy = Image.new("RGB", (10, 10), bkgColor)
    d0 = ImageDraw.Draw(dummy)


    n = len(legend)
    if n == 0:
        return Image.new("RGB", (1, max_panel_height), bkgColor)

    # Dobierz liczbę kolumn tak, by zmieścić w pionie
    # (więcej kolumn => mniej wierszy)
    colsNumber = None
    for ncols in range(min_cols, min(max_cols, n) + 1):
        nrows = math.ceil(n / ncols)
        panel_h = pad * 2 + row * nrows
        if panel_h <= max_panel_height:
            colsNumber = ncols
            break
    if colsNumber is None:
        colsNumber = min(max_cols, n)

    ncols = colsNumber
    nrows = math.ceil(n / ncols)

    
    cols = []
    for c in range(ncols):
        cols.append(legend[c * nrows:(c + 1) * nrows])

    
    colSize = []
    for col in cols:
        if not col:
            colSize.append(0)
        else:
            colSize.append(max(int(d0.textlength(str(name), font=font)) for _, name in col))

    # Szerokość kolumny: pad + box + gap + text + pad
    box_text_gap = 12
    colWidth = 260
    
    col_w = [colWidth for _ in range(ncols)]
    panelW= ncols * colWidth + col_gap * (ncols - 1)
    panelH = pad * 2 + row * nrows

    legend_panel = Image.new("RGB", (panelW, panelH), bkgColor)
    d = ImageDraw.Draw(legend_panel)

    x = 0
    for c in range(ncols):
        y = pad
        for i, name in cols[c]:

            c = colors[i % len(colors)]

            d.rectangle([x + pad, y, x+pad + box, y+ box], fill = c, outline= (0, 0, 0),  width=1)
            d.text((x + pad + box + box_text_gap, y - 2), str(name), fill=(0, 0, 0), font=font)

            y += row

        x += col_w[c] + col_gap


    lw, lh = legend_panel.size
    panel = Image.new("RGB", (w + pad_out + lw + pad_out, h), bkgColor)
    panel.paste(img, (0, 0))

   
    y0 = max(pad_out, (h - lh) // 2)
    panel.paste(legend_panel, (w + pad_out, y0))
    return panel



if __name__ == '__main__':

    for npz_path in masksPath.glob("*.npz"):

        imgName = str(int(npz_path.stem)).zfill(nameSpace)
        imgPath = next((imagesPath / f"{imgName}{e}" for e in (".png",".jpg",".jpeg") if (imagesPath / f"{imgName}{e}").exists()), None)


        imgOrig = np.array(Image.open(imgPath).convert("RGB"), dtype=np.float32)
        data = np.load(npz_path)
        masks, labels = data["masks"], data["labels"].astype(int)

        # create overlay
        img = imgOrig.copy()
        for mask, label in zip(masks, labels):
            m = mask.astype(bool)

            img[m] = np.array(colors[label % len(colors)], dtype=np.float32)

        imgNew = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        
        seen = set()
        legend = []
        for label in labels.tolist():

            if label in seen: 
                continue

            seen.add(label)

            if 0 <= label < len(classes):
                c = classes[label]

            name = c["name"] if isinstance(c, dict) else str(c)
            legend.append((label, name))

       
        colored = add_legend(path = imgNew, legend = legend)
        colored.save(outputPath / f"{imgName}_overlay_legend.png")

    
    CLASS_IDS = [3, 11, 12,13, 18, 19, 20, 29, 31, 37,40, 44, 47, 59, 60,  63, 64, 65, 76, 78, 79, 80,   91, 92, 93, 95, 97, 98]
    CLASS_IDS = [i - 1 for i in CLASS_IDS]

    legend_gt = []
    for classLabel in CLASS_IDS:
        if 0 <= classLabel < len(classes):
            c = classes[classLabel]

            name = c["name"] if isinstance(c, dict) and "name" in c else str(c)
            legend_gt.append((classLabel, name))

    # save_legend_only(OUT_DIR / "scene_legend.png", legend_valid)
    only_legend(outputPath / "scene_legend_horizontal.png", legend_gt, ncols=4)