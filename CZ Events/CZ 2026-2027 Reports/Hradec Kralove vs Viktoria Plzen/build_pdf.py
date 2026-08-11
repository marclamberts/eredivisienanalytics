"""Compile the Visuals/*.png pages into one landscape PDF, one full-bleed
image per page -- same deliverable shape as the sibling post-match reports.

Usage: python3 build_pdf.py
"""
import glob
import os

from PIL import Image
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(OUT_DIR, "Visuals")
PDF_PATH = os.path.join(OUT_DIR, "Hradec_Kralove_vs_Viktoria_Plzen_PreMatch.pdf")

PAGE_W, PAGE_H = 1920, 1080  # points, 16:9


def main():
    pages = sorted(glob.glob(os.path.join(VIS_DIR, "*.png")))
    if not pages:
        raise SystemExit("No PNGs found in Visuals/ -- run build_charts.py first")

    c = canvas.Canvas(PDF_PATH, pagesize=landscape((PAGE_H, PAGE_W)))
    for path in pages:
        img = Image.open(path)
        iw, ih = img.size
        scale = min(PAGE_W / iw, PAGE_H / ih)
        w, h = iw * scale, ih * scale
        x, y = (PAGE_W - w) / 2, (PAGE_H - h) / 2
        c.drawImage(path, x, y, width=w, height=h)
        c.showPage()
    c.save()
    print("Saved:", PDF_PATH, f"({len(pages)} pages)")


if __name__ == "__main__":
    main()
