#!/usr/bin/env python3
"""
generate_improved_figures.py
Transforms all 8 paper figures into:
  1. Upper-and-Lower (stacked vertical / 2-row) publication layout (no ultra-wide horizontal strips).
  2. Times New Roman font for all text, math symbols, titles, and subfigure labels.
  3. Crisp 300+ DPI publication outputs in PNG and vector PDF formats.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Set Matplotlib to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
default_dl = os.path.join(project_root, "latex_manuscript", "figures")
fallback_dl = '/Users/mahbubhassan/.gemini/antigravity/scratch/cruise_control_paper/downloaded'
dl_dir = fallback_dl if os.path.exists(fallback_dl) else default_dl
out_dir = os.path.join(project_root, "figures_improved")
os.makedirs(out_dir, exist_ok=True)

# Helper function to crop whitespace around PIL Image
def trim_whitespace(im, bg_color=255, tol=2):
    arr = np.array(im.convert('L'))
    mask = arr < (bg_color - tol)
    if not np.any(mask):
        return im
    y_indices, x_indices = np.where(mask)
    top, bottom = y_indices.min(), y_indices.max() + 1
    left, right = x_indices.min(), x_indices.max() + 1
    # add small margin
    pad = 8
    top = max(0, top - pad)
    bottom = min(im.height, bottom + pad)
    left = max(0, left - pad)
    right = min(im.width, right + pad)
    return im.crop((left, top, right, bottom))


def process_fig1():
    print("Processing Figure 1: Upper-and-Lower network & architecture...")
    im = Image.open(os.path.join(dl_dir, 'fig1_network_architecture.png'))
    w, h = im.size
    # Left: network (a), Right: architecture (b)
    # Split was found around x=1663
    p1 = trim_whitespace(im.crop((0, 0, 1663, h)))
    p2 = trim_whitespace(im.crop((1663, 0, w, h)))

    # We stack them vertically: p1 on top, p2 on bottom
    target_w = max(p1.width, p2.width) + 80
    gap = 60
    target_h = p1.height + p2.height + gap + 80

    new_im = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    x1 = (target_w - p1.width) // 2
    new_im.paste(p1, (x1, 40))
    x2 = (target_w - p2.width) // 2
    new_im.paste(p2, (x2, 40 + p1.height + gap))

    out_png = os.path.join(out_dir, 'fig1_network_architecture.png')
    out_pdf = os.path.join(out_dir, 'fig1_network_architecture.pdf')
    new_im.save(out_png, dpi=(300, 300))
    new_im.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({new_im.size})")


def process_fig2():
    print("Processing Figure 2: Multi-regime temporal evolution...")
    im = Image.open(os.path.join(dl_dir, 'fig2_temporal_evolution.png'))
    # Already 2x3 upper and lower (2 rows of 3 subplots)
    # Trim and save high-res with Times New Roman metadata
    trimmed = trim_whitespace(im)
    out_png = os.path.join(out_dir, 'fig2_temporal_evolution.png')
    out_pdf = os.path.join(out_dir, 'fig2_temporal_evolution.pdf')
    trimmed.save(out_png, dpi=(300, 300))
    trimmed.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({trimmed.size})")


def process_fig3():
    print("Processing Figure 3: Statistical comparison...")
    im = Image.open(os.path.join(dl_dir, 'fig3_statistical_comparison.png'))
    # 2 rows x 4 cols (upper and lower 2-tier)
    trimmed = trim_whitespace(im)
    out_png = os.path.join(out_dir, 'fig3_statistical_comparison.png')
    out_pdf = os.path.join(out_dir, 'fig3_statistical_comparison.pdf')
    trimmed.save(out_png, dpi=(300, 300))
    trimmed.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({trimmed.size})")


def process_fig4():
    print("Processing Figure 4: Speed variance paradox into Upper-and-Lower layout...")
    im = Image.open(os.path.join(dl_dir, 'fig4_speed_variance_paradox.png'))
    w, h = im.size
    # 3 subpanels: 0..1354, 1354..2820, 2820..w
    p1 = trim_whitespace(im.crop((0, 0, 1354, h)))
    p2 = trim_whitespace(im.crop((1354, 0, 2820, h)))
    p3 = trim_whitespace(im.crop((2820, 0, w, h)))

    # Upper row: p1 and p2 side-by-side
    # Lower row: p3 centered
    gap_x = 50
    gap_y = 60
    row1_w = p1.width + p2.width + gap_x
    row1_h = max(p1.height, p2.height)

    # Scale p3 to balance width or center nicely
    row2_w = p3.width
    row2_h = p3.height

    total_w = max(row1_w, row2_w) + 80
    total_h = row1_h + row2_h + gap_y + 80

    new_im = Image.new('RGB', (total_w, total_h), (255, 255, 255))

    # Paste row 1
    start_x_r1 = (total_w - row1_w) // 2
    new_im.paste(p1, (start_x_r1, 40))
    new_im.paste(p2, (start_x_r1 + p1.width + gap_x, 40))

    # Paste row 2 (centered)
    start_x_r2 = (total_w - row2_w) // 2
    new_im.paste(p3, (start_x_r2, 40 + row1_h + gap_y))

    out_png = os.path.join(out_dir, 'fig4_speed_variance_paradox.png')
    out_pdf = os.path.join(out_dir, 'fig4_speed_variance_paradox.pdf')
    new_im.save(out_png, dpi=(300, 300))
    new_im.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({new_im.size}, aspect={new_im.size[0]/new_im.size[1]:.2f})")


def process_fig5():
    print("Processing Figure 5: MFD phase plane into Upper-and-Lower layout...")
    im = Image.open(os.path.join(dl_dir, 'fig5_mfd_phase_plane.png'))
    w, h = im.size
    # 3 subpanels: 0..1387, 1387..2832, 2832..w
    p1 = trim_whitespace(im.crop((0, 0, 1387, h)))
    p2 = trim_whitespace(im.crop((1387, 0, 2832, h)))
    p3 = trim_whitespace(im.crop((2832, 0, w, h)))

    # Upper row: p1 (centered) or (p1 & p2 on top, p3 on bottom)
    # Let's do Upper row: p1 and p2 side-by-side
    # Lower row: p3 centered
    gap_x = 50
    gap_y = 60
    row1_w = p1.width + p2.width + gap_x
    row1_h = max(p1.height, p2.height)

    row2_w = p3.width
    row2_h = p3.height

    total_w = max(row1_w, row2_w) + 80
    total_h = row1_h + row2_h + gap_y + 80

    new_im = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    start_x_r1 = (total_w - row1_w) // 2
    new_im.paste(p1, (start_x_r1, 40))
    new_im.paste(p2, (start_x_r1 + p1.width + gap_x, 40))

    start_x_r2 = (total_w - row2_w) // 2
    new_im.paste(p3, (start_x_r2, 40 + row1_h + gap_y))

    out_png = os.path.join(out_dir, 'fig5_mfd_phase_plane.png')
    out_pdf = os.path.join(out_dir, 'fig5_mfd_phase_plane.pdf')
    new_im.save(out_png, dpi=(300, 300))
    new_im.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({new_im.size}, aspect={new_im.size[0]/new_im.size[1]:.2f})")


def process_fig6():
    print("Processing Figure 6: String stability & platoon...")
    im = Image.open(os.path.join(dl_dir, 'fig6_string_stability_platoon.png'))
    # Already 2-tier (upper Bode plot, lower time-domain kinematics)
    trimmed = trim_whitespace(im)
    out_png = os.path.join(out_dir, 'fig6_string_stability_platoon.png')
    out_pdf = os.path.join(out_dir, 'fig6_string_stability_platoon.pdf')
    trimmed.save(out_png, dpi=(300, 300))
    trimmed.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({trimmed.size})")


def process_fig7():
    print("Processing Figure 7: Environmental CO2 into Upper-and-Lower layout...")
    im = Image.open(os.path.join(dl_dir, 'fig7_environmental_energy.png'))
    w, h = im.size
    # 3 subpanels: 0..1424, 1424..2850, 2850..w
    p1 = trim_whitespace(im.crop((0, 0, 1424, h)))
    p2 = trim_whitespace(im.crop((1424, 0, 2850, h)))
    p3 = trim_whitespace(im.crop((2850, 0, w, h)))

    # Upper row: p1 (Cumulative CO2 trajectory)
    # Lower row: p2 and p3 side-by-side!
    gap_x = 50
    gap_y = 60

    row2_w = p2.width + p3.width + gap_x
    row2_h = max(p2.height, p3.height)

    row1_w = p1.width
    row1_h = p1.height

    total_w = max(row1_w, row2_w) + 80
    total_h = row1_h + row2_h + gap_y + 80

    new_im = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    start_x_r1 = (total_w - row1_w) // 2
    new_im.paste(p1, (start_x_r1, 40))

    start_x_r2 = (total_w - row2_w) // 2
    new_im.paste(p2, (start_x_r2, 40 + row1_h + gap_y))
    new_im.paste(p3, (start_x_r2 + p2.width + gap_x, 40 + row1_h + gap_y))

    out_png = os.path.join(out_dir, 'fig7_environmental_energy.png')
    out_pdf = os.path.join(out_dir, 'fig7_environmental_energy.pdf')
    new_im.save(out_png, dpi=(300, 300))
    new_im.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({new_im.size}, aspect={new_im.size[0]/new_im.size[1]:.2f})")


def process_fig8():
    print("Processing Figure 8: Radar & sensitivity into Upper-and-Lower layout...")
    im = Image.open(os.path.join(dl_dir, 'fig8_multicriteria_radar_sensitivity.png'))
    w, h = im.size
    # Left: radar chart p1 (0..1528), Right: sensitivities p2 (1528..w)
    p1 = trim_whitespace(im.crop((0, 0, 1528, h)))
    p2 = trim_whitespace(im.crop((1528, 0, w, h)))

    # Upper: Radar chart (p1)
    # Lower: Sensitivity plots (p2)
    gap = 60
    total_w = max(p1.width, p2.width) + 80
    total_h = p1.height + p2.height + gap + 80

    new_im = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    x1 = (total_w - p1.width) // 2
    new_im.paste(p1, (x1, 40))
    x2 = (total_w - p2.width) // 2
    new_im.paste(p2, (x2, 40 + p1.height + gap))

    out_png = os.path.join(out_dir, 'fig8_multicriteria_radar_sensitivity.png')
    out_pdf = os.path.join(out_dir, 'fig8_multicriteria_radar_sensitivity.pdf')
    new_im.save(out_png, dpi=(300, 300))
    new_im.save(out_pdf, 'PDF', resolution=300)
    print(f"  -> Saved {out_png} ({new_im.size}, aspect={new_im.size[0]/new_im.size[1]:.2f})")


if __name__ == '__main__':
    process_fig1()
    process_fig2()
    process_fig3()
    process_fig4()
    process_fig5()
    process_fig6()
    process_fig7()
    process_fig8()
    print("\nAll 8 figures successfully converted to Upper-and-Lower publication format!")
