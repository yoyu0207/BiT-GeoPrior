from __future__ import annotations

from pathlib import Path


OUT_DIR = Path(r"D:/yoyu/SA_Identification/project/fig")


def svg_header(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <style>
    .title {{ font: 700 18px 'Times New Roman', serif; fill: #1f1f1f; }}
    .label {{ font: 13px 'Times New Roman', serif; fill: #1f1f1f; }}
    .small {{ font: 11px 'Times New Roman', serif; fill: #1f1f1f; }}
    .tiny {{ font: 10px 'Times New Roman', serif; fill: #4a4a4a; }}
    .box {{ stroke: #4c4c4c; stroke-width: 1.2; rx: 10; ry: 10; }}
    .soft {{ fill: #f4efe3; }}
    .cool {{ fill: #e8f2fb; }}
    .mint {{ fill: #e8f4ee; }}
    .rose {{ fill: #f8e7e7; }}
    .sand {{ fill: #f6ecd2; }}
    .gray {{ fill: #ececec; }}
    .tokStroke {{ stroke: #4c4c4c; stroke-width: 1.0; }}
    .arrow {{ stroke: #333333; stroke-width: 1.35; fill: none; marker-end: url(#arrow); }}
    .dash {{ stroke-dasharray: 4 3; }}
  </style>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L8,3 L0,6 z" fill="#333333"/>
  </marker>
</defs>
<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
'''


def svg_footer() -> str:
    return "</svg>\n"


def rect(x: float, y: float, w: float, h: float, cls: str, extra: str = "") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" ry="10" class="box {cls}" {extra}/>\n'


def text(x: float, y: float, content: str, cls: str = "label", anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{content}</text>\n'


def line(x1: float, y1: float, x2: float, y2: float, cls: str = "arrow") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>\n'


def token(x: float, y: float, w: float, h: float, fill: str, label_txt: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" ry="6" fill="{fill}" class="tokStroke"/>\n'
        + text(x + w / 2, y + h / 2 + 4, label_txt, "small")
    )


def pill(x: float, y: float, w: float, h: float, fill: str, label_txt: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" ry="{h/2}" fill="{fill}" class="tokStroke"/>\n'
        + text(x + w / 2, y + h / 2 + 4, label_txt, "small")
    )


def draw_fig2_spg() -> str:
    W, H = 1180, 520
    s = [svg_header(W, H)]
    s.append(text(590, 34, "Fig. 2  Spatial Prior Gating (SPG) module", "title"))

    # Main feature path
    s.append(rect(70, 215, 170, 64, "cool"))
    s.append(text(155, 248, "Temporal feature map", "label"))
    s.append(text(155, 267, "F ∈ R^{C×H×W}", "small"))

    s.append(rect(355, 215, 230, 64, "sand"))
    s.append(text(470, 248, "Residual spatial gating", "label"))
    s.append(text(470, 267, "F + γ · (F ⊙ A)", "small"))

    s.append(rect(700, 215, 170, 64, "mint"))
    s.append(text(785, 248, "Prior-modulated", "label"))
    s.append(text(785, 267, "feature map", "label"))

    s.append(line(240, 247, 355, 247))
    s.append(line(585, 247, 700, 247))

    # Prior branch
    s.append(rect(110, 385, 170, 62, "rose"))
    s.append(text(195, 417, "Spatial prior", "label"))
    s.append(text(195, 436, "P ∈ R^{1×H₀×W₀}", "small"))

    s.append(rect(365, 382, 165, 68, "gray"))
    s.append(text(447.5, 412, "Bilinear resize", "label"))
    s.append(text(447.5, 431, "align to H × W", "small"))

    s.append(rect(610, 350, 135, 54, "soft"))
    s.append(text(677.5, 382, "1×1 Conv", "label"))

    s.append(rect(610, 417, 135, 54, "soft"))
    s.append(text(677.5, 449, "Sigmoid", "label"))

    s.append(rect(830, 382, 170, 68, "mint"))
    s.append(text(915, 412, "Channel attention", "label"))
    s.append(text(915, 431, "A ∈ (0,1)^{C×H×W}", "small"))

    s.append(line(280, 416, 365, 416))
    s.append(line(530, 416, 610, 377))
    s.append(line(677.5, 404, 677.5, 417))
    s.append(line(745, 444, 830, 416))
    s.append(line(915, 382, 915, 305))
    s.append(line(915, 305, 520, 305))
    s.append(line(520, 305, 520, 279))

    # annotations
    s.append(text(522, 328, "attention-guided residual modulation", "tiny"))
    s.append(text(520, 188, "identity initialization: γ = 0", "tiny"))

    # equation callout
    s.append(rect(938, 120, 180, 78, "gray"))
    s.append(text(1028, 150, "Core operation", "label"))
    s.append(text(1028, 172, "A = σ(Conv₁×₁(P))", "small"))
    s.append(text(1028, 191, "F' = F + γ · (F ⊙ A)", "small"))
    s.append(line(1000, 198, 955, 382, "arrow dash"))

    return "".join(s) + svg_footer()


def draw_frame_icon(x: float, y: float, w: float, h: float, fill: str, label_txt: str) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" ry="6" fill="{fill}" class="tokStroke"/>',
        f'<rect x="{x+10}" y="{y+10}" width="{w-20}" height="{h-20}" rx="4" ry="4" fill="white" fill-opacity="0.28" stroke="#555" stroke-width="0.8"/>',
        f'<rect x="{x+24}" y="{y+18}" width="{w-48}" height="{h-36}" rx="4" ry="4" fill="none" stroke="#666" stroke-width="0.8" stroke-dasharray="3 2"/>',
        text(x + w / 2, y + h + 16, label_txt, "small"),
    ]
    return "\n".join(parts) + "\n"


def draw_fig3_overall() -> str:
    W, H = 1380, 760
    s = [svg_header(W, H)]
    s.append(text(690, 36, "Fig. 3  Overall architecture of G-OEP-BiT", "title"))

    # Inputs
    s.append(draw_frame_icon(105, 595, 150, 82, "#d7e9f8", "T1 image (2020)"))
    s.append(draw_frame_icon(330, 595, 150, 82, "#d7e9f8", "T2 image"))

    # Shared BiT encoders
    s.append(rect(90, 450, 175, 78, "soft"))
    s.append(text(177.5, 486, "Shared BiT encoder", "label"))
    s.append(text(177.5, 506, "backbone + transformer", "small"))
    s.append(rect(315, 450, 175, 78, "soft"))
    s.append(text(402.5, 486, "Shared BiT encoder", "label"))
    s.append(text(402.5, 506, "backbone + transformer", "small"))

    s.append(line(180, 595, 177.5, 528))
    s.append(line(405, 595, 402.5, 528))

    # Prior encoder branch
    s.append(rect(620, 568, 170, 78, "rose"))
    s.append(text(705, 602, "Ecological prior", "label"))
    s.append(text(705, 621, "encoder", "label"))
    s.append(text(705, 640, "from T1", "small"))
    s.append(line(255, 636, 620, 606))

    s.append(rect(615, 430, 180, 78, "mint"))
    s.append(text(705, 463, "Online prior map", "label"))
    s.append(text(705, 483, "P ∈ R^{1×H×W}", "small"))
    s.append(line(705, 568, 705, 508))

    # SPG modules
    s.append(rect(910, 430, 145, 78, "cool"))
    s.append(text(982.5, 463, "SPG on T1", "label"))
    s.append(text(982.5, 483, "F₁' = SPG(F₁, P)", "small"))
    s.append(rect(1120, 430, 145, 78, "cool"))
    s.append(text(1192.5, 463, "SPG on T2", "label"))
    s.append(text(1192.5, 483, "F₂' = SPG(F₂, P)", "small"))

    s.append(line(265, 489, 910, 466))
    s.append(line(490, 489, 1120, 466))
    s.append(line(795, 469, 910, 469))
    s.append(line(795, 469, 1120, 469))

    # Fusion and decoder
    s.append(rect(945, 285, 290, 74, "gray"))
    s.append(text(1090, 318, "Feature fusion", "label"))
    s.append(text(1090, 337, "[F₁' ; F₂']", "small"))
    s.append(line(982.5, 430, 1018, 359))
    s.append(line(1192.5, 430, 1162, 359))

    s.append(rect(960, 162, 260, 74, "sand"))
    s.append(text(1090, 194, "Change decoder", "label"))
    s.append(text(1090, 214, "Upsample + Conv layers", "small"))
    s.append(line(1090, 285, 1090, 236))

    s.append(rect(988, 60, 205, 62, "mint"))
    s.append(text(1090, 88, "Eradication probability map", "label"))
    s.append(text(1090, 108, "G-OEP-BiT output", "small"))
    s.append(line(1090, 162, 1090, 122))

    # side notes
    s.append(rect(65, 112, 245, 92, "gray"))
    s.append(text(187.5, 142, "Image-driven branch", "label"))
    s.append(text(187.5, 164, "captures bitemporal change", "small"))
    s.append(text(187.5, 184, "features with shared BiT encoder", "small"))
    s.append(line(310, 158, 390, 450, "arrow dash"))

    s.append(rect(420, 112, 270, 92, "gray"))
    s.append(text(555, 142, "Ecological prior branch", "label"))
    s.append(text(555, 164, "learns former Spartina tendency", "small"))
    s.append(text(555, 184, "from the pre-eradication image", "small"))
    s.append(line(690, 158, 705, 430, "arrow dash"))

    s.append(rect(1090, 560, 220, 88, "gray"))
    s.append(text(1200, 590, "Key idea", "label"))
    s.append(text(1200, 612, "ecological context is injected", "small"))
    s.append(text(1200, 632, "at the feature level rather than", "small"))
    s.append(text(1200, 652, "used as post-processing", "small"))
    s.append(line(1120, 560, 1035, 508, "arrow dash"))

    return "".join(s) + svg_footer()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fig2_spg_module_rse.svg").write_text(draw_fig2_spg(), encoding="utf-8")
    (OUT_DIR / "fig3_goep_bit_architecture_rse.svg").write_text(draw_fig3_overall(), encoding="utf-8")


if __name__ == "__main__":
    main()
