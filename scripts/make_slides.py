"""
Generate Project_Progress_Feb19.pptx
Run: python scripts/make_slides.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY       = RGBColor(0x0D, 0x1B, 0x3E)   # dark navy background
ACCENT     = RGBColor(0x00, 0x8B, 0xD8)   # bright blue accent
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xD0, 0xD8, 0xE8)
GREEN      = RGBColor(0x2E, 0xCC, 0x71)
ORANGE     = RGBColor(0xF3, 0x9C, 0x12)
SLIDE_W    = Inches(13.33)
SLIDE_H    = Inches(7.5)


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_bg(slide, color: RGBColor):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, fill_color, alpha=None):
    shape = slide.shapes.add_shape(1, l, t, w, h)  # MSO_SHAPE_TYPE.RECTANGLE
    shape.line.fill.background()
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    return shape


def add_text(slide, text, l, t, w, h,
             font_size=18, bold=False, color=WHITE,
             align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_bullet_box(slide, items, l, t, w, h,
                   font_size=16, color=WHITE, bullet="▸ ", indent_items=None):
    """Add a textbox with bullet points. indent_items = set of indices to indent."""
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    indent_items = indent_items or set()

    for i, item in enumerate(items):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        prefix = "    ◦ " if i in indent_items else bullet
        run = p.add_run()
        run.text = prefix + item
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        p.space_before = Pt(4)
    return txBox


def slide_header(slide, title, subtitle=None):
    """Accent bar + title at top of slide."""
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.07), ACCENT)
    add_text(slide, title,
             Inches(0.5), Inches(0.18), Inches(12), Inches(0.65),
             font_size=28, bold=True, color=WHITE)
    if subtitle:
        add_text(slide, subtitle,
                 Inches(0.5), Inches(0.82), Inches(12), Inches(0.4),
                 font_size=14, color=LIGHT_GRAY, italic=True)
    # thin divider
    add_rect(slide, Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.03), ACCENT)


# ── Build Presentation ────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H
blank = prs.slide_layouts[6]   # completely blank layout


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank)
set_bg(s1, NAVY)

# Top accent bar
add_rect(s1, 0, 0, SLIDE_W, Inches(0.12), ACCENT)

# Left accent stripe
add_rect(s1, 0, 0, Inches(0.12), SLIDE_H, ACCENT)

# Course tag
add_text(s1, "Advanced Machine Learning  ·  Spring 2026",
         Inches(0.4), Inches(0.25), Inches(12), Inches(0.4),
         font_size=13, color=LIGHT_GRAY, italic=True)

# Main title
add_text(s1,
         "Label-Efficient Disaster Response\nUsing Foundation Models",
         Inches(0.5), Inches(1.6), Inches(12.3), Inches(2.2),
         font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# Subtitle
add_text(s1,
         "Building Damage Detection with Prithvi & Vision Transformers on xBD",
         Inches(0.5), Inches(3.7), Inches(12.3), Inches(0.6),
         font_size=18, color=ACCENT, align=PP_ALIGN.CENTER, italic=True)

# Divider
add_rect(s1, Inches(3.5), Inches(4.5), Inches(6.3), Inches(0.04), ACCENT)

# Team & date
add_text(s1,
         "Sakhi Patel   ·   Vivek Vanera   ·   Mulya Patel",
         Inches(0.5), Inches(4.7), Inches(12.3), Inches(0.5),
         font_size=17, color=WHITE, align=PP_ALIGN.CENTER, bold=True)

add_text(s1, "February 19, 2026",
         Inches(0.5), Inches(5.3), Inches(12.3), Inches(0.4),
         font_size=14, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

# Bottom bar
add_rect(s1, 0, Inches(7.3), SLIDE_W, Inches(0.2), ACCENT)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem Statement
# ════════════════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank)
set_bg(s2, NAVY)
slide_header(s2, "Problem Statement",
             "Why label-efficient damage detection matters")

# Left column — Problem
add_rect(s2, Inches(0.4), Inches(1.4), Inches(5.8), Inches(0.38), ACCENT)
add_text(s2, "The Challenge",
         Inches(0.5), Inches(1.42), Inches(5.6), Inches(0.36),
         font_size=15, bold=True, color=WHITE)

add_bullet_box(s2, [
    "Disasters strike with no warning — rapid damage assessment is critical",
    "Manual inspection of satellite imagery is slow & expensive",
    "Deep learning models require thousands of labeled examples",
    "Expert labeling of disaster imagery is scarce and costly",
], Inches(0.4), Inches(1.85), Inches(5.8), Inches(3.0), font_size=14)

# Right column — Dataset
add_rect(s2, Inches(6.9), Inches(1.4), Inches(5.9), Inches(0.38), ORANGE)
add_text(s2, "xBD Dataset",
         Inches(7.0), Inches(1.42), Inches(5.7), Inches(0.36),
         font_size=15, bold=True, color=WHITE)

add_bullet_box(s2, [
    "850,000+ building polygons",
    "19 disaster events worldwide",
    "~0.3 m/pixel satellite imagery",
    "Pre & post-disaster image pairs",
    "4-class damage labels:",
    "No damage  /  Minor  /  Major  /  Destroyed",
], Inches(6.9), Inches(1.85), Inches(5.9), Inches(3.0),
   font_size=14, indent_items={5})

# Research gap box
add_rect(s2, Inches(0.4), Inches(5.1), Inches(12.4), Inches(1.7), RGBColor(0x14, 0x2A, 0x5E))
add_text(s2, "Research Gap",
         Inches(0.6), Inches(5.18), Inches(4), Inches(0.4),
         font_size=14, bold=True, color=ACCENT)
add_text(s2,
         "Can geospatial foundation models (pre-trained on satellite imagery) "
         "achieve strong damage detection with significantly fewer labeled examples "
         "than training from scratch?",
         Inches(0.6), Inches(5.55), Inches(12.0), Inches(1.1),
         font_size=14, color=WHITE, italic=True)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Our Approach
# ════════════════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank)
set_bg(s3, NAVY)
slide_header(s3, "Our Approach", "Two-phase experimental pipeline")

# Phase boxes
phases = [
    ("Phase 1\nBaseline", "ResNet-18\nViT-B/16\nFull supervision", ACCENT),
    ("Phase 2\nFoundation", "Prithvi-100M\nFine-tuning\nFew-shot", ORANGE),
    ("Phase 3\nLabel\nEfficiency", "10% / 25% / 50%\nlabeled data\nComparison", GREEN),
]

for i, (title, body, color) in enumerate(phases):
    x = Inches(0.5 + i * 4.25)
    add_rect(s3, x, Inches(1.4), Inches(3.9), Inches(0.5), color)
    add_text(s3, title, x + Inches(0.1), Inches(1.42),
             Inches(3.7), Inches(0.46), font_size=14, bold=True, color=WHITE)
    add_rect(s3, x, Inches(1.9), Inches(3.9), Inches(2.2), RGBColor(0x14, 0x2A, 0x5E))
    add_text(s3, body, x + Inches(0.15), Inches(2.0),
             Inches(3.6), Inches(2.0), font_size=15, color=WHITE, align=PP_ALIGN.CENTER)

    # Arrow between boxes
    if i < 2:
        add_text(s3, "→", Inches(4.3 + i * 4.25), Inches(2.6),
                 Inches(0.5), Inches(0.5), font_size=28, bold=True, color=ACCENT,
                 align=PP_ALIGN.CENTER)

# Research questions
add_rect(s3, Inches(0.4), Inches(4.3), Inches(12.4), Inches(0.38), RGBColor(0x1A, 0x35, 0x6E))
add_text(s3, "Research Questions",
         Inches(0.5), Inches(4.32), Inches(5), Inches(0.36),
         font_size=14, bold=True, color=ACCENT)

add_bullet_box(s3, [
    "RQ1: Does geospatial pre-training (Prithvi) outperform ImageNet pre-training (ViT) on satellite damage detection?",
    "RQ2: How does label efficiency scale? At what labeled-data fraction does Prithvi match a fully-supervised ResNet?",
    "RQ3: Does bi-temporal (pre+post) input consistently improve over post-only classification?",
], Inches(0.4), Inches(4.75), Inches(12.4), Inches(2.5), font_size=13, bullet="Q  ")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Implementation Progress
# ════════════════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank)
set_bg(s4, NAVY)
slide_header(s4, "Implementation Progress", "Status as of February 19, 2026")

milestones = [
    (True,  "GitHub Repository",    "github.com/sakhi20/disaster-response-label-efficient\nFull project structure: models/, utils/, configs/, scripts/, paper/"),
    (True,  "xBD Dataset",          "~53 GB downloaded (6 part files)\nExtracting locally — 850K+ building annotations"),
    (True,  "Baseline Script",      "scripts/train_baseline.py — ResNet-18 on xBD\nClass-weighted loss, AdamW + cosine LR, per-class F1"),
    (True,  "Data Pipeline",        "utils/data_loader.py — XBDTileDataset\nPre/post image pairs, JSON label parsing, majority-class labeling"),
    (False, "Baseline Results",     "Training pending dataset extraction\nTarget: ~70% accuracy on 1K subset"),
    (False, "Prithvi Fine-tuning",  "Planned: Week of Feb 23\nHuggingFace weights download + ViT head replacement"),
]

for i, (done, title, desc) in enumerate(milestones):
    col = i % 2
    row = i // 2
    x = Inches(0.4 + col * 6.5)
    y = Inches(1.4 + row * 1.9)
    w = Inches(6.1)

    bg_color = RGBColor(0x0A, 0x2A, 0x1A) if done else RGBColor(0x1A, 0x1A, 0x35)
    add_rect(s4, x, y, w, Inches(1.7), bg_color)

    # Status dot
    dot_color = GREEN if done else ORANGE
    add_rect(s4, x + Inches(0.12), y + Inches(0.55), Inches(0.22), Inches(0.22), dot_color)

    status_text = "✓  COMPLETE" if done else "◷  IN PROGRESS"
    status_color = GREEN if done else ORANGE
    add_text(s4, status_text,
             x + Inches(0.42), y + Inches(0.08), Inches(2.5), Inches(0.35),
             font_size=11, bold=True, color=status_color)

    add_text(s4, title,
             x + Inches(0.15), y + Inches(0.38), Inches(5.7), Inches(0.38),
             font_size=15, bold=True, color=WHITE)

    add_text(s4, desc,
             x + Inches(0.15), y + Inches(0.82), Inches(5.8), Inches(0.82),
             font_size=11, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Code Highlights
# ════════════════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank)
set_bg(s5, NAVY)
slide_header(s5, "Code Highlights", "Key implementation components")

# Left — Dataset class
add_rect(s5, Inches(0.4), Inches(1.4), Inches(5.9), Inches(0.38), ACCENT)
add_text(s5, "utils/data_loader.py  —  XBDTileDataset",
         Inches(0.5), Inches(1.42), Inches(5.7), Inches(0.36),
         font_size=13, bold=True, color=WHITE)

code1 = (
    "class XBDTileDataset(Dataset):\n"
    "  def __init__(self, samples, transform):\n"
    "    self.samples = samples  # (img_path, label)\n"
    "    self.transform = transform\n\n"
    "  def __getitem__(self, idx):\n"
    "    img = Image.open(path).convert('RGB')\n"
    "    return self.transform(img), label\n\n"
    "# Label = majority damage class per tile\n"
    "# Parsed from xBD JSON polygon annotations"
)
add_rect(s5, Inches(0.4), Inches(1.82), Inches(5.9), Inches(3.1), RGBColor(0x08, 0x10, 0x20))
add_text(s5, code1,
         Inches(0.5), Inches(1.88), Inches(5.7), Inches(2.9),
         font_size=11, color=RGBColor(0xAA, 0xDD, 0xFF))

# Right — Model
add_rect(s5, Inches(6.9), Inches(1.4), Inches(5.9), Inches(0.38), ORANGE)
add_text(s5, "scripts/train_baseline.py  —  ResNet-18",
         Inches(7.0), Inches(1.42), Inches(5.7), Inches(0.36),
         font_size=13, bold=True, color=WHITE)

code2 = (
    "def build_resnet18(num_classes=4):\n"
    "  model = resnet18(weights=IMAGENET1K_V1)\n"
    "  model.fc = nn.Sequential(\n"
    "    nn.Dropout(p=0.3),\n"
    "    nn.Linear(512, num_classes),\n"
    "  )\n"
    "  return model\n\n"
    "# Class-weighted CrossEntropyLoss\n"
    "# weights = [0.5, 1.5, 2.0, 2.5]\n"
    "# AdamW + CosineAnnealingLR"
)
add_rect(s5, Inches(6.9), Inches(1.82), Inches(5.9), Inches(3.1), RGBColor(0x08, 0x10, 0x20))
add_text(s5, code2,
         Inches(7.0), Inches(1.88), Inches(5.7), Inches(2.9),
         font_size=11, color=RGBColor(0xAA, 0xDD, 0xFF))

# Bottom — pipeline summary
add_rect(s5, Inches(0.4), Inches(5.1), Inches(12.4), Inches(1.8), RGBColor(0x0D, 0x22, 0x44))
add_text(s5, "Training Pipeline",
         Inches(0.6), Inches(5.15), Inches(4), Inches(0.38),
         font_size=14, bold=True, color=ACCENT)
add_bullet_box(s5, [
    "Augmentation: RandomFlip + RandomRotation(15°) + ColorJitter → Resize(224×224) → ImageNet Normalize",
    "Optimizer: AdamW (lr=1e-3, wd=1e-4)  |  Scheduler: CosineAnnealingLR  |  Epochs: 10  |  Batch: 32",
    "Metrics: Per-class F1, Macro F1, Accuracy, xView2 Harmonic Mean Score",
], Inches(0.5), Inches(5.55), Inches(12.2), Inches(1.3), font_size=12)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Future Directions
# ════════════════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank)
set_bg(s6, NAVY)
slide_header(s6, "Future Directions: SAR Integration")

# SAR icon/visual placeholder rect
add_rect(s6, Inches(8.5), Inches(1.8), Inches(4.0), Inches(3.5), RGBColor(0x1A, 0x35, 0x6E))
add_text(s6, "📡", Inches(9.8), Inches(2.3), Inches(1.5), Inches(1.5), font_size=60, align=PP_ALIGN.CENTER)
add_text(s6, "Synthetic Aperture Radar", Inches(8.5), Inches(4.2), Inches(4.0), Inches(0.5), 
         font_size=16, color=ACCENT, align=PP_ALIGN.CENTER, bold=True)

# Main content
add_bullet_box(s6, [
    "Extend to SAR (Synthetic Aperture Radar) imagery for all-weather disaster monitoring",
    "SAR advantages: Works through clouds, smoke, darkness - critical during active disasters",
    "Challenge: Multi-view SAR has different appearance patterns vs. optical imagery",
    "Opportunity: Combine optical + SAR for robust multi-modal damage detection pipeline",
], Inches(0.5), Inches(1.8), Inches(7.5), Inches(4.0), font_size=18)

# Expert Citation
add_rect(s6, Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.8), RGBColor(0x0A, 0x1A, 0x35))
add_text(s6, "Collaborative Insight", Inches(0.7), Inches(5.85), Inches(3), Inches(0.3), font_size=11, bold=True, color=ACCENT)
add_text(s6, "Special thanks to Prof. Mikhail Gilman (NCSU Mathematics) for high-frequency SAR expertise and multi-view geometry insights.", 
         Inches(0.7), Inches(6.15), Inches(11.5), Inches(0.45), font_size=12, italic=True, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Next Steps
# ════════════════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank)
set_bg(s7, NAVY)
slide_header(s7, "Next Steps & Timeline")

timeline = [
    ("Week 1\nFeb 17–23",  ACCENT,  [
        "Extract xBD dataset & run baseline training",
        "Establish ResNet-18 accuracy benchmark",
        "Set up W&B experiment tracking",
    ]),
    ("Week 2\nFeb 24 – Mar 2", ORANGE, [
        "Download Prithvi-100M weights (HuggingFace)",
        "Implement Prithvi fine-tuning pipeline",
        "Run Prithvi vs ViT-B/16 comparison",
    ]),
    ("Week 3\nMar 3–9",    GREEN,  [
        "Label-efficiency experiments (10/25/50%)",
        "Few-shot learning evaluation",
        "Ablation: pre+post vs post-only input",
    ]),
]

for i, (week, color, items) in enumerate(timeline):
    x = Inches(0.4 + i * 4.25)
    add_rect(s7, x, Inches(1.4), Inches(3.9), Inches(0.55), color)
    add_text(s7, week, x + Inches(0.1), Inches(1.42),
             Inches(3.7), Inches(0.51), font_size=14, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s7, x, Inches(1.95), Inches(3.9), Inches(3.5), RGBColor(0x10, 0x22, 0x44))
    add_bullet_box(s7, items,
                   x + Inches(0.1), Inches(2.0),
                   Inches(3.7), Inches(3.3), font_size=13)

# Final goal
add_rect(s7, Inches(0.4), Inches(5.65), Inches(12.4), Inches(1.55), RGBColor(0x08, 0x18, 0x30))
add_text(s7, "🎯  Project Goal",
         Inches(0.6), Inches(5.72), Inches(4), Inches(0.4),
         font_size=14, bold=True, color=ACCENT)
add_text(s7,
         "Demonstrate that Prithvi (geospatial foundation model) achieves competitive damage detection "
         "accuracy with 25–50% fewer labels than a fully-supervised ResNet-18 baseline, "
         "enabling faster, cheaper disaster response pipelines.",
         Inches(0.6), Inches(6.1), Inches(12.0), Inches(1.0),
         font_size=13, color=WHITE, italic=True)


# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "Project_Progress_Feb19.pptx"
prs.save(out_path)
print(f"✓ Saved: {out_path}")
