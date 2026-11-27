"""Build a 48 x 36 inch landscape poster (editable SVG and print PDF)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from build_figures import diversity, jpeg, presentation, face, INK, MUTED, BLUE

OUT = Path(__file__).resolve().parent
BG = "#F2F6F8"
TEAL = "#155E70"
ACCENT = "#B85F28"


def panel(fig, x, y, w, h, fill="white"):
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure,
                                 facecolor=fill, edgecolor="none", zorder=-1))


def txt(fig, x, y, text, size=22, color=INK, weight="normal"):
    return fig.text(x, y, text, fontsize=size, color=color, weight=weight,
                    va="top", ha="left", linespacing=1.38)


def section(fig, x, number, title, finding):
    txt(fig, x, .817, number, 44, TEAL, "bold")
    txt(fig, x+.033, .818, title, 40, INK, "bold")
    txt(fig, x, .769, finding, 28, MUTED)


def add_chart(fig, fn, bounds, factor=1.55):
    ax = fig.add_axes(bounds)
    fn(ax)
    for t in ax.findobj(Text):
        t.set_fontsize(t.get_fontsize()*factor)
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth()*1.4)
        line.set_markersize(line.get_markersize()*1.3)
    return ax


def make_poster():
    fig = plt.figure(figsize=(48,36), facecolor=BG)
    panel(fig, 0, .852, 1, .148, INK)
    txt(fig, .024, .976, "How well do AI-image detectors generalize", 77, "white", "bold")
    txt(fig, .024, .917, "when generators, presentation, and manipulation change?", 48, "white")
    txt(fig, .024, .876, "Shriya Rajesh  •  Jeslyn Yang  •  Aditya Lanjewar  •  Jad Barmada  •  Saisha Hongal  •  Medini Patil  •  Rachel Lin  |  USC CSCI 566", 26, "#D2E5EE")
    for x in [.016, .345, .674]:
        panel(fig, x, .118, .31, .72)
    section(fig, .028, "01", "Generator change",
            "More generator types at the same image\nbudget improve clean test accuracy.")
    section(fig, .357, "02", "Presentation change",
            "Augmentation changes error balance;\n224-pixel input also reduces VLM recall.")
    section(fig, .686, "03", "Manipulation change",
            "Detectors that perform well on generated\nimages can fail differently on face crops.")
    add_chart(fig, diversity, [.066,.487,.237,.226])
    txt(fig,.028,.444,"192 matched runs per detector",36,TEAL,"bold")
    txt(fig,.028,.407,
        "Eight unseen generators × four diversity levels ×\nthree dataset repeats × two training modes.\n1,400 real + 1,400 fake images per training run.\n"
        "Same validation/test manifests; threshold 0.5.\nResNet-50 fine-tuned vs frozen CLIP ViT-L/14 + head.",26)
    txt(fig,.028,.301,"Matched frontier comparison (GenImage)",32,TEAL,"bold")
    txt(fig,.028,.265,"Balanced accuracy, shared 612 sources:",27,INK,"bold")
    txt(fig,.028,.237,"ViT 89.4%  •  CNN 85.6%  •  Claude 77.8%  •  GPT 63.4%",27,INK)
    txt(fig,.028,.184,"Baseline k=7/repeat-0; detector224 VLM scores.\nEqual-weight holdouts; 18 GPT invalid fakes omitted.\nPipelines differ in architecture and pretraining.",24,MUTED)
    add_chart(fig, jpeg, [.44,.552,.194,.17], factor=1.48)
    txt(fig,.357,.519,"JPEG60: fewer false alarms, more missed fakes.",25,INK,"bold")
    txt(fig,.357,.49,"All 8 holdouts; k=7. ViT AUROC ≈ 0.85 in both.\nResize accuracy rises: CNN 50→69%; ViT 57→82%.",23,MUTED)
    add_chart(fig, presentation, [.393,.24,.24,.19])
    txt(fig,.357,.213,"590 paired sources (520 fake, 70 real). Both\nVLMs flag 0/70 reals in all views. 40 sources\nexcluded for at least one invalid GPT score.",24,MUTED)
    add_chart(fig, face, [.726,.452,.231,.26], factor=1.55)
    txt(fig,.686,.422,"189 matched face crops",35,TEAL,"bold")
    txt(fig,.686,.379,
        "54 real + 135 manipulated (27 per method).\nCNN flags 53/54 real crops; GPT detects 16/135\nfakes. Balanced accuracy is ~50% for both.\n"
        "ViT and Claude also falsely flag 42/54 reals.",27)
    txt(fig,.686,.255,"*Supervised adapted head: 67.8% balanced accuracy",26,ACCENT,"bold")
    txt(fig,.686,.216,"Five methods: SimSwap, FOMM, Wav2Lip,\nInSwapper, PIRender. 27 recorded video-pair groups.\nIdentity and mirror labels remain unverified; adapted\nhead trained on three methods (prior test inspection).",25,MUTED)
    panel(fig,.016,.022,.968,.08,INK)
    txt(fig,.028,.091,"Takeaway",30,"#A4DDE4","bold")
    txt(fig,.119,.092,"Generalization is conditional: report the test domain, image preparation, and both class error rates.",36,"white","bold")
    txt(fig,.028,.047,"Context: GenImage (Zhu et al., 2023); Community Forensics (Park & Owens, 2025); Fake or JPEG? (Grommelt et al., 2024); DF40 (Yan et al., 2024).    •    Results & reproducibility: github.com/JBarmada/CSCI566-AI-Image-Classifier",20,"#D2E5EE")
    fig.savefig(OUT/"poster_review.pdf")
    fig.savefig(OUT/"poster_review.svg")
    fig.savefig(OUT/"poster_review.png",dpi=45)
    plt.close(fig)


if __name__ == "__main__":
    make_poster()
