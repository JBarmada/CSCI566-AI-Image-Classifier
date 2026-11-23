# Exploratory still-photo transfer: DeepFakeFace

CARC job `14488658` completed with exit code 0 on a Tesla P100. The existing
GenImage-trained ADM holdout, k=7, repeat-0, seed-17 CNN and frozen CLIP
ViT-L/14 heads were evaluated in baseline and augmented modes. No model or
threshold was trained or selected using these photos. The decision threshold
was fixed at 0.5. Each pipeline used its own original preprocessing.

Fifty relative paths were sampled uniformly from the sorted intersection of
the 30,000 paths in the released `wiki`, `insight`, and `inpainting` archives
using Python's random seed 5662026. The archives were pinned to revision
`eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c`. The 50 real photos and their
50 samples from each fake archive form a shared 150-image evaluation. The
first six paths per group were used for the 18-image smoke and were also
included in the full set, without tuning on smoke scores. Same-path matching
follows the dataset's released structure; it does not verify the success of
each edit, person independence, or absence of training-data overlap.

| Model | Real falsely flagged | InsightFace detected | Inpainting detected | InsightFace AUROC | Inpainting AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN baseline | 23/50 | 31/50 | 34/50 | 0.6404 | 0.6702 |
| CNN augmented | 18/50 | 24/50 | 19/50 | 0.5576 | 0.5036 |
| ViT baseline | 4/50 | 28/50 | 33/50 | 0.8452 | 0.8956 |
| ViT augmented | 5/50 | 20/50 | 20/50 | 0.7168 | 0.7712 |

False-positive counts use the same 50 real photos for both method comparisons.
AUROC compares all fake-real score pairs. This is one model configuration per
pipeline and a small, non-independent celebrity-photo sample, not a population
estimate or an architecture-only comparison. All fake labels are dataset
provenance; no individual image was manually verified as visibly manipulated.
Comparing with DF40 video crops changes source, selection, and processing at
once, so it cannot isolate the cause of the observed real-score shift.

All 150 JPEGs decoded and had distinct SHA-256 hashes. Critically, the selected
real files have 40 different raw image sizes, while all 100 fake files are
512x512. This is a class-correlated preprocessing difference and a potential
shortcut even though both models resize/crop during their own preprocessing.
Thus these AUROCs and recall values cannot establish that the detectors rely
on manipulation features. Confirm the pattern on source-matched, consistently
processed data before making a strong generalization claim. Identity and source
overlap with training data remain unverified; the inpainting generator is
related to Stable Diffusion, a family represented in GenImage.

The archived manifest contains filenames and hashes but no image bytes.
`provenance.json` records selection and transfer sizes. `ACCEPTANCE.json`
records checkpoint hashes, original-reference replays, and repeat-smoke
checks. Both CNN reference and repeat differences were zero; ViT reference
differences were at most 2.79e-8 and repeat difference was zero. Predictions
were matched to every manifest row for all four models; `summarize.py`
recomputes the metrics and verifies the evaluated manifest hash.

To recompute the published metrics in this checkout:

```powershell
cd "C:\Users\creat\OneDrive - University of Southern California\Documents\ChatGPT\CSC566 Proj\csci566_research_continuation"
python results\df40_transfer\still_face_v1\summarize.py
```

The image archives are third-party data and are not committed. Dataset-card
license labels conflict; check the [dataset card](https://huggingface.co/datasets/OpenRL/DeepFakeFace)
and original image terms before redistributing source photos. See
[dataset and error analysis](../STILL_FACE_RESEARCH.md).
