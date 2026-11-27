# CSCI 566 final delivery package

**Team-review drafts, November 27, 2026.** The source of truth is the
[consolidated report](../../results/df40_transfer/FINAL_REPORT_DRAFT.md)
and the [November 26 independent audit](../../results/llm_eval/team_readiness_2026-11-26/README.md).

## Open the deliverables

| Artifact | Purpose |
| --- | --- |
| [Poster PDF](poster_review.pdf) · [preview PNG](poster_review.png) · [editable SVG](poster_review.svg) | Single 48 × 36 inch landscape review poster; print size is an assumption pending course confirmation. |
| [Offline demonstration](demo.html) | Download the HTML and open it locally in a browser. Includes real saved prediction scores, no servers, GPU, images, or API calls. |
| [Presenter guide](PRESENTER_GUIDE.md) | Two-minute walkthrough, timing, key questions and backup demo plan. |
| [Final report PDF](final_report_review.pdf) · [editable Markdown](../../results/df40_transfer/FINAL_REPORT_DRAFT.md) | Full review draft with results, methods, caveats, and linked references. |
| [Contribution statement draft](CONTRIBUTIONS_REVIEW.md) | Evidence-based one-page contribution statement for all seven members to confirm. |
| [Decision register](REVIEW_DECISIONS.md) | Items for team review and the deferred user discussion. |

## Experimental evidence

The main question is how detectors generalize when generators, image
presentation and manipulation change. The poster gives equal visual emphasis
to those three tests.

1. **Generator change:** [clean CNN/ViT matched table](../../results/df40_transfer/clean_cnn_audit/matched_confidence_summary.csv),
   192 matched run pairs; 96 split manifests, 8 holdouts. The figure uses the
   four training curves; the report adds F1 and AUROC.
2. **Presentation change:** [CNN/ViT audited robustness table](../../results/df40_transfer/cnn_robustness_v100/paired_macro.csv),
   [class-specific JPEG analysis](../../results/df40_transfer/poster_assets/JPEG_ERROR_ANALYSIS.md),
   and [paired 590-source GPT/Claude comparison](../../results/llm_eval/team_readiness_2026-11-26/README.md#paired-image-presentation-comparison).
   The 612-source detector224 study and 88-source pilot are separate analyses,
   not pooled with the 590-source cohort.
3. **Manipulation change:** [matched GPT face scores](../../results/df40_transfer/openai_face/df40_five_method_v3_v2/comparison/matched_predictions.csv)
   and [Claude face scores](../../results/df40_transfer/claude_face/df40_five_method_v3_v1/comparison/matched_predictions.csv),
   all seven configurations on 189 crops; source, identity and mirror-label
   limitations in the report.
4. **Supporting findings:** [ViT calibration and error analysis](../../results/analysis/README.md),
   [still face transfer](../../results/df40_transfer/still_face_v1/README.md),
   and [adaptation review](../../results/df40_transfer/adaptation_paired_review/README.md).

## Reproduce

From the repository root using Python with pandas and matplotlib:

```powershell
python docs/final_delivery/build_figures.py
python docs/final_delivery/build_poster.py
python docs/final_delivery/build_demo.py
python docs/final_delivery/verify_metrics.py
```

The four individual figures are available in PNG and SVG:
[diversity](diversity.svg), [JPEG changes](jpeg.svg), [presentation](presentation.svg),
[face errors](face.svg). The poster can be edited as vector SVG or by changing
the source and rerunning the script. Source plots are generated anew.

For the report PDF and browser smoke check, use Node with
`marked` and `playwright` installed and Google Chrome installed at the path
in the two CJS scripts (adjust the Chrome path on other platforms).
The optional final command argument is a screenshot output path.

```powershell
node docs/final_delivery/build_report.cjs report-preview.png
node docs/final_delivery/verify_demo.cjs demo-preview.png
```

Verification: numerical verifier passed for clean, robustness, matched frontier,
all six presentation counts and all seven face configurations; figure/PDF renders inspected, poster dimensions checked (3456 ×
2592 PDF points), report text and four figures verified, report relative links
resolved, browser demo verified at threshold 0.5 and after threshold reset.
The demo reproduces 590 presentation source rows and 189 face source rows.
No new training or model API calls were needed. Original datasets are not
redistributed by this package.

## Review, not submission

The proposal and submitted midterm establish the original project scope and
authorship order. The public [course schedule](https://usc-fortis.github.io/CSCI566-F26/schedule.html)
lists poster/demo November 30 and final report December 7. Team members must
check current Piazza/Gradescope submission instructions, confirm poster
dimensions, approve the contribution statement, rehearse and complete the required peer evaluation. The [syllabus](https://usc-fortis.github.io/CSCI566-F26/syllabus.pdf) describes a roughly five-minute poster visit, individual questions, a research-paper-style report and the agreed one-page contribution statement.
