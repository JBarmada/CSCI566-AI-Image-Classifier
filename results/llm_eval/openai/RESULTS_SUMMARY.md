# GPT-6.2 Sol GenImage Evaluation Results

**Integration review:** The original outputs below are preserved. The pilot
comparison mixed 88 valid GPT images with 90-image Gemini/Qwen summaries, and
the ViT discovery selected an old ADM pilot with only all-real intersections.
Use the [corrected offline comparisons](integration_review/README.md) for those
two analyses. Raw per-view GPT/Claude results have not been changed. The original
request-attempt log linked below is not committed, so its retry/usage provenance
remains to be independently checked.

Run date: 2026-11-26
Model requested and returned: `gpt-6.2-sol`
API: OpenAI Responses API; `detail=auto`, `max_output_tokens=128`, temperature omitted, no tools, `store=false`. The prompt and decision threshold were fixed before scoring; scores `p_fake >= 50` are classified as fake. `p_fake` is a model-reported score, not a calibrated probability.

## Dataset and run completion

The experiment uses the exact 630 unique images in Claude's shared list: 70 real and 560 fake images (70 from each of eight generators). The same images were evaluated in three views, for 1,890 planned image/view requests. These are repeated measurements on 630 images, not 1,890 independent images.

The log contains 1,893 API attempts: 1,890 unique image/view/configuration keys and three retry attempts. All 1,890 keys have a final response: 1,844 valid scores (97.6%) and 46 invalid outputs (2.4%). There were no refusals and no requests left missing; three API failure attempts were retried successfully. Metrics below use valid scores only and retain the expected request counts for coverage.

## Overall metrics by view

The dataset is imbalanced: 70 real images (11.1%) and 560 fake images (88.9%) per view. Confusion counts and threshold metrics use valid responses only. `n valid` is the denominator for accuracy, precision, recall, specificity, F1, and balanced accuracy; AUROC also uses valid responses.

| View | Valid / expected | AUROC | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | TN | FP | FN | TP | Real FPR | Fake miss rate | Valid-score coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 617 / 630 | 0.9910 | 0.6029 | 1.0000 | 0.5521 | 1.0000 | 0.7114 | 0.7761 | 70 | 0 | 245 | 302 | 0.0% | 44.8% | 97.9% |
| Square crop | 615 / 630 | 0.9881 | 0.6016 | 1.0000 | 0.5505 | 1.0000 | 0.7101 | 0.7752 | 70 | 0 | 245 | 300 | 0.0% | 45.0% | 97.6% |
| 224×224 | 612 / 630 | 0.9307 | 0.3513 | 1.0000 | 0.2675 | 1.0000 | 0.4221 | 0.6338 | 70 | 0 | 397 | 145 | 0.0% | 73.2% | 97.1% |

The high AUROC coexists with low recall at the fixed 50 threshold: many fakes received scores below threshold. No valid real image was labeled fake in any view.

## Per-generator results

Each generator row compares that generator's valid fake images with the same set of up to 70 valid real images. Thus real images recur across rows; these per-generator results are not independent samples. Values are fake recall and AUROC.

| Generator | Original recall / AUROC | Square recall / AUROC | 224×224 recall / AUROC |
|---|---:|---:|---:|
| ADM | 19.1% / 0.9629 | 19.1% / 0.9457 | 16.2% / 0.8380 |
| BigGAN | 22.1% / 0.9956 | 21.4% / 0.9920 | 24.6% / 0.9843 |
| Midjourney | 88.1% / 0.9835 | 86.8% / 0.9892 | 39.7% / 0.9286 |
| VQDM | 35.7% / 0.9915 | 30.3% / 0.9850 | 20.3% / 0.9531 |
| glide | 27.1% / 0.9981 | 26.1% / 0.9963 | 25.7% / 0.9696 |
| Stable Diffusion 1.4 | 89.7% / 1.0000 | 91.4% / 1.0000 | 33.3% / 0.9353 |
| Stable Diffusion 1.5 | 83.8% / 0.9976 | 83.8% / 0.9991 | 28.4% / 0.9183 |
| Wukong | 77.9% / 0.9985 | 81.8% / 0.9971 | 26.2% / 0.9155 |

Coverage varies by generator because invalid responses are excluded from valid-score metrics; see the linked CSV for exact denominators.

## Baselines and comparisons

### Shape-only baseline

Using original-view dimensions and classifying square images as fake gives AUROC 0.9929 and accuracy 0.9984 (TN=69, FP=1, FN=0, TP=560). In this dataset all fakes are square while 69 of 70 real images are non-square. The original-view GPT AUROC is close to this baseline, so it is not evidence by itself of semantic AI-image detection. The square crop addresses this shape shortcut, while the 224×224 view also changes resolution and image content.

### Claude

Comparisons are on exact matching source-image IDs and views with valid responses from both models.

| View | Matched unique images | Score correlation | Threshold agreement | GPT AUROC | Claude AUROC |
|---|---:|---:|---:|---:|---:|
| Original | 617 | 0.7463 | 76.3% | 0.9910 | 0.9998 |
| Square crop | 615 | 0.7607 | 78.0% | 0.9881 | 0.9963 |
| 224×224 | 612 | 0.5447 | 67.5% | 0.9307 | 0.9521 |

### Gemini and Qwen pilot

The pilot intersection contains 90 unique images. GPT returned valid scores for 88 (97.8% coverage); both comparisons below use those 88 matched images. Comparisons are original view only. The saved comparison records that source-byte encodings differ, client pixel equivalence is not verified there, and API-internal preprocessing cannot be observed. Do not treat these as fully controlled identical-model-input comparisons.

| Comparison | Matched images | Score correlation | Threshold agreement | GPT AUROC | Other model AUROC |
|---|---:|---:|---:|---:|---:|
| GPT vs Gemini | 88 | 0.5274 | 70.5% | 0.9949 | 0.9788 |
| GPT vs Qwen (Groq) | 88 | 0.1826 | 58.0% | 0.9949 | 0.8319 |

The legacy pilot task-row table contains 160 rows but only 90 unique images because real images recur across generator tasks. It is included separately in the metrics folder and should not be interpreted as 160 independent observations.

### CNN/ViT

The exact-hash comparison found 24 saved clean ViT test exports, covering 72 matched rows in total. Individual export intersections are only 3–4 images, too small for a meaningful aggregate performance claim. The export's saved threshold and holdout/provenance fields are retained in the comparison CSV. Input-preparation equivalence is not established, and API-internal preprocessing is unobservable. No matching CNN prediction export was available.


## Interpretation and limitations

- The 224×224 results are lower than original and square-view results, but this transformation changes more than size. Sending a 224×224 file does not guarantee identical internal preprocessing across models or match CNN/ViT normalization.
- The fixed 50 threshold and prompt were not tuned after seeing results.
- Invalid responses are retained and reduce valid-score coverage; they are not silently replaced or scored.
- The sample is a fixed curated subset, not the separate 2,520-image manifest or a population-calibrated probability study.
- The API log uses `store=false`, which does not itself disable default abuse-monitoring retention. Review the account's data controls before sharing or reusing submitted images.

## Result files

- [Per-view metrics](./metrics/openai_view_metrics.csv)
- [Per-generator metrics](./metrics/openai_per_generator_metrics.csv)
- [Response status and coverage](./metrics/openai_response_status.csv)
- [Predictions](./metrics/openai_predictions.csv)
- [Shape-only baseline](./metrics/openai_shape_baseline_metrics.csv)
- [GPT vs Claude](./metrics/openai_vs_claude_matched.csv)
- [GPT vs Gemini/Qwen pilot](./metrics/openai_vs_pilot_models.csv)
- [Legacy pilot task-row metrics](./metrics/pilot_legacy_task_row_metrics.csv)
- [GPT vs saved ViT predictions](./metrics/openai_vs_vit_matched.csv)
- [Token and available cost summary](./metrics/openai_cost_summary.csv)
- [Raw request log](./logs/requests_gpt-6.2-sol.jsonl)
