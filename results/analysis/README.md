# Confidence and Failure Analysis

For this part of the project, we wanted to look at which images the ViT model gets wrong and how confident it is in its predictions.

We used the existing predictions in [`results/vit_benchmark/predictions/`](../vit_benchmark/predictions/), so we didn't need to retrain any models. We also recalculated the test metrics and checked that they matched the recorded results for all 192 runs ([verification.json](calibration/verification.json)).

This builds on the [CNN and ViT comparison](../df40_transfer/clean_cnn_audit/README.md), which already includes the raw ECE and Brier scores for both models, along with the disagreement counts at k=7. Our ViT results match the earlier analysis (baseline ECE of 0.2816, with an average of 48.3 missed fakes and 17.2 false alarms per run at k=7).

Here, we focus more on why the ViT has a high ECE, whether recalibration helps, how the errors change with k, and whether the model keeps getting the same images wrong.

## How to Run

```bash
python code/analysis/calibration.py   # Calibration metrics and plots
python code/analysis/recalibrate.py   # Temperature scaling, Platt scaling, and thresholds
python code/analysis/errors.py        # Error analysis
python -m pytest tests/test_analysis.py
```

All of these scripts take less than a minute to run on a laptop. To also plot the images that the model gets wrong with the highest confidence, run `errors.py --image-root <GenImage_Compact>`. This requires access to the original images, which are stored on CARC.

## Main Findings

### 1. The ViT is underconfident

One thing we noticed was that the ViT rarely predicts a probability of an image being fake outside the 0.2–0.8 range, even when it makes the correct prediction. At k=7, the baseline model has a test accuracy of 88.3%, but its average confidence is only around 60%.

In the reliability plots, the predicted probabilities are consistently closer to 0.5 than the actual fraction of fake images. Almost all images in the bins below 0.5 are real, while almost all images in the bins above 0.5 are fake.

Plots: [Reliability by k](figures/reliability_by_k.png) and [Probability histograms at k=7](figures/probability_histograms_k7.png)

This helps explain the high ECE of around 0.28 from our earlier analysis. As we increase k, the baseline ECE actually increases from 0.229 to 0.282, even though the Brier score improves from 0.199 to 0.173. The model becomes more accurate with more training generators, but its confidence doesn't increase by much. This makes the gap between accuracy and confidence larger.

[Calibration metrics across k](figures/calibration_vs_k.png)

| k=7 test (averaged across 8 holdouts) | Baseline | Augmented |
|---|---:|---:|
| Accuracy | 88.3% | 82.8% |
| Mean confidence − accuracy | −0.282 | −0.246 |
| ECE | 0.282 | 0.251 |
| Brier score | 0.173 | 0.193 |

### 2. Temperature scaling works better with more training generators

We tried temperature scaling to see if we could improve the model's confidence scores without retraining it. For each run, we fitted a temperature using only the validation set and then applied that same temperature to the test set.

The fitted temperatures ranged from 0.12 to 0.15, which makes the logits about 7–8 times larger in magnitude. Temperature scaling doesn't change which side of 0.5 a prediction falls on, so the predicted labels and accuracy stay the same.

| Test results (averaged across holdouts) | Raw ECE | Scaled ECE | Raw log loss | Scaled log loss |
|---|---:|---:|---:|---:|
| k=1 baseline | 0.229 | 0.224 | 0.586 | 1.043 |
| k=1 augmented | 0.204 | 0.216 | 0.599 | 1.053 |
| k=7 baseline | 0.282 | 0.068 | 0.534 | 0.291 |
| k=7 augmented | 0.251 | 0.121 | 0.576 | 0.399 |

At k=7, temperature scaling helped both the baseline and augmented models. We also ran a paired bootstrap over the test images to check the ECE improvement. The baseline ECE decreased by 0.209 (95% CI: 0.197–0.221), while the augmented ECE decreased by 0.130 (95% CI: 0.118–0.142).

[Bootstrap results](recalibration/bootstrap_k7_raw_vs_temperature.csv)

However, at k=1 and k=2, temperature scaling nearly doubled the log loss. Since the fake images in the validation set come from generators the model has already seen during training, the model can classify them more easily. Applying a temperature fitted on those images can make the model too confident when it gets an image from an unseen generator wrong. Our results suggest that training with more generators can help with both accuracy and how well calibration transfers to unseen generators.

Even at k=7, the model still tends to underestimate the probability that an image from an unseen generator is fake. This can be seen in the [reliability plot after temperature scaling](figures/reliability_k7_temperature.png), where the curve is above the diagonal. We also tried Platt scaling, which fits both a slope and an intercept. It performed similarly to temperature scaling at k=7, but performed worse at k=1.

We also tested changing the classification threshold using the validation set. At k=7, this improved augmented model accuracy from 82.8% to 84.6% and fake recall from 69.9% to 76.4%. For the baseline, there was almost no change at k=7 (88.3% to 88.4%), and accuracy actually decreased at k=1 (73.0% to 71.2%).

[Results for each recalibration method](recalibration/test_macro_by_method.csv)

### 3. Most of the mistakes are fake images classified as real

At k=7, 68% of baseline errors and 86% of augmented errors come from the model classifying fake images as real. As we add more training generators, the model misses fewer fake images. The false-alarm rate for real images stays fairly consistent at around 4–8% across k values.

[Error rates across k](figures/error_rates_vs_k.png)

Augmentation reduces false alarms, but it also causes the model to miss more fake images. At k=7, the miss rate increases from 17.3% to 30.1%, while the false-alarm rate decreases from 6.2% to 4.3%. Some of the largest increases in the miss rate are for generators that the baseline model does relatively well on, including BigGAN (5.6% to 27.5%), GLIDE (3.6% to 14.2%), and ADM (16.0% to 36.0%).

| Held-out generator (k=7 baseline) | Missed fakes | Confidently missed* |
|---|---:|---:|
| SD1.4 | 26.1% | 4.2% |
| SD1.5 | 23.6% | 4.5% |
| Wukong | 22.7% | 9.0% |
| Midjourney | 21.3% | 6.2% |
| VQDM | 19.3% | 5.4% |
| ADM | 16.0% | 1.0% |
| BigGAN | 5.6% | 0.5% |
| GLIDE | 3.6% | 0.0% |

\* We counted an incorrect prediction as confident if its confidence after temperature scaling was at least 0.9. We used the scaled confidence because the raw probabilities never reach that level.

SD1.4, SD1.5, and Wukong were the hardest generators for the model to detect. Wukong and Midjourney also had the highest rates of confident misses. Interestingly, even though SD1.5 is included in the training set when SD1.4 is held out at k=7, the model still misses about a quarter of the SD1.4 images. This shows that training on a similar generator doesn't necessarily mean the model will generalize well to it.

When grouped by generator family, the baseline miss rate at k=7 was 5.6% for GANs (BigGAN only), 18.5% for diffusion models, and 21.3% for Midjourney (not assigned to a family).

[Error rates by generator family](errors/error_rates_by_family_k_mode.csv)

### 4. The model gets some of the same images wrong across runs

Since we use the same test images across the three repeats, we wanted to see whether the model was making different mistakes each time or consistently getting certain images wrong. Looking at individual images instead of just the overall error rates, we found that a large portion of the errors came from the same images.

- At k=7, 15% of the fake test images were missed in all three baseline repeats. These images made up 87% of all missed-fake errors. For the augmented model, 26% of fake images were missed in all three repeats, accounting for 85% of missed-fake errors.
- All holdouts use the same 280 real test images. Across the 24 baseline runs at k=7, 13% of these images were incorrectly classified as fake at least once, and 2.9% were classified as fake in every run.
- We also noticed that false-alarm rates differed depending on where the real images were originally stored. Images in the ADM and Wukong folders had false-alarm rates of 18% and 10%, compared to 1–7% for the other folders. Since the ADM and Wukong groups involved only 8 of 34 and 10 of 41 images, respectively, this is something that would need more investigation. It could be worth looking at image format, size, or compression.

Related results: [Error concentration](errors/k7_error_concentration.csv), [Repeated missed fakes](figures/k7_missed_fake_persistence.png), and [False alarms by source folder](errors/k7_false_alarms_by_real_source_folder.csv).

We saved the individual image paths in [k7_fake_image_persistence.csv](errors/k7_fake_image_persistence.csv) and [k7_baseline_most_confident_mistakes.csv](errors/k7_baseline_most_confident_mistakes.csv) so that the images can be viewed on CARC.

## Notes on the Analysis

- All runs use the same classifier seed of 17. The three repeats use different generator combinations, or different sampled images at k=7. This means the standard deviation across repeats reflects differences in the sampled datasets rather than randomness during training.
- We first average the three repeats for each held-out generator, then average the results across all eight holdouts so that each generator is weighted equally. The standard deviation across holdouts shows how much the results differ between generators and should not be interpreted as a confidence interval.
- For the bootstrap, we resample test images within each holdout separately by class and use the same sampled images for all three repeats. This avoids treating the repeats as independent test sets. The confidence intervals only account for test-image sampling, not the choice of generators.
- All recalibration parameters and decision thresholds are fitted using only the validation set. The tests in `tests/test_analysis.py` check that changing the test data doesn't affect any of the fitted parameters.
- We use 10 equal-width bins to calculate ECE, which is consistent with the original training pipelines. We chose 0.9 as the confidence cutoff for reporting confident mistakes.

## CNN Comparison

The disagreement counts between the CNN and ViT are available in the [clean CNN audit](../df40_transfer/clean_cnn_audit/README.md). However, the CNN's individual image predictions are stored in a tarball on CARC and aren't included in the repository. Because of this, the recalibration and repeated-error analysis in this folder only cover the ViT. We could run the same analysis on the CNN once its predictions are exported in the same format.

## Folder Structure

| Folder | Contents |
|---|---|
| `calibration/` | Validation and test metrics for each run, holdout and overall summaries, validation-to-test differences, reliability bins, and metric verification |
| `recalibration/` | Fitted temperatures, Platt scaling parameters, decision thresholds, test metrics, and bootstrap confidence intervals |
| `errors/` | Miss and false-alarm rates, summaries by generator, generator family and k, repeated image errors, and confident mistakes |
| `figures/` | Plots used in the analysis |

The `recalibration/test_predictions_recalibrated.csv.gz` file (6 MB) is generated by `recalibrate.py` and is excluded from Git.
