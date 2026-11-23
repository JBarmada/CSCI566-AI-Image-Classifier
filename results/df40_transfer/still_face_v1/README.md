# Can our models detect deepfakes in still photos?

We tested our existing CNN and ViT on **DeepFakeFace**, a dataset of real celebrity photos and manipulated images. We used 50 real photos, 50 InsightFace face swaps, and 50 inpainted images. Both models saw the same images. We did not retrain them or change their decision threshold.

| Model | Real photos falsely called fake | Face swaps detected | Inpainted images detected |
| --- | ---: | ---: | ---: |
| CNN baseline | 23/50 | 31/50 | 34/50 |
| CNN augmented | 18/50 | 24/50 | 19/50 |
| ViT baseline | 4/50 | 28/50 | 33/50 |
| ViT augmented | 5/50 | 20/50 | 20/50 |

The baseline ViT falsely flagged fewer real photos than the CNN in this small test. It also missed some manipulated images. The augmented models had lower fake-detection counts than their baseline counterparts in this sample.

The earlier DF40 test used video crops; the baseline ViT falsely flagged 90 of 100 real crops there. It flagged 4 of 50 real photos here. This suggests its real-image performance varies across datasets, but the datasets differ in source and processing, so we cannot tell what caused the change.

**These results are exploratory.** The sample is small, the real and fake files differ in size, and we have not independently verified each manipulation or whether people appear in other data. We need a larger, carefully checked test before making broad claims.

For how the data and models were checked, AUROC scores, reproduction commands, and other details, see [technical details](TECHNICAL_DETAILS.md). See also the [processing follow-up](../still_face_geometry_v1/README.md) and [dataset and error research](../STILL_FACE_RESEARCH.md).
