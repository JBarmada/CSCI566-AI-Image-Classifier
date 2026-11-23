# Still-photo transfer and real-image false positives

Research continuation, 23 November 2026. This document distinguishes released
dataset descriptions, our sample checks, and hypotheses about model errors.

## Dataset comparison

| Candidate | Contents and benefit | Limitations for this project |
| --- | --- | --- |
| [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) | Authors describe 30,000 IMDB-WIKI real photos and 30,000 outputs each from InsightFace, Stable Diffusion inpainting, and Stable Diffusion 1.5 text-to-image. Three corresponding image archives share relative filenames. | Text-to-image is created from scratch and is not an edited photograph. Stable Diffusion 1.5 also appears in our GenImage benchmark; an inference run is not automatically unseen-generator transfer. Identity and source-image overlap with GenImage are unverified. Archive labels do not independently prove every face edit succeeded. The dataset page has conflicting license declarations (`openrail` metadata and `apache-2.0` in embedded text); check original-image and dataset terms before redistribution. |
| [OpenForensics](https://zenodo.org/records/5528418) | 115,000 in-the-wild images and 334,000 face annotations, including face-level boxes and masks. Tests realistic multi-face scenes. | The full release is 56.4 GB. Face labels must be mapped to individual crops; assigning a single scene label to every face is invalid. Relevant test-image archives and annotations are large. |
| [CelebHQ-FM / FFHQ-FM](https://github.com/stresearch/face-manipulation-datasets) | CelebHQ-FM includes original portraits and attribute-edited variants with an identity-disjoint train/test split. FFHQ-FM includes localized edits in more varied scenes. | The available edits include expression, age and gender attributes, rather than only identity swaps; whole-scene FFHQ images can include multiple people. Download and reuse terms require verification. The repository's FFHQ-FM aggregate and per-class counts differ by one, so check manifests before reporting counts. |

DeepFakeFace was selected for a **small exploratory inference comparison**:
the original archives expose consistent path names and can be read with HTTP
range requests, avoiding a multi-gigabyte download. It complements the existing
video-frame study rather than replacing it. The authors' [paper](https://arxiv.org/html/2309.02218)
describes InsightFace swapping, facial inpainting, and the IMDB-WIKI source.

## Real-image error hypotheses

In our 100-image DF40 real subset at the existing 0.5 threshold, false positives
were 99/100 for the baseline CNN, 57/100 for augmented CNN, 90/100 for baseline
ViT, and 83/100 for augmented ViT. This is a small selected video-crop subset;
it does not demonstrate that all real photos or all people are misclassified.
The FaceForensics++ follow-up uses separate validation-selected thresholds but
its test was examined previously; see [the report draft](FINAL_REPORT_DRAFT.md)
and [experiment log](FINAL_REPORT_EXPERIMENT_LOG.md).

For the separate FaceForensics++ crop split, thresholds chosen using 70
validation real images targeted at most 10% validation false positives.
At the selected thresholds the baseline ViT flagged 5/70 test reals but
detected only 9/70 SimSwap, 17/70 FOMM, and 3/70
Wav2Lip samples; the baseline CNN flagged 5/70 and detected 2/70, 6/70, and
11/70 respectively. These are different fake-detection tradeoffs from the
original 0.5 cutoff. This test was later inspected for other analyses and
identity independence is unverified. SimSwap and InsightFace are distinct
methods, so those results are not interchangeable.

[Grommelt et al.](https://arxiv.org/abs/2403.17608) show that GenImage detectors
can learn compression and image-size shortcuts. [Dang et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dang_On_the_Detection_of_Digital_Face_Manipulation_CVPR_2020_paper.pdf)
use the same real-image face landmarks to crop a real image and its derived fake,
illustrating one way to limit crop differences. These findings justify testing
processing as a possible confound; neither establishes the cause of our errors.
Camera model, compression, video extraction, crop geometry, image size,
population, and threshold can change together. We lack controlled camera
metadata and have not isolated their separate causal effects.

An EXIF count on the 50 selected DeepFakeFace real files found camera-make
and camera-model fields in only four images, and any EXIF in six. None of the
100 selected fake files had EXIF. The fields are incomplete and have not been
independently verified, making camera-specific comparisons infeasible in this
sample. The count can be reproduced with
[`audit_camera_metadata.py`](still_face_v1/audit_camera_metadata.py) and the
locally obtained, hash-checked images; it does not publish metadata contents.

Suggested controlled follow-up: take held-out real photographs with documented
provenance; apply the same detector and crop once, then vary only JPEG quality
or resize per copy and compare paired score changes. For camera effects use
documented camera/device groups and balance people, scenes, and processing as
far as possible; observational metadata alone cannot establish causation.
Select any new threshold on separate validation samples and report fake recall
alongside real false-positive rate. Do not tune on the final test.

## Reproducibility and interpretation

The local continuation samples 50 shared source paths from three archives at
revision `eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c`, using a fixed random seed.
The first six paths per group are the operational smoke and are included in the
full descriptive sample. The same 50 real images are reused for both fake
methods, so method-level comparisons are dependent. File hashes and ZIP CRC
checks establish byte consistency and archive provenance, not independent
identity separation, face-edit success, or licensing permission to redistribute
photographs. Methods, checkpoint replays, scores and run status are reported in
the associated results once available.
