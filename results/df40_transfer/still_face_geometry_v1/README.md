# Separate geometry and JPEG checks

These inference-only analyses follow [the original still-photo evaluation](../still_face_v1/README.md)
and the [combined-processing experiment](../still_face_uniform_v1/README.md).
They use the same 150 source images, four saved model configurations, fixed
0.5 threshold, reference replay gates and 18-image repeat smoke. CARC jobs
`14489331` (geometry) and `14489332` (JPEG) completed with exit code 0.

Two variants were made independently from each original input:

- **Geometry:** convert to RGB, center-crop to a square, bicubically resize
  to 512x512, and save losslessly as PNG. This changes geometry and
  interpolation without an additional lossy JPEG encoding.
- **JPEG:** convert to RGB and re-encode at the original dimensions with
  quality 95 and chroma subsampling disabled, without a geometric resize.

| Model | Original real FP | Geometry real FP | JPEG real FP |
| --- | ---: | ---: | ---: |
| CNN baseline | 23/50 | 28/50 | 23/50 |
| CNN augmented | 18/50 | 20/50 | 18/50 |
| ViT baseline | 4/50 | 5/50 | 4/50 |
| ViT augmented | 5/50 | 5/50 | 5/50 |

At threshold 0.5 the geometry change flipped five baseline CNN and one
baseline ViT real decisions; the tested JPEG change flipped none of the 50
real decisions for any model. Scores can change without crossing the
threshold. The geometry change is not identical across input groups: all
fake images were already 512x512, while real image dimensions varied. The
fake-image geometry scores matched their original scores exactly. This small,
post-hoc test cannot generalize to other JPEG qualities, cameras or data, and
does not establish why the earlier video-crop sample produced high false
positives. Source/identity overlap and individual edit success are unknown.

Per-image predictions, manifest hashes, replay evidence, and paired score
changes are in this folder and [the JPEG folder](../still_face_jpeg95_v1/).
The transformation code is [transform_factors.py](../still_face_v1/transform_factors.py).
Images are not committed.

```powershell
cd "C:\Users\creat\OneDrive - University of Southern California\Documents\ChatGPT\CSC566 Proj\csci566_research_continuation"
python results\df40_transfer\still_face_uniform_v1\compare.py results\df40_transfer\still_face_geometry_v1
python results\df40_transfer\still_face_uniform_v1\compare.py results\df40_transfer\still_face_jpeg95_v1
```
