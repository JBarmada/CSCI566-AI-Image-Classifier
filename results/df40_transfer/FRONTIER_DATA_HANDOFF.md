# Face datasets for the frontier-model comparison

For the teammate running frontier-model tests. These are **recommendations for
discussion**, not a request to change your existing protocol. The later OpenAI
run completed all 189 five-method images and the matched comparison; see the
[results](openai_face/df40_five_method_v3_v2/README.md) and
[current workflow](../../docs/llm-evaluation.md). The separate 150-image
DeepFakeFace API evaluation remains optional and has not been run.
Keep the same image hashes when
comparing against our CNN and ViT predictions.

## Recommended first comparison: still-photo pilot

The [DeepFakeFace pilot](still_face_v1/README.md) contains 50 real photos,
50 InsightFace swaps and 50 inpainted images. It gives a complementary
still-image test and already has predictions for the baseline and augmented
CNN and ViT on the exact images. The [manifest](still_face_v1/manifest.csv)
records paths and SHA-256 hashes; [technical details](still_face_v1/TECHNICAL_DETAILS.md)
document selection, metrics and provenance. Compare each fake method against
the **same** 50 real images, reporting real false alarms out of 50 and fake
detections out of 50. The real set is reused across both comparisons.

The copy used for the completed evaluation is at
`/home1/barmada/still_face_transfer_20261123_1225/deepfakeface_sample_v1/`
on CARC. The first manifest image's file hash was checked against that copy;
the 150 hashes were checked during the original evaluation. A teammate's access
to this home-directory copy has not been verified. Image bytes are not in
GitHub; coordinate access and permitted use with Jad before testing.

This is a small, previously inspected exploratory sample, not a pristine
confirmation set. The real images have varying dimensions while the fake
images are all 512×512, so image dimensions or processing may provide an
unintended class cue. Labels reflect dataset provenance, not individual proof
of visible manipulation; identities and possible pretraining overlap remain
unverified. Results do not establish why the models succeed or fail. The
dataset card's license labels conflict; check original terms and the provider's
data-handling rules before uploading photos to any external service.

## Optional follow-up: five-method face crops

The [189-image v3 manifest](new_data/fresh_candidate_review_v3/manifest.csv)
contains 54 real crops and 27 each from SimSwap, FOMM, Wav2Lip, InSwapper and
PIRender. It uses 27 source-video-ID pairs, with the same real crops reused
across methods. Image bytes are in the project-restricted CARC directory
`/project2/yzhao010_1531/csci566-genimage/data/df40_fresh_candidate_review_v3_20261123/`;
images are not in GitHub. Verify the manifest and image hashes before use.
The existing-model P100 comparison has completed and passed its replay checks;
match model predictions by image hash. See the [results](new_data/fresh_candidate_results_v3/README.md),
[frozen evaluation protocol](new_data/FRESH_EVALUATION_PROTOCOL.md) and
[data audit](new_data/FRESH_CANDIDATE_REVIEW.md).

Use this set only as an exploratory follow-up. The known video IDs differ from
earlier project plans, but person-level and cross-dataset independence are not
established. Wav2Lip source roles, individual manipulation success, and mirror
byte equivalence and method labels have outstanding questions. Three candidate
pairs were excluded before scoring; Jad's review of two exclusions is open.
It is not an external, identity-independent test. Check original FaceForensics++
terms and provider upload policies before using any third-party API.

## Coordinate the test before scoring

1. Agree on the exact manifest, model/version, image preprocessing, prompt,
   output format, decoding settings and treatment of refusals before seeing
   evaluation outputs. Preserve original image hashes and test all selected
   examples, including negative findings. Give the model image bytes and a
   neutral identifier: do not expose directory names, filenames, metadata,
   method labels, ground-truth labels or existing model predictions.
2. Clarify what “one-shot” means in the team's study. **Zero-shot** supplies
   no labeled example; **one-shot prompting** supplies a labeled example in
   context. If using a labeled example, choose and freeze it from a separate
   development source, not the evaluated manifest. Exclude known linked
   source-video groups and check for identity overlap where feasible; disclose
   remaining unknowns. A real/fake example pair is two labeled examples, so
   report that design explicitly. Never reuse the test images as demonstrations
   or adjust the prompt after reviewing their outcomes.
3. Save per-image hash, model/version, prompt identifier, raw response,
   parsed label and errors/refusals. Report attempted, answered, refused and
   unparseable counts; do not silently drop failures. Show real false positives
   and per-method fake detections with denominators. If the model provides only
   class labels, do not calculate AUROC from those labels as though they were
   calibrated continuous confidence scores. Any reported scores need their
   meaning and extraction method documented.
4. Report provider/model preprocessing and any compression or resizing; use
   the same source images but do not claim the model pipelines are identical.
   Identity and pretraining overlap are unknown, and group-linked images and
   repeated real examples are not independent observations. Observe dataset
   terms and the team's restrictions before sending any images externally.

For context, the existing [Qwen pilot](multimodal_qwen/README.md) was a
**zero-shot** 64-image GenImage experiment with one open model and one prompt;
it did not evaluate GPT or Claude or either face set above.
