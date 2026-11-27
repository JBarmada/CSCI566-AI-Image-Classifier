# Decisions awaiting team review

Drafted independently during the five-hour no-question window. This record
does not require immediate user input.

| Topic | Working decision | Why / evidence | What team should review |
| --- | --- | --- | --- |
| Research story | Three generalization shifts | Jad selected this question; proposal, midterm and Nov 26 audit support it | Approve final captions and allocating equal panels |
| Generator result | Four clean curves + matched frontier comparison | Fixed-budget study anchors project; frontier comparisons use verified common scores | Add a generator-level heatmap if useful; it is available in poster_assets |
| Presentation result | JPEG class-error delta + paired VLM three-view bars | Direct differences rather than connecting two points; recall and false alarms visible | Confirm poster readability; report includes resize and calibration |
| Manipulation result | Face error-space plot with baseline CNN/ViT, GPT, Claude and optional supervised reference | Opposite error modes are scientifically informative; 7-model full table in report | Keep 67.78% reference as exploratory or omit if space is tight |
| Fresh face data | Defer until provenance and identity checks pass | Five-method 189-image candidate is already scored; it is not person-disjoint | If collecting more data, predeclare independent selection, source groups and untouched test |
| Extra methods | Report existing five methods; no additional acquisition required for deadline | SimSwap, FOMM, Wav2Lip, InSwapper, PIRender are covered; mirror-label/identity limits remain | Do not add more until label provenance is established |
| Adaptation | No new training before poster | Three-seed exploratory adversarial and contrastive objectives show no reliable improvement; test already examined | Protect a genuinely independent test before future trials |
| Frontier | Adopt Nov 26 matched comparison; no new paid runs | 612 matched detector224 sources and 88-source pilot; 590 common sources for image-view study | Owner should supply GPT attempt-level JSONL, or disclose it cannot be independently checked |
| Geometry | Document shape correlation as a methodological caveat, not headline | Square-crop control preserves much ranking; detector224 recall falls | No causal image-shape or resolution-only claim |
| Real errors | Report tests without claiming camera cause | Geometry and JPEG95 tests on the same still-face subset do not isolate acquisition effects | Rachel should verify phrasing and add evidence only if available |
| Poster format | 48 × 36 landscape PDF + editable SVG | Reversible format assumption; course public schedule did not establish dimensions | Confirm printing/submission requirements on Piazza |
| Demo | Tested local replay, clearly labeled, no provider or GPU dependency | All source scores retained; browser smoke passed | Rehearse; add live checkpoint inference only if required and verified |
| Final report | Consolidated editable Markdown plus 11-page review PDF | Full methods, limitations, class errors, F1/AUROC, matched frontier and calibration | Confirm prescribed format, author order, contribution statement, references |

No new training, data downloads, or API calls were necessary. Evidence and
scope are in the [proposal/midterm coverage audit](../../results/llm_eval/team_readiness_2026-11-26/README.md).
