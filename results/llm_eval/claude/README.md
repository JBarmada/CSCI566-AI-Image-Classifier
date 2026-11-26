# Claude zero-shot run on GenImage

Scores the shared 630-image GenImage set with Claude: 70 real images plus 70 fakes
from each of the 8 generators. The set includes all 90 pilot images, so Claude can
be compared directly with the Gemini and Qwen pilot results. The prompt is copied
word for word from `results/llm_eval/run_full.py`.

Claude only sees neutrally named PNGs (`img_0001.png` …). File names, folders and
metadata are removed so the answer can't be read off the path. The pixels are
unchanged, and `blind_images.py` checks this for every image.

## Steps

### 1. Pick the images (already done, re-run to check)

```bash
python results/llm_eval/claude/select_images.py
```

This writes `shared_image_list.csv` and `carc_paths.txt`. The selection is fixed
(seed 566), so everyone who runs it gets the same 630 images. Other models should
use this same list.

### 2. Download the images from CARC

```bash
mkdir -p ~/genimage_subset
rsync -av --files-from=results/llm_eval/claude/carc_paths.txt \
  NETID@discovery.usc.edu:/project2/yzhao010_1531/csci566-genimage/data/extracted/GenImage_Compact/ \
  ~/genimage_subset/
```

Replace `NETID` with your USC NetID. It should copy 630 files.

### 3. Make the blind folder

```bash
python results/llm_eval/claude/blind_images.py \
  --image-root ~/genimage_subset \
  --blind-dir ~/claude_blind
```

`~/claude_blind` gets 630 PNGs and nothing else. The answer key is saved as
`results/llm_eval/claude/answer_key.csv`, in the repo and **not** in the blind folder.

### 4a. Run Claude through the API (main result)

This sends **one isolated API call per image**, the same setup as the Gemini/Qwen
runs in `run_full.py`. It uses the monthly API credits included with Claude Max
(claim them by linking a Claude Console organization at platform.claude.com),
and all 630 images should cost roughly $5–10.

```bash
pip install anthropic
read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY   # paste the key, press Enter (nothing shows)
python results/llm_eval/claude/run_claude_api.py --blind-dir ~/"cs566 proj"/claude_blind --limit 10
```

Check the 10 rows in `claude_api_responses.csv`, then run the same command without
`--limit 10` to finish the rest. It skips images that are already done, so you can
re-run it after any errors or interruptions. Then score it:

```bash
python results/llm_eval/claude/score_claude.py --raw results/llm_eval/claude/claude_api_responses.csv \
  --model claude-opus-5-5 --prefix claude_api
```

The key only lives in that terminal session. It is never written to a file.

### 4c. Shape and resolution control (API)

In this dataset every fake is square (128–1024 px) and the real photos are not
(69 of 70 non-square, ~500 px wide), so image shape alone separates real from fake.
To check whether Claude relies on that, the same 630 images are re-run in two more
views, keeping the same `img_XXXX` IDs:

- `square`: center crop to a square at the original resolution (removes shape)
- `detector224`: bicubic resize of the short side to 224, then a 224 center crop,
  the same input the CNN/ViT see (removes shape and resolution)

```bash
for v in square detector224; do
  python results/llm_eval/claude/blind_images.py --image-root ~/genimage_subset \
    --blind-dir ~/"cs566 proj"/claude_blind_$v --view $v
  python results/llm_eval/claude/run_claude_api.py --blind-dir ~/"cs566 proj"/claude_blind_$v \
    --output results/llm_eval/claude/claude_api_${v}_responses.csv
  python results/llm_eval/claude/score_claude.py --raw results/llm_eval/claude/claude_api_${v}_responses.csv \
    --key results/llm_eval/claude/answer_key_$v.csv --model claude-opus-5-5 --prefix claude_api_$v
done
python results/llm_eval/claude/compare_views.py
```

`compare_views.py` puts the three runs side by side with a "shape-only baseline"
(calling every square image fake) and writes `claude_api_view_comparison*.csv`.

### 4b. Run Claude in Claude Code (side check)

In Claude Desktop, open the **Code** tab, choose the `~/claude_blind` folder (not the
repo), and select the model you want to report (note its exact name).

Run 9 batches of 70 images, starting a **new session for each batch** so earlier
images don't pile up in context. Paste this prompt and change the range each time
(`img_0001`–`img_0070`, then `img_0071`–`img_0140`, … up to `img_0630`):

````
Please rate images img_0001.png through img_0070.png in this folder, one at a time.

For each image:
1. Open it with the Read tool and look at it.
2. Answer the question below, judging only from what you see in the image.
3. Immediately append one line to claude_responses.txt in this folder:
   img_XXXX | <your exact answer>
   e.g. img_0001 | {"p_fake": 35}
   If you can't or won't give a number, write your actual reply on that line instead.

Rules:
- Do not write or run any code to analyse the images, and do not look at file
  sizes, metadata or any other files. Only look at the pixels.
- Do not search the web.
- Skip any image that already has a line in claude_responses.txt.
- Don't summarise or comment between images; just keep going until the batch is done.

The question for every image:

Analyze the provided image and estimate the probability that it was
generated by an AI system rather than captured by a real camera.

Return ONLY a JSON object in this format:
{"p_fake": 0}

p_fake must be a number from 0 to 100:
0 means certainly real; 100 means certainly AI-generated.

Do not assume that unusual visual characteristics necessarily
indicate AI generation.
````

After each batch, check that `claude_responses.txt` has the expected number of lines.

### 5. Score it

```bash
cp ~/claude_blind/claude_responses.txt results/llm_eval/claude/
python results/llm_eval/claude/score_claude.py --model "<exact model name>"
```

This writes, in Shriya's format:
- `claude_predictions.csv`
- `claude_overall_metrics.csv`
- `claude_per_generator_metrics.csv`, where each generator's fakes are scored
  against all real images
- `claude_response_status.csv`

It lists any image with no answer or an unusable answer. Re-run those images in a
new session, or keep the refusal and report it.

### 6. Commit

Commit everything in this folder **except the images**: the scripts, image list,
answer key, `claude_responses.txt`, predictions, metrics and this README. Add the
model name and the dates you ran it to the section below.

## Run details

- **API run (main result):** model `claude-opus-5-5` (confirmed in every API response),
  default settings (adaptive thinking, default effort), one isolated call per image,
  4 parallel requests. Run on 2026-11-26.
- **Views:** `original` (as distributed), `square` (center crop, original resolution),
  `detector224` (bicubic resize + 224 center crop, same input as the CNN/ViT).
  Same 630 images and `img_XXXX` IDs in all three.
- **Responses:** 630/630 successful in every view; no refusals, no invalid answers, no re-runs needed.
- **Claude Code run (side check):** Claude Opus 5.5 in Claude Desktop (Code tab),
  batch 1 only (img_0001–img_0070), 2026-11-26. Responses in `claude_responses.txt`;
  not used for the main results.

### Results (threshold p_fake ≥ 50)

| View | AUROC | Fakes caught | Real called real |
|---|---:|---:|---:|
| original | 0.9998 | 75.5% | 100% |
| square | 0.9963 | 75.5% | 100% |
| detector224 | 0.9535 | 56.3% | 100% |
| shape-only baseline (square = fake) | 0.9929 | 100% | 98.6% |

Shape alone separates this dataset almost perfectly, but Claude's results barely change
when shape is removed (square), so it is not relying on it. Reducing every image to
224×224 lowers its scores, mostly for high-resolution generators (SD1.4/1.5, Wukong,
Midjourney), so resolution matters. Claude never flagged a real photo in any view;
its errors are all missed fakes, many scored just under 50.

## Limitations

- The API run matches the Gemini/Qwen setup (one isolated call per image). The
  Claude Code run rated ~70 images per session, so earlier images may affect later
  answers and interface settings are not controlled; it is only used as a check.
- Fakes and real photos differ in shape and resolution in this dataset; the
  `square` and `detector224` runs (4c) control for this.
- Opus 5.5 always uses adaptive thinking, so its answers come after hidden reasoning,
  unlike a plain single-shot answer.
- Images were converted to PNG with identical pixels, whereas the pilot sent the
  original JPEG files. Models decode both to the same pixels.
- 70 per group still gives about ±11 percentage points of uncertainty on a per-generator rate.
