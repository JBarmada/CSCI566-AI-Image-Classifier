"""Convert fixed-prompt CLIP scores to the shared calibration format."""
import csv
import sys
from pathlib import Path


def main():
    source, destination = map(Path, sys.argv[1:3])
    with source.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['model'] = 'clip_zero_shot'
        row['mode'] = 'fixed_prompts'
        row['prob_fake'] = row['prob_fake_prompt']
    with destination.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
