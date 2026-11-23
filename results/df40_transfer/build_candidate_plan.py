"""Plan source-video membership from public metadata; retrieve no media."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'metadata/ffpp_official_test_pairs.json'
    pairs = json.loads(source.read_text(encoding='utf-8'))
    canonical = [tuple(sorted(pair)) for pair in pairs]
    ids = [video for pair in canonical for video in pair]
    if len(canonical) != 70 or len(set(ids)) != 140:
        raise ValueError('Expected 70 disjoint official source-video pairs')
    ranked = sorted(canonical, key=lambda pair: hashlib.sha256(
        ('566:' + '_'.join(pair)).encode()).hexdigest())
    rows = []
    for rank, pair in enumerate(ranked[:22]):
        group = '_'.join(pair)
        digest = hashlib.sha256(('566:' + group).encode()).hexdigest()
        orientation = int(digest[:8], 16) % 2
        target, donor = pair[orientation], pair[1 - orientation]
        role = 'smoke' if rank < 2 else 'evaluation'
        shared = {'role': role, 'source_group_id': group, 'target_video_id': target,
                  'source_video_id': donor, 'compression': 'c23',
                  'status': 'metadata_candidate_not_downloaded'}
        rows.append({**shared, 'video_id': target, 'label': 0, 'manipulation': 'original',
                     'relative_video_path': f'original_sequences/youtube/c23/videos/{target}.mp4'})
        rows.append({**shared, 'video_id': f'{target}_{donor}', 'label': 1,
                     'manipulation': 'Deepfakes', 'relative_video_path':
                     f'manipulated_sequences/Deepfakes/c23/videos/{target}_{donor}.mp4'})
    destination = ROOT / 'candidate_video_plan.csv'
    with destination.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {'source_url': 'https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/test.json',
                  'retrieved_date': '2026-11-20', 'official_pair_count': len(pairs),
                  'snapshot_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'plan_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                  'smoke_videos': 4, 'evaluation_videos': 40,
                  'media_retrieved': False, 'identity_disjointness_verified': False}
    (ROOT / 'metadata/PROVENANCE.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
