"""Collect complete clean robustness results from the isolated replay directory."""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    results = [json.loads(p.read_text()) for p in sorted(root.glob('runs/*/clean/result.json'))]
    markers = list(root.glob('runs/*/clean/COMPLETE.json'))
    if len(results) != 24 or len(markers) != 24:
        raise ValueError(f'Expected 24 completed results, got {len(results)} and {len(markers)}')
    if any(r['n'] != 560 or r['source_rows'] != 560 or r['threshold'] != 0.5
           or not r['source_comparison_available'] for r in results):
        raise ValueError('A result failed the expected reference comparison checks')
    output = {'job_id': '14491618', 'tolerance': 0.001,
              'run_count': len(results), 'rows': sum(r['n'] for r in results),
              'max_abs_probability_difference': max(r['max_abs_probability_difference'] for r in results),
              'runs_within_tolerance': sum(r['max_abs_probability_difference'] <= 0.001 for r in results),
              'results': results}
    Path('robust_clean_results.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != 'results'}))


if __name__ == '__main__':
    main()
