"""Check a fixed selection of matching frame names across two mirror archives."""
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from collect_fake_crops import RangeArchive
from remote_zip import RemoteZipFile
from retrieve_adaptation import REV
from verify_real_archive import crop
from audit_shared_pairs import load

GROUP = '628_568'
SIZES = {'simswap': 2922323911, 'inswap': 2502440755}


def members(method):
    data, _ = load(method)
    group = data[f'{method}_Fake']['train'][GROUP + '_' + method]
    return {Path(path).name for path in group['frames']}


def digests(method, frames):
    url = f'https://huggingface.co/datasets/ManhQuangAI/DF40_train/resolve/{REV}/{method}.zip?download=true'
    with ZipFile(RemoteZipFile(None, SIZES[method], url)) as archive:
        index = {i.filename: i for i in archive.infolist()}
    source = RangeArchive(url, SIZES[method])
    return {name: crop(source, index[f'{method}/frames/{GROUP}/{name}'])[1] for name in frames}


def main():
    common = sorted(members('simswap') & members('inswap'), key=lambda s: int(Path(s).stem))
    frames = [common[i] for i in sorted({0, len(common)//3, 2*len(common)//3, len(common)-1})]
    hashes = {method: digests(method, frames) for method in SIZES}
    report = {'group': GROUP, 'common_frame_names': len(common), 'sampled_frames': frames,
              'sha256': hashes, 'identical_frames': [f for f in frames if hashes['simswap'][f] == hashes['inswap'][f]],
              'limitations': 'Small selected sample; equal bytes do not establish which method label is wrong.'}
    (ROOT / 'duplicate_pair_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
