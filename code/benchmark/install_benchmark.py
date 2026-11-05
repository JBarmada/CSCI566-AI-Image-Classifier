"""Safely install supplied archives; never overwrite different existing files."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(archive_path, root, prefix):
    root = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for index, entry in enumerate(archive.infolist(), 1):
            path = (root / entry.filename).resolve()
            if not entry.filename.startswith(prefix + '/') or not path.is_relative_to(root):
                raise ValueError(f'Unsafe archive path: {entry.filename}')
            if entry.is_dir():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.installing')
            with archive.open(entry) as source, temporary.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            if path.exists():
                if sha(path) != sha(temporary):
                    temporary.unlink()
                    raise FileExistsError(f'Refusing to overwrite different data: {path}')
                temporary.unlink()
            else:
                temporary.rename(path)
            if index % 2000 == 0:
                print(f'Installed {index}/{len(archive.infolist())}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    install(args.archive, args.data_root, args.prefix)


if __name__ == '__main__':
    main()
