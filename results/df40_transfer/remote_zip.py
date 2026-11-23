"""Read individual public Drive ZIP members with bounded HTTP ranges."""
import io
import re
import urllib.request


class RemoteZipFile(io.RawIOBase):
    transferred = 0
    budget = 160 * 1024 * 1024

    def __init__(self, file_id, size, source_url=None):
        self.url = source_url or ('https://drive.usercontent.google.com/download?id='
                                  + file_id + '&export=download&confirm=t')
        if not self.url.startswith('https://'):
            raise ValueError('HTTPS source required')
        self.size = int(size)
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: self.size}
        self.position = bases[whence] + offset
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, amount=-1):
        if amount < 0:
            amount = self.size - self.position
        amount = min(amount, self.size - self.position)
        if amount <= 0:
            return b''
        if amount > 20 * 1024 * 1024:
            raise ValueError('Refusing an unexpectedly large ZIP read')
        if self.transferred + amount > self.budget:
            raise ValueError('Smoke retrieval budget exceeded')
        if amount > 512 * 1024:
            pieces = []
            remaining = amount
            while remaining:
                piece = self.read(min(remaining, 512 * 1024))
                pieces.append(piece)
                remaining -= len(piece)
            return b''.join(pieces)
        start, end = self.position, self.position + amount - 1
        request = urllib.request.Request(
            self.url + f'&_range={start}-{end}',
            headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=45) as response:
            expected = f'bytes {start}-{end}/{self.size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                if 'text/html' in response.headers.get('Content-Type', ''):
                    page = response.read(8192).decode('utf-8', errors='replace')
                    if 'Quota exceeded' in page:
                        raise ValueError('Google Drive download quota exceeded for this archive')
                raise ValueError(f'Range refused: expected {expected}; got status '
                                 f'{response.status}, range {response.headers.get("Content-Range")}')
            data = response.read(amount + 1)
        if len(data) != amount:
            raise ValueError('Unexpected range length')
        RemoteZipFile.transferred += len(data)
        self.position += len(data)
        return data


def canonical_path(path):
    marker = 'deepfakes_detection_datasets/'
    if marker not in path:
        raise ValueError(f'Unrecognized official path: {path}')
    return path.split(marker, 1)[1].replace('\\', '/')


def identities(group):
    return sorted(set(re.findall(r'(?:^|_)id(\d+)(?=_|$)', group)))
