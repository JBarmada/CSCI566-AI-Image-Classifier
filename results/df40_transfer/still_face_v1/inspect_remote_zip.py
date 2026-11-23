"""Inspect ZIP directory using bounded HTTP range requests, without downloading it."""
import argparse
import io
import urllib.request
import zipfile


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as response:
            self.length = int(response.headers["Content-Length"])
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.length)[whence] + offset
        if self.pos < 0:
            raise ValueError("negative position")
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.pos
        size = min(size, self.length - self.pos)
        if not size:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + size - 1}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206:
                raise ValueError(f"server ignored range: {response.status}")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError(f"expected {size} bytes, got {len(data)}")
        self.pos += len(data)
        self.bytes_read += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    remote = RemoteZip(args.url)
    with zipfile.ZipFile(remote) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        print(f"archive_bytes={remote.length} entries={len(files)}")
        for info in files[:args.limit]:
            print(f"{info.filename}\t{info.file_size}\t{info.compress_size}")
    print(f"transferred_bytes={remote.bytes_read}")


if __name__ == "__main__":
    main()
