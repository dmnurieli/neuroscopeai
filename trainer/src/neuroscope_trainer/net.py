"""Small HTTP helpers (standard library only): whole-file downloads, a resumable gzip line stream, and a seekable
range-read file that pyarrow can read parquet from without downloading the whole file."""

import hashlib
import io
import os
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

USER_AGENT = "neuroscope-trainer/0.1"
RETRIES = 6


class NetworkError(RuntimeError):
    pass


def _open(url: str, start: int = 0, end: int | None = None, timeout: float = 60.0):
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start or end is not None:
        headers["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def _retry(fn, what: str):
    delay = 1.0
    for attempt in range(RETRIES):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise NetworkError(f"{what}: HTTP {e.code}") from e
            if attempt == RETRIES - 1:
                raise NetworkError(f"{what}: {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 30)


def download(url: str, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def go():
        with _open(url) as r:
            data = r.read()
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return dest

    return _retry(go, url)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GzipLines:
    """Iterates the lines of a remote .gz file as it downloads. A dropped connection resumes from the compressed
    byte it stopped at (HTTP Range), so the decompressor never notices. `bytes_read` counts compressed bytes."""

    def __init__(self, url: str, chunk: int = 1 << 20):
        self.url, self.chunk = url, chunk
        self.bytes_read = 0

    def __iter__(self):
        z = zlib.decompressobj(16 + zlib.MAX_WBITS)
        tail = b""
        resp = None
        failures = 0
        while True:
            try:
                if resp is None:
                    resp = _retry(lambda: _open(self.url, start=self.bytes_read), self.url)
                block = resp.read(self.chunk)
                failures = 0
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                failures += 1
                if failures > RETRIES:
                    raise NetworkError(f"{self.url}: {e}") from e
                resp = None
                time.sleep(min(2**failures, 30))
                continue
            if not block:
                break
            self.bytes_read += len(block)
            data = tail + z.decompress(block)
            lines = data.split(b"\n")
            tail = lines.pop()
            yield from lines
            if z.eof:  # a multi-member gzip continues with a fresh member
                rest = z.unused_data
                z = zlib.decompressobj(16 + zlib.MAX_WBITS)
                if rest:
                    data = tail + z.decompress(rest)
                    lines = data.split(b"\n")
                    tail = lines.pop()
                    yield from lines
        tail += z.flush()
        if tail:
            yield tail


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP Range requests, with a read-ahead block cache."""

    def __init__(self, url: str, block: int = 4 << 20):
        super().__init__()
        self.block = block
        # Resolve redirects once (the CDN URL is what accepts ranges).
        with _retry(lambda: _open(url, 0, 0), url) as r:
            self.url = r.geturl()
            cr = r.headers.get("Content-Range", "")
            self.size = int(cr.rsplit("/", 1)[1]) if "/" in cr else int(r.headers["Content-Length"])
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + off))
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
            lo, hi = i * self.block, min(self.size, (i + 1) * self.block) - 1

            def go():
                with _open(self.url, lo, hi) as r:
                    return r.read()

            data = _retry(go, self.url)
            self.bytes_read += len(data)
            self.cache[i] = data
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, self.block)
            b = self._block(i)[off : off + n]
            out += b
            self.pos += len(b)
            n -= len(b)
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)
