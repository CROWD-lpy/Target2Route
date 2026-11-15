import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


PAIR_COLUMNS = ("target_seq", "target_smi")


@contextmanager
def atomic_file(path, *, binary=False, replace=False):
    """Publish a complete file in one operation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Output already exists: {path}")
    options = {} if binary else {"encoding": "utf-8", "newline": ""}
    with NamedTemporaryFile(
        mode="wb" if binary else "w", dir=path.parent, delete=False, **options
    ) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.close()
            if replace:
                temporary.replace(path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def new_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        staging = Path(temporary) / path.name
        staging.mkdir()
        yield staging
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
        staging.rename(path)


def write_json(path, value, *, replace=False):
    with atomic_file(path, replace=replace) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def describe_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
