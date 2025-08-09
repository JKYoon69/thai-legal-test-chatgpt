import io
import json
import zipfile
from typing import Iterable


def to_jsonl(objs: Iterable, path):
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o.model_dump(), ensure_ascii=False) + "\n")


def make_zip_bundle(zip_path, files: dict[str, bytes]):
    # files: {"name.ext": b"...", ...}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
