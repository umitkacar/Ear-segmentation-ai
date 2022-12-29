import functools
import shutil
from os import path as ospath
from pathlib import Path

import requests
from requests import Response
from tqdm.auto import tqdm

from earsegmentationai.const import MODEL_PATH, MODELURL


def get_model() -> None:

    if ospath.exists(MODEL_PATH):
        return

    print("Downloading necessary deep learning model \n")
    res: Response = requests.get(MODELURL, stream=True, allow_redirects=True)
    if res.status_code != 200:
        res.raise_for_status()
        raise RuntimeError(
            f"Request to {MODELURL} returned status code {res.status_code}"
        )

    file_size: int = int(res.headers.get("Content-Length", 0))
    path: Path = Path(MODEL_PATH).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    res.raw.read = functools.partial(
        res.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(res.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
