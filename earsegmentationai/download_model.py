"""
Earsegmentationai

Copyright (c) 2023 Ear-Segmentation-Ai Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import functools
import shutil
from os import path as ospath
from pathlib import Path

import requests
from requests import Response
from tqdm.auto import tqdm

from earsegmentationai import MODEL_PATH, MODELURL


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
