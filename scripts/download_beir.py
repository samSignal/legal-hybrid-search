"""Download a public BEIR benchmark dataset (e.g. scifact, nfcorpus, fiqa) into data/<name>.

    python scripts/download_beir.py scifact
    legalsearch --data data/scifact eval
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    dest = Path(__file__).resolve().parent.parent / "data"
    print(f"Downloading {name} ...")
    with urllib.request.urlopen(URL.format(name)) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    print(f"Saved to {dest / name}")
