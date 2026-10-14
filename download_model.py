"""从 ModelScope（魔搭）下载 Embedding 模型到 ./models/，适合 HuggingFace 及其镜像都访问不了的网络环境。

用法：python download_model.py [模型ID]，默认 BAAI/bge-small-zh-v1.5
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://www.modelscope.cn/api/v1/models/{model}/repo"
SKIP = {".gitattributes", "README.md", "configuration.json", "pytorch_model.bin"}


def list_files(model: str) -> list[dict]:
    url = API.format(model=model) + "/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    return [f for f in data["Data"]["Files"] if f["Type"] == "blob" and f["Path"] not in SKIP]


def download(model: str, path: str, dest: Path, size: int) -> None:
    if dest.exists() and dest.stat().st_size == size:
        print(f"  已存在 {path}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"Revision": "master", "FilePath": path})
    url = API.format(model=model) + "?" + query
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {path}: {done / 1e6:.1f}/{size / 1e6:.1f} MB", end="", flush=True)
    print()
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{path} 大小不符：{tmp.stat().st_size} != {size}")
    tmp.replace(dest)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-zh-v1.5"
    target = ROOT / "models" / model.split("/")[-1]
    print(f"下载 {model} -> {target}")
    for f in list_files(model):
        download(model, f["Path"], target / f["Path"], f["Size"])
    print("完成")


if __name__ == "__main__":
    main()
