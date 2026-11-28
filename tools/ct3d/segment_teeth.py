"""TotalSegmentator의 teeth 작업으로 CT에서 치아를 번호별로 나눈다.

setup_segmentation.py로 만든 가상환경의 python으로 실행한다.

사용법:
    python segment_teeth.py <ct.nii.gz> <출력 폴더> [--device gpu|cpu]

출력 폴더에 구조마다 NIfTI 파일이 하나씩 생긴다 (예: upper_right_central_incisor_fdi11.nii.gz).
걸린 시간과 찾은 구조 목록은 segmentation_info.json에 적는다.

모든 계산은 이 PC 안에서 한다. 처음 실행할 때 모델 가중치만 내려받는다.
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from totalsegmentator.python_api import totalsegmentator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--device", default="gpu" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"장치: {args.device}" + (f" ({gpu})" if gpu and args.device != "cpu" else ""), flush=True)

    start = time.time()
    totalsegmentator(args.volume, args.out_dir, task="teeth", device=args.device)
    seconds = round(time.time() - start)

    found = {}
    for f in sorted(args.out_dir.glob("*.nii.gz")):
        voxels = int(np.count_nonzero(np.asanyarray(nib.load(f).dataobj)))
        if voxels:
            found[f.name[: -len(".nii.gz")]] = voxels

    info = {
        "task": "teeth",
        "device": args.device,
        "gpu": gpu,
        "seconds": seconds,
        "structures_found": len(found),
        "voxels": found,
    }
    (args.out_dir / "segmentation_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"끝: {seconds}초, 찾은 구조 {len(found)}개")
    for name in found:
        print("  ", name)


if __name__ == "__main__":
    main()
