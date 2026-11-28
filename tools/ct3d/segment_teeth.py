"""TotalSegmentator의 teeth 작업으로 CT에서 치아를 번호별로 나눈다.

setup_segmentation.py로 만든 가상환경의 python으로 실행한다.

사용법:
    python segment_teeth.py <ct.nii.gz> <출력 폴더> [--device gpu|cpu]

출력:
    teeth_labels.nii.gz       화소마다 구조 번호가 들어 있는 볼륨. 격자는 입력 CT와 같다
    segmentation_info.json    구조 번호와 이름, 찾은 구조별 화소 수, 걸린 시간

GPU가 없어도 돌아간다. RAM 여유가 10GB쯤 필요하다.
모든 계산은 이 PC 안에서 한다. 처음 실행할 때 모델 가중치만 내려받는다.
TotalSegmentator는 기본으로 실행 통계(작업 이름, 버전, 운영체제 같은 값. 영상은 아님)를
자기 서버로 보내는데, 이 스크립트는 그 기능을 끄고 실행한다.
"""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import totalsegmentator.python_api as ts_api
from totalsegmentator.config import set_config_key, setup_totalseg
from totalsegmentator.map_to_binary import class_map


def device_name(device) -> str:
    if hasattr(device, "type"):  # torch.device
        return "gpu" if device.type == "cuda" else device.type
    return str(device)


# TotalSegmentator 2.18.0은 CPU로 돌릴 때 치아 범위를 찾는 단계로 장치 이름 대신 None을 넘겨서 멈춘다.
# 문자열도 처리하는 함수로 바꿔 끼운다. GPU에서는 원래 함수와 결과가 같다.
ts_api.convert_device_to_string = device_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--device", default="gpu" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 실행 통계 전송을 끈다. 설정 파일은 사용자 폴더의 .totalsegmentator/config.json에 있다.
    setup_totalseg()
    set_config_key("send_usage_stats", False)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"장치: {args.device}" + (f" ({gpu})" if gpu and args.device != "cpu" else ""), flush=True)

    # 구조마다 파일을 따로 저장하는 기본 방식은 원본 해상도 볼륨을 여러 벌 메모리에 올려서
    # RAM 16GB인 PC에서 메모리 부족으로 멈춘다. 번호 볼륨 하나(ml=True)로 저장한다.
    labels_path = args.out_dir / "teeth_labels.nii.gz"
    start = time.time()
    ts_api.totalsegmentator(args.volume, labels_path, task="teeth", device=args.device, ml=True)
    seconds = round(time.time() - start)

    names = class_map["teeth"]  # 번호 → 이름
    seg = np.asanyarray(nib.load(labels_path).dataobj)
    counts = np.bincount(seg.ravel(), minlength=max(names) + 1)
    found = {names[i]: int(counts[i]) for i in sorted(names) if counts[i]}

    info = {
        "task": "teeth",
        "totalsegmentator_version": importlib.metadata.version("TotalSegmentator"),
        "device": args.device,
        "gpu": gpu,
        "seconds": seconds,
        "labels": {str(i): names[i] for i in sorted(names)},
        "structures_found": len(found),
        "voxels": found,
    }
    (args.out_dir / "segmentation_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"끝: {seconds}초, 찾은 구조 {len(found)}개")
    for name in found:
        print("  ", name)


if __name__ == "__main__":
    main()
