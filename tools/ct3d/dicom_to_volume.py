"""DICOM 묶음을 하나의 볼륨 파일(NIfTI)로 바꾸고 확인용 그림을 만든다.

사용법:
    python dicom_to_volume.py <DICOM 폴더> <출력 폴더>

출력:
    ct.nii.gz        원본 해상도 볼륨 (자동 분할 도구의 입력)
    preview_*.png    세 방향 가운데 단면과 최대값 투영(MIP)
    volume_info.json 크기, 간격, 화소 값 분포

환자 식별 정보는 출력에 넣지 않는다. NIfTI에는 형상과 화소 값만 들어간다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


def read_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise SystemExit(f"DICOM 시리즈를 찾지 못했습니다: {dicom_dir}")
    # 시리즈가 여러 개면 장수가 가장 많은 것을 쓴다.
    files = max(
        (reader.GetGDCMSeriesFileNames(str(dicom_dir), sid) for sid in series_ids),
        key=len,
    )
    reader.SetFileNames(files)
    return reader.Execute()


def to_png(arr2d: np.ndarray, lo: float, hi: float, path: Path) -> None:
    a = np.clip((arr2d.astype(np.float32) - lo) / (hi - lo), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def main() -> None:
    dicom_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    img = read_series(dicom_dir)
    vol = sitk.GetArrayViewFromImage(img)  # 축 순서: z, y, x
    size, spacing = img.GetSize(), img.GetSpacing()

    pct = [0.1, 1, 5, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9]
    sample = vol[::2, ::2, ::2]
    info = {
        "size_xyz": list(size),
        "spacing_mm_xyz": [round(s, 4) for s in spacing],
        "extent_mm_xyz": [round(n * s, 1) for n, s in zip(size, spacing)],
        "direction": [round(d, 4) for d in img.GetDirection()],
        "dtype": str(vol.dtype),
        "min": int(vol.min()),
        "max": int(vol.max()),
        "percentiles": {str(p): float(np.percentile(sample, p)) for p in pct},
    }
    (out_dir / "volume_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))

    sitk.WriteImage(img, str(out_dir / "ct.nii.gz"), useCompression=True)

    # 확인용 그림. SimpleITK 배열은 z가 아래에서 위로 가므로 관상면과 시상면은 뒤집어 머리가 위로 오게 한다.
    lo, hi = info["percentiles"]["5"], info["percentiles"]["99.9"]
    nz, ny, nx = vol.shape
    to_png(vol[nz // 2], lo, hi, out_dir / "preview_axial.png")
    to_png(vol[::-1, ny // 2, :], lo, hi, out_dir / "preview_coronal.png")
    to_png(vol[::-1, :, nx // 2], lo, hi, out_dir / "preview_sagittal.png")
    to_png(vol.max(axis=1)[::-1], lo, hi, out_dir / "preview_mip_front.png")
    to_png(vol.max(axis=2)[::-1], lo, hi, out_dir / "preview_mip_side.png")
    to_png(vol.max(axis=0), lo, hi, out_dir / "preview_mip_top.png")
    print("저장:", out_dir)


if __name__ == "__main__":
    main()
