"""자동 분할한 치아를 치열 모형(index.html)에 끼워 넣을 수 있는 데이터 파일로 바꾼다.

사용법:
    python teeth_for_model.py <분할 결과 폴더> <ct.nii.gz> <출력 ct-teeth.js> [--remap 42:41 ...]

치열 모형은 치아를 표준 치열궁 위에 한 개씩 세운다. 그래서 치아마다 자기 좌표로 바꿔서 내보낸다.
    원점  치아 목(머리와 뿌리의 경계)의 가운데
    +y    뿌리에서 머리 쪽 (치아의 긴 축)
    +z    볼·입술 쪽 (치열궁의 바깥)
    +x    y × z
머리(crown)와 뿌리(root)는 y=0에서 잘라 따로 담는다. 모형이 치료 단계에 따라 둘을 따로 켜고 끄기 때문이다.

--remap은 분할 모델이 붙인 번호를 모형의 번호로 바꿀 때 쓴다(예: 42:41).

출력 파일에는 환자의 CT에서 뽑은 치아 모양이 들어 있다. 공개할지는 따로 정한다.
"""
import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv
import SimpleITK as sitk

from labels_to_mesh import label_to_surface

UNIT_MM = 0.01          # 좌표를 0.01mm 단위 정수로 담는다
MIN_VOLUME_MM3 = 100    # 이보다 작으면 조각으로 보고 뺀다
UPPER = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
LOWER = [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37]


def lps_to_three(p: np.ndarray) -> np.ndarray:
    """환자 좌표(x 왼쪽, y 뒤, z 위)를 모형 좌표(x 왼쪽, y 위, z 앞)로 돌린다."""
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def to_part(clipped: pv.PolyData) -> dict:
    """잘라 낸 메시에서 꼭짓점과 삼각형을 꺼낸다."""
    tri = clipped.triangulate().clean()
    return {"v": np.asarray(tri.points), "f": tri.faces.reshape(-1, 4)[:, 1:]}


def pack(part: dict) -> dict:
    return {
        "p": b64(np.round(part["v"] / UNIT_MM).astype("<i2")),
        "i": b64(part["f"].astype("<u2")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seg_dir", type=Path)
    ap.add_argument("volume", type=Path)
    ap.add_argument("out_js", type=Path)
    ap.add_argument("--remap", nargs="*", default=[], help="분할 번호:모형 번호")
    ap.add_argument("--enamel", type=float, default=3300.0, help="법랑질로 보는 밝기. 치아 목 높이를 정하는 데 쓴다")
    ap.add_argument("--tris", type=int, default=2600, help="치아 하나의 삼각형 수")
    args = ap.parse_args()
    remap = {int(a): int(b) for a, b in (item.split(":") for item in args.remap)}

    info = json.loads((args.seg_dir / "segmentation_info.json").read_text(encoding="utf-8"))
    names = {int(i): n for i, n in info["labels"].items()}
    labels = sitk.ReadImage(str(args.seg_dir / "teeth_labels.nii.gz"), sitk.sitkUInt8)
    ct = sitk.ReadImage(str(args.volume))
    lab_arr = sitk.GetArrayViewFromImage(labels)
    ct_arr = sitk.GetArrayViewFromImage(ct)
    spacing, origin = np.array(labels.GetSpacing()), np.array(labels.GetOrigin())
    voxel_mm3 = float(spacing.prod())
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)

    # 1) 치아마다 화소 좌표, 긴 축, 밝기를 모은다.
    teeth = {}
    for label in stats.GetLabels():
        name = names.get(label, "")
        m = re.search(r"fdi(\d+)$", name)
        if not m or "pulp" in name or "canal" in name or not 11 <= int(m.group(1)) <= 48:
            continue
        volume = stats.GetNumberOfPixels(label) * voxel_mm3
        fdi = remap.get(int(m.group(1)), int(m.group(1)))
        if volume < MIN_VOLUME_MM3:
            print(f"{m.group(1)}번: {volume:.0f}mm3, 너무 작아 뺌")
            continue
        x, y, z, sx, sy, sz = stats.GetBoundingBox(label)
        sub = lab_arr[z:z + sz, y:y + sy, x:x + sx] == label
        zz, yy, xx = np.nonzero(sub)
        pts = lps_to_three(np.column_stack([xx + x, yy + y, zz + z]) * spacing + origin)
        centroid = pts.mean(axis=0)
        axis = np.linalg.svd(pts - centroid, full_matrices=False)[2][0]
        upper = fdi < 30
        if (axis[1] > 0) == upper:      # 위 치아는 머리가 아래로, 아래 치아는 머리가 위로 간다
            axis = -axis
        teeth[fdi] = {
            "label": label, "bbox": (x, y, z, sx, sy, sz), "pts": pts, "c": centroid, "y": axis,
            "hu": ct_arr[z:z + sz, y:y + sy, x:x + sx][sub], "volume": volume, "from": int(m.group(1)),
        }

    # 2) 치열궁의 바깥 방향: 이웃 치아의 중심을 이은 선에 수직이고, 치열 중심에서 멀어지는 쪽.
    for order in (UPPER, LOWER):
        seq = [f for f in order if f in teeth]
        if not seq:
            continue
        centre = np.mean([teeth[f]["c"] for f in seq], axis=0)
        for i, f in enumerate(seq):
            a, b = teeth[seq[max(i - 1, 0)]]["c"], teeth[seq[min(i + 1, len(seq) - 1)]]["c"]
            tangent = np.array([b[0] - a[0], 0.0, b[2] - a[2]])
            normal = np.array([tangent[2], 0.0, -tangent[0]])
            if normal @ (teeth[f]["c"] - centre) < 0:
                normal = -normal
            t = teeth[f]
            z_axis = normal - (normal @ t["y"]) * t["y"]
            t["z"] = z_axis / np.linalg.norm(z_axis)
            t["x"] = np.cross(t["y"], t["z"])

    # 3) 치아 목 높이를 정하고, 자기 좌표의 메시로 만들어 머리와 뿌리로 자른다.
    out = {}
    print("번호  폭    두께  머리높이 뿌리길이  목 높이 기준   삼각형(머리+뿌리)")
    for fdi in sorted(teeth):
        t = teeth[fdi]
        frame = np.column_stack([t["x"], t["y"], t["z"]])      # 열이 자기 좌표의 축
        local = (t["pts"] - t["c"]) @ frame
        surf = label_to_surface(labels, t["label"], t["bbox"], args.tris, 0.5)
        verts = (lps_to_three(np.asarray(surf.points)) - t["c"]) @ frame
        tip, apex = verts[:, 1].max(), verts[:, 1].min()
        length = tip - apex

        # 법랑질이 끝나는 높이를 치아 목으로 본다. 법랑질이 거의 없으면 표준 비율(길이의 42%)을 쓴다.
        enamel_y = local[t["hu"] > args.enamel, 1]
        if len(enamel_y) >= 200:
            crown_h, basis = tip - np.percentile(enamel_y, 5), "법랑질"
        else:
            crown_h, basis = 0.42 * length, "표준 비율"
        clamped = float(np.clip(crown_h, 0.30 * length, 0.55 * length))
        if clamped != crown_h:
            basis += "(범위 제한)"
        neck = tip - clamped
        ring = local[np.abs(local[:, 1] - neck) < 0.6]
        offset = np.array([ring[:, 0].mean(), neck, ring[:, 2].mean()])

        faces = surf.faces.reshape(-1, 4)[:, 1:]
        v = verts - offset
        signed_volume = np.einsum("ij,ij->i", v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]])).sum() / 6
        if signed_volume < 0:           # 면이 바깥을 향하게 맞춘다
            faces = faces[:, ::-1]
        whole = pv.PolyData(v, np.column_stack([np.full(len(faces), 3), faces]).ravel())
        crown = to_part(whole.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False))
        root = to_part(whole.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=True))
        w = float(crown["v"][:, 0].max() - crown["v"][:, 0].min())
        d = float(crown["v"][:, 2].max() - crown["v"][:, 2].min())
        out[str(fdi)] = {
            "w": round(w, 2), "d": round(d, 2), "h": round(clamped, 2), "root": round(float(length - clamped), 2),
            "crown": pack(crown), "rootMesh": pack(root),
        }
        note = f"  (분할 번호 {t['from']})" if t["from"] != fdi else ""
        print(f"{fdi:3d} {w:6.1f} {d:6.1f} {clamped:7.1f} {length - clamped:8.1f}   {basis:14s} {len(crown['f'])}+{len(root['f'])}{note}")

    data = {"unit": UNIT_MM, "teeth": out}
    header = (
        "// CT에서 뽑은 치아별 실제 모양. tools/ct3d/teeth_for_model.py가 만든 파일이라 직접 고치지 않는다.\n"
        "// 치아마다 자기 좌표(원점: 치아 목, +y: 머리 쪽, +z: 볼·입술 쪽)이고 좌표 단위는 unit(mm)이다.\n"
    )
    args.out_js.write_text(header + "window.CT_TEETH = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"\n치아 {len(out)}개, 저장: {args.out_js} ({args.out_js.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
