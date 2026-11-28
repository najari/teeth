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

find_remnants.py에서 번호를 붙인 뿌리 조각도 같은 형식으로 담고 remnant 표시를 붙인다.
뿌리 조각은 위쪽 일부를 잇몸 위로 나온 부분(crown)으로, 나머지를 뿌리(root)로 나눈다.

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

from labels_to_mesh import REMNANT_HU, hu_surface, label_to_surface

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


# 모형 좌표의 기준. index.html의 표준 치열과 맞춘 값이다.
FRONT_NECK_Z = -3.4                 # 위 가운데 앞니 목의 앞뒤 위치
UPPER_TIP_Y, LOWER_TIP_Y = 0.4, -0.4  # 위아래 치아 끝의 평균 높이 (두 끝 사이 0.8mm)
PIVOT = np.array([0.0, 14.0, -80.0])  # index.html이 아래턱을 벌릴 때 쓰는 회전 중심


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 방향을 b 방향으로 돌리는 가장 작은 회전."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def plane_normal(points: np.ndarray) -> np.ndarray:
    """점들에 가장 잘 맞는 평면의 법선. 위(+y)를 향하게 한다."""
    n = np.linalg.svd(points - points.mean(axis=0))[2][-1]
    return n if n[1] > 0 else -n


def fit_arch(necks: np.ndarray, start) -> list:
    """index.html의 치열궁 곡선 x = ±a·sin t, z = z0 + b·(cos t − 1)을 치아 목 위치에 맞춘다."""
    from scipy.optimize import minimize
    pts = np.column_stack([np.abs(necks[:, 0]), necks[:, 2]])
    t = np.linspace(0, np.pi * 0.6, 800)

    def cost(p):
        a, b, z0 = p
        curve = np.column_stack([a * np.sin(t), z0 + b * (np.cos(t) - 1)])
        return np.sum(np.min(np.linalg.norm(pts[:, None, :] - curve[None], axis=2), axis=1) ** 2)

    res = minimize(cost, start, method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-4, "maxiter": 4000})
    rms = float(np.sqrt(res.fun / len(pts)))
    return [round(float(v), 3) for v in res.x], rms


def place_in_model(place: dict, out: dict) -> dict:
    """CT 좌표의 치아 자리를 모형 좌표로 옮겨 out에 pos, axes를 넣고, 위아래 치열궁을 돌려준다.

    1) 위 치아 끝들이 이루는 교합면을 수평으로, 정중선을 앞(+z)으로 돌린다.
    2) 위 가운데 앞니 사이를 x = 0, 그 목을 z = FRONT_NECK_Z에 두고, 위 치아 끝의 평균 높이를 UPPER_TIP_Y에 둔다.
    3) CT를 찍을 때 입을 조금 벌리고 있었으므로, 아래 치아는 모형의 턱 회전 중심(PIVOT)을 기준으로
       교합면이 수평이 되게 돌린 뒤 치아 끝의 평균 높이를 LOWER_TIP_Y에 둔다.
    """
    upper = {f: p for f, p in place.items() if f < 30}
    lower = {f: p for f, p in place.items() if f > 30}
    up_tips = np.array([p["tip"] for p in upper.values() if not p["remnant"]])

    r1 = rotation_between(plane_normal(up_tips), np.array([0.0, 1.0, 0.0]))

    def side_back(quads):        # 그 쪽에서 가장 안쪽 치아 목
        cand = [f for f in upper if f // 10 in quads and not upper[f]["remnant"]]
        return r1 @ upper[max(cand, key=lambda f: f % 10)]["neck"]

    incisors = [f for f in (11, 21) if f in upper] or sorted(upper, key=lambda f: f % 10)[:2]
    front = np.mean([r1 @ upper[f]["neck"] for f in incisors], axis=0)
    back = (side_back((1,)) + side_back((2,))) / 2
    d = front - back
    rot = rotation_y(-np.arctan2(d[0], d[2])) @ r1

    front = rot @ np.mean([upper[f]["neck"] for f in incisors], axis=0)
    shift = np.array([-front[0], UPPER_TIP_Y - np.mean((up_tips @ rot.T)[:, 1]), FRONT_NECK_Z - front[2]])

    def to_model(p):
        return p @ rot.T + shift

    # 아래턱: 위턱 기준으로 옮긴 뒤, 턱 회전 중심을 기준으로 다물게 돌린다.
    low_tips = to_model(np.array([p["tip"] for p in lower.values() if not p["remnant"]]))
    r_low = rotation_between(plane_normal(low_tips), np.array([0.0, 1.0, 0.0]))
    closed_tips = (low_tips - PIVOT) @ r_low.T + PIVOT
    low_shift = np.array([0.0, LOWER_TIP_Y - closed_tips[:, 1].mean(), 0.0])
    opened = np.degrees(np.arccos(np.clip(plane_normal(low_tips)[1], -1, 1)))
    print(f"\n자리 맞추기: CT에서 아래턱은 위턱보다 {opened:.1f}° 벌어져 있어 모형의 턱 회전 중심으로 다물게 돌렸습니다.")

    necks = {"U": [], "L": []}
    for f, p in place.items():
        axes = rot @ p["axes"]
        neck = to_model(p["neck"])
        if f > 30:
            axes = r_low @ axes
            neck = (neck - PIVOT) @ r_low.T + PIVOT + low_shift
        out[str(f)]["pos"] = [round(float(v), 3) for v in neck]
        out[str(f)]["axes"] = [round(float(v), 4) for v in axes.T.ravel()]   # x축, y축, z축 순서
        necks["U" if f < 30 else "L"].append(neck)

    arch = {}
    for jaw, start in (("U", (28, 46, 0)), ("L", (26, 43, -3))):
        params, rms = fit_arch(np.array(necks[jaw]), start)
        arch[jaw] = params
        print(f"{'위' if jaw == 'U' else '아래'}턱 치열궁 a={params[0]:.1f} b={params[1]:.1f} z0={params[2]:.1f}  (치아 목과의 거리 RMS {rms:.2f}mm)")
    return arch


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

    # 1-2) 뿌리만 남은 치아 (find_remnants.py에서 번호를 붙인 것만).
    #       짧고 뭉툭해서 긴 축을 모양으로 잡을 수 없다. 그 턱 교합면에 수직인 방향을 긴 축으로 쓴다.
    #       모형은 이 축으로 그 자리의 임플란트와 보철을 세우므로, 이웃 치아의 기울기를 빌리면 임플란트가 함께 기운다.
    def occlusal_axis(upper: bool) -> np.ndarray:
        tips = [t["pts"][np.argmax((t["pts"] - t["c"]) @ t["y"])] for f, t in teeth.items() if (f < 30) == upper]
        n = plane_normal(np.array(tips))                 # 위(+y)를 향함
        return -n if upper else n                        # 위 치아는 머리가 아래로 간다
    remnant_path = args.seg_dir / "remnant_labels.nii.gz"
    if remnant_path.exists():
        remnants = sitk.ReadImage(str(remnant_path), sitk.sitkUInt8)
        rem_arr = sitk.GetArrayViewFromImage(remnants)
        rem_stats = sitk.LabelShapeStatisticsImageFilter()
        rem_stats.Execute(remnants)
        axes_by_jaw = {upper: occlusal_axis(upper) for upper in (True, False) if any((f < 30) == upper for f in teeth)}
        for cand in json.loads((args.seg_dir / "remnants_info.json").read_text(encoding="utf-8"))["candidates"]:
            fdi = cand["fdi"]
            if not fdi:
                continue
            if fdi in teeth:
                print(f"{fdi}번: 자동 분할한 치아가 이미 있어 뿌리 조각 {cand['id']}번을 쓰지 않음")
                continue
            x, y, z, sx, sy, sz = rem_stats.GetBoundingBox(cand["id"])
            sub = rem_arr[z:z + sz, y:y + sy, x:x + sx] == cand["id"]
            zz, yy, xx = np.nonzero(sub)
            pts = lps_to_three(np.column_stack([xx + x, yy + y, zz + z]) * spacing + origin)
            teeth[fdi] = {
                "label": cand["id"], "bbox": (x, y, z, sx, sy, sz), "pts": pts, "c": pts.mean(axis=0),
                "y": axes_by_jaw[fdi < 30], "hu": None, "volume": cand["volume_mm3"], "from": fdi, "remnant": remnants,
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
    out, place = {}, {}
    print("번호  폭    두께  머리높이 뿌리길이  목 높이 기준   삼각형(머리+뿌리)")
    for fdi in sorted(teeth):
        t = teeth[fdi]
        frame = np.column_stack([t["x"], t["y"], t["z"]])      # 열이 자기 좌표의 축
        local = (t["pts"] - t["c"]) @ frame
        is_remnant = "remnant" in t
        if is_remnant:
            surf = hu_surface(ct, t["remnant"], t["label"], t["bbox"], REMNANT_HU, args.tris)
        else:
            surf = label_to_surface(labels, t["label"], t["bbox"], args.tris, 0.5)
        verts = (lps_to_three(np.asarray(surf.points)) - t["c"]) @ frame
        tip, apex = verts[:, 1].max(), verts[:, 1].min()
        length = tip - apex

        if is_remnant:
            # 머리가 없으므로 치아 목을 정할 근거가 없다. 위쪽 35%(1.5~3mm)를 잇몸 위로 나온 부분으로 본다.
            clamped, basis = float(np.clip(0.35 * length, 1.5, 3.0)), "뿌리 조각"
            neck = tip - clamped
            offset = np.array([local[:, 0].mean(), neck, local[:, 2].mean()])
        else:
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
        widest = v if is_remnant else crown["v"]      # 뿌리 조각은 전체 폭을, 치아는 머리 폭을 자리 폭으로 쓴다
        w = float(widest[:, 0].max() - widest[:, 0].min())
        d = float(widest[:, 2].max() - widest[:, 2].min())
        out[str(fdi)] = {
            "w": round(w, 2), "d": round(d, 2), "h": round(clamped, 2), "root": round(float(length - clamped), 2),
            "crown": pack(crown), "rootMesh": pack(root),
        }
        if is_remnant:
            out[str(fdi)]["remnant"] = True
        place[fdi] = {
            "neck": t["c"] + frame @ offset,
            "tip": t["c"] + frame @ np.array([offset[0], tip, offset[2]]),
            "axes": frame, "remnant": is_remnant,
        }
        note = f"  (분할 번호 {t['from']})" if t["from"] != fdi else ""
        print(f"{fdi:3d} {w:6.1f} {d:6.1f} {clamped:7.1f} {length - clamped:8.1f}   {basis:14s} {len(crown['f'])}+{len(root['f'])}{note}")

    # 4) 치아마다 실제 자리(치아 목의 위치와 축)를 모형 좌표로 옮긴다.
    arch = place_in_model(place, out)

    n_remnants = sum(1 for t in out.values() if t.get("remnant"))
    data = {"unit": UNIT_MM, "arch": arch, "teeth": out}
    header = (
        "// CT에서 뽑은 치아별 실제 모양과 자리. tools/ct3d/teeth_for_model.py가 만든 파일이라 직접 고치지 않는다.\n"
        "// 모양은 치아마다 자기 좌표(원점: 치아 목, +y: 머리 쪽, +z: 볼·입술 쪽)이고 좌표 단위는 unit(mm)이다.\n"
        "// pos와 axes는 모형 좌표에서 그 자기 좌표의 원점과 x, y, z축이다. arch는 치아 목에 맞춘 치열궁 [a, b, z0]이다.\n"
    )
    args.out_js.write_text(header + "window.CT_TEETH = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"\n치아 {len(out) - n_remnants}개, 뿌리만 남은 치아 {n_remnants}개, 저장: {args.out_js} ({args.out_js.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
