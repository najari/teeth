"""구강 스캔(위턱·아래턱 STL)을 CT 치아 메시에 맞춰(정합) CT 뷰어에서 함께 보이게 한다.

사용법:
    python scan_to_ct.py <스캔 폴더> <teeth_parts.glb> <출력 폴더> [--tris 150000]

스캔 폴더에는 이름에 UpperJaw, LowerJaw가 들어간 STL이 있어야 한다(CEREC/inLab에서 내보낸 형식).
스캔 좌표는 z가 교합 방향(위턱 쪽이 +z), 앞니가 -y 쪽, 환자 왼쪽이 +x라고 본다.

맞추는 방법
    CT 쪽: labels_to_mesh.py가 만든 치아 메시에서 치아마다 끝에서 6mm(머리 부분)의 점. 뿌리 조각은 뺀다.
    스캔 쪽: 교합면에서 7mm 안의 띠. 잇몸과 입천장이 넓어 전체 중심이 치아보다 한참 위(아래)에 있기 때문이다.
    강체 ICP로 맞추되 매번 가장 먼 30%는 버린다. 시작 위치에 민감해서 수평 회전 5가지 × 좌우·앞뒤 9가지로
    시작해 보고 가장 잘 맞는 것을 다듬는다.

출력:
    teeth_parts_scan.glb     teeth_parts.glb에 upper_scan, lower_scan을 더한 것. 스캔은 각 턱의 CT 치아에 맞춘 자리에 놓인다
    scan_registration.json   턱마다 스캔 → CT 뷰어 좌표 변환(4×4), 치아별 거리, CT와 스캔의 물림 차이

출력에는 환자의 형상이 들어 있다. 공개 저장소에 올리지 않는다.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

CROWN_MM = 6.0        # CT 치아 끝에서 이만큼을 머리로 본다
BAND_MM = 7.0         # 스캔 교합면에서 이만큼을 치아 띠로 본다
KEEP = 0.7            # ICP에서 가까운 쪽 이 비율만 쓴다
SCAN_COLOR = (0.85, 0.55, 0.55)
SCAN_ALPHA = 0.4
# 스캔 좌표 = BASE @ CT 뷰어 좌표 (CT 뷰어: x 환자 왼쪽, y 위, z 앞 / 스캔: x 환자 왼쪽, y 뒤, z 위턱 쪽)
BASE = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)


def kabsch(a: np.ndarray, b: np.ndarray):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cb - r @ ca


def icp(src, tree, tgt, r, t, iters):
    for _ in range(iters):
        moved = src @ r.T + t
        d, i = tree.query(moved)
        k = d <= np.quantile(d, KEEP)
        dr, dt = kabsch(moved[k], tgt[i[k]])
        r, t = dr @ r, dr @ t + dt
    return r, t


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def fdi_of(name: str):
    m = re.search(r"fdi(\d+)$", name)
    return int(m.group(1)) if m else None


def crown_points(teeth: dict, upper: bool, rng):
    pts, owner = [], []
    for f, g in teeth.items():
        if (f < 30) != upper:
            continue
        s, _ = trimesh.sample.sample_surface(g, 5000, seed=f)
        keep = s[:, 1] < s[:, 1].min() + CROWN_MM if upper else s[:, 1] > s[:, 1].max() - CROWN_MM   # 위 치아는 머리가 아래
        pts.append(s[keep])
        owner.append(np.full(keep.sum(), f))
    return np.concatenate(pts), np.concatenate(owner)


def register(scan: trimesh.Trimesh, teeth: dict, upper: bool, rng) -> tuple[np.ndarray, dict]:
    """CT 뷰어 좌표 → 스캔 좌표 변환(4×4)과 치아별 거리를 돌려준다."""
    full = np.asarray(scan.vertices)
    z = full[:, 2]
    band = z < np.percentile(z, 0.5) + BAND_MM if upper else z > np.percentile(z, 99.5) - BAND_MM
    tgt = full[band]
    tree = cKDTree(tgt)
    src, owner = crown_points(teeth, upper, rng)
    sub = src[rng.choice(len(src), min(2500, len(src)), replace=False)]

    starts = []
    for yaw in (-10, -5, 0, 5, 10):
        r0 = rot_z(yaw) @ BASE
        t0 = tgt.mean(0) - r0 @ src.mean(0)
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                r, t = icp(sub, tree, tgt, r0, t0 + np.array([dx, dy, 0]), 20)
                d, _ = tree.query(sub @ r.T + t)
                starts.append((np.median(d), r, t))
    starts.sort(key=lambda s: s[0])
    best = None
    for _, r, t in starts[:3]:              # 앞선 세 개만 모든 점으로 다듬는다
        r, t = icp(src, tree, tgt, r, t, 60)
        d, _ = tree.query(src @ r.T + t)
        if best is None or np.median(d) < best[0]:
            best = (np.median(d), r, t)
    _, r, t = best
    d, _ = cKDTree(full).query(src @ r.T + t)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = r, t
    per_tooth = {int(f): round(float(np.median(d[owner == f])), 3) for f in np.unique(owner)}
    return T, {"median_mm": round(float(np.median(d)), 3), "p90_mm": round(float(np.percentile(d, 90)), 3), "per_tooth_mm": per_tooth}


def scan_mesh(scan: trimesh.Trimesh, to_ct: np.ndarray, tris: int, name: str) -> trimesh.Trimesh:
    poly = pv.wrap(scan)
    if poly.n_cells > tris:
        poly = poly.decimate(1.0 - tris / poly.n_cells)
    poly = poly.compute_normals(auto_orient_normals=False, consistent_normals=True, split_vertices=False)
    v = poly.points @ to_ct[:3, :3].T + to_ct[:3, 3]
    n = poly.point_data["Normals"] @ to_ct[:3, :3].T
    m = trimesh.Trimesh(vertices=v.astype(np.float32), faces=poly.faces.reshape(-1, 4)[:, 1:], vertex_normals=n.astype(np.float32), process=False)
    m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=[*SCAN_COLOR, SCAN_ALPHA], metallicFactor=0.0, roughnessFactor=0.7, alphaMode="BLEND", doubleSided=True))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("teeth_glb", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tris", type=int, default=150_000, help="스캔 하나의 삼각형 수(뷰어용으로 줄임)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    scene = trimesh.load(args.teeth_glb)
    teeth = {}
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        f = fdi_of(node)
        if f and 11 <= f <= 48 and "pulp" not in node and "canal" not in node and not node.startswith(("remnant", "fragment")):
            g = scene.geometry[geom_name].copy()
            g.apply_transform(transform)
            teeth[f] = g

    result, transforms = {}, {}
    for jaw, key in (("UpperJaw", "upper"), ("LowerJaw", "lower")):
        path = next((p for p in args.scan_dir.glob("*.stl") if jaw.lower() in p.name.lower()), None)
        if path is None:
            raise SystemExit(f"{args.scan_dir}에 이름에 {jaw}가 들어간 STL이 없습니다.")
        scan = trimesh.load(path)     # STL은 삼각형마다 꼭짓점을 따로 가지므로 합쳐서 읽는다(process=True)
        T, fit = register(scan, teeth, key == "upper", rng)
        to_ct = np.linalg.inv(T)
        transforms[key] = T
        result[key] = {"scan_to_ct": np.round(to_ct, 6).tolist(), **fit}
        far = [f for f, v in fit["per_tooth_mm"].items() if v > 0.6]
        print(f"{'위' if key == 'upper' else '아래'}턱: 삼각형 {len(scan.faces)}개, 치아 머리에서 스캔까지 거리 중앙값 {fit['median_mm']:.2f}mm, 90% {fit['p90_mm']:.2f}mm")
        print("   치아별(mm): " + " ".join(f"{f}:{v:.2f}" for f, v in fit["per_tooth_mm"].items()))
        if fit["median_mm"] > 0.4:
            print("   ※ 잘 맞지 않았습니다. 스캔의 좌표 방향이 다르거나 CT와 다른 시기의 스캔일 수 있습니다.")
        if far:
            print(f"   ※ 0.6mm 넘게 어긋난 치아: {far}. 두 촬영 사이에 바뀌었거나 분할이 다를 수 있습니다.")
        name = f"{key}_scan"
        scene.add_geometry(scan_mesh(scan, to_ct, args.tris, name), node_name=name, geom_name=name)

    # 물림: 스캔은 물린 채 찍었다. 아래턱 CT 자리와, 위턱 기준으로 본 스캔의 아래턱 자리의 차이
    lower_pts = np.concatenate([g.vertices for f, g in teeth.items() if f > 30])
    low_in_scan = lower_pts @ transforms["lower"][:3, :3].T + transforms["lower"][:3, 3]
    bite = (low_in_scan - transforms["upper"][:3, 3]) @ transforms["upper"][:3, :3]      # 위턱 기준으로 CT 좌표에 되돌림
    move = bite - lower_pts
    R = transforms["upper"][:3, :3].T @ transforms["lower"][:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    result["bite"] = {
        "note": "CT의 아래턱을 스캔의 물린 자리로 옮기는 움직임(CT 뷰어 좌표: x 환자 왼쪽, y 위, z 앞)",
        "rotation_deg": round(angle, 2),
        "mean_move_mm": np.round(move.mean(0), 2).tolist(),
        "median_distance_mm": round(float(np.median(np.linalg.norm(move, axis=1))), 2),
    }
    print(f"물림: CT 아래턱을 스캔의 물린 자리로 옮기려면 회전 {angle:.2f}°, 평균 이동 {np.round(move.mean(0), 1).tolist()}mm (x 왼쪽, y 위, z 앞)")

    out_glb = args.out_dir / "teeth_parts_scan.glb"
    out_glb.write_bytes(scene.export(file_type="glb"))
    (args.out_dir / "scan_registration.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {out_glb} ({out_glb.stat().st_size / 1e6:.1f} MB), {args.out_dir / 'scan_registration.json'}")


if __name__ == "__main__":
    main()
