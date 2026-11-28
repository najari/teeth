"""자동 분할 결과(segment_teeth.py의 출력)를 구조별 메시로 바꿔 GLB 파일 하나로 저장한다.

사용법:
    python labels_to_mesh.py <분할 결과 폴더> <ct.nii.gz> <출력 폴더>

분할 결과 폴더에는 teeth_labels.nii.gz와 segmentation_info.json이 있어야 한다.

출력:
    teeth_parts.glb     구조마다 메시 하나. 메시 이름이 구조 이름이다 (예: upper_right_central_incisor_fdi11)
    parts_info.json     구조별 종류, FDI 번호, 부피, 삼각형 수
    parts_*.png         확인용 렌더링

좌표는 volume_to_mesh.py와 같다. 단위 mm, three.js 좌표(Y가 위, +Z가 얼굴 앞), 원점은 CT 볼륨의 중심.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import SimpleITK as sitk
import trimesh

# 종류별 색. 뷰어도 같은 종류 이름으로 켜고 끈다.
KIND_COLOR = {
    "tooth": (0.97, 0.95, 0.90),
    "pulp": (0.85, 0.35, 0.33),
    "jaw": (0.72, 0.65, 0.48),
    "canal": (0.95, 0.75, 0.20),
    "sinus": (0.55, 0.75, 0.90),
    "implant": (0.67, 0.70, 0.74),
    "crown": (0.80, 0.86, 0.93),
    "bridge": (0.90, 0.85, 0.94),
    "other": (0.60, 0.80, 0.70),
}
KIND_ALPHA = {"jaw": 0.45, "sinus": 0.35, "other": 0.35}
# 삼각형 상한. 턱뼈처럼 큰 구조만 많이 준다.
KIND_TRIS = {"jaw": 90_000, "sinus": 15_000, "other": 15_000, "tooth": 6_000}
DEFAULT_TRIS = 3_000
# 분할은 0.5mm 격자로 계산돼서 표면에 계단이 진다. 표면을 뽑기 전에 이만큼(mm) 흐리게 해서 없앤다.
# 치수와 신경관은 가늘어서 덜 흐리게 한다.
KIND_SIGMA = {"pulp": 0.3, "canal": 0.3}
DEFAULT_SIGMA = 0.5


def classify(name: str) -> tuple[str, int | None]:
    """구조 이름에서 종류와 FDI 번호를 뽑는다. 치수는 치아 번호에 100을 더한 번호를 쓴다(예: fdi111 → 11번 치아의 치수)."""
    m = re.search(r"fdi(\d+)$", name)
    fdi = int(m.group(1)) if m else None
    if "pulp" in name:
        return "pulp", (fdi - 100 if fdi else None)
    if "canal" in name:
        return "canal", None
    if fdi and 11 <= fdi <= 48:
        return "tooth", fdi
    if "jawbone" in name:
        return "jaw", None
    if "sinus" in name:
        return "sinus", None
    if name in ("implant", "crown", "bridge"):
        return name, None
    return "other", None


def label_to_surface(labels: sitk.Image, label: int, bbox, target_tris: int, sigma_mm: float) -> pv.PolyData | None:
    """번호 볼륨에서 한 구조의 표면을 뽑는다. 그 구조가 있는 범위만 잘라서 처리한다."""
    x, y, z, sx, sy, sz = bbox
    pad = 8
    size = labels.GetSize()
    lo = [max(0, v - pad) for v in (x, y, z)]
    hi = [min(n, v + s + pad) for n, v, s in zip(size, (x, y, z), (sx, sy, sz))]
    crop = sitk.RegionOfInterest(labels, [h - l for l, h in zip(lo, hi)], lo)
    soft = sitk.SmoothingRecursiveGaussian(sitk.Cast(crop == label, sitk.sitkFloat32), sigma_mm)
    grid = pv.ImageData(dimensions=soft.GetSize(), spacing=soft.GetSpacing(), origin=soft.GetOrigin())
    grid.point_data["v"] = sitk.GetArrayFromImage(soft).ravel()
    surf = grid.contour([0.5], scalars="v", method="flying_edges")
    if surf.n_cells == 0:
        return None
    surf = surf.triangulate().smooth_taubin(n_iter=30, pass_band=0.08)
    if surf.n_cells > target_tris:
        surf = surf.decimate(1.0 - target_tris / surf.n_cells)
    return surf.compute_normals(auto_orient_normals=True, split_vertices=False, consistent_normals=True)


def to_trimesh(surf: pv.PolyData, center, color, alpha, name) -> trimesh.Trimesh:
    p = surf.points - center
    verts = np.column_stack([p[:, 0], p[:, 2], -p[:, 1]]).astype(np.float32)
    n = surf.point_data["Normals"]
    normals = np.column_stack([n[:, 0], n[:, 2], -n[:, 1]]).astype(np.float32)
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=[*color, alpha],
            metallicFactor=0.0,
            roughnessFactor=0.5,
            alphaMode="BLEND" if alpha < 1 else "OPAQUE",
            doubleSided=True,
        )
    )
    return mesh


def render_previews(surfaces: dict, kinds: dict, center, out_dir: Path) -> None:
    views = {
        "front": ((0, -1, 0), (0, 0, 1)),
        "right": ((-1, 0, 0), (0, 0, 1)),
        "left": ((1, 0, 0), (0, 0, 1)),
        "below": ((0, 0, -1), (0, -1, 0)),
    }
    teeth_pts = [s.points for n, s in surfaces.items() if kinds[n] == "tooth"]
    focus = np.vstack(teeth_pts).mean(axis=0) if teeth_pts else np.array(center)
    for view, (direction, up) in views.items():
        pl = pv.Plotter(off_screen=True, window_size=(1100, 850))
        pl.set_background("#e7ecea")
        for name, surf in surfaces.items():
            kind = kinds[name]
            pl.add_mesh(surf, color=KIND_COLOR[kind], opacity=KIND_ALPHA.get(kind, 1.0), smooth_shading=True)
        pl.camera_position = [tuple(focus + np.array(direction) * 260.0), tuple(focus), up]
        pl.camera.view_angle = 28
        pl.screenshot(str(out_dir / f"parts_{view}.png"))
        pl.close()


def main() -> None:
    seg_dir, volume, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 메시 좌표의 원점을 volume_to_mesh.py와 맞추려고 CT 볼륨의 중심을 쓴다.
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(volume))
    reader.ReadImageInformation()
    center = np.array([o + 0.5 * n * s for o, n, s in zip(reader.GetOrigin(), reader.GetSize(), reader.GetSpacing())])

    seg_info = json.loads((seg_dir / "segmentation_info.json").read_text(encoding="utf-8"))
    names = {int(i): name for i, name in seg_info["labels"].items()}
    labels = sitk.ReadImage(str(seg_dir / "teeth_labels.nii.gz"), sitk.sitkUInt8)
    voxel_mm3 = float(np.prod(labels.GetSpacing()))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)

    scene = trimesh.Scene()
    surfaces, kinds, parts = {}, {}, []
    for label in sorted(stats.GetLabels()):
        if label not in names:
            continue
        name = names[label]
        kind, fdi = classify(name)
        volume_mm3 = stats.GetNumberOfPixels(label) * voxel_mm3
        surf = label_to_surface(
            labels, label, stats.GetBoundingBox(label),
            KIND_TRIS.get(kind, DEFAULT_TRIS), KIND_SIGMA.get(kind, DEFAULT_SIGMA),
        )
        if surf is None:
            print(f"{name:55s} {kind:8s} {volume_mm3:9.1f} mm3  너무 작아 메시에서 빠짐", flush=True)
            continue
        scene.add_geometry(
            to_trimesh(surf, center, KIND_COLOR[kind], KIND_ALPHA.get(kind, 1.0), name),
            node_name=name,
            geom_name=name,
        )
        surfaces[name], kinds[name] = surf, kind
        parts.append({"name": name, "kind": kind, "fdi": fdi, "volume_mm3": round(volume_mm3, 1), "triangles": int(surf.n_cells)})
        print(f"{name:55s} {kind:8s} {volume_mm3:9.1f} mm3 {surf.n_cells:7d} tris", flush=True)

    if not parts:
        sys.exit("분할 결과에 비어 있지 않은 구조가 없습니다.")

    glb_path = out_dir / "teeth_parts.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))
    info = {
        "units": "mm",
        "axes": "three.js: +X 환자 왼쪽, +Y 위, +Z 얼굴 앞",
        "glb_mb": round(glb_path.stat().st_size / 1e6, 2),
        "parts": parts,
    }
    (out_dir / "parts_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    render_previews(surfaces, kinds, center, out_dir)
    teeth = sorted(p["fdi"] for p in parts if p["kind"] == "tooth")
    print(f"\n구조 {len(parts)}개, 치아 {len(teeth)}개: {teeth}")
    print(f"저장: {glb_path} ({info['glb_mb']} MB)")


if __name__ == "__main__":
    main()
