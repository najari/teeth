"""CT 볼륨에서 화소 값 임계값으로 뼈와 치아 표면을 뽑아 GLB 파일로 저장한다.

사용법:
    python volume_to_mesh.py <ct.nii.gz> <출력 폴더> [--bone 값] [--teeth 값] [--enamel 값]

출력:
    ct_surface.glb    뼈(bone)와 치아(teeth) 메시 두 개. 단위 mm, three.js 좌표(Y가 위, +Z가 얼굴 앞)
    mesh_*.png        확인용 렌더링
    mesh_info.json    임계값과 삼각형 수

GPU가 필요 없다. 임계값 분할이라 치아 뿌리와 턱뼈는 나뉘지 않는다.
치아를 하나씩 나누려면 segment_teeth.py(자동 분할)를 쓴다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import SimpleITK as sitk
import trimesh

BONE_COLOR = (0.72, 0.65, 0.48)
TEETH_COLOR = (0.97, 0.95, 0.90)
AIR = -1000.0


def smooth(img: sitk.Image, sigma_mm: float) -> sitk.Image:
    return sitk.SmoothingRecursiveGaussian(sitk.Cast(img, sitk.sitkFloat32), sigma_mm)


def find_teeth(smoothed: sitk.Image, threshold: float, enamel: float, min_enamel_vox: int) -> sitk.Image:
    """치아 마스크를 만든다.

    턱뼈 겉면의 단단한 뼈도 치아 임계값을 넘는다. 그래서 임계값을 넘는 덩어리 가운데
    법랑질 밝기(enamel)인 화소가 충분히 들어 있는 덩어리만 치아로 본다.
    """
    labels = sitk.ConnectedComponent(smoothed > threshold)
    lab = sitk.GetArrayViewFromImage(labels)
    arr = sitk.GetArrayViewFromImage(smoothed)
    counts = np.bincount(lab[arr > enamel], minlength=int(lab.max()) + 1)
    keep = np.flatnonzero(counts >= min_enamel_vox)
    mask = sitk.GetImageFromArray(np.isin(lab, keep[keep > 0]).astype(np.uint8))
    mask.CopyInformation(smoothed)
    return mask


def crop_to_mask(img: sitk.Image, mask: sitk.Image, margin_mm: float = 3.0) -> sitk.Image:
    """마스크가 있는 범위만 img에서 잘라 낸다. img와 mask의 격자 간격은 달라도 된다."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    lo_mm = mask.TransformIndexToPhysicalPoint((x, y, z))
    hi_mm = mask.TransformIndexToPhysicalPoint((x + sx - 1, y + sy - 1, z + sz - 1))
    lo = img.TransformPhysicalPointToIndex([v - margin_mm for v in lo_mm])
    hi = img.TransformPhysicalPointToIndex([v + margin_mm for v in hi_mm])
    lo = [max(0, v) for v in lo]
    hi = [min(n - 1, v) for n, v in zip(img.GetSize(), hi)]
    return sitk.RegionOfInterest(img, [h - l + 1 for l, h in zip(lo, hi)], lo)


def to_grid(img: sitk.Image) -> pv.ImageData:
    grid = pv.ImageData(dimensions=img.GetSize(), spacing=img.GetSpacing(), origin=img.GetOrigin())
    grid.point_data["v"] = sitk.GetArrayFromImage(img).ravel()  # z, y, x 순서라 x가 가장 빨리 변한다
    return grid


def extract_surface(smoothed, threshold, min_cells, keep_largest, target_tris):
    surf = to_grid(smoothed).contour([threshold], scalars="v", method="flying_edges")
    surf = surf.connectivity(extraction_mode="all")
    counts = np.bincount(surf.cell_data["RegionId"])
    if keep_largest:
        keep = np.argsort(counts)[::-1][:keep_largest]
        keep = keep[counts[keep] >= min_cells]
    else:
        keep = np.flatnonzero(counts >= min_cells)
    kept = surf.extract_cells(np.isin(surf.cell_data["RegionId"], keep))
    surf = kept.extract_surface(algorithm="dataset_surface")
    surf = surf.triangulate().smooth_taubin(n_iter=30, pass_band=0.08)
    if surf.n_cells > target_tris:
        surf = surf.decimate(1.0 - target_tris / surf.n_cells)
    surf = surf.compute_normals(auto_orient_normals=True, split_vertices=False, consistent_normals=True)
    return surf, len(keep)


def to_trimesh(surf: pv.PolyData, center, color, alpha, name) -> trimesh.Trimesh:
    # 환자 좌표(LPS: x 왼쪽, y 뒤, z 위)를 three.js 좌표(x 오른쪽 화면, y 위, z 앞)로 돌린다.
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
            roughnessFactor=0.55,
            alphaMode="BLEND" if alpha < 1 else "OPAQUE",
            doubleSided=True,
        )
    )
    return mesh


def render_previews(bone, teeth, center, out_dir: Path) -> None:
    views = {
        "front": ((0, -1, 0), (0, 0, 1)),
        "right": ((-1, 0, 0), (0, 0, 1)),
        "left": ((1, 0, 0), (0, 0, 1)),
        "top": ((0, 0, 1), (0, 1, 0)),
    }
    for name, (direction, up) in views.items():
        pl = pv.Plotter(off_screen=True, window_size=(1100, 850))
        pl.set_background("#e7ecea")
        pl.add_mesh(bone, color=BONE_COLOR, opacity=0.55, smooth_shading=True)
        pl.add_mesh(teeth, color=TEETH_COLOR, smooth_shading=True, specular=0.3)
        pos = np.array(center) + np.array(direction) * 330.0
        pl.camera_position = [tuple(pos), tuple(center), up]
        pl.camera.view_angle = 28
        pl.screenshot(str(out_dir / f"mesh_{name}.png"))
        pl.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--bone", type=float, default=1000.0, help="뼈 표면 임계값")
    ap.add_argument("--teeth", type=float, default=2800.0, help="치아 표면 임계값")
    ap.add_argument("--enamel", type=float, default=3300.0, help="법랑질로 보는 밝기. 치아 덩어리를 가려내는 데 쓴다")
    ap.add_argument("--bone-tris", type=int, default=220_000)
    ap.add_argument("--teeth-tris", type=int, default=180_000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img = sitk.ReadImage(str(args.volume))
    size, spacing, origin = img.GetSize(), img.GetSpacing(), img.GetOrigin()
    center = np.array([o + 0.5 * n * s for o, n, s in zip(origin, size, spacing)])

    # 2배로 줄인 볼륨(0.4mm)에서 치아 덩어리를 찾는다.
    half = sitk.BinShrink(img, [2, 2, 2])
    half_smooth = smooth(half, 0.3)
    teeth_mask = find_teeth(half_smooth, args.teeth, args.enamel, min_enamel_vox=100)

    # 뼈: 줄인 볼륨에서 뽑는다. 치아로 잡힌 부분은 빼서 치아 메시가 뼈 껍질에 덮이지 않게 한다.
    bone_src = sitk.MaskNegated(half_smooth, sitk.BinaryDilate(teeth_mask, [2, 2, 2]), outsideValue=AIR)
    bone, n_bone = extract_surface(bone_src, args.bone, 4000, 6, args.bone_tris)

    # 치아: 원본 해상도(0.2mm)에서 치아가 있는 범위만 잘라서 뽑는다.
    crop = smooth(crop_to_mask(img, teeth_mask), 0.2)
    near_teeth = sitk.Resample(sitk.BinaryDilate(teeth_mask, [1, 1, 1]), crop, sitk.Transform(), sitk.sitkNearestNeighbor)
    teeth_src = sitk.Mask(crop, near_teeth, outsideValue=AIR)
    teeth, n_teeth = extract_surface(teeth_src, args.teeth, 800, 0, args.teeth_tris)

    scene = trimesh.Scene()
    scene.add_geometry(to_trimesh(bone, center, BONE_COLOR, 0.55, "bone"), node_name="bone", geom_name="bone")
    scene.add_geometry(to_trimesh(teeth, center, TEETH_COLOR, 1.0, "teeth"), node_name="teeth", geom_name="teeth")
    glb_path = args.out_dir / "ct_surface.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    render_previews(bone, teeth, center, args.out_dir)

    info = {
        "bone_threshold": args.bone,
        "teeth_threshold": args.teeth,
        "enamel_threshold": args.enamel,
        "bone_triangles": int(bone.n_cells),
        "bone_pieces": int(n_bone),
        "teeth_triangles": int(teeth.n_cells),
        "teeth_pieces": int(n_teeth),
        "glb_mb": round(glb_path.stat().st_size / 1e6, 2),
        "units": "mm",
        "axes": "three.js: +X 환자 왼쪽, +Y 위, +Z 얼굴 앞",
    }
    (args.out_dir / "mesh_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
