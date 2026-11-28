"""자동 분할 결과(segment_teeth.py의 출력)를 구조별 메시로 바꿔 GLB 파일 하나로 저장한다.

사용법:
    python labels_to_mesh.py <분할 결과 폴더> <ct.nii.gz> <출력 폴더> [--ai-jaws] [--bone 값]

분할 결과 폴더에는 teeth_labels.nii.gz와 segmentation_info.json이 있어야 한다.
find_remnants.py의 출력(remnant_labels.nii.gz, remnants_info.json)이 있으면 뿌리만 남은 치아도 넣는다.

턱뼈는 CT 밝기로 따로 뽑는다. 분할 모델은 치아 둘레만 잘라서 계산하므로 턱뼈가 상자 모양으로 잘려 나오기 때문이다.
--ai-jaws를 주면 분할 모델의 턱뼈를 그대로 쓴다.

출력:
    teeth_parts.glb     구조마다 메시 하나. 메시 이름이 구조 이름이다 (예: upper_right_central_incisor_fdi11)
    parts_info.json     구조별 종류, FDI 번호, 부피, 삼각형 수
    parts_*.png         확인용 렌더링

좌표는 volume_to_mesh.py와 같다. 단위 mm, three.js 좌표(Y가 위, +Z가 얼굴 앞), 원점은 CT 볼륨의 중심.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi

# 종류별 색. 뷰어도 같은 종류 이름으로 켜고 끈다.
KIND_COLOR = {
    "tooth": (0.97, 0.95, 0.90),
    "remnant": (0.54, 0.42, 0.27),
    "pulp": (0.85, 0.35, 0.33),
    "jaw": (0.72, 0.65, 0.48),
    "canal": (0.95, 0.75, 0.20),
    "sinus": (0.55, 0.75, 0.90),
    "implant": (0.67, 0.70, 0.74),
    "crown": (0.80, 0.86, 0.93),
    "bridge": (0.90, 0.85, 0.94),
    "other": (0.60, 0.80, 0.70),
}
KIND_ALPHA = {"jaw": 0.45, "sinus": 0.35}
# 삼각형 상한. 턱뼈처럼 큰 구조만 많이 준다.
KIND_TRIS = {"jaw": 110_000, "sinus": 15_000, "other": 6_000, "tooth": 6_000, "remnant": 4_000}
DEFAULT_TRIS = 3_000
# 분할은 0.5mm 격자로 계산돼서 표면에 계단이 진다. 표면을 뽑기 전에 이만큼(mm) 흐리게 해서 없앤다.
# 치수와 신경관은 가늘어서 덜 흐리게 한다.
KIND_SIGMA = {"pulp": 0.3, "canal": 0.3}
DEFAULT_SIGMA = 0.5
REMNANT_HU = 1500.0     # 뿌리만 남은 치아의 표면으로 보는 밝기
AIR = -1000.0


def classify(name: str) -> tuple[str, int | None]:
    """구조 이름에서 종류와 FDI 번호를 뽑는다. 치수는 치아 번호에 100을 더한 번호를 쓴다(예: fdi111 → 11번 치아의 치수)."""
    m = re.search(r"fdi(\d+)$", name)
    fdi = int(m.group(1)) if m else None
    if name.startswith("remnant"):
        return "remnant", fdi
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


def contour_to_mesh(scalar: sitk.Image, level: float, target_tris: int) -> pv.PolyData | None:
    """볼륨에서 값이 level인 면을 뽑아 다듬는다."""
    grid = pv.ImageData(dimensions=scalar.GetSize(), spacing=scalar.GetSpacing(), origin=scalar.GetOrigin())
    grid.point_data["v"] = sitk.GetArrayFromImage(scalar).ravel()   # z, y, x 순서라 x가 가장 빨리 변한다
    surf = grid.contour([level], scalars="v", method="flying_edges")
    if surf.n_cells == 0:
        return None
    surf = surf.triangulate().smooth_taubin(n_iter=30, pass_band=0.08)
    if surf.n_cells > target_tris:
        surf = surf.decimate(1.0 - target_tris / surf.n_cells)
    return surf.compute_normals(auto_orient_normals=True, split_vertices=False, consistent_normals=True)


def crop_box(size, bbox, pad: int):
    x, y, z, sx, sy, sz = bbox
    lo = [max(0, v - pad) for v in (x, y, z)]
    hi = [min(n, v + s + pad) for n, v, s in zip(size, (x, y, z), (sx, sy, sz))]
    return lo, [h - l for l, h in zip(lo, hi)]


def label_to_surface(labels: sitk.Image, label: int, bbox, target_tris: int, sigma_mm: float) -> pv.PolyData | None:
    """번호 볼륨에서 한 구조의 표면을 뽑는다. 그 구조가 있는 범위만 잘라서 처리한다."""
    lo, size = crop_box(labels.GetSize(), bbox, 8)
    crop = sitk.RegionOfInterest(labels, size, lo)
    soft = sitk.SmoothingRecursiveGaussian(sitk.Cast(crop == label, sitk.sitkFloat32), sigma_mm)
    return contour_to_mesh(soft, 0.5, target_tris)


def hu_surface(ct: sitk.Image, region: sitk.Image, label: int, bbox, level: float, target_tris: int) -> pv.PolyData | None:
    """CT 밝기가 level인 면을 뽑는다. region에서 번호가 label인 덩어리 둘레만 본다.

    속이 빈 얇은 껍질처럼 생긴 조각은 마스크보다 밝기로 뽑아야 모양이 덜 뭉개진다.
    """
    lo, size = crop_box(ct.GetSize(), bbox, 8)
    hu = sitk.SmoothingRecursiveGaussian(sitk.Cast(sitk.RegionOfInterest(ct, size, lo), sitk.sitkFloat32), 0.2)
    near = sitk.BinaryDilate(sitk.RegionOfInterest(region, size, lo) == label, [2, 2, 2])
    return contour_to_mesh(sitk.Mask(hu, near, outsideValue=AIR), level, target_tris)


def whole_jaws(ct: sitk.Image, labels: np.ndarray, lower_id: int, upper_id: int, lower_teeth: np.ndarray, upper_teeth: np.ndarray, bone_hu: float, target_tris: int):
    """CT 밝기로 아래턱뼈와 위턱뼈(머리뼈와 이어진 부분 포함)를 통째로 뽑는다.

    뼈로 보이는 화소의 덩어리 가운데 분할 모델의 턱뼈와 가장 많이 겹치는 것을 고른다.
    lower_teeth, upper_teeth(치아와 뿌리 조각)는 뼈에서 뺀다. 두 턱뼈를 가르지 못하면 None을 돌려준다.
    """
    half = sitk.SmoothingRecursiveGaussian(sitk.Cast(sitk.BinShrink(ct, [2, 2, 2]), sitk.sitkFloat32), 0.3)
    hu = sitk.GetArrayViewFromImage(half)
    nz, ny, nx = hu.shape

    def shrink(mask: np.ndarray) -> np.ndarray:      # 2x2x2 칸 가운데 하나라도 차 있으면 참
        return mask[: nz * 2, : ny * 2, : nx * 2].reshape(nz, 2, ny, 2, nx, 2).any(axis=(1, 3, 5))

    plugs = {"lower_jawbone": ndi.binary_dilation(shrink(lower_teeth), iterations=1),
             "upper_jawbone": ndi.binary_dilation(shrink(upper_teeth), iterations=1)}
    holes = plugs["lower_jawbone"] | plugs["upper_jawbone"]
    hints = {"lower_jawbone": shrink(labels == lower_id), "upper_jawbone": shrink(labels == upper_id)}

    def pick(comp: np.ndarray):
        chosen = {}
        for name, hint in hints.items():
            overlap = np.bincount(comp[hint])
            overlap[0] = 0
            if overlap.sum() == 0:
                return None
            chosen[name] = int(overlap.argmax())
        return chosen

    # 턱관절처럼 두 뼈가 가까운 곳은 뼈 기준 밝기에서 한 덩어리로 붙는다.
    # 기준을 조금씩 올려 두 뼈가 떨어지는 값을 찾고, 그때의 덩어리를 씨앗으로 쓴다.
    for level in np.arange(bone_hu, bone_hu + 700, 100):
        comp, _ = ndi.label((hu > level) & ~holes)
        picked = pick(comp)
        if picked is None:
            return None
        if picked["lower_jawbone"] != picked["upper_jawbone"]:
            break
    else:
        return None
    masks = {name: comp == c for name, c in picked.items()}
    if level > bone_hu:
        # 원래 기준에서 뼈인 화소는 가까운 쪽 씨앗에 붙인다.
        base, _ = ndi.label((hu > bone_hu) & ~holes)
        merged = base == pick(base)["lower_jawbone"]
        seeds = np.zeros(hu.shape, np.uint8)
        seeds[masks["lower_jawbone"]] = 1
        seeds[masks["upper_jawbone"]] = 2
        nearest = seeds[tuple(ndi.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True))]
        masks = {"lower_jawbone": merged & (nearest == 1), "upper_jawbone": merged & (nearest == 2)}
        print(f"턱뼈: 밝기 {bone_hu:.0f}에서는 위아래가 붙어 있어 {level:.0f}에서 갈랐습니다.", flush=True)

    result = {}
    voxel_mm3 = float(np.prod(half.GetSpacing()))
    for name, mask in masks.items():
        # 속의 해면골 틈을 메워 겉모양만 남긴다. 치아가 박힌 자리로 속이 바깥과 통하므로
        # 그 턱의 치아로 구멍을 막은 채 메우고, 치아 자리는 다시 비운다.
        solid = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=3) | plugs[name])
        if name == "lower_jawbone":
            # 아래턱뼈는 촬영 범위 끝에서 잘리고 신경관 구멍도 있어서 통째로는 속이 메워지지 않는다.
            # 가로 단면마다 따로 메운다. 아래턱뼈는 뒤가 트인 U자라 단면에서 혀 쪽 공간이 갇히지 않는다.
            # 위턱뼈에 같은 방법을 쓰면 상악동과 코 안이 메워지므로 쓰지 않는다.
            for k in range(solid.shape[0]):
                solid[k] = ndi.binary_fill_holes(solid[k])
        solid &= ~holes
        img = sitk.GetImageFromArray(solid.astype(np.float32))
        img.CopyInformation(half)
        surf = contour_to_mesh(sitk.SmoothingRecursiveGaussian(img, 0.5), 0.5, target_tris)
        if surf is not None:
            result[name] = (surf, float(solid.sum()) * voxel_mm3)
    return result


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
        pl.camera_position = [tuple(focus + np.array(direction) * 340.0), tuple(focus), up]
        pl.camera.view_angle = 28
        pl.screenshot(str(out_dir / f"parts_{view}.png"))
        pl.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seg_dir", type=Path)
    ap.add_argument("volume", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--ai-jaws", action="store_true", help="분할 모델의 턱뼈를 그대로 쓴다")
    ap.add_argument("--bone", type=float, default=1000.0, help="뼈로 보는 밝기")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ct = sitk.ReadImage(str(args.volume))
    # 메시 좌표의 원점을 volume_to_mesh.py와 맞추려고 CT 볼륨의 중심을 쓴다.
    center = np.array([o + 0.5 * n * s for o, n, s in zip(ct.GetOrigin(), ct.GetSize(), ct.GetSpacing())])

    seg_info = json.loads((args.seg_dir / "segmentation_info.json").read_text(encoding="utf-8"))
    names = {int(i): name for i, name in seg_info["labels"].items()}
    labels = sitk.ReadImage(str(args.seg_dir / "teeth_labels.nii.gz"), sitk.sitkUInt8)
    label_arr = sitk.GetArrayViewFromImage(labels)
    voxel_mm3 = float(np.prod(labels.GetSpacing()))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)

    scene = trimesh.Scene()
    surfaces, kinds, parts = {}, {}, []

    def add(name: str, surf: pv.PolyData, volume_mm3: float) -> None:
        kind, fdi = classify(name)
        scene.add_geometry(to_trimesh(surf, center, KIND_COLOR[kind], KIND_ALPHA.get(kind, 1.0), name), node_name=name, geom_name=name)
        surfaces[name], kinds[name] = surf, kind
        parts.append({"name": name, "kind": kind, "fdi": fdi, "volume_mm3": round(volume_mm3, 1), "triangles": int(surf.n_cells)})
        print(f"{name:55s} {kind:8s} {volume_mm3:9.1f} mm3 {surf.n_cells:7d} tris", flush=True)

    # 1) 뿌리만 남은 치아와 그 밖의 조각 (find_remnants.py의 결과)
    remnant_arr, remnant_info = None, []
    remnant_path = args.seg_dir / "remnant_labels.nii.gz"
    if remnant_path.exists():
        remnants = sitk.ReadImage(str(remnant_path), sitk.sitkUInt8)
        remnant_arr = sitk.GetArrayViewFromImage(remnants)
        remnant_info = json.loads((args.seg_dir / "remnants_info.json").read_text(encoding="utf-8"))["candidates"]
        r_stats = sitk.LabelShapeStatisticsImageFilter()
        r_stats.Execute(remnants)
        for cand in remnant_info:
            name = f"remnant_fdi{cand['fdi']}" if cand["fdi"] else f"fragment_{cand['id']}"
            kind = classify(name)[0]
            surf = hu_surface(ct, remnants, cand["id"], r_stats.GetBoundingBox(cand["id"]), REMNANT_HU, KIND_TRIS[kind])
            if surf is not None:
                add(name, surf, cand["volume_mm3"])

    # 2) 분할 모델이 찾은 구조
    jaw_ids = {name: label for label, name in names.items() if name in ("lower_jawbone", "upper_jawbone")}
    for label in sorted(stats.GetLabels()):
        name = names.get(label)
        if name is None or (name in jaw_ids and not args.ai_jaws):
            continue
        kind, _ = classify(name)
        volume_mm3 = stats.GetNumberOfPixels(label) * voxel_mm3
        # 뿌리 조각의 일부를 분할 모델이 다른 이름으로 잡은 것은 겹치므로 뺀다.
        if remnant_arr is not None and np.count_nonzero(remnant_arr[label_arr == label]) > 0.5 * stats.GetNumberOfPixels(label):
            print(f"{name:55s} {kind:8s} {volume_mm3:9.1f} mm3  뿌리 조각과 겹쳐서 뺌", flush=True)
            continue
        surf = label_to_surface(labels, label, stats.GetBoundingBox(label), KIND_TRIS.get(kind, DEFAULT_TRIS), KIND_SIGMA.get(kind, DEFAULT_SIGMA))
        if surf is None:
            print(f"{name:55s} {kind:8s} {volume_mm3:9.1f} mm3  너무 작아 메시에서 빠짐", flush=True)
            continue
        add(name, surf, volume_mm3)

    # 3) 턱뼈 전체
    if not args.ai_jaws and len(jaw_ids) == 2:
        def teeth_of(upper: bool) -> np.ndarray:
            ids = [l for l, n in names.items() if classify(n)[0] in ("tooth", "pulp") and (classify(n)[1] < 30) == upper]
            mask = np.isin(label_arr, ids)
            for cand in remnant_info:       # 번호를 붙이지 않은 조각은 find_remnants.py가 적어 둔 턱을 따른다
                if (cand["jaw"] == "upper") == upper:
                    mask = mask | (remnant_arr == cand["id"])
            return mask

        jaws = whole_jaws(ct, label_arr, jaw_ids["lower_jawbone"], jaw_ids["upper_jawbone"],
                          teeth_of(False), teeth_of(True), args.bone, KIND_TRIS["jaw"])
        if jaws is None:
            raise SystemExit("위아래 턱뼈를 가르지 못했습니다. --ai-jaws로 다시 실행하세요.")
        for name, (surf, volume_mm3) in jaws.items():
            add(name, surf, volume_mm3)

    if not parts:
        raise SystemExit("분할 결과에 비어 있지 않은 구조가 없습니다.")

    glb_path = args.out_dir / "teeth_parts.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))
    info = {
        "units": "mm",
        "axes": "three.js: +X 환자 왼쪽, +Y 위, +Z 얼굴 앞",
        "glb_mb": round(glb_path.stat().st_size / 1e6, 2),
        "parts": parts,
    }
    (args.out_dir / "parts_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    render_previews(surfaces, kinds, center, args.out_dir)
    teeth = sorted(p["fdi"] for p in parts if p["kind"] == "tooth")
    remnant_fdi = sorted(p["fdi"] for p in parts if p["kind"] == "remnant")
    print(f"\n구조 {len(parts)}개, 치아 {len(teeth)}개: {teeth}, 뿌리만 남은 치아: {remnant_fdi}")
    print(f"저장: {glb_path} ({info['glb_mb']} MB)")


if __name__ == "__main__":
    main()
