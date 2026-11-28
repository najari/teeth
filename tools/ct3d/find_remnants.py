"""자동 분할이 놓친 치아 조각(뿌리만 남은 치아)을 CT에서 찾는다.

사용법:
    python find_remnants.py <분할 결과 폴더> <ct.nii.gz> [--assign 후보번호:FDI ...]

찾는 규칙: 치아가 있는 범위 안에서, 뼈와 치아에 붙어 있지 않은 작고 밝은 덩어리.
뼈 속에 박혀 있는 뿌리는 뼈와 한 덩어리로 이어져서 이 방법으로 찾지 못한다.

출력 (분할 결과 폴더에 저장):
    remnant_labels.nii.gz    화소마다 후보 번호가 들어 있는 볼륨
    remnants_info.json       후보별 위치, 부피, 밝기, 가장 가까운 치아, 붙인 FDI 번호
    remnants_preview.png     위에서 내려다본 그림에 후보 번호를 적은 것

후보가 치아 조각인지, 몇 번 치아인지는 사람이 정한다. 그림과 목록을 보고
--assign으로 번호를 붙여 다시 실행한다(예: --assign 1:47 2:46 3:45).
번호를 붙이지 않은 후보는 다음 단계에서 쓰지 않는다.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

BONE_HU = 1200.0        # 이 값을 넘으면 단단한 조직으로 본다
CORE_HU = 2200.0        # 덩어리 안에 이만큼 밝은 화소가 있어야 치아 조각 후보로 본다
MIN_MM3, MAX_MM3 = 12.0, 800.0
MIN_TOOTH_MM3 = 100.0   # 자동 분할 결과에서 이보다 작은 치아는 조각으로 보고 치아로 치지 않는다


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seg_dir", type=Path)
    ap.add_argument("volume", type=Path)
    ap.add_argument("--assign", nargs="*", default=[], help="후보번호:FDI")
    args = ap.parse_args()
    assign = {int(a): int(b) for a, b in (item.split(":") for item in args.assign)}

    info = json.loads((args.seg_dir / "segmentation_info.json").read_text(encoding="utf-8"))
    names = {int(i): n for i, n in info["labels"].items()}
    ct = sitk.ReadImage(str(args.volume))
    labels_img = sitk.ReadImage(str(args.seg_dir / "teeth_labels.nii.gz"), sitk.sitkUInt8)
    labels = sitk.GetArrayFromImage(labels_img)                       # z, y, x
    hu = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(sitk.Cast(ct, sitk.sitkFloat32), 0.2))
    sx, sy, sz = ct.GetSpacing()
    ox, oy, oz = ct.GetOrigin()
    voxel_mm3 = sx * sy * sz

    def vox(mm: float, spacing: float) -> int:
        return max(1, int(round(mm / spacing)))

    # 자동 분할이 치아로 잡은 것. 아주 작게 잡힌 것은 조각일 수 있으므로 치아에서 뺀다.
    counts = np.bincount(labels.ravel(), minlength=256)
    tooth_fdi = {}
    for i, n in names.items():
        m = re.search(r"fdi(\d+)$", n)
        if m and "pulp" not in n and "canal" not in n and counts[i] * voxel_mm3 >= MIN_TOOTH_MM3:
            tooth_fdi[i] = int(m.group(1))
    pulp_ids = [i for i, n in names.items() if "pulp" in n]
    teeth = np.isin(labels, list(tooth_fdi) + pulp_ids)
    near_teeth = ndi.binary_dilation(teeth, iterations=vox(0.6, sx))

    # 치아가 있는 범위에 여유를 둔 상자 안만 본다. 뒤쪽(어금니 뒤)으로는 넉넉하게 잡는다.
    zz, yy, xx = np.nonzero(teeth)
    lo = [max(0, zz.min() - vox(3, sz)), max(0, yy.min() - vox(5, sy)), max(0, xx.min() - vox(15, sx))]
    hi = [min(labels.shape[0], zz.max() + vox(3, sz)), min(labels.shape[1], yy.max() + vox(25, sy)),
          min(labels.shape[2], xx.max() + vox(15, sx))]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    hu_box = hu[box]

    hard = (hu_box > BONE_HU) & ~near_teeth[box]
    comp, n = ndi.label(hard)
    sizes = ndi.sum(hard, comp, index=np.arange(1, n + 1)) * voxel_mm3
    objects = ndi.find_objects(comp)

    tooth_centres = {f: np.array(ndi.center_of_mass(labels == i))[::-1] * (sx, sy, sz) + (ox, oy, oz) for i, f in tooth_fdi.items()}
    upper_z = np.mean([c[2] for f, c in tooth_centres.items() if f < 30]) if any(f < 30 for f in tooth_centres) else np.inf
    lower_z = np.mean([c[2] for f, c in tooth_centres.items() if f > 30]) if any(f > 30 for f in tooth_centres) else -np.inf

    found = []
    for g in np.flatnonzero((sizes >= MIN_MM3) & (sizes <= MAX_MM3)) + 1:
        sl = objects[g - 1]
        mask = comp[sl] == g
        values = hu_box[sl][mask]
        if values.max() < CORE_HU:
            continue
        cz, cy, cx = [c + s.start + o for c, s, o in zip(ndi.center_of_mass(mask), sl, lo)]
        centre = np.array([cx * sx + ox, cy * sy + oy, cz * sz + oz])
        ring = ndi.binary_dilation(mask, iterations=2) & ~mask
        jaw = "upper" if abs(centre[2] - upper_z) < abs(centre[2] - lower_z) else "lower"
        same_jaw = [f for f in tooth_centres if (f < 30) == (jaw == "upper")] or list(tooth_centres)
        nearest = min(same_jaw, key=lambda f: np.linalg.norm(tooth_centres[f][:2] - centre[:2]))
        found.append({
            "g": int(g), "slices": sl,
            "volume_mm3": round(float(sizes[g - 1]), 1),
            "centroid_lps_mm": [round(float(v), 1) for v in centre],
            "size_mm": [round((s.stop - s.start) * sp, 1) for s, sp in zip(sl[::-1], (sx, sy, sz))],
            "mean_hu": round(float(values.mean())), "max_hu": round(float(values.max())),
            # 둘레가 살(연조직)인 비율. 뼈에서 떨어져 있을수록 크다.
            "soft_tissue_around": round(float((hu_box[sl][ring] < 800).mean()), 2),
            "jaw": jaw,
            "side": "right" if centre[0] < np.mean([c[0] for c in tooth_centres.values()]) else "left",
            "nearest_tooth": int(nearest),
            "nearest_tooth_mm": round(float(np.linalg.norm(tooth_centres[nearest][:2] - centre[:2])), 1),
        })

    # 환자의 오른쪽에서 왼쪽 순서로 번호를 매긴다. 같은 데이터에서는 번호가 바뀌지 않는다.
    found.sort(key=lambda c: (c["centroid_lps_mm"][0], c["centroid_lps_mm"][1]))
    out = np.zeros(labels.shape, np.uint8)
    for number, c in enumerate(found, start=1):
        sl = tuple(slice(s.start + o, s.stop + o) for s, o in zip(c["slices"], lo))
        out[sl][comp[c["slices"]] == c.pop("g")] = number
        del c["slices"]
        c["id"] = number
        c["fdi"] = assign.get(number)

    img = sitk.GetImageFromArray(out)
    img.CopyInformation(labels_img)
    sitk.WriteImage(img, str(args.seg_dir / "remnant_labels.nii.gz"), useCompression=True)
    (args.seg_dir / "remnants_info.json").write_text(
        json.dumps({"bone_hu": BONE_HU, "core_hu": CORE_HU, "candidates": found}, indent=2, ensure_ascii=False), encoding="utf-8")

    # 위에서 내려다본 그림 두 장(아래턱, 위턱): 치아 번호(노랑)와 후보 번호(빨강).
    # 위아래 치아가 겹쳐 보이지 않게 위아래 치아의 가운데 높이에서 나눠 그린다.
    split = int(round(((lower_z + upper_z) / 2 - oz) / sz)) - lo[0] if np.isfinite(lower_z + upper_z) else hu_box.shape[0] // 2
    split = int(np.clip(split, 1, hu_box.shape[0] - 1))
    scale = 3
    panels = []
    for jaw, part in (("lower", hu_box[:split]), ("upper", hu_box[split:])):
        mip = np.clip((part.max(axis=0) + 200) / 4200, 0, 1)
        rgb = (np.stack([mip, mip, mip], -1) * 255).astype(np.uint8)
        panel = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.LANCZOS)
        draw = ImageDraw.Draw(panel)
        for f, c in tooth_centres.items():
            if (f < 30) == (jaw == "upper"):
                draw.text((((c[0] - ox) / sx - lo[2]) * scale - 6, ((c[1] - oy) / sy - lo[1]) * scale - 5), str(f), fill=(255, 220, 0))
        for c in found:
            if c["jaw"] != jaw:
                continue
            px = ((c["centroid_lps_mm"][0] - ox) / sx - lo[2]) * scale
            py = ((c["centroid_lps_mm"][1] - oy) / sy - lo[1]) * scale
            r = max(c["size_mm"][:2]) / sx * scale / 2 + 6
            draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 60, 60), width=2)
            draw.text((px + r + 4, py - 6), f"#{c['id']}" + (f" = {c['fdi']}" if c["fdi"] else ""), fill=(255, 90, 90))
        draw.text((6, 6), f"{jaw} jaw, seen from above. left of image = patient's right", fill=(255, 255, 255))
        panels.append(panel)
    picture = Image.new("RGB", (panels[0].width * 2, panels[0].height))
    picture.paste(panels[0], (0, 0))
    picture.paste(panels[1], (panels[0].width, 0))
    picture.save(args.seg_dir / "remnants_preview.png")

    print("후보  부피mm3  평균/최대 밝기  위치(x,y,z mm)        크기(mm)          둘레가 살  가까운 치아   붙인 번호")
    for c in found:
        p, s = c["centroid_lps_mm"], c["size_mm"]
        print(f"#{c['id']:<3d} {c['volume_mm3']:7.1f}  {c['mean_hu']:5d}/{c['max_hu']:<5d}  ({p[0]:5.1f},{p[1]:5.1f},{p[2]:5.1f})"
              f"  {s[0]:4.1f}x{s[1]:4.1f}x{s[2]:4.1f}   {c['soft_tissue_around']:.2f}     "
              f"{c['nearest_tooth']}번 {c['nearest_tooth_mm']:4.1f}mm   {c['fdi'] or '-'}")
    print(f"\n후보 {len(found)}개. 그림: {args.seg_dir / 'remnants_preview.png'}")


if __name__ == "__main__":
    main()
