"""구강 스캔을 CT에 맞춘 결과로 치과의사에게 보여 줄 그림을 만든다.

사용법:
    python scan_figures.py <출력 폴더> [--teeth 45,46,47] [--check 42]

<출력 폴더>에는 앞 단계의 결과가 있어야 한다: ct.nii.gz, teeth_parts_scan.glb(scan_to_ct.py), scan_registration.json.

만드는 그림 (<출력 폴더>/scan_figures/)
    1) 뿌리 조각 자리: 아래턱을 위에서, 볼 쪽에서 본 3D 그림. 스캔(잇몸)은 반투명
    2) 뿌리 조각마다 CT 단면: 볼-혀 방향으로 세로로 자른 CT 위에 스캔 표면, CT 치아·조각 윤곽을 겹침
    3) 확인할 치아(--check): CT 치아 머리를 스캔과의 거리로 칠한 3D 그림과, 긴 축을 따라 자른 CT 단면
    4) 위 그림을 모은 HTML 한 장(dentist_figures.html)

좌표는 CT 뷰어와 같다(mm, x 환자 왼쪽, y 위, z 얼굴 앞, 원점은 CT 볼륨 중심).
그림에는 환자의 형상이 들어 있다. 공개 저장소에 올리지 않는다. 설명용이며 진단 자료가 아니다.
"""
import argparse
import base64
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"scan": "#e0607e", "tooth": "#ffe066", "remnant": "#ff9a3c"}


def fdi_of(name: str):
    m = re.search(r"fdi(\d+)$", name)
    return int(m.group(1)) if m else None


def load_parts(glb: Path) -> dict:
    scene = trimesh.load(glb)
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        g = scene.geometry[geom].copy()
        g.apply_transform(transform)
        parts[node] = g
    return parts


class CT:
    """ct.nii.gz를 CT 뷰어 좌표로 읽는다."""

    def __init__(self, path: Path):
        img = sitk.ReadImage(str(path))
        self.arr = sitk.GetArrayFromImage(img).astype(np.float32)          # z, y, x
        self.origin, self.spacing = np.array(img.GetOrigin()), np.array(img.GetSpacing())
        self.center = self.origin + 0.5 * np.array(img.GetSize()) * self.spacing

    def sample(self, pts_view: np.ndarray) -> np.ndarray:
        # 뷰어 (x, y, z) = (Lx - cx, Lz - cz, -(Ly - cy))
        lps = np.column_stack([pts_view[:, 0], -pts_view[:, 2], pts_view[:, 1]]) + self.center
        ijk = (lps - self.origin) / self.spacing
        return map_coordinates(self.arr, [ijk[:, 2], ijk[:, 1], ijk[:, 0]], order=1, cval=-1000.0)


def section_figure(ct: CT, meshes: list, p0, u, w, half_u, half_w, title, path, marks=()):
    """p0를 지나고 u(가로), w(세로)가 놓인 평면으로 자른 CT와 메시 윤곽."""
    u, w = u / np.linalg.norm(u), w / np.linalg.norm(w)
    n = np.cross(u, w)
    step = 0.1
    su, sw = np.arange(-half_u, half_u + step, step), np.arange(-half_w, half_w + step, step)
    gu, gw = np.meshgrid(su, sw)
    pts = p0 + gu.ravel()[:, None] * u + gw.ravel()[:, None] * w
    img = ct.sample(pts).reshape(gw.shape)
    fig, ax = plt.subplots(figsize=(6.2, 6.2 * half_w / half_u if half_u > half_w else 6.2))
    ax.imshow(img, cmap="gray", vmin=-600, vmax=3200, origin="lower", extent=[-half_u, half_u, -half_w, half_w])
    handles = []
    for mesh, color, label in meshes:
        seg = trimesh.intersections.mesh_plane(mesh, n, p0)
        if len(seg) == 0:
            continue
        rel = seg - p0
        xy = np.stack([rel @ u, rel @ w], axis=-1)
        ax.add_collection(LineCollection(xy, colors=color, linewidths=1.6))
        handles.append(plt.Line2D([], [], color=color, lw=2, label=label))
    for (x, y, text) in marks:
        ax.annotate(text, (x, y), color="white", fontsize=9, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, lw=0))
    ax.set_xlim(-half_u, half_u)
    ax.set_ylim(-half_w, half_w)
    ax.set_xlabel("← 혀 쪽          (mm)          볼·입술 쪽 →")
    ax.set_ylabel("← 뿌리 쪽     (mm)     머리 쪽 →")
    ax.set_title(title, fontsize=11)
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render(meshes: list, camera, path, scalars=None, size=(900, 700)):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("#eef2f1")
    for item in meshes:
        mesh, kw = item
        p.add_mesh(pv.wrap(mesh), smooth_shading=True, **kw)
    if scalars:
        p.add_scalar_bar(**scalars)
    p.camera_position = camera
    p.screenshot(str(path))
    p.close()


def top_contact(mesh: trimesh.Trimesh, scan_tree: cKDTree, band=1.0):
    """조각의 위쪽 band mm 띠의 점에서 가장 가까운 스캔 표면까지 거리의 중앙값과, 0.2mm 안인 점의 비율.

    0.2mm는 정합 오차 수준이다. 이 안에서 겹치면 스캔이 조각 표면 자체를 찍었거나 아주 얇게 덮인 것인데, 둘은 가를 수 없다.
    """
    v = mesh.vertices
    d = scan_tree.query(v[v[:, 1] > v[:, 1].max() - band])[0]
    return float(np.median(d)), float((d < 0.2).mean())


def horizontal_normal(tangent: np.ndarray, away_from: np.ndarray, at: np.ndarray) -> np.ndarray:
    """수평이고 tangent에 수직이며 away_from에서 멀어지는 쪽(치열궁 바깥 = 볼·입술 쪽) 방향."""
    t = np.array([tangent[0], 0.0, tangent[2]])
    nrm = np.array([t[2], 0.0, -t[0]])
    nrm /= np.linalg.norm(nrm)
    return nrm if nrm @ (at - away_from) > 0 else -nrm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--teeth", default="45,46,47", help="뿌리 조각으로 볼 번호")
    ap.add_argument("--check", default="42", help="스캔과 비교해 볼 치아 번호(AI 번호)")
    args = ap.parse_args()
    out = args.out_dir / "scan_figures"
    out.mkdir(exist_ok=True)
    remnant_ids = [int(x) for x in args.teeth.split(",") if x]
    check = int(args.check) if args.check else None

    parts = load_parts(args.out_dir / "teeth_parts_scan.glb")
    reg = json.loads((args.out_dir / "scan_registration.json").read_text(encoding="utf-8"))
    ct = CT(args.out_dir / "ct.nii.gz")
    scan = parts["lower_scan"]
    teeth = {fdi_of(n): g for n, g in parts.items()
             if fdi_of(n) and 31 <= fdi_of(n) <= 48 and "pulp" not in n and "canal" not in n and not n.startswith(("remnant", "fragment"))}
    remnants = {fdi_of(n): g for n, g in parts.items() if n.startswith("remnant") and fdi_of(n) in remnant_ids}
    arch_centre = np.mean([g.vertices.mean(0) for g in teeth.values()], axis=0)
    figures = []          # (파일 이름, 설명)
    scan_tree = cKDTree(scan.vertices)

    # 1) 뿌리 조각 자리 3D
    if remnants:
        site = np.mean([g.vertices.mean(0) for g in remnants.values()], axis=0)
        near = [(g, dict(color="#f4efe2")) for f, g in teeth.items() if np.linalg.norm(g.vertices.mean(0) - site) < 25]
        rem = [(g, dict(color="#8a6a45")) for g in remnants.values()]
        sc = [(scan, dict(color="#d98c8c", opacity=0.45))]
        top = site + np.array([0, 60, 0])
        buccal = horizontal_normal(np.array([1.0, 0, 0]) if len(remnants) < 2 else
                                   remnants[max(remnants)].vertices.mean(0) - remnants[min(remnants)].vertices.mean(0), arch_centre, site)
        render(near + rem + sc, [tuple(top), tuple(site), (0, 0, -1)], out / "remnants_top.png")
        render(near + rem + sc, [tuple(site + 55 * buccal + np.array([0, 8, 0])), tuple(site), (0, 1, 0)], out / "remnants_side.png")
        render(near + rem, [tuple(site + 55 * buccal + np.array([0, 8, 0])), tuple(site), (0, 1, 0)], out / "remnants_side_noscan.png")
        figures += [("remnants_top.png", "뿌리 조각 자리를 위에서 본 것. 분홍 반투명은 구강 스캔(잇몸과 치아 표면), 갈색은 CT에서 찾은 뿌리 조각, 흰색은 CT 치아."),
                    ("remnants_side.png", "같은 자리를 볼 쪽에서 본 것."),
                    ("remnants_side_noscan.png", "스캔을 끄고 본 것. 뿌리 조각의 길이와 기울기를 볼 수 있습니다.")]

    # 2) 뿌리 조각마다 CT 단면
    remnant_gaps = {}
    ordered = sorted(remnants)
    for k, f in enumerate(ordered):
        g = remnants[f]
        c = g.vertices.mean(0)
        a = remnants[ordered[max(k - 1, 0)]].vertices.mean(0)
        b = remnants[ordered[min(k + 1, len(ordered) - 1)]].vertices.mean(0)
        tangent = b - a if np.linalg.norm(b - a) > 1 else np.array([0.0, 0, 1])
        u = horizontal_normal(tangent, arch_centre, c)
        topv = g.vertices[np.argmax(g.vertices[:, 1])]
        med, share = top_contact(g, scan_tree)
        gap_text = f"가장 가까운 스캔 표면까지 중앙값 {med:.2f}mm, 0.2mm 안인 점 {100 * share:.0f}%"
        # 단면은 조각 꼭대기를 지나게 한다(가운데를 자르면 꼭대기가 빠진다)
        c = np.array([topv[0], c[1], topv[2]])
        mark_u = (topv - c) @ u
        mark_w = (topv - c) @ np.array([0, 1.0, 0])
        meshes = [(scan, COLORS["scan"], "구강 스캔 표면"), (g, COLORS["remnant"], f"뿌리 조각 {f}번 (CT)")]
        meshes += [(t, COLORS["tooth"], "CT 치아") for t in teeth.values() if np.linalg.norm(t.vertices.mean(0) - c) < 12][:1]
        name = f"section_{f}.png"
        section_figure(ct, meshes, c, u, np.array([0, 1.0, 0]), 14, 14,
                       f"{f}번 자리 세로 단면 (볼-혀 방향)", out / name,
                       marks=[(mark_u + 0.5, mark_w + 0.8, "조각 꼭대기")])
        remnant_gaps[f] = f"{med:.2f}mm({100 * share:.0f}%)"
        figures.append((name, f"{f}번 자리를 조각 꼭대기를 지나게 볼-혀 방향으로 세로로 자른 CT. 분홍 선은 구강 스캔 표면(잇몸·치아), "
                              f"주황 선은 CT에서 찾은 뿌리 조각, 아래쪽 얼룩진 밝은 부분이 턱뼈입니다. "
                              f"조각 위쪽 1mm 띠에서 {gap_text}입니다(0.2mm는 정합 오차 수준)."))

    # 3) 확인할 치아
    if check and check in teeth:
        g = teeth[check]
        v = g.vertices
        top_part = v[:, 1] > v[:, 1].max() - 6
        d, _ = scan_tree.query(v)
        colored = g.copy()
        dd = np.where(top_part, d, np.nan)
        neighbours = {f: t for f, t in teeth.items() if f != check and np.linalg.norm(t.vertices.mean(0) - v.mean(0)) < 14}
        nb = [(t, dict(color="#f4efe2")) for t in neighbours.values()]
        cm = [(colored, dict(scalars=np.nan_to_num(dd, nan=0.0), cmap="viridis", clim=[0, 2.0], show_scalar_bar=False))]
        sc = [(scan, dict(color="#d98c8c", opacity=0.35))]
        c = v.mean(0)
        front = c + np.array([0, 5, 60])
        bar = dict(title="스캔까지 거리(mm)", n_labels=5, fmt="%.1f", vertical=True, position_x=0.85, position_y=0.2, height=0.6, width=0.06)
        render(nb + cm, [tuple(front), tuple(c), (0, 1, 0)], out / f"tooth_{check}_front.png", scalars=bar)
        render(nb + cm + sc, [tuple(front), tuple(c), (0, 1, 0)], out / f"tooth_{check}_front_scan.png")
        render(nb + cm + sc, [tuple(c + np.array([0, 60, 8])), tuple(c), (0, 0, -1)], out / f"tooth_{check}_top_scan.png")
        med = reg["lower"]["per_tooth_mm"].get(str(check))
        others = [x for k2, x in reg["lower"]["per_tooth_mm"].items() if int(k2) != check]
        figures += [(f"tooth_{check}_front.png", f"AI가 {check}번으로 나눈 아래 앞니(가운데)를 앞에서 본 것. 머리 부분을 구강 스캔 표면까지의 거리로 칠했습니다"
                                                 f"(보라 0mm ~ 노랑 2mm 이상). 이 치아의 거리 중앙값은 {med}mm로, 다른 아래 치아({min(others):.2f}~{max(others):.2f}mm)보다 큽니다."),
                    (f"tooth_{check}_front_scan.png", "같은 그림에 구강 스캔(분홍 반투명)을 겹친 것."),
                    (f"tooth_{check}_top_scan.png", "위에서 본 것.")]
        # 긴 축을 따라 자른 단면
        cv = v - c
        axis = np.linalg.svd(cv, full_matrices=False)[2][0]
        axis = axis if axis[1] > 0 else -axis
        nb_c = [t.vertices.mean(0) for t in neighbours.values()]
        tangent = (max(nb_c, key=lambda p: p[0]) - min(nb_c, key=lambda p: p[0])) if len(nb_c) >= 2 else np.array([1.0, 0, 0])
        u = horizontal_normal(tangent, arch_centre, c)
        u = u - (u @ axis) * axis
        u /= np.linalg.norm(u)
        # 스캔의 치아 머리가 CT 치아 머리에서 어느 쪽으로 비켜 있나: 머리 점마다 가장 가까운 스캔 점으로 가는 벡터의 평균
        crown = v[top_part]
        dist, idx = scan_tree.query(crown)
        close = dist < 3.0
        shift = (scan.vertices[idx[close]] - crown[close]).mean(0)
        side = np.cross(axis, u)
        def word(x, pos, neg):
            return f"{pos if x >= 0 else neg} {abs(x):.1f}mm"
        shift_text = f"{word(shift @ u, '입술 쪽', '혀 쪽')}, {word(shift @ axis, '머리 쪽', '뿌리 쪽')}, 옆으로 {abs(shift @ side):.1f}mm"
        meshes = [(scan, COLORS["scan"], "구강 스캔 표면"), (g, COLORS["tooth"], f"CT 치아 {check}번(AI 번호)")]
        name = f"section_{check}.png"
        section_figure(ct, meshes, c, u, axis, 10, 15, f"{check}번(AI 번호) 긴 축 단면 (입술-혀 방향)", out / name)
        figures.append((name, f"이 치아를 긴 축을 따라 입술-혀 방향으로 자른 CT. 노란 선은 AI가 나눈 치아 윤곽, 분홍 선은 구강 스캔 표면입니다. "
                              f"노란 선은 CT 영상의 치아와 맞습니다. 스캔 표면은 치아 머리 쪽에서 CT 치아와 어긋나 있고, "
                              f"머리 부분에서 가장 가까운 스캔 표면으로 가는 평균 방향은 {shift_text}입니다. "
                              "두 촬영 사이에 이 치아의 자리가 달랐을 수 있습니다."))

    # 4) HTML 한 장
    rows = []
    for fname, caption in figures:
        data = base64.b64encode((out / fname).read_bytes()).decode("ascii")
        rows.append(f'<figure><img src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>')
    fit = reg["lower"]
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>구강 스캔과 CT 겹침 그림</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1d2b2c;line-height:1.6}}
figure{{margin:24px 0}}img{{max-width:100%;border:1px solid #d4dcda;border-radius:6px}}figcaption{{font-size:14px;color:#44575a}}
table{{border-collapse:collapse}}td,th{{border:1px solid #d4dcda;padding:4px 10px;font-size:14px}}</style></head><body>
<h1>구강 스캔과 CT 겹침 그림</h1>
<p>구강 스캔(위턱·아래턱)을 CT에서 AI가 나눈 치아의 머리 부분에 맞춘 뒤 만든 그림입니다. 설명용이며 진단 자료가 아닙니다.
치아 번호는 AI가 붙인 것이고, 뿌리 조각 번호는 자리를 보고 붙인 것입니다.</p>
<table><tr><th>맞춘 정도</th><th>위턱</th><th>아래턱</th></tr>
<tr><td>치아 머리에서 스캔 표면까지 거리 중앙값</td><td>{reg['upper']['median_mm']}mm</td><td>{fit['median_mm']}mm</td></tr></table>
<h2>질문</h2><ol>
<li>오른쪽 아래 뿌리 조각 세 개(45·46·47번 자리)의 꼭대기 부분은 구강 스캔 표면과 거의 겹칩니다
(위쪽 1mm 띠에서 가장 가까운 스캔 표면까지 중앙값과 0.2mm 안인 점의 비율: {', '.join(f'{k}번 {v}' for k, v in remnant_gaps.items())}).
조각이 잇몸 위로 드러나 있습니까, 아주 얇은 잇몸에 덮여 있습니까? 이 자료로는 둘을 가를 수 없습니다.</li>
<li>AI가 42번으로 나눈 아래 앞니는 41번입니까, 42번입니까? 이 치아만 스캔과 1mm 넘게 어긋납니다. CT와 스캔을 찍는 사이에 이 치아가 움직였을 수 있습니까?</li>
<li>CT에서 번호를 붙이지 못한 조각 두 개(아래 앞니 쪽의 띠 모양 조각, 26번 뒤쪽의 작은 덩어리)는 무엇입니까? (이 그림에는 없습니다)</li></ol>
{''.join(rows)}</body></html>"""
    (out / "dentist_figures.html").write_text(html, encoding="utf-8")
    print(f"그림 {len(figures)}장, 저장: {out / 'dentist_figures.html'}")


if __name__ == "__main__":
    main()
