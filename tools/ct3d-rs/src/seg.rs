//! AI 분할 결과로 구조별 메시를 만든다 (tools/ct3d의 find_remnants.py, labels_to_mesh.py를 옮긴 것).
//!
//! 입력: CT(DICOM에서 읽은 볼륨)와 segment_teeth.py의 출력 폴더(teeth_labels.nii.gz, segmentation_info.json).
//! 순서
//!   1) 번호 볼륨을 읽고 CT와 격자가 같은지 확인한다
//!   2) 뿌리 조각 후보: 치아가 있는 범위에서 뼈와 치아에 붙어 있지 않은 작고 밝은 덩어리
//!   3) 구조마다 메시: 뿌리 조각은 CT 밝기로, AI가 나눈 구조는 번호 마스크로
//!   4) 턱뼈 전체: AI 턱뼈는 치아 둘레만 상자 모양으로 나오므로 CT 밝기로 다시 뽑는다

use crate::ctproc::{self, Log, Mesh, Volume, AIR};
use crate::scene::{kind_of, Part, KINDS};
use glam::Vec3;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

const BONE_HU: f32 = 1200.0; // 이 값을 넘으면 단단한 조직
const CORE_HU: f32 = 2200.0; // 덩어리 안에 이만큼 밝은 칸이 있어야 치아 조각 후보
const MIN_MM3: f32 = 12.0;
const MAX_MM3: f32 = 800.0;
const MIN_TOOTH_MM3: f32 = 100.0;
pub(crate) const REMNANT_HU: f32 = 1500.0; // 뿌리 조각 표면으로 보는 밝기
const JAW_BONE_HU: f32 = 1000.0;

pub struct Seg {
    pub labels: Vec<u8>,
    pub names: BTreeMap<u8, String>,
}

/// segment_teeth.py의 출력 폴더를 읽는다. 번호 볼륨의 격자가 CT와 같아야 한다.
pub fn read_seg(dir: &Path, ct: &Volume) -> Result<Seg, String> {
    let info_text = std::fs::read_to_string(dir.join("segmentation_info.json")).map_err(|e| format!("segmentation_info.json을 읽지 못했습니다: {e}"))?;
    let info: serde_json::Value = serde_json::from_str(&info_text).map_err(|e| e.to_string())?;
    let names: BTreeMap<u8, String> = info["labels"]
        .as_object()
        .ok_or("segmentation_info.json에 labels가 없습니다")?
        .iter()
        .filter_map(|(k, v)| Some((k.parse().ok()?, v.as_str()?.to_string())))
        .collect();

    use nifti::{NiftiObject, NiftiType, ReaderOptions};
    let obj = ReaderOptions::new().read_file(dir.join("teeth_labels.nii.gz")).map_err(|e| format!("teeth_labels.nii.gz를 읽지 못했습니다: {e}"))?;
    let h = obj.header();
    let dim = h.dim().map_err(|e| e.to_string())?;
    if dim.len() < 3 || [dim[0] as usize, dim[1] as usize, dim[2] as usize] != ct.dims {
        return Err(format!("번호 볼륨 크기 {:?}가 CT {:?}와 다릅니다", &dim[..dim.len().min(3)], ct.dims));
    }
    let dt = h.data_type().map_err(|e| e.to_string())?;
    let raw = obj.into_volume().into_raw_data();
    // NIfTI도 x가 가장 빨리 변하는 순서라 CT 볼륨과 배치가 같다
    let labels: Vec<u8> = match dt {
        NiftiType::Uint8 | NiftiType::Int8 => raw,
        NiftiType::Int16 | NiftiType::Uint16 => raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as u8).collect(),
        NiftiType::Int32 | NiftiType::Uint32 => raw.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u8).collect(),
        NiftiType::Float32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).round() as u8).collect(),
        other => return Err(format!("번호 볼륨의 자료형 {other:?}는 읽지 못합니다")),
    };
    Ok(Seg { labels, names })
}

pub(crate) fn fdi_of(name: &str) -> Option<u32> {
    name.rfind("fdi").and_then(|p| name[p + 3..].parse().ok())
}

pub(crate) struct LabelStats {
    pub count: usize,
    pub lo: [usize; 3],
    pub hi: [usize; 3],
    sum: [f64; 3],
}

pub(crate) fn label_stats(labels: &[u8], dims: [usize; 3]) -> HashMap<u8, LabelStats> {
    let [nx, ny, _] = dims;
    let mut s: HashMap<u8, LabelStats> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let c = [i % nx, (i / nx) % ny, i / (nx * ny)];
        let e = s.entry(l).or_insert(LabelStats { count: 0, lo: c, hi: c, sum: [0.0; 3] });
        e.count += 1;
        for a in 0..3 {
            e.lo[a] = e.lo[a].min(c[a]);
            e.hi[a] = e.hi[a].max(c[a]);
            e.sum[a] += c[a] as f64;
        }
    }
    s
}

pub struct Candidate {
    pub id: usize,
    pub fdi: Option<u32>,
    pub voxels: Vec<usize>, // CT 볼륨의 칸 번호
    pub volume_mm3: f32,
    pub centre: Vec3, // 환자 좌표(LPS)
    pub size_mm: [f32; 3],
    pub mean_hu: f32,
    pub max_hu: f32,
    pub soft_around: f32,
    pub upper: bool,
    pub nearest: u32,
    pub nearest_mm: f32,
}

fn crop(v: &Volume, lo: [usize; 3], hi: [usize; 3]) -> Volume {
    let d = [0, 1, 2].map(|a| hi[a] - lo[a]);
    let mut data = vec![0f32; d[0] * d[1] * d[2]];
    data.par_chunks_mut(d[0] * d[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..d[1] {
            let src = v.idx(lo[0], y + lo[1], z + lo[2]);
            plane[y * d[0]..(y + 1) * d[0]].copy_from_slice(&v.data[src..src + d[0]]);
        }
    });
    Volume { dims: d, spacing: v.spacing, origin: [0, 1, 2].map(|a| v.origin[a] + lo[a] as f32 * v.spacing[a]), data }
}

/// 뿌리 조각 후보를 찾는다. 번호는 환자 오른쪽에서 왼쪽 순으로 매겨 같은 데이터에서는 바뀌지 않는다.
pub fn find_remnants(ct: &Volume, seg: &Seg, assign: &HashMap<usize, u32>, log: Log) -> Vec<Candidate> {
    let dims = ct.dims;
    let [nx, ny, nz] = dims;
    let sp = ct.spacing;
    let voxel_mm3 = sp[0] * sp[1] * sp[2];
    let stats = label_stats(&seg.labels, dims);

    // 자동 분할이 치아로 잡은 것. 아주 작게 잡힌 것은 조각일 수 있으므로 치아에서 뺀다
    let mut tooth_fdi: HashMap<u8, u32> = HashMap::new();
    let mut pulp: HashSet<u8> = HashSet::new();
    for (&l, name) in &seg.names {
        let Some(s) = stats.get(&l) else { continue };
        if name.contains("pulp") {
            pulp.insert(l);
        } else if let Some(f) = fdi_of(name) {
            if !name.contains("canal") && s.count as f32 * voxel_mm3 >= MIN_TOOTH_MM3 {
                tooth_fdi.insert(l, f);
            }
        }
    }
    let world = |c: [f64; 3]| Vec3::new(ct.origin[0] + c[0] as f32 * sp[0], ct.origin[1] + c[1] as f32 * sp[1], ct.origin[2] + c[2] as f32 * sp[2]);
    let centres: BTreeMap<u32, Vec3> = tooth_fdi
        .iter()
        .map(|(l, f)| {
            let s = &stats[l];
            (*f, world(s.sum.map(|v| v / s.count as f64)))
        })
        .collect();
    let is_tooth = |l: u8| tooth_fdi.contains_key(&l) || pulp.contains(&l);

    // 치아가 있는 범위에 여유를 둔 상자: 위아래 3mm, 앞 5mm, 뒤 25mm, 좌우 15mm
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for (l, s) in &stats {
        if is_tooth(*l) {
            for a in 0..3 {
                lo[a] = lo[a].min(s.lo[a]);
                hi[a] = hi[a].max(s.hi[a] + 1);
            }
        }
    }
    let mm = |v: f32, a: usize| (v / sp[a]).round() as usize;
    let blo = [lo[0].saturating_sub(mm(15.0, 0)), lo[1].saturating_sub(mm(5.0, 1)), lo[2].saturating_sub(mm(3.0, 2))];
    let bhi = [(hi[0] + mm(15.0, 0)).min(nx), (hi[1] + mm(25.0, 1)).min(ny), (hi[2] + mm(3.0, 2)).min(nz)];
    let mut hu = crop(ct, blo, bhi);
    ctproc::gaussian(&mut hu, 0.2);
    let bd = hu.dims;
    let to_full = |i: usize| -> usize {
        let (x, y, z) = (i % bd[0], (i / bd[0]) % bd[1], i / (bd[0] * bd[1]));
        ((z + blo[2]) * ny + y + blo[1]) * nx + x + blo[0]
    };
    let teeth_box: Vec<bool> = (0..hu.data.len()).into_par_iter().map(|i| is_tooth(seg.labels[to_full(i)])).collect();
    let near = ctproc::dilate_l1(&teeth_box, bd, (0.6 / sp[0]).round() as isize);
    let hard: Vec<bool> = hu.data.par_iter().zip(near.par_iter()).map(|(&v, &n)| v > BONE_HU && !n).collect();
    let (comp, sizes) = ctproc::label(&hard, bd);

    // 덩어리별 칸 목록 (작은 것만)
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &c) in comp.iter().enumerate() {
        if c != 0 {
            let v = sizes[c as usize] as f32 * voxel_mm3;
            if (MIN_MM3..=MAX_MM3).contains(&v) {
                members.entry(c).or_default().push(i);
            }
        }
    }
    let mean_z = |upper: bool| {
        let z: Vec<f32> = centres.iter().filter(|(f, _)| (**f < 30) == upper).map(|(_, c)| c.z).collect();
        if z.is_empty() { f32::NAN } else { z.iter().sum::<f32>() / z.len() as f32 }
    };
    let (upper_z, lower_z) = (mean_z(true), mean_z(false));

    let mut found: Vec<Candidate> = Vec::new();
    for (c, list) in members {
        let max_hu = list.iter().map(|&i| hu.data[i]).fold(f32::MIN, f32::max);
        if max_hu < CORE_HU {
            continue;
        }
        let mean_hu = list.iter().map(|&i| hu.data[i]).sum::<f32>() / list.len() as f32;
        let (mut l, mut h, mut sum) = ([usize::MAX; 3], [0usize; 3], [0f64; 3]);
        for &i in &list {
            let p = [i % bd[0], (i / bd[0]) % bd[1], i / (bd[0] * bd[1])];
            for a in 0..3 {
                l[a] = l[a].min(p[a]);
                h[a] = h[a].max(p[a]);
                sum[a] += p[a] as f64;
            }
        }
        let n = list.len() as f64;
        let centre = Vec3::new(
            hu.origin[0] + (sum[0] / n) as f32 * sp[0],
            hu.origin[1] + (sum[1] / n) as f32 * sp[1],
            hu.origin[2] + (sum[2] / n) as f32 * sp[2],
        );
        // 둘레(2칸)가 살인 비율. 뼈에서 떨어져 있을수록 크다
        let rl = l.map(|v| v.saturating_sub(3));
        let rh = [0, 1, 2].map(|a| (h[a] + 4).min(bd[a]));
        let rd = [0, 1, 2].map(|a| rh[a] - rl[a]);
        let mut m = vec![false; rd[0] * rd[1] * rd[2]];
        for &i in &list {
            let p = [i % bd[0] - rl[0], (i / bd[0]) % bd[1] - rl[1], i / (bd[0] * bd[1]) - rl[2]];
            m[(p[2] * rd[1] + p[1]) * rd[0] + p[0]] = true;
        }
        let grown = ctproc::dilate_l1(&m, rd, 2);
        let (mut ring, mut soft) = (0usize, 0usize);
        for (j, (&g, &o)) in grown.iter().zip(&m).enumerate() {
            if g && !o {
                ring += 1;
                let (x, y, z) = (j % rd[0] + rl[0], (j / rd[0]) % rd[1] + rl[1], j / (rd[0] * rd[1]) + rl[2]);
                if hu.data[(z * bd[1] + y) * bd[0] + x] < 800.0 {
                    soft += 1;
                }
            }
        }
        let upper = (centre.z - upper_z).abs() < (centre.z - lower_z).abs();
        let (nearest, nearest_mm) = centres
            .iter()
            .filter(|(f, _)| (**f < 30) == upper)
            .map(|(f, p)| (*f, (p.truncate() - centre.truncate()).length()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::NAN));
        found.push(Candidate {
            id: 0,
            fdi: None,
            voxels: list.iter().map(|&i| to_full(i)).collect(),
            volume_mm3: sizes[c as usize] as f32 * voxel_mm3,
            centre,
            size_mm: [0, 1, 2].map(|a| (h[a] - l[a] + 1) as f32 * sp[a]),
            mean_hu,
            max_hu,
            soft_around: if ring > 0 { soft as f32 / ring as f32 } else { 0.0 },
            upper,
            nearest,
            nearest_mm,
        });
    }
    found.sort_by(|a, b| a.centre.x.total_cmp(&b.centre.x).then(a.centre.y.total_cmp(&b.centre.y)));
    for (k, c) in found.iter_mut().enumerate() {
        c.id = k + 1;
        c.fdi = assign.get(&c.id).copied();
    }
    log(format!("뿌리 조각 후보 {}개", found.len()));
    for c in &found {
        log(format!(
            "  #{} {:.1}mm³ 밝기 {:.0}/{:.0} 위치 ({:.1},{:.1},{:.1}) 크기 {:.1}×{:.1}×{:.1} 둘레가 살 {:.2} {}턱 {}번에서 {:.1}mm 붙인 번호 {}",
            c.id, c.volume_mm3, c.mean_hu, c.max_hu, c.centre.x, c.centre.y, c.centre.z, c.size_mm[0], c.size_mm[1], c.size_mm[2],
            c.soft_around, if c.upper { "위" } else { "아래" }, c.nearest, c.nearest_mm,
            c.fdi.map_or("-".to_string(), |f| f.to_string())
        ));
    }
    found
}

// 종류별 삼각형 상한과 흐리기(mm). labels_to_mesh.py와 같다
fn target_tris(kind: &str) -> usize {
    match kind {
        "jaw" => 110_000,
        "sinus" => 15_000,
        "other" | "tooth" => 6_000,
        "remnant" => 4_000,
        _ => 3_000,
    }
}

fn finish(mut m: Mesh, target: usize) -> Option<Mesh> {
    if m.idx.is_empty() {
        return None;
    }
    ctproc::taubin(&mut m, 10);
    Some(ctproc::simplify(&m, target))
}

/// 칸 목록의 둘레를 잘라 CT 밝기 level의 면을 뽑는다 (속이 빈 얇은 조각은 마스크보다 밝기가 모양을 덜 뭉갠다)
pub(crate) fn hu_surface(ct: &Volume, voxels: &[usize], level: f32, target: usize) -> Option<Mesh> {
    let [nx, ny, nz] = ct.dims;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for &i in voxels {
        let p = [i % nx, (i / nx) % ny, i / (nx * ny)];
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a] + 1);
        }
    }
    let lo = lo.map(|v| v.saturating_sub(8));
    let hi = [(hi[0] + 8).min(nx), (hi[1] + 8).min(ny), (hi[2] + 8).min(nz)];
    let mut v = crop(ct, lo, hi);
    ctproc::gaussian(&mut v, 0.2);
    let d = v.dims;
    let mut m = vec![false; v.data.len()];
    for &i in voxels {
        let p = [i % nx - lo[0], (i / nx) % ny - lo[1], i / (nx * ny) - lo[2]];
        m[(p[2] * d[1] + p[1]) * d[0] + p[0]] = true;
    }
    let near = ctproc::dilate(&m, d, 2);
    v.data.par_iter_mut().zip(near.par_iter()).for_each(|(x, &n)| {
        if !n {
            *x = AIR;
        }
    });
    finish(ctproc::isosurface(&v, level), target)
}

/// 번호 볼륨에서 한 구조의 면을 뽑는다. 0.5mm 격자 결과라 계단이 지므로 살짝 흐린 뒤 0.5에서 뽑는다
pub(crate) fn label_surface(ct: &Volume, labels: &[u8], label: u8, s: &LabelStats, sigma_mm: f32, target: usize) -> Option<Mesh> {
    let [nx, ny, nz] = ct.dims;
    let lo = s.lo.map(|v| v.saturating_sub(8));
    let hi = [(s.hi[0] + 9).min(nx), (s.hi[1] + 9).min(ny), (s.hi[2] + 9).min(nz)];
    let d = [0, 1, 2].map(|a| hi[a] - lo[a]);
    let mut data = vec![0f32; d[0] * d[1] * d[2]];
    data.par_chunks_mut(d[0] * d[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..d[1] {
            for x in 0..d[0] {
                if labels[((z + lo[2]) * ny + y + lo[1]) * nx + x + lo[0]] == label {
                    plane[y * d[0] + x] = 1.0;
                }
            }
        }
    });
    let mut v = Volume { dims: d, spacing: ct.spacing, origin: [0, 1, 2].map(|a| ct.origin[a] + lo[a] as f32 * ct.spacing[a]), data };
    ctproc::gaussian(&mut v, sigma_mm);
    finish(ctproc::isosurface(&v, 0.5), target)
}

/// 2×2×2 칸 가운데 하나라도 차 있으면 참
fn shrink_mask(m: &[bool], dims: [usize; 3]) -> Vec<bool> {
    let d = dims.map(|n| n / 2);
    let mut out = vec![false; d[0] * d[1] * d[2]];
    out.par_chunks_mut(d[0] * d[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..d[1] {
            for x in 0..d[0] {
                plane[y * d[0] + x] = (0..8).any(|k| m[((2 * z + (k >> 2)) * dims[1] + 2 * y + ((k >> 1) & 1)) * dims[0] + 2 * x + (k & 1)]);
            }
        }
    });
    out
}

/// CT 밝기로 아래턱뼈와 위턱뼈를 통째로 뽑는다 (labels_to_mesh.py의 whole_jaws)
fn whole_jaws(ct: &Volume, seg: &Seg, lower_id: u8, upper_id: u8, lower_teeth: &[bool], upper_teeth: &[bool], log: Log) -> Option<Vec<(String, Mesh, f32)>> {
    let mut half = ctproc::shrink2(ct);
    ctproc::gaussian(&mut half, 0.3);
    let d = half.dims;
    let plugs = [ctproc::dilate_l1(&shrink_mask(lower_teeth, ct.dims), d, 1), ctproc::dilate_l1(&shrink_mask(upper_teeth, ct.dims), d, 1)];
    let holes: Vec<bool> = plugs[0].par_iter().zip(plugs[1].par_iter()).map(|(a, b)| *a || *b).collect();
    let hints = [
        shrink_mask(&seg.labels.par_iter().map(|&l| l == lower_id).collect::<Vec<_>>(), ct.dims),
        shrink_mask(&seg.labels.par_iter().map(|&l| l == upper_id).collect::<Vec<_>>(), ct.dims),
    ];
    let pick = |comp: &[u32], n: usize| -> Option<[u32; 2]> {
        let mut out = [0u32; 2];
        for j in 0..2 {
            let mut overlap = vec![0usize; n];
            for (c, &h) in comp.iter().zip(&hints[j]) {
                if h && *c != 0 {
                    overlap[*c as usize] += 1;
                }
            }
            let best = (1..n).max_by_key(|&k| overlap[k])?;
            if overlap[best] == 0 {
                return None;
            }
            out[j] = best as u32;
        }
        Some(out)
    };
    let components = |level: f32| {
        let m: Vec<bool> = half.data.par_iter().zip(holes.par_iter()).map(|(&v, &h)| v > level && !h).collect();
        ctproc::label(&m, d)
    };
    // 턱관절처럼 두 뼈가 가까운 곳은 뼈 기준 밝기에서 한 덩어리로 붙는다.
    // 기준을 100씩 올려 두 뼈가 떨어지는 값을 찾고, 그때의 덩어리를 씨앗으로 쓴다
    let mut found = None;
    for step in 0..7 {
        let level = JAW_BONE_HU + 100.0 * step as f32;
        let (comp, sizes) = components(level);
        let p = pick(&comp, sizes.len())?;
        if p[0] != p[1] {
            found = Some((level, comp, p));
            break;
        }
    }
    let (level, comp, p) = found?;
    let mut masks: [Vec<bool>; 2] = [comp.par_iter().map(|&c| c == p[0]).collect(), comp.par_iter().map(|&c| c == p[1]).collect()];
    if level > JAW_BONE_HU {
        // 원래 기준에서 뼈인 칸은 가까운 쪽 씨앗에 붙인다 (씨앗에서 퍼져 나가는 거리로 정함)
        let (base, bsizes) = components(JAW_BONE_HU);
        let merged_id = pick(&base, bsizes.len())?[0];
        let mut owner = vec![0u8; base.len()];
        let mut queue = std::collections::VecDeque::new();
        for i in 0..owner.len() {
            for j in 0..2 {
                if masks[j][i] {
                    owner[i] = j as u8 + 1;
                    queue.push_back(i);
                }
            }
        }
        let [nx, ny, nz] = d;
        while let Some(i) = queue.pop_front() {
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            let o = owner[i];
            let mut visit = |k: usize| {
                if owner[k] == 0 {
                    owner[k] = o;
                    queue.push_back(k);
                }
            };
            if x > 0 { visit(i - 1) }
            if x + 1 < nx { visit(i + 1) }
            if y > 0 { visit(i - nx) }
            if y + 1 < ny { visit(i + nx) }
            if z > 0 { visit(i - nx * ny) }
            if z + 1 < nz { visit(i + nx * ny) }
        }
        for j in 0..2 {
            masks[j] = base.par_iter().zip(owner.par_iter()).map(|(&b, &o)| b == merged_id && o == j as u8 + 1).collect();
        }
        log(format!("턱뼈: 밝기 {JAW_BONE_HU:.0}에서는 위아래가 붙어 있어 {level:.0}에서 갈랐습니다"));
    }
    let voxel_mm3 = half.spacing.iter().product::<f32>();
    let mut out = Vec::new();
    for (j, name) in ["lower_jawbone", "upper_jawbone"].iter().enumerate() {
        // 속의 해면골 틈을 메워 겉모양만 남긴다. 치아가 박힌 자리로 속이 바깥과 통하므로
        // 그 턱의 치아로 구멍을 막은 채 메우고, 치아 자리는 다시 비운다
        let closed = ctproc::erode_l1(&ctproc::dilate_l1(&masks[j], d, 3), d, 3);
        let plugged: Vec<bool> = closed.par_iter().zip(plugs[j].par_iter()).map(|(a, b)| *a || *b).collect();
        let mut solid = ctproc::fill_holes(&plugged, d);
        if j == 0 {
            // 아래턱뼈는 촬영 범위 끝에서 잘리고 신경관 구멍도 있어 통째로는 메워지지 않는다. 가로 단면마다 메운다
            let plane = d[0] * d[1];
            solid.par_chunks_mut(plane).for_each(|s| {
                let filled = ctproc::fill_holes(s, [d[0], d[1], 1]);
                s.copy_from_slice(&filled);
            });
        }
        solid.par_iter_mut().zip(holes.par_iter()).for_each(|(s, &h)| *s = *s && !h);
        let vol = solid.iter().filter(|&&s| s).count() as f32 * voxel_mm3;
        let mut v = Volume { dims: d, spacing: half.spacing, origin: half.origin, data: solid.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect() };
        ctproc::gaussian(&mut v, 0.5);
        if let Some(m) = finish(ctproc::isosurface(&v, 0.5), target_tris("jaw")) {
            out.push((name.to_string(), m, vol));
        }
    }
    Some(out)
}

/// 구조별 메시를 만든다. 이름은 labels_to_mesh.py와 같다 (치아는 AI 이름, 뿌리 조각은 remnant_fdi47, 번호 없는 조각은 fragment_4)
pub fn part_meshes(ct: &Volume, seg: &Seg, cands: &[Candidate], ai_jaws: bool, log: Log) -> Vec<Part> {
    let t = Instant::now();
    let centre = Vec3::from(ct.origin) + 0.5 * Vec3::from(ct.dims.map(|d| d as f32)) * Vec3::from(ct.spacing);
    let voxel_mm3 = ct.spacing.iter().product::<f32>();
    let stats = label_stats(&seg.labels, ct.dims);
    let mut parts = Vec::new();
    let mut push = |name: String, mesh: Mesh, vol: f32| {
        let mut p = ctproc::to_part(&name, mesh, centre);
        p.volume_mm3 = Some(vol as f64);
        parts.push(p);
    };

    // 1) 뿌리 조각
    let remnant_voxels: HashSet<usize> = cands.iter().flat_map(|c| c.voxels.iter().copied()).collect();
    let mesh_list: Vec<(String, Option<Mesh>, f32)> = cands
        .par_iter()
        .map(|c| {
            let name = c.fdi.map_or(format!("fragment_{}", c.id), |f| format!("remnant_fdi{f}"));
            let kind = KINDS[kind_of(&name)].key;
            (name, hu_surface(ct, &c.voxels, REMNANT_HU, target_tris(kind)), c.volume_mm3)
        })
        .collect();
    for (name, m, vol) in mesh_list {
        if let Some(m) = m {
            push(name, m, vol);
        }
    }

    // 2) AI가 나눈 구조
    let jaw_ids: HashMap<&str, u8> = seg.names.iter().filter(|(_, n)| n.contains("jawbone")).map(|(l, n)| (n.as_str(), *l)).collect();
    let todo: Vec<(u8, &String)> = seg.names.iter().filter(|(l, n)| stats.contains_key(l) && (ai_jaws || !jaw_ids.contains_key(n.as_str()))).map(|(l, n)| (*l, n)).collect();
    let results: Vec<(String, Option<Mesh>, f32, &'static str)> = todo
        .par_iter()
        .map(|(l, name)| {
            let s = &stats[l];
            let vol = s.count as f32 * voxel_mm3;
            // 뿌리 조각의 일부를 다른 이름으로 잡은 것은 겹치므로 뺀다
            if !remnant_voxels.is_empty() {
                let [nx, ny, _] = ct.dims;
                let mut overlap = 0usize;
                for z in s.lo[2]..=s.hi[2] {
                    for y in s.lo[1]..=s.hi[1] {
                        for x in s.lo[0]..=s.hi[0] {
                            let i = (z * ny + y) * nx + x;
                            if seg.labels[i] == *l && remnant_voxels.contains(&i) {
                                overlap += 1;
                            }
                        }
                    }
                }
                if overlap * 2 > s.count {
                    return (name.to_string(), None, vol, "뿌리 조각과 겹쳐서 뺌");
                }
            }
            let kind = KINDS[kind_of(name)].key;
            let sigma = if matches!(kind, "pulp" | "canal") { 0.3 } else { 0.5 };
            let m = label_surface(ct, &seg.labels, *l, s, sigma, target_tris(kind));
            let why = if m.is_none() { "너무 작아 메시에서 빠짐" } else { "" };
            (name.to_string(), m, vol, why)
        })
        .collect();
    for (name, m, vol, why) in results {
        match m {
            Some(m) => push(name, m, vol),
            None => log(format!("{name}: {vol:.1}mm³, {why}")),
        }
    }

    // 3) 턱뼈 전체
    if !ai_jaws {
        if let (Some(&lo), Some(&up)) = (jaw_ids.get("lower_jawbone"), jaw_ids.get("upper_jawbone")) {
            let tooth_of = |upper: bool| -> Vec<bool> {
                let ids: HashSet<u8> = seg
                    .names
                    .iter()
                    .filter(|(_, n)| matches!(KINDS[kind_of(n)].key, "tooth" | "pulp"))
                    .filter(|(_, n)| {
                        let f = fdi_of(n).unwrap_or(0);
                        let f = if f > 100 { f - 100 } else { f };
                        (f < 30) == upper
                    })
                    .map(|(l, _)| *l)
                    .collect();
                let mut m: Vec<bool> = seg.labels.par_iter().map(|l| ids.contains(l)).collect();
                for c in cands.iter().filter(|c| c.upper == upper) {
                    for &i in &c.voxels {
                        m[i] = true;
                    }
                }
                m
            };
            match whole_jaws(ct, seg, lo, up, &tooth_of(false), &tooth_of(true), log) {
                Some(jaws) => {
                    for (name, m, vol) in jaws {
                        push(name, m, vol);
                    }
                }
                None => log("위아래 턱뼈를 가르지 못했습니다. AI 턱뼈 없이 진행합니다".into()),
            }
        }
    }
    log(format!("구조별 메시 {}개 ({:.1}초)", parts.len(), t.elapsed().as_secs_f32()));
    parts
}
