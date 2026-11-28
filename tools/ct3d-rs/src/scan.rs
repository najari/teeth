//! 구강 스캔(위턱·아래턱 STL)을 CT 치아에 맞춘다 (tools/ct3d의 scan_to_ct.py, teeth_for_model.py의 스캔 부분을 옮긴 것).
//!
//! 스캔 좌표는 CEREC/inLab 형식으로 본다: z가 교합 방향(위턱 쪽이 +z), 앞니가 -y 쪽, 환자 왼쪽이 +x.
//! CT 쪽 좌표는 y가 위, z가 얼굴 앞, x가 환자 왼쪽이면 원점은 어디든 된다(CT 뷰어 좌표, 모형용 좌표 모두).
//!
//! 맞추는 방법: CT 치아마다 끝에서 6mm(머리)의 점을 스캔의 치아 띠(교합면에서 7mm)에 강체 ICP로 맞춘다.
//! 매번 가장 먼 30%는 버리고, 시작 위치에 민감해서 수평 회전 5가지 × 좌우·앞뒤 9가지로 시작해 본 뒤 앞선 셋을 다듬는다.

use glam::{DMat3, DQuat, DVec3};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

const CROWN_MM: f64 = 6.0;
const BAND_MM: f64 = 7.0;
const KEEP: f64 = 0.7;
pub const SNAP_MM: f64 = 0.7;
pub const FIT_LIMIT_MM: f64 = 0.4;
pub const GUM_GAP_POOR_MM: f64 = 2.0;

/// 열린 표면 메시 (꼭짓점은 합쳐져 있음)
pub struct Surface {
    pub pos: Vec<DVec3>,
    pub faces: Vec<[u32; 3]>,
}

impl Surface {
    pub fn vertex_normals(&self) -> Vec<DVec3> {
        let mut n = vec![DVec3::ZERO; self.pos.len()];
        for f in &self.faces {
            let [a, b, c] = f.map(|i| self.pos[i as usize]);
            let fnm = (b - a).cross(c - a);
            for &i in f {
                n[i as usize] += fnm;
            }
        }
        n.into_iter().map(|v| v.normalize_or_zero()).collect()
    }
}

/// 이진 STL을 읽고 같은 자리의 꼭짓점을 합친다 (STL은 삼각형마다 꼭짓점을 따로 가진다)
pub fn read_stl(path: &Path) -> Result<Surface, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}을 읽지 못했습니다: {e}", path.display()))?;
    if raw.len() < 84 {
        return Err(format!("{}: STL이 너무 짧습니다", path.display()));
    }
    let n = u32::from_le_bytes([raw[80], raw[81], raw[82], raw[83]]) as usize;
    if raw.len() < 84 + 50 * n {
        return Err(format!("{}: 이진 STL이 아니거나 잘렸습니다(텍스트 STL은 읽지 못합니다)", path.display()));
    }
    let mut index: HashMap<[u32; 3], u32> = HashMap::with_capacity(n);
    let mut pos = Vec::with_capacity(n / 2);
    let mut faces = Vec::with_capacity(n);
    let f32_at = |o: usize| f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
    for t in 0..n {
        let base = 84 + 50 * t + 12;
        let mut tri = [0u32; 3];
        for (k, slot) in tri.iter_mut().enumerate() {
            let o = base + 12 * k;
            let key = [0, 4, 8].map(|d| f32_at(o + d).to_bits());
            *slot = *index.entry(key).or_insert_with(|| {
                pos.push(DVec3::new(f32_at(o) as f64, f32_at(o + 4) as f64, f32_at(o + 8) as f64));
                (pos.len() - 1) as u32
            });
        }
        if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
            faces.push(tri);
        }
    }
    Ok(Surface { pos, faces })
}

/// 고른 칸으로 나눈 점 집합. 가까운 점 찾기용
pub struct Grid {
    h: f64,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
    pub pts: Vec<DVec3>,
}

impl Grid {
    pub fn new(pts: Vec<DVec3>, h: f64) -> Self {
        let mut cells: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            cells.entry(Self::key_of(*p, h)).or_default().push(i as u32);
        }
        Self { h, cells, pts }
    }
    fn key_of(p: DVec3, h: f64) -> (i32, i32, i32) {
        ((p.x / h).floor() as i32, (p.y / h).floor() as i32, (p.z / h).floor() as i32)
    }
    /// max_r 안에서 가장 가까운 점(번호, 거리)
    pub fn nearest(&self, p: DVec3, max_r: f64) -> Option<(usize, f64)> {
        let (kx, ky, kz) = Self::key_of(p, self.h);
        let mut best: Option<(usize, f64)> = None;
        let mut best_d2 = max_r * max_r;
        let rings = (max_r / self.h).ceil() as i32 + 1;
        for r in 0..=rings {
            if best.is_some() && ((r - 1).max(0) as f64 * self.h).powi(2) > best_d2 {
                break;
            }
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        if let Some(list) = self.cells.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &i in list {
                                let d2 = (self.pts[i as usize] - p).length_squared();
                                if d2 <= best_d2 {
                                    best_d2 = d2;
                                    best = Some((i as usize, d2));
                                }
                            }
                        }
                    }
                }
            }
        }
        best.map(|(i, d2)| (i, d2.sqrt()))
    }
}

/// 대칭 n×n 행렬의 고유값과 고유벡터(열). 야코비 회전
fn jacobi<const N: usize>(mut a: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut v = [[0f64; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..N).flat_map(|p| (p + 1..N).map(move |q| (p, q))).map(|(p, q)| a[p][q] * a[p][q]).sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..N {
            for q in p + 1..N {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (kp, kq) = (a[k][p], a[k][q]);
                    a[k][p] = c * kp - s * kq;
                    a[k][q] = s * kp + c * kq;
                }
                for k in 0..N {
                    let (pk, qk) = (a[p][k], a[q][k]);
                    a[p][k] = c * pk - s * qk;
                    a[q][k] = s * pk + c * qk;
                }
                for row in v.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
            }
        }
    }
    let mut vals = [0f64; N];
    for i in 0..N {
        vals[i] = a[i][i];
    }
    (vals, v)
}

/// a를 b에 가장 잘 겹치는 회전과 이동 (Horn의 사원수 방법)
pub fn kabsch(a: &[DVec3], b: &[DVec3]) -> (DMat3, DVec3) {
    let n = a.len() as f64;
    let ca = a.iter().copied().sum::<DVec3>() / n;
    let cb = b.iter().copied().sum::<DVec3>() / n;
    let mut s = [[0f64; 3]; 3];
    for (p, q) in a.iter().zip(b) {
        let (p, q) = ((*p - ca).to_array(), (*q - cb).to_array());
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += p[r] * q[c];
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = s;
    let m = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (vals, vecs) = jacobi(m);
    let k = (0..4).max_by(|&i, &j| vals[i].total_cmp(&vals[j])).unwrap();
    let q = DQuat::from_xyzw(vecs[1][k], vecs[2][k], vecs[3][k], vecs[0][k]).normalize();
    let r = DMat3::from_quat(q);
    (r, cb - r * ca)
}

/// 강체 변환 (p → r p + t)
#[derive(Clone, Copy)]
pub struct Rigid {
    pub r: DMat3,
    pub t: DVec3,
}

impl Rigid {
    pub fn apply(&self, p: DVec3) -> DVec3 {
        self.r * p + self.t
    }
    pub fn inverse(&self) -> Rigid {
        let rt = self.r.transpose();
        Rigid { r: rt, t: -(rt * self.t) }
    }
}

fn icp(src: &[DVec3], grid: &Grid, mut x: Rigid, iters: usize) -> Rigid {
    for _ in 0..iters {
        let pairs: Vec<(DVec3, DVec3, f64)> = src
            .par_iter()
            .filter_map(|p| {
                let m = x.apply(*p);
                grid.nearest(m, 10.0).map(|(i, d)| (m, grid.pts[i], d))
            })
            .collect();
        if pairs.len() < 10 {
            break;
        }
        let mut ds: Vec<f64> = pairs.iter().map(|p| p.2).collect();
        let k = ((ds.len() as f64 * KEEP) as usize).min(ds.len() - 1);
        let cut = *ds.select_nth_unstable_by(k, f64::total_cmp).1;
        let (a, b): (Vec<DVec3>, Vec<DVec3>) = pairs.iter().filter(|p| p.2 <= cut).map(|p| (p.0, p.1)).unzip();
        let (dr, dt) = kabsch(&a, &b);
        x = Rigid { r: dr * x.r, t: dr * x.t + dt };
    }
    x
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let k = v.len() / 2;
    *v.select_nth_unstable_by(k, f64::total_cmp).1
}

pub struct Fit {
    /// CT 쪽 좌표 → 스캔 좌표
    pub ct_to_scan: Rigid,
    pub median_mm: f64,
    pub p90_mm: f64,
    pub per_tooth: Vec<(u32, f64)>,
}

/// 치아(번호, 메시 꼭짓점)를 스캔에 맞춘다. upper면 위턱
pub fn register(scan: &Surface, teeth: &[(u32, Vec<DVec3>)], upper: bool) -> Result<Fit, String> {
    // CT 치아 머리 점: 위 치아는 머리가 아래
    let mut src = Vec::new();
    let mut owner = Vec::new();
    for (f, pts) in teeth {
        let (lo, hi) = pts.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.y), b.max(p.y)));
        for p in pts {
            if (upper && p.y < lo + CROWN_MM) || (!upper && p.y > hi - CROWN_MM) {
                src.push(*p);
                owner.push(*f);
            }
        }
    }
    if src.len() < 100 {
        return Err("스캔에 맞출 CT 치아가 없습니다".into());
    }
    let mut zs: Vec<f64> = scan.pos.iter().map(|p| p.z).collect();
    zs.sort_by(f64::total_cmp);
    let band: Vec<DVec3> = if upper {
        let lim = zs[(zs.len() as f64 * 0.005) as usize] + BAND_MM;
        scan.pos.iter().copied().filter(|p| p.z < lim).collect()
    } else {
        let lim = zs[(zs.len() as f64 * 0.995) as usize] - BAND_MM;
        scan.pos.iter().copied().filter(|p| p.z > lim).collect()
    };
    let tgt_mean = band.iter().copied().sum::<DVec3>() / band.len() as f64;
    let grid = Grid::new(band, 1.0);
    let src_mean = src.iter().copied().sum::<DVec3>() / src.len() as f64;
    let step = (src.len() / 2500).max(1);
    let sub: Vec<DVec3> = src.iter().step_by(step).copied().collect();
    // 스캔 좌표 = BASE · CT 좌표 (CT: x 왼쪽, y 위, z 앞 / 스캔: x 왼쪽, y 뒤, z 위턱 쪽)
    let base = DMat3::from_cols(DVec3::X, DVec3::Z, -DVec3::Y);
    let mut starts: Vec<(f64, Rigid)> = Vec::new();
    for yaw in [-10.0f64, -5.0, 0.0, 5.0, 10.0] {
        let r0 = DMat3::from_rotation_z(yaw.to_radians()) * base;
        let t0 = tgt_mean - r0 * src_mean;
        for dx in [-3.0, 0.0, 3.0] {
            for dy in [-3.0, 0.0, 3.0] {
                let x = icp(&sub, &grid, Rigid { r: r0, t: t0 + DVec3::new(dx, dy, 0.0) }, 20);
                let mut d: Vec<f64> = sub.par_iter().filter_map(|p| grid.nearest(x.apply(*p), 10.0).map(|n| n.1)).collect();
                starts.push((median(&mut d), x));
            }
        }
    }
    starts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut best: Option<(f64, Rigid)> = None;
    for (_, x0) in starts.iter().take(3) {
        let x = icp(&src, &grid, *x0, 60);
        let mut d: Vec<f64> = src.par_iter().filter_map(|p| grid.nearest(x.apply(*p), 10.0).map(|n| n.1)).collect();
        let m = median(&mut d);
        if best.as_ref().map_or(true, |b| m < b.0) {
            best = Some((m, x));
        }
    }
    let x = best.unwrap().1;
    // 평가는 스캔 전체에 대해
    let full = Grid::new(scan.pos.clone(), 0.5);
    let d: Vec<f64> = src.par_iter().map(|p| full.nearest(x.apply(*p), 10.0).map_or(10.0, |n| n.1)).collect();
    let mut all = d.clone();
    let med = median(&mut all);
    all.sort_by(f64::total_cmp);
    let p90 = all[(all.len() as f64 * 0.9) as usize];
    let mut per: HashMap<u32, Vec<f64>> = HashMap::new();
    for (f, dd) in owner.iter().zip(&d) {
        per.entry(*f).or_default().push(*dd);
    }
    let mut per_tooth: Vec<(u32, f64)> = per.into_iter().map(|(f, mut v)| (f, median(&mut v))).collect();
    per_tooth.sort_by_key(|p| p.0);
    Ok(Fit { ct_to_scan: x, median_mm: med, p90_mm: p90, per_tooth })
}

/// 스캔 폴더에서 이름에 UpperJaw / LowerJaw가 들어간 STL을 찾는다
pub fn find_stl(dir: &Path, jaw: &str) -> Result<std::path::PathBuf, String> {
    std::fs::read_dir(dir)
        .map_err(|e| format!("{}를 읽지 못했습니다: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            let n = p.file_name().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
            n.ends_with(".stl") && n.contains(&jaw.to_lowercase())
        })
        .ok_or(format!("{}에 이름에 {jaw}가 들어간 STL이 없습니다", dir.display()))
}

/// 위·아래 스캔을 읽어 맞춘다. teeth는 (번호, CT 쪽 좌표의 치아 꼭짓점). 돌려주는 스캔은 CT 쪽 좌표로 옮긴 것
pub fn load_and_register(dir: &Path, teeth: &[(u32, Vec<DVec3>)], log: crate::ctproc::Log) -> Result<[(Surface, Fit); 2], String> {
    let mut out = Vec::new();
    for (jaw, name, upper) in [("UpperJaw", "위", true), ("LowerJaw", "아래", false)] {
        let t = std::time::Instant::now();
        let mut s = read_stl(&find_stl(dir, jaw)?)?;
        let mine: Vec<(u32, Vec<DVec3>)> = teeth.iter().filter(|(f, _)| (*f < 30) == upper).cloned().collect();
        let fit = register(&s, &mine, upper)?;
        let back = fit.ct_to_scan.inverse();
        s.pos.par_iter_mut().for_each(|p| *p = back.apply(*p));
        log(format!(
            "{name}턱 스캔: 삼각형 {}개, 치아 머리에서 스캔까지 거리 중앙값 {:.2}mm, 90% {:.2}mm ({:.1}초)",
            s.faces.len(),
            fit.median_mm,
            fit.p90_mm,
            t.elapsed().as_secs_f32()
        ));
        log(format!("   치아별(mm): {}", fit.per_tooth.iter().map(|(f, d)| format!("{f}:{d:.2}")).collect::<Vec<_>>().join(" ")));
        let far: Vec<String> = fit.per_tooth.iter().filter(|p| p.1 > 0.6).map(|p| p.0.to_string()).collect();
        if !far.is_empty() {
            log(format!("   ※ 0.6mm 넘게 어긋난 치아: {}. 두 촬영 사이에 바뀌었거나 분할이 다를 수 있습니다.", far.join(", ")));
        }
        out.push((s, fit));
    }
    let lower = out.pop().unwrap();
    let upper = out.pop().unwrap();
    Ok([upper, lower])
}

/// 치아 머리(자기 좌표)를 한 번 잘게 나눈 뒤 가까운 스캔 표면에 붙인다. world_of는 자기 좌표 → 스캔과 같은 좌표,
/// local_of는 그 반대. 붙인 점의 비율도 돌려준다
pub fn snap_crown(
    v: &[DVec3],
    f: &[[u32; 3]],
    scan: &Grid,
    scan_normals: &[DVec3],
    world_of: impl Fn(DVec3) -> DVec3 + Sync,
    local_of: impl Fn(DVec3) -> DVec3 + Sync,
    dir_of: impl Fn(DVec3) -> DVec3 + Sync,
) -> (Vec<DVec3>, Vec<[u32; 3]>, f64) {
    // 가운데 점으로 나누기: 삼각형 하나 → 넷. 변의 가운데 점은 이웃 삼각형과 함께 쓴다
    let mut pos = v.to_vec();
    let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut faces = Vec::with_capacity(f.len() * 4);
    for t in f {
        let mut m = [0u32; 3];
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            m[k] = *mid.entry(key).or_insert_with(|| {
                pos.push((pos[a as usize] + pos[b as usize]) * 0.5);
                (pos.len() - 1) as u32
            });
        }
        faces.push([t[0], m[0], m[2]]);
        faces.push([t[1], m[1], m[0]]);
        faces.push([t[2], m[2], m[1]]);
        faces.push([m[0], m[1], m[2]]);
    }
    let normals = Surface { pos: pos.clone(), faces: faces.clone() }.vertex_normals();
    let hits: Vec<Option<(usize, f64, f64)>> = pos
        .par_iter()
        .zip(normals.par_iter())
        .map(|(p, n)| scan.nearest(world_of(*p), SNAP_MM).map(|(i, d)| (i, d, dir_of(*n).dot(scan_normals[i]))))
        .collect();
    // 스캔 면의 앞뒤 방향이 CT 쪽과 반대일 수 있다: 아주 가까운 점들의 방향으로 정한다
    let mut close: Vec<f64> = hits.iter().flatten().filter(|h| h.1 < 0.3).map(|h| h.2).collect();
    let sign = if close.is_empty() { 1.0 } else { median(&mut close).signum() };
    let mut snapped = 0usize;
    let out: Vec<DVec3> = pos
        .iter()
        .zip(&hits)
        .map(|(p, h)| match h {
            Some((i, d, dot)) if dot * sign > 0.3 => {
                let w = ((SNAP_MM - d) / 0.35).clamp(0.0, 1.0) * (p.y / 1.0).clamp(0.0, 1.0); // 치아 목(y=0) 쪽은 그대로
                if w > 0.99 {
                    snapped += 1;
                }
                *p + w * (local_of(scan.pts[*i]) - *p)
            }
            _ => *p,
        })
        .collect();
    let share = snapped as f64 / pos.len() as f64;
    (out, faces, share)
}

#[derive(Copy, Clone, PartialEq)]
struct Item(f64, u32);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0) // 작은 거리가 먼저 나오게 뒤집는다
    }
}

/// 여러 씨앗에서 표면을 따라 잰 가장 가까운 거리
fn geodesic(n: usize, adj: &[Vec<(u32, f64)>], seeds: &[u32]) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    for &s in seeds {
        dist[s as usize] = 0.0;
        heap.push(Item(0.0, s));
    }
    while let Some(Item(d, u)) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        for &(v, w) in &adj[u as usize] {
            let nd = d + w;
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap.push(Item(nd, v));
            }
        }
    }
    dist
}

/// 스캔에서 치아 면을 뺀 나머지(잇몸). teeth는 (치아 면의 점, 스캔과 잘 맞지 않은 치아인가).
/// 확실한 치아 면(0.25mm 안)과 확실한 잇몸 면(1mm 밖)을 씨앗으로 정하고, 나머지는 표면을 따라 더 가까운 씨앗 쪽에 붙인다
pub fn gum(scan: &Surface, teeth: &[(Vec<DVec3>, bool)]) -> Surface {
    let all = Grid::new(teeth.iter().flat_map(|t| t.0.iter().copied()).collect(), 0.5);
    let poor = Grid::new(teeth.iter().filter(|t| t.1).flat_map(|t| t.0.iter().copied()).collect(), 0.5);
    let dist: Vec<(f64, bool)> = scan
        .pos
        .par_iter()
        .map(|p| {
            let d = all.nearest(*p, 1.5).map_or(f64::INFINITY, |n| n.1);
            let is_poor = poor.nearest(*p, GUM_GAP_POOR_MM).is_some();
            (d, is_poor)
        })
        .collect();
    let tooth_seed: Vec<u32> = (0..dist.len() as u32).filter(|&i| dist[i as usize].0 < 0.25 || dist[i as usize].1).collect();
    let gum_seed: Vec<u32> = (0..dist.len() as u32).filter(|&i| dist[i as usize].0 > 1.0 && !dist[i as usize].1).collect();
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); scan.pos.len()];
    for f in &scan.faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let w = (scan.pos[a as usize] - scan.pos[b as usize]).length() + 1e-6;
            adj[a as usize].push((b, w));
            adj[b as usize].push((a, w));
        }
    }
    let (dt, dg) = rayon::join(|| geodesic(scan.pos.len(), &adj, &tooth_seed), || geodesic(scan.pos.len(), &adj, &gum_seed));
    let keep: Vec<bool> = dt.iter().zip(&dg).map(|(t, g)| g.is_finite() && g < t).collect();
    let faces: Vec<[u32; 3]> = scan.faces.iter().copied().filter(|f| f.iter().all(|&i| keep[i as usize])).collect();
    // 작은 섬(치아 사이에 남은 조각 등)은 버린다: 삼각형 2,000개 미만
    let mut parent: Vec<u32> = (0..scan.pos.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for f in &faces {
        let r0 = find(&mut parent, f[0]);
        for &v in &f[1..] {
            let r = find(&mut parent, v);
            if r != r0 {
                parent[r as usize] = r0;
            }
        }
    }
    let mut count: HashMap<u32, usize> = HashMap::new();
    let roots: Vec<u32> = faces.iter().map(|f| find(&mut parent, f[0])).collect();
    for r in &roots {
        *count.entry(*r).or_default() += 1;
    }
    let faces: Vec<[u32; 3]> = faces.iter().zip(&roots).filter(|(_, r)| count[r] >= 2000).map(|(f, _)| *f).collect();
    // 쓰는 꼭짓점만 남긴다
    let mut remap = vec![u32::MAX; scan.pos.len()];
    let mut pos = Vec::new();
    let faces = faces
        .iter()
        .map(|f| {
            f.map(|i| {
                if remap[i as usize] == u32::MAX {
                    remap[i as usize] = pos.len() as u32;
                    pos.push(scan.pos[i as usize]);
                }
                remap[i as usize]
            })
        })
        .collect();
    Surface { pos, faces }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kabsch_recovers_rotation() {
        let r = DMat3::from_euler(glam::EulerRot::XYZ, 0.3, -0.2, 0.7);
        let t = DVec3::new(1.0, -2.0, 0.5);
        let a: Vec<DVec3> = (0..50).map(|i| DVec3::new((i as f64 * 0.37).sin() * 5.0, (i as f64 * 0.91).cos() * 3.0, i as f64 * 0.1)).collect();
        let b: Vec<DVec3> = a.iter().map(|p| r * *p + t).collect();
        let (r2, t2) = kabsch(&a, &b);
        assert!((r2 - r).abs().to_cols_array().iter().all(|v| *v < 1e-9));
        assert!((t2 - t).length() < 1e-9);
    }

    #[test]
    fn grid_matches_brute_force() {
        let pts: Vec<DVec3> = (0..2000).map(|i| DVec3::new((i as f64 * 1.3).sin() * 20.0, (i as f64 * 0.7).cos() * 20.0, (i as f64 * 0.11).sin() * 5.0)).collect();
        let g = Grid::new(pts.clone(), 0.8);
        for k in 0..50 {
            let q = DVec3::new(k as f64 * 0.7 - 15.0, 3.0 - k as f64 * 0.3, 1.0);
            let brute = pts.iter().map(|p| (*p - q).length()).fold(f64::MAX, f64::min);
            let got = g.nearest(q, 30.0).unwrap().1;
            assert!((got - brute).abs() < 1e-12, "{got} vs {brute}");
        }
    }
}
