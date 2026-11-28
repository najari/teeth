//! 자동 분할한 치아를 치열 모형(index.html)에 끼워 넣을 데이터 파일 ct-teeth.js로 바꾼다
//! (tools/ct3d의 teeth_for_model.py를 옮긴 것).
//!
//! 치아마다 자기 좌표로 바꿔서 내보낸다.
//!     원점  치아 목(머리와 뿌리의 경계)의 가운데
//!     +y    뿌리에서 머리 쪽 (치아의 긴 축)
//!     +z    볼·입술 쪽 (치열궁의 바깥)
//!     +x    y × z
//! 머리(crown)와 뿌리(root)는 y=0에서 잘라 따로 담는다. 모형이 치료 단계에 따라 둘을 따로 켜고 끄기 때문이다.
//! 뿌리 조각(find_remnants에서 번호를 붙인 것)도 같은 형식으로 담고 remnant 표시를 붙인다.
//!
//! 출력 파일에는 환자의 CT에서 뽑은 치아 모양이 들어 있다. 공개할지는 따로 정한다.

use crate::ctproc::{self, Log, Volume};
use crate::scan;
use crate::seg::{self, Candidate, Seg};
use glam::{DMat3, DVec3};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

const UNIT_MM: f64 = 0.01; // 좌표를 0.01mm 단위 정수로 담는다
const MIN_VOLUME_MM3: f64 = 100.0; // 이보다 작으면 조각으로 보고 뺀다
const UPPER: [u32; 14] = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27];
const LOWER: [u32; 14] = [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37];

// 모형 좌표의 기준. index.html의 표준 치열과 맞춘 값이다
const FRONT_NECK_Z: f64 = -3.4; // 위 가운데 앞니 목의 앞뒤 위치
const UPPER_TIP_Y: f64 = 0.4; // 위아래 치아 끝의 평균 높이 (두 끝 사이 0.8mm)
const LOWER_TIP_Y: f64 = -0.4;
const PIVOT: DVec3 = DVec3::new(0.0, 14.0, -80.0); // index.html이 아래턱을 벌릴 때 쓰는 회전 중심
const Y: DVec3 = DVec3::Y;

pub struct Options {
    pub remap: HashMap<u32, u32>, // 분할 번호 → 모형 번호 (예: 42 → 41)
    pub enamel: f32,              // 법랑질로 보는 밝기. 치아 목 높이를 정하는 데 쓴다
    pub tris: usize,              // 치아 하나의 삼각형 수
    pub scan: Option<std::path::PathBuf>, // 구강 스캔 폴더(UpperJaw, LowerJaw STL). 있으면 치아 머리를 다듬고 잇몸을 넣는다
}

impl Default for Options {
    fn default() -> Self {
        Self { remap: HashMap::new(), enamel: 3300.0, tris: 2600, scan: None }
    }
}

/// 환자 좌표(x 왼쪽, y 뒤, z 위)를 모형 좌표(x 왼쪽, y 위, z 앞)로 돌린다
fn three(x: f64, y: f64, z: f64) -> DVec3 {
    DVec3::new(x, z, -y)
}

fn mean(p: impl IntoIterator<Item = DVec3>) -> DVec3 {
    let (s, n) = p.into_iter().fold((DVec3::ZERO, 0usize), |(s, n), v| (s + v, n + 1));
    s / n as f64
}

/// 대칭 3×3 행렬의 고유값과 고유벡터(야코비 회전). 고유값이 큰 것부터
fn eigh(m: DMat3) -> ([f64; 3], [DVec3; 3]) {
    let mut a = [[0f64; 3]; 3];
    for (r, row) in a.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = m.col(c)[r];
        }
    }
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let off = a[0][1].powi(2) + a[0][2].powi(2) + a[1][2].powi(2);
        let scale = a[0][0].powi(2) + a[1][1].powi(2) + a[2][2].powi(2);
        if off <= 1e-30 * scale.max(1e-300) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (kp, kq) = (a[k][p], a[k][q]);
                a[k][p] = c * kp - s * kq;
                a[k][q] = s * kp + c * kq;
            }
            for k in 0..3 {
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
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    (order.map(|i| a[i][i]), order.map(|i| DVec3::new(v[0][i], v[1][i], v[2][i])))
}

fn covariance(pts: &[DVec3]) -> (DVec3, DMat3) {
    let c = mean(pts.iter().copied());
    let mut m = [[0f64; 3]; 3];
    for p in pts {
        let d = (*p - c).to_array();
        for r in 0..3 {
            for k in 0..3 {
                m[r][k] += d[r] * d[k];
            }
        }
    }
    (c, DMat3::from_cols_array_2d(&m))
}

/// 점들이 가장 길게 퍼진 방향
fn principal_axis(pts: &[DVec3]) -> DVec3 {
    eigh(covariance(pts).1).1[0]
}

/// 점들에 가장 잘 맞는 평면의 법선. 위(+y)를 향하게 한다
fn plane_normal(pts: &[DVec3]) -> DVec3 {
    let n = eigh(covariance(pts).1).1[2];
    if n.y > 0.0 { n } else { -n }
}

/// a 방향을 b 방향으로 돌리는 가장 작은 회전
fn rotation_between(a: DVec3, b: DVec3) -> DMat3 {
    let (a, b) = (a.normalize(), b.normalize());
    let (v, c) = (a.cross(b), a.dot(b));
    let s = v.length();
    if s < 1e-9 {
        return DMat3::IDENTITY;
    }
    let vx = DMat3::from_cols(DVec3::new(0.0, v.z, -v.y), DVec3::new(-v.z, 0.0, v.x), DVec3::new(v.y, -v.x, 0.0));
    DMat3::IDENTITY + vx + vx * vx * ((1.0 - c) / (s * s))
}

fn rotation_y(theta: f64) -> DMat3 {
    let (s, c) = theta.sin_cos();
    DMat3::from_cols(DVec3::new(c, 0.0, -s), DVec3::Y, DVec3::new(s, 0.0, c))
}

/// numpy.percentile의 기본(선형 보간)과 같은 값
fn percentile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let rank = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (rank - lo as f64)
}

fn round(v: f64, digits: i32) -> f64 {
    let k = 10f64.powi(digits);
    (v * k).round() / k
}

/// scipy.optimize.minimize(method="Nelder-Mead")와 같은 순서로 찾는다 (같은 시작점이면 같은 답이 나오게)
fn nelder_mead(f: impl Fn(&[f64; 3]) -> f64, x0: [f64; 3], xatol: f64, fatol: f64, maxiter: usize) -> ([f64; 3], f64) {
    const N: usize = 3;
    let (rho, chi, psi, sigma) = (1.0, 2.0, 0.5, 0.5);
    let mut sim = [x0; N + 1];
    for k in 0..N {
        sim[k + 1][k] = if x0[k] != 0.0 { 1.05 * x0[k] } else { 0.00025 };
    }
    let mut fsim: [f64; N + 1] = sim.map(|x| f(&x));
    let sort = |sim: &mut [[f64; 3]; N + 1], fsim: &mut [f64; N + 1]| {
        let mut ind = [0usize, 1, 2, 3];
        ind.sort_by(|&i, &j| fsim[i].total_cmp(&fsim[j]));
        *sim = ind.map(|i| sim[i]);
        *fsim = ind.map(|i| fsim[i]);
    };
    let comb = |a: f64, x: &[f64; 3], b: f64, y: &[f64; 3]| -> [f64; 3] { [0, 1, 2].map(|k| a * x[k] + b * y[k]) };
    sort(&mut sim, &mut fsim);
    let mut iterations = 1;
    while iterations < maxiter {
        let xspread = (1..=N).flat_map(|j| (0..N).map(move |k| (j, k))).map(|(j, k)| (sim[j][k] - sim[0][k]).abs()).fold(0.0, f64::max);
        let fspread = (1..=N).map(|j| (fsim[0] - fsim[j]).abs()).fold(0.0, f64::max);
        if xspread <= xatol && fspread <= fatol {
            break;
        }
        let mut xbar = [0f64; 3];
        for s in &sim[..N] {
            for k in 0..N {
                xbar[k] += s[k];
            }
        }
        let xbar = xbar.map(|v| v / N as f64);
        let xr = comb(1.0 + rho, &xbar, -rho, &sim[N]);
        let fxr = f(&xr);
        let mut shrink = false;
        if fxr < fsim[0] {
            let xe = comb(1.0 + rho * chi, &xbar, -rho * chi, &sim[N]);
            let fxe = f(&xe);
            if fxe < fxr {
                (sim[N], fsim[N]) = (xe, fxe);
            } else {
                (sim[N], fsim[N]) = (xr, fxr);
            }
        } else if fxr < fsim[N - 1] {
            (sim[N], fsim[N]) = (xr, fxr);
        } else if fxr < fsim[N] {
            let xc = comb(1.0 + psi * rho, &xbar, -psi * rho, &sim[N]);
            let fxc = f(&xc);
            if fxc <= fxr {
                (sim[N], fsim[N]) = (xc, fxc);
            } else {
                shrink = true;
            }
        } else {
            let xcc = comb(1.0 - psi, &xbar, psi, &sim[N]);
            let fxcc = f(&xcc);
            if fxcc < fsim[N] {
                (sim[N], fsim[N]) = (xcc, fxcc);
            } else {
                shrink = true;
            }
        }
        if shrink {
            for j in 1..=N {
                sim[j] = comb(1.0 - sigma, &sim[0], sigma, &sim[j]);
                fsim[j] = f(&sim[j]);
            }
        }
        iterations += 1;
        sort(&mut sim, &mut fsim);
    }
    (sim[0], fsim[0])
}

/// index.html의 치열궁 곡선 x = ±a·sin t, z = z0 + b·(cos t − 1)을 치아 목 위치에 맞춘다
fn fit_arch(necks: &[DVec3], start: [f64; 3]) -> ([f64; 3], f64) {
    let pts: Vec<(f64, f64)> = necks.iter().map(|p| (p.x.abs(), p.z)).collect();
    let step = std::f64::consts::PI * 0.6 / 799.0; // numpy.linspace(0, 0.6π, 800)와 같은 값
    let t: Vec<(f64, f64)> = (0..800).map(|i| (i as f64 * step).sin_cos()).collect();
    let cost = |p: &[f64; 3]| {
        let [a, b, z0] = *p;
        pts.iter()
            .map(|&(x, z)| t.iter().map(|&(s, c)| (x - a * s).powi(2) + (z - z0 - b * (c - 1.0)).powi(2)).fold(f64::MAX, f64::min))
            .sum::<f64>()
    };
    let (x, fun) = nelder_mead(cost, start, 0.01, 1e-4, 4000);
    (x.map(|v| round(v, 3)), (fun / pts.len() as f64).sqrt())
}

struct Tooth {
    from: u32,           // 분할 결과의 번호
    label: Option<u8>,   // AI 번호 (뿌리 조각이면 없음)
    voxels: Vec<usize>,  // CT 볼륨의 칸 번호
    pts: Vec<DVec3>,     // 칸의 모형 방향 좌표
    c: DVec3,
    y: DVec3,
    x: DVec3,
    z: DVec3,
}

impl Tooth {
    fn local(&self, p: DVec3) -> DVec3 {
        let d = p - self.c;
        DVec3::new(d.dot(self.x), d.dot(self.y), d.dot(self.z))
    }
    fn world(&self, l: DVec3) -> DVec3 {
        self.c + self.x * l.x + self.y * l.y + self.z * l.z
    }
}

/// 면을 평면 y=0으로 자른다. above면 y ≥ 0 쪽을 남긴다. 자른 자리의 꼭짓점은 이웃 삼각형이 함께 쓴다
fn clip(v: &[DVec3], faces: &[[u32; 3]], above: bool) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let side = |i: u32| if above { v[i as usize].y } else { -v[i as usize].y };
    let mut out_v: Vec<DVec3> = Vec::new();
    let mut kept: HashMap<u32, u32> = HashMap::new();
    let mut cut: HashMap<(u32, u32), u32> = HashMap::new();
    let mut out_f = Vec::new();
    for f in faces {
        let mut poly: Vec<u32> = Vec::with_capacity(4);
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let (sa, sb) = (side(a), side(b));
            if sa >= 0.0 {
                let n = out_v.len() as u32;
                poly.push(*kept.entry(a).or_insert_with(|| {
                    out_v.push(v[a as usize]);
                    n
                }));
            }
            if (sa >= 0.0) != (sb >= 0.0) {
                let key = (a.min(b), a.max(b));
                let n = out_v.len() as u32;
                poly.push(*cut.entry(key).or_insert_with(|| {
                    let (p, q) = (v[key.0 as usize], v[key.1 as usize]);
                    let t = p.y / (p.y - q.y);
                    let mut x = p + (q - p) * t;
                    x.y = 0.0;
                    out_v.push(x);
                    n
                }));
            }
        }
        for i in 1..poly.len().saturating_sub(1) {
            let t = [poly[0], poly[i], poly[i + 1]];
            if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                out_f.push(t);
            }
        }
    }
    (out_v, out_f)
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= c.len() {
                s.push(B64[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

/// {"p": 0.01mm 단위 int16 좌표, "i": uint16 삼각형}
fn pack(v: &[DVec3], f: &[[u32; 3]]) -> Result<String, String> {
    if v.len() > u16::MAX as usize {
        return Err(format!("꼭짓점이 {}개라 16비트 번호로 담을 수 없습니다", v.len()));
    }
    let p: Vec<u8> = v.iter().flat_map(|q| q.to_array()).flat_map(|x| ((x / UNIT_MM).round() as i16).to_le_bytes()).collect();
    let i: Vec<u8> = f.iter().flatten().flat_map(|&k| (k as u16).to_le_bytes()).collect();
    Ok(format!("{{\"p\":\"{}\",\"i\":\"{}\"}}", base64(&p), base64(&i)))
}

fn num(v: f64) -> String {
    serde_json::to_string(&v).unwrap_or_else(|_| "null".into())
}

struct Shape {
    fdi: u32,
    neck: DVec3,     // CT 쪽 모형 방향 좌표
    tip: DVec3,
    axes: DMat3,     // 열이 자기 좌표의 x, y, z축
    remnant: bool,
    from: u32,       // 분할 결과의 번호
    c: DVec3,        // 자기 좌표의 기준점(치아 칸의 가운데)
    offset: DVec3,   // 자기 좌표에서 치아 목
    whole: Vec<DVec3>,              // 자기 좌표의 치아 전체 꼭짓점 (뿌리 조각의 폭에 씀)
    crown: (Vec<DVec3>, Vec<[u32; 3]>),
    root: (Vec<DVec3>, Vec<[u32; 3]>),
    clamped: f64,
    length: f64,
    basis: String,
    surface: Vec<DVec3>,            // CT 쪽 모형 방향 좌표의 치아 표면 (스캔 정합과 잇몸 가르기에 씀)
    scan_note: String,
}

impl Shape {
    fn world(&self, l: DVec3) -> DVec3 {
        self.c + self.axes * (l + self.offset)
    }
    fn local(&self, w: DVec3) -> DVec3 {
        self.axes.transpose() * (w - self.c) - self.offset
    }
    /// ct-teeth.js의 치아 하나 (pos, axes 앞까지)와 표의 한 줄
    fn record(&self) -> Result<(String, String), String> {
        let widest = if self.remnant { &self.whole } else { &self.crown.0 }; // 뿌리 조각은 전체 폭을, 치아는 머리 폭을 자리 폭으로 쓴다
        let span = |g: fn(&DVec3) -> f64| widest.iter().map(g).fold(f64::MIN, f64::max) - widest.iter().map(g).fold(f64::MAX, f64::min);
        let (w, d) = (span(|p| p.x), span(|p| p.z));
        let mut record = format!(
            "\"w\":{},\"d\":{},\"h\":{},\"root\":{},\"crown\":{},\"rootMesh\":{}",
            num(round(w, 2)),
            num(round(d, 2)),
            num(round(self.clamped, 2)),
            num(round(self.length - self.clamped, 2)),
            pack(&self.crown.0, &self.crown.1)?,
            pack(&self.root.0, &self.root.1)?
        );
        if self.remnant {
            record += ",\"remnant\":true";
        }
        let note = if self.from != self.fdi { format!("  (분할 번호 {})", self.from) } else { String::new() };
        let line = format!(
            "{:3} {w:6.1} {d:6.1} {:7.1} {:8.1}   {:14} {}+{}{note}{}",
            self.fdi,
            self.clamped,
            self.length - self.clamped,
            self.basis,
            self.crown.1.len(),
            self.root.1.len(),
            self.scan_note
        );
        Ok((record, line))
    }
}

/// 치아 목 높이를 정하고, 자기 좌표의 메시로 만들어 머리와 뿌리로 자른다
fn shape(ct: &Volume, seg: &Seg, stats: &HashMap<u8, seg::LabelStats>, fdi: u32, t: &Tooth, o: &Options) -> Result<Shape, String> {
    let local: Vec<DVec3> = t.pts.iter().map(|p| t.local(*p)).collect();
    let mesh = match t.label {
        None => seg::hu_surface(ct, &t.voxels, seg::REMNANT_HU, o.tris),
        Some(l) => seg::label_surface(ct, &seg.labels, l, &stats[&l], 0.5, o.tris),
    }
    .ok_or(format!("{fdi}번: 표면을 뽑지 못했습니다"))?;
    let surface: Vec<DVec3> = mesh.pos.iter().map(|p| three(p.x as f64, p.y as f64, p.z as f64)).collect();
    let verts: Vec<DVec3> = surface.iter().map(|p| t.local(*p)).collect();
    let tip = verts.iter().map(|v| v.y).fold(f64::MIN, f64::max);
    let apex = verts.iter().map(|v| v.y).fold(f64::MAX, f64::min);
    let length = tip - apex;
    let remnant = t.label.is_none();

    let (clamped, basis, offset);
    if remnant {
        // 머리가 없으므로 치아 목을 정할 근거가 없다. 위쪽 35%(1.5~3mm)를 잇몸 위로 나온 부분으로 본다
        clamped = (0.35 * length).clamp(1.5, 3.0);
        basis = "뿌리 조각".to_string();
        let m = mean(local.iter().copied());
        offset = DVec3::new(m.x, tip - clamped, m.z);
    } else {
        // 법랑질이 끝나는 높이를 치아 목으로 본다. 법랑질이 거의 없으면 표준 비율(길이의 42%)을 쓴다
        let mut enamel_y: Vec<f64> = t.voxels.iter().zip(&local).filter(|(&i, _)| ct.data[i] > o.enamel).map(|(_, l)| l.y).collect();
        let (crown_h, mut b) = if enamel_y.len() >= 200 {
            (tip - percentile(&mut enamel_y, 5.0), "법랑질".to_string())
        } else {
            (0.42 * length, "표준 비율".to_string())
        };
        clamped = crown_h.clamp(0.30 * length, 0.55 * length);
        if clamped != crown_h {
            b += "(범위 제한)";
        }
        basis = b;
        let neck = tip - clamped;
        let m = mean(local.iter().copied().filter(|l| (l.y - neck).abs() < 0.6));
        offset = DVec3::new(m.x, neck, m.z);
    }

    let v: Vec<DVec3> = verts.iter().map(|p| *p - offset).collect();
    let mut faces: Vec<[u32; 3]> = mesh.idx.chunks_exact(3).map(|f| [f[0], f[1], f[2]]).collect();
    let signed: f64 = faces.iter().map(|f| v[f[0] as usize].dot(v[f[1] as usize].cross(v[f[2] as usize]))).sum::<f64>() / 6.0;
    if signed < 0.0 {
        // 면이 바깥을 향하게 맞춘다
        faces.iter_mut().for_each(|f| f.reverse());
    }
    let crown = clip(&v, &faces, true);
    let root = clip(&v, &faces, false);
    Ok(Shape {
        fdi,
        neck: t.world(offset),
        tip: t.world(DVec3::new(offset.x, tip, offset.z)),
        axes: DMat3::from_cols(t.x, t.y, t.z),
        remnant,
        from: t.from,
        c: t.c,
        offset,
        whole: v,
        crown,
        root,
        clamped,
        length,
        basis,
        surface,
        scan_note: String::new(),
    })
}

/// 모형 좌표로 옮기는 변환 (place_in_model이 정한 것)
struct ModelXf {
    rot: DMat3,
    shift: DVec3,
    r_low: DMat3,
    low_shift: DVec3,
}

impl ModelXf {
    fn apply(&self, p: DVec3, lower: bool) -> DVec3 {
        let m = self.rot * p + self.shift;
        if lower { self.r_low * (m - PIVOT) + PIVOT + self.low_shift } else { m }
    }
}

struct Placed {
    pos: DVec3,
    axes: DMat3,
}

/// CT 좌표의 치아 자리를 모형 좌표로 옮기고 위아래 치열궁을 맞춘다.
///   1) 위 치아 끝들이 이루는 교합면을 수평으로, 정중선을 앞(+z)으로 돌린다
///   2) 위 가운데 앞니 사이를 x = 0, 그 목을 z = FRONT_NECK_Z에 두고, 위 치아 끝의 평균 높이를 UPPER_TIP_Y에 둔다
///   3) CT를 찍을 때 입을 조금 벌리고 있었으므로, 아래 치아는 모형의 턱 회전 중심(PIVOT)을 기준으로
///      교합면이 수평이 되게 돌린 뒤 치아 끝의 평균 높이를 LOWER_TIP_Y에 둔다
fn place_in_model(shapes: &[Shape], log: Log) -> Result<(Vec<Placed>, [[f64; 3]; 2], ModelXf), String> {
    let upper: Vec<&Shape> = shapes.iter().filter(|s| s.fdi < 30).collect();
    let up_tips: Vec<DVec3> = upper.iter().filter(|s| !s.remnant).map(|s| s.tip).collect();
    let low_tips: Vec<DVec3> = shapes.iter().filter(|s| s.fdi > 30 && !s.remnant).map(|s| s.tip).collect();
    if up_tips.len() < 3 || low_tips.len() < 3 {
        return Err("위아래 치아가 3개 이상씩 있어야 자리를 맞출 수 있습니다".into());
    }
    let r1 = rotation_between(plane_normal(&up_tips), Y);
    let side_back = |quad: u32| -> Result<DVec3, String> {
        upper
            .iter()
            .filter(|s| s.fdi / 10 == quad && !s.remnant)
            .max_by_key(|s| s.fdi % 10)
            .map(|s| r1 * s.neck)
            .ok_or(format!("{quad}사분면에 치아가 없어 정중선을 정할 수 없습니다"))
    };
    let mut incisors: Vec<&Shape> = upper.iter().copied().filter(|s| s.fdi == 11 || s.fdi == 21).collect();
    if incisors.is_empty() {
        let mut by_pos = upper.clone();
        by_pos.sort_by_key(|s| s.fdi % 10);
        incisors = by_pos.into_iter().take(2).collect();
    }
    let front = mean(incisors.iter().map(|s| r1 * s.neck));
    let back = (side_back(1)? + side_back(2)?) / 2.0;
    let d = front - back;
    let rot = rotation_y(-d.x.atan2(d.z)) * r1;

    let front = rot * mean(incisors.iter().map(|s| s.neck));
    let shift = DVec3::new(-front.x, UPPER_TIP_Y - mean(up_tips.iter().map(|p| rot * *p)).y, FRONT_NECK_Z - front.z);
    let to_model = |p: DVec3| rot * p + shift;

    // 아래턱: 위턱 기준으로 옮긴 뒤, 턱 회전 중심을 기준으로 다물게 돌린다
    let low_tips: Vec<DVec3> = low_tips.into_iter().map(to_model).collect();
    let n_low = plane_normal(&low_tips);
    let r_low = rotation_between(n_low, Y);
    let closed_y = mean(low_tips.iter().map(|p| r_low * (*p - PIVOT) + PIVOT)).y;
    let low_shift = DVec3::new(0.0, LOWER_TIP_Y - closed_y, 0.0);
    log(format!("자리 맞추기: CT에서 아래턱은 위턱보다 {:.1}° 벌어져 있어 모형의 턱 회전 중심으로 다물게 돌렸습니다.", n_low.y.clamp(-1.0, 1.0).acos().to_degrees()));

    let mut placed = Vec::new();
    let mut necks: [Vec<DVec3>; 2] = [Vec::new(), Vec::new()];
    for s in shapes {
        let mut axes = rot * s.axes;
        let mut neck = to_model(s.neck);
        if s.fdi > 30 {
            axes = r_low * axes;
            neck = r_low * (neck - PIVOT) + PIVOT + low_shift;
        }
        necks[(s.fdi > 30) as usize].push(neck);
        placed.push(Placed { pos: neck, axes });
    }
    let mut arch = [[0f64; 3]; 2];
    for (j, (name, start)) in [("위", [28.0, 46.0, 0.0]), ("아래", [26.0, 43.0, -3.0])].iter().enumerate() {
        let (p, rms) = fit_arch(&necks[j], *start);
        arch[j] = p;
        log(format!("{name}턱 치열궁 a={:.1} b={:.1} z0={:.1}  (치아 목과의 거리 RMS {rms:.2}mm)", p[0], p[1], p[2]));
    }
    Ok((placed, arch, ModelXf { rot, shift, r_low, low_shift }))
}

/// 치아를 모아 ct-teeth.js를 만든다. 뿌리 조각 후보는 번호(fdi)를 붙인 것만 쓴다
pub fn write_model_js(ct: &Volume, seg: &Seg, cands: &[Candidate], o: &Options, out: &Path, log: Log) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let stats = seg::label_stats(&seg.labels, ct.dims);
    let sp = ct.spacing.map(|v| v as f64);
    let org = ct.origin.map(|v| v as f64);
    let voxel_mm3 = sp.iter().product::<f64>();
    let [nx, ny, _] = ct.dims;
    let point = |i: usize| three(org[0] + (i % nx) as f64 * sp[0], org[1] + ((i / nx) % ny) as f64 * sp[1], org[2] + (i / (nx * ny)) as f64 * sp[2]);

    // 1) 치아마다 칸 좌표와 긴 축을 모은다
    let mut teeth: BTreeMap<u32, Tooth> = BTreeMap::new();
    for (&l, name) in &seg.names {
        let Some(f) = seg::fdi_of(name) else { continue };
        if name.contains("pulp") || name.contains("canal") || !(11..=48).contains(&f) {
            continue;
        }
        let Some(s) = stats.get(&l) else { continue };
        let volume = s.count as f64 * voxel_mm3;
        if volume < MIN_VOLUME_MM3 {
            log(format!("{f}번: {volume:.0}mm3, 너무 작아 뺌"));
            continue;
        }
        let fdi = *o.remap.get(&f).unwrap_or(&f);
        let mut voxels = Vec::with_capacity(s.count);
        for z in s.lo[2]..=s.hi[2] {
            for y in s.lo[1]..=s.hi[1] {
                let row = (z * ny + y) * nx;
                for x in s.lo[0]..=s.hi[0] {
                    if seg.labels[row + x] == l {
                        voxels.push(row + x);
                    }
                }
            }
        }
        let pts: Vec<DVec3> = voxels.par_iter().map(|&i| point(i)).collect();
        let c = mean(pts.iter().copied());
        let mut axis = principal_axis(&pts);
        if (axis.y > 0.0) == (fdi < 30) {
            // 위 치아는 머리가 아래로, 아래 치아는 머리가 위로 간다
            axis = -axis;
        }
        teeth.insert(fdi, Tooth { from: f, label: Some(l), voxels, pts, c, y: axis, x: DVec3::ZERO, z: DVec3::ZERO });
    }

    // 1-2) 뿌리만 남은 치아. 짧고 뭉툭해서 긴 축을 모양으로 잡을 수 없으므로 그 턱 교합면에 수직인 방향을 쓴다.
    //       모형은 이 축으로 그 자리의 임플란트와 보철을 세우므로, 이웃 치아의 기울기를 빌리면 임플란트가 함께 기운다
    let occlusal_axis = |upper: bool| -> Option<DVec3> {
        let tips: Vec<DVec3> = teeth
            .iter()
            .filter(|(f, _)| (**f < 30) == upper)
            .map(|(_, t)| *t.pts.iter().max_by(|a, b| (**a - t.c).dot(t.y).total_cmp(&(**b - t.c).dot(t.y))).unwrap())
            .collect();
        (tips.len() >= 3).then(|| {
            let n = plane_normal(&tips);
            if upper { -n } else { n }
        })
    };
    let axes_by_jaw = [occlusal_axis(false), occlusal_axis(true)];
    for c in cands {
        let Some(fdi) = c.fdi else { continue };
        if teeth.contains_key(&fdi) {
            log(format!("{fdi}번: 자동 분할한 치아가 이미 있어 뿌리 조각 {}번을 쓰지 않음", c.id));
            continue;
        }
        let Some(axis) = axes_by_jaw[(fdi < 30) as usize] else {
            log(format!("{fdi}번: 그 턱에 치아가 적어 교합면을 정할 수 없어 뿌리 조각 {}번을 쓰지 않음", c.id));
            continue;
        };
        let pts: Vec<DVec3> = c.voxels.iter().map(|&i| point(i)).collect();
        let centre = mean(pts.iter().copied());
        teeth.insert(fdi, Tooth { from: fdi, label: None, voxels: c.voxels.clone(), pts, c: centre, y: axis, x: DVec3::ZERO, z: DVec3::ZERO });
    }

    // 2) 치열궁의 바깥 방향: 이웃 치아의 중심을 이은 선에 수직이고, 치열 중심에서 멀어지는 쪽
    for order in [UPPER, LOWER] {
        let seq: Vec<u32> = order.iter().copied().filter(|f| teeth.contains_key(f)).collect();
        if seq.is_empty() {
            continue;
        }
        let centre = mean(seq.iter().map(|f| teeth[f].c));
        for (i, f) in seq.iter().enumerate() {
            let a = teeth[&seq[i.saturating_sub(1)]].c;
            let b = teeth[&seq[(i + 1).min(seq.len() - 1)]].c;
            let mut normal = DVec3::new(b.z - a.z, 0.0, -(b.x - a.x));
            let t = teeth.get_mut(f).unwrap();
            if normal.dot(t.c - centre) < 0.0 {
                normal = -normal;
            }
            t.z = (normal - normal.dot(t.y) * t.y).normalize();
            t.x = t.y.cross(t.z);
        }
    }
    let missing: Vec<u32> = teeth.iter().filter(|(_, t)| t.z == DVec3::ZERO).map(|(f, _)| *f).collect();
    for f in missing {
        log(format!("{f}번: 모형의 치열에 없는 번호라 뺌"));
        teeth.remove(&f);
    }

    // 3) 치아마다 모양 (서로 독립이라 함께 돌린다)
    let list: Vec<(&u32, &Tooth)> = teeth.iter().collect();
    let mut shapes: Vec<Shape> = list.par_iter().map(|(f, t)| shape(ct, seg, &stats, **f, t, o)).collect::<Result<_, _>>()?;

    // 3-2) 구강 스캔: 치아 머리를 스캔 표면에 붙이고, 잇몸을 가를 치아 면을 모은다
    let mut scans = None;
    if let Some(dir) = &o.scan {
        let teeth_pts: Vec<(u32, Vec<DVec3>)> = shapes.iter().filter(|s| !s.remnant).map(|s| (s.fdi, s.surface.clone())).collect();
        let [(up, fit_u), (lo, fit_l)] = scan::load_and_register(dir, &teeth_pts, log)?; // 스캔은 이 함수 안의 좌표(lps→three)로 옮겨져 온다
        let grids = [scan::Grid::new(up.pos.clone(), 0.5), scan::Grid::new(lo.pos.clone(), 0.5)];
        let normals = [up.vertex_normals(), lo.vertex_normals()];
        let fits: HashMap<u32, f64> = fit_u.per_tooth.iter().chain(&fit_l.per_tooth).copied().collect();
        shapes.par_iter_mut().for_each(|s| {
            let j = (s.fdi > 30) as usize;
            let fit = fits.get(&s.fdi).copied();
            if s.remnant {
                s.scan_note = "  스캔: 뿌리 조각이라 그대로".into();
            } else if fit.map_or(true, |f| f > scan::FIT_LIMIT_MM) {
                s.scan_note = format!("  스캔: 정합 거리 {}mm라 그대로", fit.map_or("-".into(), |f| format!("{f:.3}")));
            } else {
                let (v, f, share) = scan::snap_crown(&s.crown.0, &s.crown.1, &grids[j], &normals[j], |l| s.world(l), |w| s.local(w), |n| s.axes * n);
                s.crown = (v, f);
                s.scan_note = format!("  스캔: 머리 점의 {:.0}%를 스캔에 붙임", 100.0 * share);
            }
        });
        scans = Some(([up, lo], fits));
    }
    log("번호  폭    두께  머리높이 뿌리길이  목 높이 기준   삼각형(머리+뿌리)".into());
    let mut records = Vec::new();
    for s in &shapes {
        let (record, line) = s.record()?;
        log(line);
        records.push(record);
    }

    // 4) 실제 자리(치아 목의 위치와 축)를 모형 좌표로
    let (placed, arch, xf) = place_in_model(&shapes, log)?;

    // 5) 잇몸: 스캔에서 치아 면(스캔에 붙인 머리 포함)을 뺀 나머지를 모형 좌표로
    let mut gum_json = String::new();
    if let Some((surfaces, fits)) = &scans {
        let mut parts = Vec::new();
        for (j, jaw) in ["U", "L"].iter().enumerate() {
            let teeth: Vec<(Vec<DVec3>, bool)> = shapes
                .iter()
                .filter(|s| (s.fdi > 30) == (j == 1))
                .flat_map(|s| {
                    let poor = !s.remnant && fits.get(&s.fdi).map_or(true, |f| *f > scan::FIT_LIMIT_MM);
                    let mut v = vec![(s.surface.clone(), poor)];
                    if s.scan_note.contains("붙임") {
                        v.push((s.crown.0.iter().map(|l| s.world(*l)).collect(), false));
                    }
                    v
                })
                .collect();
            let g = scan::gum(&surfaces[j], &teeth);
            let m = ctproc::simplify(
                &ctproc::Mesh { pos: g.pos.iter().map(|p| p.as_vec3()).collect(), idx: g.faces.iter().flatten().copied().collect() },
                30_000,
            );
            let pos: Vec<DVec3> = m.pos.iter().map(|p| xf.apply(p.as_dvec3(), j == 1)).collect();
            let faces: Vec<[u32; 3]> = m.idx.chunks_exact(3).map(|f| [f[0], f[1], f[2]]).collect();
            log(format!("{}턱 잇몸: 삼각형 {}개", if j == 0 { "위" } else { "아래" }, faces.len()));
            parts.push(format!("\"{jaw}\":{}", pack(&pos, &faces)?));
        }
        gum_json = format!(",\"gum\":{{{}}}", parts.join(","));
    }

    let mut js = String::new();
    js += "// CT에서 뽑은 치아별 실제 모양과 자리. tools/ct3d/teeth_for_model.py(또는 tools/ct3d-rs의 --model-js)가 만든 파일이라 직접 고치지 않는다.\n";
    js += "// 모양은 치아마다 자기 좌표(원점: 치아 목, +y: 머리 쪽, +z: 볼·입술 쪽)이고 좌표 단위는 unit(mm)이다.\n";
    js += "// pos와 axes는 모형 좌표에서 그 자기 좌표의 원점과 x, y, z축이다. arch는 치아 목에 맞춘 치열궁 [a, b, z0]이다.\n";
    if scans.is_some() {
        js += "// gum은 구강 스캔에서 치아를 뺀 잇몸이고 모형 좌표다. 치아 머리는 구강 스캔으로 다듬었다.\n";
    }
    let arr = |v: &[f64]| v.iter().map(|x| num(*x)).collect::<Vec<_>>().join(",");
    let _ = write!(js, "window.CT_TEETH = {{\"unit\":{},\"arch\":{{\"U\":[{}],\"L\":[{}]}},\"teeth\":{{", num(UNIT_MM), arr(&arch[0]), arr(&arch[1]));
    for (k, ((s, p), record)) in shapes.iter().zip(&placed).zip(&records).enumerate() {
        let pos = p.pos.to_array().map(|v| round(v, 3));
        let axes: Vec<f64> = (0..3).flat_map(|c| p.axes.col(c).to_array()).map(|v| round(v, 4)).collect();
        let _ = write!(js, "{}\"{}\":{{{},\"pos\":[{}],\"axes\":[{}]}}", if k > 0 { "," } else { "" }, s.fdi, record, arr(&pos), arr(&axes));
    }
    js += "}";
    js += &gum_json;
    js += "};\n";
    std::fs::write(out, &js).map_err(|e| format!("{}에 저장하지 못했습니다: {e}", out.display()))?;
    let n_rem = shapes.iter().filter(|s| s.remnant).count();
    log(format!(
        "치아 {}개, 뿌리만 남은 치아 {n_rem}개, 저장: {} ({:.2} MB, {:.1}초)",
        shapes.len() - n_rem,
        out.display(),
        js.len() as f64 / 1e6,
        t0.elapsed().as_secs_f32()
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigh_diagonalises() {
        let m = DMat3::from_cols_array_2d(&[[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]]);
        let (vals, vecs) = eigh(m);
        assert!(vals[0] >= vals[1] && vals[1] >= vals[2]);
        for k in 0..3 {
            assert!((m * vecs[k] - vecs[k] * vals[k]).length() < 1e-9);
        }
    }

    #[test]
    fn base64_matches_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn rotation_between_maps_a_to_b() {
        let a = DVec3::new(0.3, 0.9, -0.2).normalize();
        assert!((rotation_between(a, Y) * a - Y).length() < 1e-12);
    }

    #[test]
    fn clip_halves_keep_area() {
        let v = vec![DVec3::new(0.0, -1.0, 0.0), DVec3::new(1.0, 1.0, 0.0), DVec3::new(-1.0, 1.0, 0.0)];
        let f = vec![[0, 1, 2]];
        let area = |v: &[DVec3], f: &[[u32; 3]]| f.iter().map(|t| (v[t[1] as usize] - v[t[0] as usize]).cross(v[t[2] as usize] - v[t[0] as usize]).length() / 2.0).sum::<f64>();
        let (av, af) = clip(&v, &f, true);
        let (bv, bf) = clip(&v, &f, false);
        assert!((area(&av, &af) + area(&bv, &bf) - area(&v, &f)).abs() < 1e-12);
        assert_eq!(af.len(), 2);
        assert_eq!(bf.len(), 1);
    }
}
