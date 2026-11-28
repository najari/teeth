//! DICOM 폴더를 읽어 뼈와 치아의 표면 메시를 만든다 (tools/ct3d/volume_to_mesh.py를 옮긴 것).
//!
//! 순서
//!   1) DICOM 슬라이스를 모두 풀어 볼륨 하나로 (dicom-rs, 순수 Rust)
//!   2) 2배로 줄이고(0.4mm) 가우스로 살짝 흐린다
//!   3) 치아: 치아 임계값을 넘는 덩어리 가운데 법랑질 밝기 화소가 충분한 것
//!   4) 뼈: 치아 자리를 뺀 뒤 뼈 임계값으로, 큰 덩어리 6개까지만
//!   5) 치아: 원본 해상도(0.2mm)에서 치아 둘레만 잘라 다시 뽑는다
//!   6) 등가면(Surface Nets) → 매끄럽게(Taubin) → 삼각형 줄이기(meshopt) → 법선

use crate::scene::{kind_of, Part};
use dicom_dictionary_std::tags;
use dicom_object::open_file;
use dicom_pixeldata::PixelDecoder;
use fast_surface_nets::ndshape::{RuntimeShape, Shape};
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
use glam::Vec3;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

pub const AIR: f32 = -1000.0;

pub struct Params {
    pub bone: f32,
    pub teeth: f32,
    pub enamel: f32,
    pub bone_tris: usize,
    pub teeth_tris: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self { bone: 1000.0, teeth: 2800.0, enamel: 3300.0, bone_tris: 220_000, teeth_tris: 180_000 }
    }
}

/// x가 가장 빨리 변하는 3차원 볼륨. 좌표는 환자 좌표(LPS, mm)
#[derive(Clone)]
pub struct Volume {
    pub dims: [usize; 3],
    pub spacing: [f32; 3],
    pub origin: [f32; 3],
    pub data: Vec<f32>,
}

impl Volume {
    pub(crate) fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }
}

/// 진행 상황을 알려 주는 함수
pub type Log<'a> = &'a (dyn Fn(String) + Sync);

pub fn read_dicom_dir(dir: &Path) -> Result<Volume, String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("폴더를 읽지 못했습니다: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    // 확장자가 없는 DICOM도 있으므로, 열어 보고 영상이 있는 것만 쓴다
    let mut slices: Vec<(f32, [f32; 2], [f32; 2], usize, usize, Vec<f32>)> = files
        .par_iter()
        .filter_map(|p| {
            let obj = open_file(p).ok()?;
            let ipp = obj.element(tags::IMAGE_POSITION_PATIENT).ok()?.to_multi_float32().ok()?;
            let ps = obj.element(tags::PIXEL_SPACING).ok()?.to_multi_float32().ok()?;
            let rows = obj.element(tags::ROWS).ok()?.to_int::<usize>().ok()?;
            let cols = obj.element(tags::COLUMNS).ok()?.to_int::<usize>().ok()?;
            let pix = obj.decode_pixel_data().ok()?;
            // 기본 변환은 Rescale Slope/Intercept를 적용한다
            let arr = pix.to_ndarray::<f32>().ok()?;
            Some((ipp[2], [ipp[0], ipp[1]], [ps[1], ps[0]], cols, rows, arr.iter().copied().collect()))
        })
        .collect();
    if slices.len() < 2 {
        return Err("영상이 있는 DICOM 파일이 두 장 이상 없습니다".into());
    }
    slices.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (nx, ny) = (slices[0].3, slices[0].4);
    if slices.iter().any(|s| s.3 != nx || s.4 != ny) {
        return Err("슬라이스 크기가 서로 다릅니다".into());
    }
    let nz = slices.len();
    let sz = (slices[nz - 1].0 - slices[0].0) / (nz - 1) as f32;
    let mut data = Vec::with_capacity(nx * ny * nz);
    for s in &slices {
        data.extend_from_slice(&s.5);
    }
    Ok(Volume { dims: [nx, ny, nz], spacing: [slices[0].2[0], slices[0].2[1], sz], origin: [slices[0].1[0], slices[0].1[1], slices[0].0], data })
}

/// 2×2×2 칸의 평균으로 줄인다 (SimpleITK BinShrink와 같음)
pub(crate) fn shrink2(v: &Volume) -> Volume {
    let d = [v.dims[0] / 2, v.dims[1] / 2, v.dims[2] / 2];
    let mut out = vec![0f32; d[0] * d[1] * d[2]];
    out.par_chunks_mut(d[0] * d[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let mut s = 0.0;
                for (dz, dy, dx) in (0..8).map(|k| (k >> 2, (k >> 1) & 1, k & 1)) {
                    s += v.data[v.idx(2 * x + dx, 2 * y + dy, 2 * z + dz)];
                }
                plane[y * d[0] + x] = s / 8.0;
            }
        }
    });
    Volume {
        dims: d,
        spacing: v.spacing.map(|s| s * 2.0),
        origin: [0, 1, 2].map(|a| v.origin[a] + 0.5 * v.spacing[a]),
        data: out,
    }
}

/// 축마다 따로 하는 가우스 흐리기. sigma는 mm
pub(crate) fn gaussian(v: &mut Volume, sigma_mm: f32) {
    let [nx, ny, nz] = v.dims;
    for axis in 0..3 {
        let sigma = sigma_mm / v.spacing[axis];
        let r = (3.0 * sigma).ceil() as isize;
        let w: Vec<f32> = (-r..=r).map(|k| (-(k * k) as f32 / (2.0 * sigma * sigma)).exp()).collect();
        let sum: f32 = w.iter().sum();
        let w: Vec<f32> = w.into_iter().map(|x| x / sum).collect();
        let n = v.dims[axis] as isize;
        let src = std::mem::take(&mut v.data);
        let mut out = vec![0f32; src.len()];
        out.par_chunks_mut(nx * ny).enumerate().for_each(|(z, plane)| {
            for y in 0..ny {
                for x in 0..nx {
                    let pos = [x as isize, y as isize, z as isize];
                    let mut acc = 0.0;
                    for (k, wk) in (-r..=r).zip(&w) {
                        let mut p = pos;
                        p[axis] = (p[axis] + k).clamp(0, n - 1);
                        acc += wk * src[(p[2] as usize * ny + p[1] as usize) * nx + p[0] as usize];
                    }
                    plane[y * nx + x] = acc;
                }
            }
        });
        v.data = out;
    }
    debug_assert_eq!(v.data.len(), nx * ny * nz);
}

/// 면으로 이웃한(6방향) 칸끼리 이은 덩어리 번호. 0은 바깥. 두 번째 값은 덩어리별 칸 수(0번은 비움)
pub(crate) fn label(mask: &[bool], dims: [usize; 3]) -> (Vec<u32>, Vec<usize>) {
    let [nx, ny, nz] = dims;
    let mut lab = vec![0u32; mask.len()];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || lab[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut count = 0;
        lab[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            count += 1;
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            let mut visit = |j: usize| {
                if mask[j] && lab[j] == 0 {
                    lab[j] = id;
                    stack.push(j);
                }
            };
            if x > 0 { visit(i - 1) }
            if x + 1 < nx { visit(i + 1) }
            if y > 0 { visit(i - nx) }
            if y + 1 < ny { visit(i + nx) }
            if z > 0 { visit(i - nx * ny) }
            if z + 1 < nz { visit(i + nx * ny) }
        }
        sizes.push(count);
    }
    (lab, sizes)
}

/// 반지름 r칸의 공 모양으로 넓힌다 (SimpleITK BinaryDilate의 기본 모양)
pub(crate) fn dilate(mask: &[bool], dims: [usize; 3], r: isize) -> Vec<bool> {
    dilate_with(mask, dims, &ball(r))
}

/// 반지름 r칸의 마름모 모양으로 넓힌다 (scipy binary_dilation을 iterations=r로 돌린 것과 같음)
pub(crate) fn dilate_l1(mask: &[bool], dims: [usize; 3], r: isize) -> Vec<bool> {
    dilate_with(mask, dims, &diamond(r))
}

type Offsets = Vec<(isize, isize, isize)>;

fn cube(r: isize) -> impl Iterator<Item = (isize, isize, isize)> {
    (-r..=r).flat_map(move |z| (-r..=r).flat_map(move |y| (-r..=r).map(move |x| (x, y, z))))
}

fn ball(r: isize) -> Offsets {
    cube(r).filter(|(x, y, z)| x * x + y * y + z * z <= r * r).collect()
}

fn diamond(r: isize) -> Offsets {
    cube(r).filter(|(x, y, z)| x.abs() + y.abs() + z.abs() <= r).collect()
}

/// 마름모 모양으로 깎는다. 볼륨 바깥은 비어 있다고 본다 (scipy binary_erosion의 기본)
pub(crate) fn erode_l1(mask: &[bool], dims: [usize; 3], r: isize) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let offsets = diamond(r);
    let mut out = vec![false; mask.len()];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        if !mask[i] {
            return;
        }
        let (x, y, z) = ((i % nx) as isize, ((i / nx) % ny) as isize, (i / (nx * ny)) as isize);
        *o = offsets.iter().all(|(dx, dy, dz)| {
            let (a, b, c) = (x + dx, y + dy, z + dz);
            a >= 0 && b >= 0 && c >= 0 && (a as usize) < nx && (b as usize) < ny && (c as usize) < nz
                && mask[(c as usize * ny + b as usize) * nx + a as usize]
        });
    });
    out
}

/// 바깥과 이어지지 않은 빈 곳을 메운다. dims의 축 수만큼(2차원이면 nz = 1) 면 이웃으로 본다
pub(crate) fn fill_holes(mask: &[bool], dims: [usize; 3]) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let mut outside = vec![false; mask.len()];
    let mut stack: Vec<usize> = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let border = x == 0 || y == 0 || x + 1 == nx || y + 1 == ny || (nz > 1 && (z == 0 || z + 1 == nz));
                let i = (z * ny + y) * nx + x;
                if border && !mask[i] && !outside[i] {
                    outside[i] = true;
                    stack.push(i);
                }
            }
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
        let mut visit = |j: usize| {
            if !mask[j] && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        };
        if x > 0 { visit(i - 1) }
        if x + 1 < nx { visit(i + 1) }
        if y > 0 { visit(i - nx) }
        if y + 1 < ny { visit(i + nx) }
        if z > 0 { visit(i - nx * ny) }
        if z + 1 < nz { visit(i + nx * ny) }
    }
    outside.into_iter().map(|o| !o).collect()
}

fn dilate_with(mask: &[bool], dims: [usize; 3], offsets: &Offsets) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let mut out = vec![false; mask.len()];
    for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
        let (x, y, z) = ((i % nx) as isize, ((i / nx) % ny) as isize, (i / (nx * ny)) as isize);
        for (dx, dy, dz) in offsets {
            let (a, b, c) = (x + dx, y + dy, z + dz);
            if a >= 0 && b >= 0 && c >= 0 && (a as usize) < nx && (b as usize) < ny && (c as usize) < nz {
                out[(c as usize * ny + b as usize) * nx + a as usize] = true;
            }
        }
    }
    out
}

pub(crate) struct Mesh {
    pub(crate) pos: Vec<Vec3>, // 환자 좌표(mm)
    pub(crate) idx: Vec<u32>,
}

/// 값이 level인 면을 Surface Nets로 뽑는다. 테두리 한 칸을 바깥으로 채워 닫힌 면이 되게 한다.
pub(crate) fn isosurface(v: &Volume, level: f32) -> Mesh {
    let [nx, ny, nz] = v.dims;
    let shape = RuntimeShape::<u32, 3>::new([nx as u32 + 2, ny as u32 + 2, nz as u32 + 2]);
    let mut sdf = vec![1.0f32; shape.size() as usize];
    sdf.par_chunks_mut(((nx + 2) * (ny + 2)) as usize).enumerate().for_each(|(z, plane)| {
        if z == 0 || z > nz {
            return;
        }
        for y in 1..=ny {
            for x in 1..=nx {
                // 안쪽(값이 level보다 큼)이 음수. 크기는 보간에만 쓰인다
                plane[y * (nx + 2) + x] = (level - v.data[v.idx(x - 1, y - 1, z - 1)]) / 1000.0;
            }
        }
    });
    let mut buf = SurfaceNetsBuffer::default();
    surface_nets(&sdf, &shape, [0; 3], [nx as u32 + 1, ny as u32 + 1, nz as u32 + 1], &mut buf);
    let pos = buf
        .positions
        .iter()
        .map(|p| Vec3::new(
            v.origin[0] + (p[0] - 1.0) * v.spacing[0],
            v.origin[1] + (p[1] - 1.0) * v.spacing[1],
            v.origin[2] + (p[2] - 1.0) * v.spacing[2],
        ))
        .collect();
    Mesh { pos, idx: buf.indices }
}

/// Taubin 매끄럽게: 이웃 평균 쪽으로 당겼다가(λ) 반대로 밀어(μ) 부피가 줄지 않게 한다
pub(crate) fn taubin(m: &mut Mesh, pairs: usize) {
    let n = m.pos.len();
    let mut nbr: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in m.idx.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            nbr[a as usize].push(b);
            nbr[b as usize].push(a);
        }
    }
    nbr.par_iter_mut().for_each(|l| {
        l.sort_unstable();
        l.dedup();
    });
    for _ in 0..pairs {
        for f in [0.5f32, -0.53] {
            let old = m.pos.clone();
            m.pos.par_iter_mut().enumerate().for_each(|(i, p)| {
                let l = &nbr[i];
                if !l.is_empty() {
                    let avg = l.iter().map(|&j| old[j as usize]).sum::<Vec3>() / l.len() as f32;
                    *p += f * (avg - *p);
                }
            });
        }
    }
}

/// 삼각형 수를 target 가까이로 줄이고, 쓰지 않는 꼭짓점을 버린다
pub(crate) fn simplify(m: &Mesh, target: usize) -> Mesh {
    if m.idx.len() / 3 <= target {
        return Mesh { pos: m.pos.clone(), idx: m.idx.clone() };
    }
    let flat: Vec<f32> = m.pos.iter().flat_map(|p| p.to_array()).collect();
    let adapter = meshopt::VertexDataAdapter::new(bytemuck::cast_slice(&flat), 12, 0).expect("vertex data");
    let idx = meshopt::simplify(&m.idx, &adapter, target * 3, 0.02, meshopt::SimplifyOptions::None, None);
    let mut remap = vec![u32::MAX; m.pos.len()];
    let mut pos = Vec::new();
    let idx = idx
        .iter()
        .map(|&i| {
            if remap[i as usize] == u32::MAX {
                remap[i as usize] = pos.len() as u32;
                pos.push(m.pos[i as usize]);
            }
            remap[i as usize]
        })
        .collect();
    Mesh { pos, idx }
}

fn normals(pos: &[Vec3], idx: &[u32]) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; pos.len()];
    for t in idx.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let f = (pos[b] - pos[a]).cross(pos[c] - pos[a]);
        n[a] += f;
        n[b] += f;
        n[c] += f;
    }
    n.into_iter().map(|v| v.normalize_or_zero()).collect()
}

/// 환자 좌표(LPS)를 뷰어 좌표(x 환자 왼쪽, y 위, z 얼굴 앞)로. 원점은 볼륨 중심
pub(crate) fn to_part(name: &str, m: Mesh, centre: Vec3) -> Part {
    let pos: Vec<Vec3> = m.pos.iter().map(|p| {
        let q = *p - centre;
        Vec3::new(q.x, q.z, -q.y)
    }).collect();
    let nrm = normals(&pos, &m.idx);
    let (min, max) = pos.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    Part { name: name.into(), kind: kind_of(name), positions: pos, normals: nrm, indices: m.idx, min, max, volume_mm3: None }
}

/// 큰 덩어리만 남기고 나머지의 밝은 칸은 공기로 바꾼다
fn keep_components(v: &mut Volume, level: f32, keep_largest: usize, min_voxels: usize) -> usize {
    let mask: Vec<bool> = v.data.iter().map(|&x| x > level).collect();
    let (lab, sizes) = label(&mask, v.dims);
    let mut order: Vec<usize> = (1..sizes.len()).filter(|&i| sizes[i] >= min_voxels).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));
    if keep_largest > 0 {
        order.truncate(keep_largest);
    }
    let mut keep = vec![false; sizes.len()];
    for &i in &order {
        keep[i] = true;
    }
    v.data.par_iter_mut().zip(lab.par_iter()).for_each(|(x, &l)| {
        if *x > level && !keep[l as usize] {
            *x = AIR;
        }
    });
    order.len()
}

pub fn threshold_meshes(dir: &Path, p: &Params, log: Log) -> Result<Vec<Part>, String> {
    let t = Instant::now();
    let lap = |what: &str| log(format!("{what} ({:.1}초)", t.elapsed().as_secs_f32()));

    let full = read_dicom_dir(dir)?;
    lap(&format!("DICOM {}장 읽음, {}×{}×{}, {:.2}mm", full.dims[2], full.dims[0], full.dims[1], full.dims[2], full.spacing[0]));
    let centre = Vec3::from(full.origin) + 0.5 * Vec3::from(full.dims.map(|d| d as f32)) * Vec3::from(full.spacing);

    // 2) 0.4mm로 줄이고 흐리기
    let mut half = shrink2(&full);
    gaussian(&mut half, 0.3);
    lap("줄이고 흐림");

    // 3) 치아 덩어리: 치아 임계값을 넘는 덩어리 가운데 법랑질 밝기 칸이 100개 이상인 것
    let above: Vec<bool> = half.data.iter().map(|&x| x > p.teeth).collect();
    let (lab, sizes) = label(&above, half.dims);
    let mut enamel = vec![0usize; sizes.len()];
    for (l, &x) in lab.iter().zip(&half.data) {
        if *l != 0 && x > p.enamel {
            enamel[*l as usize] += 1;
        }
    }
    let teeth_mask: Vec<bool> = lab.iter().map(|&l| l != 0 && enamel[l as usize] >= 100).collect();
    let n_teeth = enamel.iter().filter(|&&c| c >= 100).count();
    lap(&format!("치아 덩어리 {n_teeth}개"));

    // 4) 뼈: 치아 자리를 빼고, 큰 덩어리 6개까지
    let mut bone_src = half.clone();
    let holes = dilate(&teeth_mask, half.dims, 2);
    bone_src.data.par_iter_mut().zip(holes.par_iter()).for_each(|(x, &h)| {
        if h {
            *x = AIR;
        }
    });
    let n_bone = keep_components(&mut bone_src, p.bone, 6, 2000);
    let mut bone = isosurface(&bone_src, p.bone);
    let raw_bone = bone.idx.len() / 3;
    taubin(&mut bone, 10);
    let bone = simplify(&bone, p.bone_tris);
    lap(&format!("뼈 표면: 덩어리 {n_bone}개, 삼각형 {}만 → {}만", raw_bone / 10_000, bone.idx.len() / 3 / 10_000));

    // 5) 치아: 원본 해상도에서 치아 둘레(3mm 여유)만 잘라서
    let near = dilate(&teeth_mask, half.dims, 1);
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for (i, _) in teeth_mask.iter().enumerate().filter(|(_, m)| **m) {
        let c = [i % half.dims[0], (i / half.dims[0]) % half.dims[1], i / (half.dims[0] * half.dims[1])];
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    if lo[0] == usize::MAX {
        return Err("치아를 찾지 못했습니다. 치아 임계값을 확인하세요".into());
    }
    let pad = [0, 1, 2].map(|a| (3.0 / full.spacing[a]).ceil() as usize);
    let flo = [0, 1, 2].map(|a| (lo[a] * 2).saturating_sub(pad[a]));
    let fhi = [0, 1, 2].map(|a| (hi[a] * 2 + 1 + pad[a]).min(full.dims[a] - 1));
    let cd = [0, 1, 2].map(|a| fhi[a] - flo[a] + 1);
    let mut crop = Volume {
        dims: cd,
        spacing: full.spacing,
        origin: [0, 1, 2].map(|a| full.origin[a] + flo[a] as f32 * full.spacing[a]),
        data: vec![0.0; cd[0] * cd[1] * cd[2]],
    };
    crop.data.par_chunks_mut(cd[0] * cd[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..cd[1] {
            for x in 0..cd[0] {
                plane[y * cd[0] + x] = full.data[full.idx(x + flo[0], y + flo[1], z + flo[2])];
            }
        }
    });
    gaussian(&mut crop, 0.2);
    // 치아 둘레 밖은 공기로 (0.4mm 격자의 표시를 0.2mm 격자로 옮겨 씀)
    let hd = half.dims;
    crop.data.par_chunks_mut(cd[0] * cd[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..cd[1] {
            for x in 0..cd[0] {
                let h = [(x + flo[0]) / 2, (y + flo[1]) / 2, (z + flo[2]) / 2];
                let inside = h[0] < hd[0] && h[1] < hd[1] && h[2] < hd[2] && near[(h[2] * hd[1] + h[1]) * hd[0] + h[0]];
                if !inside {
                    plane[y * cd[0] + x] = AIR;
                }
            }
        }
    });
    let n_pieces = keep_components(&mut crop, p.teeth, 0, 400);
    let mut teeth = isosurface(&crop, p.teeth);
    let raw_teeth = teeth.idx.len() / 3;
    taubin(&mut teeth, 10);
    let teeth = simplify(&teeth, p.teeth_tris);
    lap(&format!("치아 표면: 덩어리 {n_pieces}개, 삼각형 {}만 → {}만", raw_teeth / 10_000, teeth.idx.len() / 3 / 10_000));

    Ok(vec![to_part("bone", bone, centre), to_part("teeth", teeth, centre)])
}

/// 메시를 GLB로 저장한다 (Python 결과와 비교하거나 다른 프로그램에서 열 때)
pub fn write_glb(parts: &[Part], path: &Path) -> std::io::Result<()> {
    let mut bin: Vec<u8> = Vec::new();
    let (mut views, mut accessors, mut meshes, mut nodes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, p) in parts.iter().enumerate() {
        let mut push = |bytes: &[u8], target: u32| {
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            views.push(serde_json::json!({"buffer": 0, "byteOffset": bin.len(), "byteLength": bytes.len(), "target": target}));
            bin.extend_from_slice(bytes);
            views.len() - 1
        };
        let pv = push(bytemuck::cast_slice(&p.positions), 34962);
        let nv = push(bytemuck::cast_slice(&p.normals), 34962);
        let iv = push(bytemuck::cast_slice(&p.indices), 34963);
        let a = accessors.len();
        accessors.push(serde_json::json!({"bufferView": pv, "componentType": 5126, "count": p.positions.len(), "type": "VEC3",
            "min": p.min.to_array(), "max": p.max.to_array()}));
        accessors.push(serde_json::json!({"bufferView": nv, "componentType": 5126, "count": p.normals.len(), "type": "VEC3"}));
        accessors.push(serde_json::json!({"bufferView": iv, "componentType": 5125, "count": p.indices.len(), "type": "SCALAR"}));
        meshes.push(serde_json::json!({"name": p.name, "primitives": [{"attributes": {"POSITION": a, "NORMAL": a + 1}, "indices": a + 2}]}));
        nodes.push(serde_json::json!({"name": p.name, "mesh": i}));
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let json = serde_json::json!({
        "asset": {"version": "2.0", "generator": "ct3d-viewer"},
        "scene": 0, "scenes": [{"nodes": (0..parts.len()).collect::<Vec<_>>()}],
        "nodes": nodes, "meshes": meshes, "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": bin.len()}],
    });
    let mut js = serde_json::to_vec(&json)?;
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&js);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    std::fs::write(path, out)
}
