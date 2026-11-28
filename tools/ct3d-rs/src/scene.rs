//! GLB 파일을 읽어 구조별 메시로 나누고, 이름으로 종류와 한글 이름을 정한다.
//! 규칙은 tools/ct3d/viewer_template.html과 같다.

use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::path::Path;

/// 구조의 종류. 화면에서 종류별로 켜고 끈다.
pub struct Kind {
    pub key: &'static str,
    pub label: &'static str,
    pub color: [u8; 3], // sRGB
    pub soft: bool,     // 반투명으로 그림
}

pub const KINDS: &[Kind] = &[
    Kind { key: "bone", label: "뼈", color: [0xb9, 0xa5, 0x7a], soft: true },
    Kind { key: "teeth", label: "치아", color: [0xf7, 0xf3, 0xe6], soft: false },
    Kind { key: "jaw", label: "턱뼈", color: [0xb9, 0xa5, 0x7a], soft: true },
    Kind { key: "tooth", label: "치아", color: [0xf7, 0xf3, 0xe6], soft: false },
    Kind { key: "remnant", label: "뿌리만 남음", color: [0x8a, 0x6a, 0x45], soft: false },
    Kind { key: "pulp", label: "치수", color: [0xd9, 0x5a, 0x54], soft: false },
    Kind { key: "canal", label: "신경관", color: [0xf2, 0xbf, 0x33], soft: false },
    Kind { key: "implant", label: "임플란트", color: [0xaa, 0xb3, 0xbd], soft: false },
    Kind { key: "crown", label: "크라운", color: [0xc9, 0xdc, 0xee], soft: false },
    Kind { key: "bridge", label: "브릿지", color: [0xe6, 0xd9, 0xf0], soft: false },
    Kind { key: "sinus", label: "상악동", color: [0x8c, 0xbf, 0xe6], soft: true },
    Kind { key: "other", label: "기타", color: [0x99, 0xcc, 0xb3], soft: true },
];

pub fn kind_index(key: &str) -> usize {
    KINDS.iter().position(|k| k.key == key).unwrap_or(KINDS.len() - 1)
}

fn fdi_of(name: &str) -> Option<u32> {
    let pos = name.rfind("fdi")?;
    name[pos + 3..].parse().ok()
}

/// 메시 이름으로 종류를 정한다.
pub fn kind_of(name: &str) -> usize {
    let key = if name == "bone" || name == "teeth" {
        name
    } else if name.starts_with("remnant") {
        "remnant"
    } else if name.contains("pulp") {
        "pulp"
    } else if name.contains("canal") {
        "canal"
    } else if matches!(fdi_of(name), Some(n) if (11..=48).contains(&n)) {
        "tooth"
    } else if name.contains("jawbone") {
        "jaw"
    } else if name.contains("sinus") {
        "sinus"
    } else if matches!(name, "implant" | "crown" | "bridge") {
        name
    } else {
        "other"
    };
    kind_index(key)
}

/// 화면에 보여 줄 한글 이름.
pub fn label_of(name: &str, kind: usize) -> String {
    const QUAD: [&str; 5] = ["", "위 오른쪽", "위 왼쪽", "아래 왼쪽", "아래 오른쪽"];
    const TOOTH: [&str; 8] = ["가운데 앞니", "옆 앞니", "송곳니", "첫째 작은어금니", "둘째 작은어금니", "첫째 큰어금니", "둘째 큰어금니", "사랑니"];
    let key = KINDS[kind].key;
    if let (Some(n), true) = (fdi_of(name), matches!(key, "tooth" | "pulp" | "remnant")) {
        let n = n % 100;
        let (q, i) = ((n / 10) as usize, (n % 10) as usize);
        if (1..=4).contains(&q) && (1..=8).contains(&i) {
            let suffix = match key {
                "pulp" => "의 치수",
                _ => "",
            };
            let note = if key == "remnant" { ", 뿌리만 남음" } else { "" };
            return format!("{} {}{} ({}번{})", QUAD[q], TOOTH[i - 1], suffix, n, note);
        }
    }
    if let Some(rest) = name.strip_prefix("fragment_") {
        return format!("치아 번호를 붙이지 않은 조각 {rest}");
    }
    let fixed: HashMap<&str, &str> = [
        ("lower_jawbone", "아래턱뼈"),
        ("upper_jawbone", "위턱뼈"),
        ("left_inferior_alveolar_canal", "왼쪽 아래턱 신경관"),
        ("right_inferior_alveolar_canal", "오른쪽 아래턱 신경관"),
        ("left_mandibular_incisive_canal_fdi103", "왼쪽 앞니 신경관"),
        ("right_mandibular_incisive_canal_fdi104", "오른쪽 앞니 신경관"),
        ("lingual_canal", "혀 쪽 혈관 구멍"),
        ("left_maxillary_sinus", "왼쪽 상악동"),
        ("right_maxillary_sinus", "오른쪽 상악동"),
        ("pharynx", "목 안 공간"),
        ("implant", "임플란트"),
        ("crown", "크라운"),
        ("bridge", "브릿지"),
        ("bone", "뼈"),
        ("teeth", "치아"),
    ]
    .into_iter()
    .collect();
    fixed.get(name).map(|s| s.to_string()).unwrap_or_else(|| name.to_string())
}

pub struct Part {
    pub name: String,
    pub kind: usize,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub min: Vec3,
    pub max: Vec3,
    pub volume_mm3: Option<f64>,
}

impl Part {
    pub fn centre(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

/// GLB를 읽는다. 노드마다 메시 하나라고 보고, 노드 이름(없으면 메시 이름)을 구조 이름으로 쓴다.
pub fn load_glb(path: &Path) -> Result<Vec<Part>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("GLB를 읽지 못했습니다: {e}"))?;
    // parts_info.json(labels_to_mesh.py의 출력)이 옆에 있으면 부피를 읽는다.
    let info = std::fs::read_to_string(path.with_file_name("parts_info.json")).ok();
    load_glb_bytes(&bytes, info.as_deref())
}

/// 메모리에 있는 GLB를 읽는다(웹에서 받은 파일, 끌어다 놓은 파일). info는 parts_info.json의 내용
pub fn load_glb_bytes(bytes: &[u8], info: Option<&str>) -> Result<Vec<Part>, String> {
    let (doc, buffers, _) = gltf::import_slice(bytes).map_err(|e| format!("GLB를 읽지 못했습니다: {e}"))?;
    let mut parts = Vec::new();
    let scene = doc.default_scene().or_else(|| doc.scenes().next()).ok_or("장면이 없습니다")?;
    for node in scene.nodes() {
        walk(&node, Mat4::IDENTITY, &buffers, &mut parts);
    }
    if parts.is_empty() {
        return Err("메시가 없습니다".into());
    }

    if let Some(text) = info {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
            for p in v["parts"].as_array().into_iter().flatten() {
                if let (Some(name), Some(vol)) = (p["name"].as_str(), p["volume_mm3"].as_f64()) {
                    if let Some(part) = parts.iter_mut().find(|x| x.name == name) {
                        part.volume_mm3 = Some(vol);
                    }
                }
            }
        }
    }
    Ok(parts)
}

fn walk(node: &gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], out: &mut Vec<Part>) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        let name = node.name().or(mesh.name()).unwrap_or("part").to_string();
        let normal_m = world.inverse().transpose();
        let (mut pos, mut nrm, mut idx) = (Vec::new(), Vec::new(), Vec::new());
        for prim in mesh.primitives() {
            let r = prim.reader(|b| Some(&buffers[b.index()]));
            let base = pos.len() as u32;
            let Some(ps) = r.read_positions() else { continue };
            pos.extend(ps.map(|p| world.transform_point3(Vec3::from(p))));
            match r.read_normals() {
                Some(ns) => nrm.extend(ns.map(|n| normal_m.transform_vector3(Vec3::from(n)).normalize_or_zero())),
                None => nrm.resize(pos.len(), Vec3::ZERO),
            }
            match r.read_indices() {
                Some(ix) => idx.extend(ix.into_u32().map(|i| i + base)),
                None => idx.extend(base..pos.len() as u32),
            }
        }
        if nrm.iter().all(|n| *n == Vec3::ZERO) {
            nrm = vertex_normals(&pos, &idx);
        }
        if !pos.is_empty() {
            let (min, max) = pos.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
            out.push(Part { kind: kind_of(&name), name, positions: pos, normals: nrm, indices: idx, min, max, volume_mm3: None });
        }
    }
    for child in node.children() {
        walk(&child, world, buffers, out);
    }
}

fn vertex_normals(pos: &[Vec3], idx: &[u32]) -> Vec<Vec3> {
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

/// 광선이 구조에 처음 닿는 거리. 닿지 않으면 None.
pub fn ray_hit(part: &Part, origin: Vec3, dir: Vec3, cut_z: Option<f32>) -> Option<f32> {
    if !ray_box(origin, dir, part.min, part.max) {
        return None;
    }
    let mut best: Option<f32> = None;
    for t in part.indices.chunks_exact(3) {
        let (a, b, c) = (part.positions[t[0] as usize], part.positions[t[1] as usize], part.positions[t[2] as usize]);
        if let Some(d) = ray_triangle(origin, dir, a, b, c) {
            if let Some(z) = cut_z {
                if (origin + dir * d).z > z {
                    continue; // 잘려 나가 안 보이는 부분
                }
            }
            if best.map_or(true, |x| d < x) {
                best = Some(d);
            }
        }
    }
    best
}

fn ray_box(o: Vec3, d: Vec3, min: Vec3, max: Vec3) -> bool {
    let inv = d.recip();
    let (t1, t2) = ((min - o) * inv, (max - o) * inv);
    let tmin = t1.min(t2).max_element();
    let tmax = t1.max(t2).min_element();
    tmax >= tmin.max(0.0)
}

/// Möller–Trumbore
fn ray_triangle(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 1e-4).then_some(t)
}
