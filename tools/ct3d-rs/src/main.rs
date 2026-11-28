//! CT 3D 보기 (Rust·egui 판)
//!
//! tools/ct3d의 Python 스크립트가 만든 GLB(ct_surface.glb, teeth_parts.glb)를 연다.
//! 웹 뷰어(viewer_template.html)와 같은 조작을 한다: 보는 방향, 종류별 켜고 끄기, 뼈 농도, 자르기, 누르면 이름.
//!
//! 사용법:
//!     ct3d-viewer [파일.glb | DICOM 폴더] [--view front|right|left|top|bottom] [--hide jaw,sinus]
//!                 [--bone 0..100] [--cut 0..100] [--pick 구조이름] [--export 저장.glb] [--screenshot 저장.png]
//!                 [--seg 분할폴더 [--assign 1:47,2:46] [--ai-jaws] [--model-js ct-teeth.js [--remap 42:41]]]
//! DICOM 폴더를 주면 뼈와 치아 표면을 그 자리에서 만든다(tools/ct3d/volume_to_mesh.py와 같은 순서).
//! --seg로 segment_teeth.py의 출력 폴더를 함께 주면 구조별 메시를 만든다(find_remnants.py, labels_to_mesh.py와 같은 순서).
//! --model-js를 주면 창을 열지 않고 치열 모형용 ct-teeth.js만 만든다(teeth_for_model.py와 같은 순서).
//! --click, --drag, --wheel은 누르기·끌기·휠을 흉내 낸다(확인용).
//! 웹(wasm) 빌드는 GLB 보기만 한다. 주소의 ?glb=파일.glb로 받는다(web/index.html).
//! 파일이나 폴더는 창에 끌어다 놓아도 열린다. --screenshot을 주면 화면을 PNG로 저장하고 닫는다(확인용).

#[cfg(not(target_arch = "wasm32"))]
mod ctproc;
#[cfg(not(target_arch = "wasm32"))]
mod model;
mod render;
mod scene;
#[cfg(not(target_arch = "wasm32"))]
mod seg;
#[cfg(not(target_arch = "wasm32"))]
mod scan;

use eframe::egui::{self, Color32, RichText, Sense, Vec2};
use glam::Vec3;
use render::Renderer;
use scene::{Part, KINDS};
use std::path::PathBuf;

const VIEWS: [(&str, &str, f32, f32); 5] = [
    ("front", "정면", 0.0, 0.0),
    ("right", "오른쪽", -std::f32::consts::FRAC_PI_2, 0.0),
    ("left", "왼쪽", std::f32::consts::FRAC_PI_2, 0.0),
    ("top", "위에서", 0.0, 1.55),
    ("bottom", "아래에서", 0.0, -1.55),
];

#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // 웹은 DICOM 처리와 명령줄 인자를 쓰지 않는다
#[derive(Default)]
struct Args {
    file: Option<PathBuf>,
    view: Option<String>,
    hide: Vec<String>,
    bone: Option<f32>,
    cut: Option<f32>,
    pick: Option<String>,
    screenshot: Option<PathBuf>,
    click: Option<(f32, f32)>, // 3D 화면 안의 비율 좌표(0..1)를 누른 것처럼 한다 (확인용)
    drag: Option<(f32, f32)>,  // 3D 화면 가운데에서 이만큼(점) 끈 것처럼 한다 (확인용)
    wheel: Option<f32>,        // 휠을 이만큼(줄) 굴린 것처럼 한다. 양수가 앞으로(확대) (확인용)
    export: Option<PathBuf>,   // DICOM에서 만든 메시를 GLB로 저장
    seg: Option<PathBuf>,      // segment_teeth.py의 출력 폴더
    assign: std::collections::HashMap<usize, u32>, // 뿌리 조각 후보 번호 → FDI 번호
    ai_jaws: bool,
    scan: Option<PathBuf>,     // 구강 스캔 폴더(UpperJaw, LowerJaw STL)
    model_js: Option<PathBuf>, // 치열 모형용 ct-teeth.js를 만들고 끝낸다
    remap: std::collections::HashMap<u32, u32>, // 분할 번호 → 모형 번호
}

/// "1:47,2:46" 꼴을 읽는다
#[cfg(not(target_arch = "wasm32"))]
fn pairs<K: std::str::FromStr + std::hash::Hash + Eq, V: std::str::FromStr>(s: Option<String>) -> std::collections::HashMap<K, V> {
    s.unwrap_or_default()
        .split(',')
        .filter_map(|kv| {
            let (k, v) = kv.split_once(':')?;
            Some((k.trim().parse().ok()?, v.trim().parse().ok()?))
        })
        .collect()
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // 웹은 DICOM 처리와 명령줄 인자를 쓰지 않는다
#[derive(Clone)]
struct SegOptions {
    dir: PathBuf,
    assign: std::collections::HashMap<usize, u32>,
    ai_jaws: bool,
    scan: Option<PathBuf>,
}

type Job = std::sync::mpsc::Receiver<Result<Vec<Part>, String>>;

#[cfg(not(target_arch = "wasm32"))]
fn parse_args() -> Args {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(s) = it.next() {
        match s.as_str() {
            "--view" => a.view = it.next(),
            "--hide" => a.hide = it.next().unwrap_or_default().split(',').map(str::to_string).collect(),
            "--bone" => a.bone = it.next().and_then(|v| v.parse().ok()),
            "--cut" => a.cut = it.next().and_then(|v| v.parse().ok()),
            "--pick" => a.pick = it.next(),
            "--screenshot" => a.screenshot = it.next().map(PathBuf::from),
            "--export" => a.export = it.next().map(PathBuf::from),
            "--seg" => a.seg = it.next().map(PathBuf::from),
            "--ai-jaws" => a.ai_jaws = true,
            "--scan" => a.scan = it.next().map(PathBuf::from),
            "--assign" => a.assign = pairs(it.next()),
            "--remap" => a.remap = pairs(it.next()),
            "--model-js" => a.model_js = it.next().map(PathBuf::from),
            "--drag" => a.drag = it.next().and_then(|v| {
                let (x, y) = v.split_once(',')?;
                Some((x.parse().ok()?, y.parse().ok()?))
            }),
            "--wheel" => a.wheel = it.next().and_then(|v| v.parse().ok()),
            "--click" => {
                a.click = it.next().and_then(|v| {
                    let (x, y) = v.split_once(',')?;
                    Some((x.parse().ok()?, y.parse().ok()?))
                })
            }
            _ => a.file = Some(PathBuf::from(s)),
        }
    }
    a
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // 웹은 DICOM 처리와 명령줄 인자를 쓰지 않는다
struct App {
    renderer: Option<Renderer>,
    parts: Vec<Part>,
    visible: Vec<bool>,
    kind_on: Vec<bool>,
    file_name: String,
    message: String,
    // 카메라
    target: Vec3,
    fit: f32,
    yaw: f32,
    pitch: f32,
    dist_scale: f32,
    view_name: &'static str,
    // 조작
    bone: f32,
    cut: f32,
    z_range: (f32, f32),
    picked: Option<usize>,
    // 확인용 화면 저장
    screenshot: Option<PathBuf>,
    closing: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pending_pick: Option<String>,
    pending_click: Option<(f32, f32)>,
    pending_drag: Option<(f32, f32)>,
    pending_wheel: Option<f32>,
    scene_rect: Option<egui::Rect>,
    export: Option<PathBuf>,
    seg: Option<SegOptions>,
    job: Option<Job>,
    progress: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    frames: u32,
    bench_start: Option<std::time::Instant>,
}

const SHOT_FRAME: u32 = 70; // 확인용 화면 저장은 이 프레임에서. 10~70프레임의 평균 시간을 잰다

impl App {
    fn new(cc: &eframe::CreationContext, args: Args) -> Self {
        install_korean_font(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        let renderer = cc.wgpu_render_state.as_ref().map(Renderer::new);
        let mut app = Self {
            renderer,
            parts: Vec::new(),
            visible: Vec::new(),
            kind_on: vec![true; KINDS.len()],
            file_name: String::new(),
            message: "GLB 파일을 이 창에 끌어다 놓으세요.".into(),
            target: Vec3::ZERO,
            fit: 60.0,
            yaw: 0.0,
            pitch: 0.0,
            dist_scale: 1.0,
            view_name: "front",
            bone: args.bone.unwrap_or(55.0),
            cut: args.cut.unwrap_or(0.0),
            z_range: (-60.0, 60.0),
            picked: None,
            screenshot: args.screenshot,
            closing: Default::default(),
            pending_pick: args.pick,
            pending_click: args.click,
            pending_drag: args.drag,
            pending_wheel: args.wheel,
            scene_rect: None,
            export: args.export,
            seg: args.seg.map(|dir| SegOptions { dir, assign: args.assign, ai_jaws: args.ai_jaws, scan: args.scan }),
            job: None,
            progress: Default::default(),
            frames: 0,
            bench_start: None,
        };
        if app.renderer.is_none() {
            app.message = "3D 화면(wgpu)을 열 수 없습니다.".into();
        }
        for key in &args.hide {
            if let Some(i) = KINDS.iter().position(|k| k.key == key) {
                app.kind_on[i] = false;
            }
        }
        if let Some(f) = args.file {
            app.open(f);
        }
        if let Some(v) = args.view.as_deref().and_then(|v| VIEWS.iter().position(|x| x.0 == v)) {
            app.set_view(v);
        }
        app
    }

    fn open(&mut self, path: PathBuf) {
        self.file_name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        #[cfg(not(target_arch = "wasm32"))]
        if path.is_dir() {
            // DICOM 폴더: 시간이 걸리므로 따로 돌리고, 진행 상황은 화면에 보여 준다
            let (tx, rx) = std::sync::mpsc::channel();
            let progress = self.progress.clone();
            progress.lock().unwrap().clear();
            self.message = "CT를 읽는 중입니다.".into();
            let seg = self.seg.clone();
            std::thread::spawn(move || {
                let log = move |s: String| {
                    println!("{s}");
                    progress.lock().unwrap().push(s);
                };
                let result = match seg {
                    None => ctproc::threshold_meshes(&path, &ctproc::Params::default(), &log),
                    Some(o) => segmented_meshes(&path, &o, &log),
                };
                let _ = tx.send(result);
            });
            self.job = Some(rx);
            return;
        }
        let result = scene::load_glb(&path);
        self.set_parts(result);
    }

    fn set_parts(&mut self, result: Result<Vec<Part>, String>) {
        match result {
            Ok(parts) => {
                let (min, max) = parts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(p.min), b.max(p.max)));
                self.target = (min + max) * 0.5;
                self.fit = 0.5 * (max - min).max_element();
                self.z_range = (min.z, max.z);
                self.visible = vec![true; parts.len()];
                if let Some(r) = self.renderer.as_mut() {
                    r.upload(&parts);
                }
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(path) = self.export.take() {
                    match ctproc::write_glb(&parts, &path) {
                        Ok(()) => println!("저장: {}", path.display()),
                        Err(e) => println!("GLB를 저장하지 못했습니다: {e}"),
                    }
                }
                self.parts = parts;
                self.picked = self.pending_pick.take().and_then(|n| self.parts.iter().position(|p| p.name == n));
                self.message.clear();
                let view = VIEWS.iter().position(|v| v.0 == self.view_name).unwrap_or(0);
                self.set_view(view);
            }
            Err(e) => {
                println!("{e}");
                self.message = e;
            }
        }
    }

    fn set_view(&mut self, i: usize) {
        let (name, _, yaw, pitch) = VIEWS[i];
        self.view_name = name;
        self.yaw = yaw;
        self.pitch = pitch;
        self.dist_scale = 1.0;
    }

    fn dist(&self, aspect: f32) -> f32 {
        self.fit / (14f32.to_radians()).tan() * 1.2 / aspect.min(1.0) * self.dist_scale
    }

    fn cut_z(&self) -> Option<f32> {
        (self.cut > 0.0).then(|| self.z_range.1 - self.cut / 100.0 * (self.z_range.1 - self.z_range.0))
    }

    fn stats(&self) -> String {
        if self.parts.is_empty() {
            return String::new();
        }
        let tris: usize = self.parts.iter().map(|p| p.indices.len() / 3).sum();
        let teeth = self.parts.iter().filter(|p| KINDS[p.kind].key == "tooth").count();
        let mut s = Vec::new();
        if teeth > 0 {
            s.push(format!("치아 {teeth}개"));
            s.push(format!("구조 {}개", self.parts.len()));
        }
        s.push(format!("삼각형 {}천 개", tris / 1000));
        s.push(self.file_name.clone());
        s.join(" · ")
    }

    fn pick_at(&mut self, view_proj: glam::Mat4, ndc: (f32, f32)) {
        let (o, d) = render::unproject(view_proj, ndc.0, ndc.1);
        let cut = self.cut_z();
        let soft_hidden = self.bone < 100.0;
        let mut hits: Vec<(f32, usize)> = self
            .parts
            .iter()
            .enumerate()
            .filter(|(i, p)| self.visible[*i] && !matches!(KINDS[p.kind].key, "bone" | "teeth"))
            .filter_map(|(i, p)| scene::ray_hit(p, o, d, cut).map(|t| (t, i)))
            .collect();
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        // 반투명한 턱뼈 뒤의 치아를 고를 수 있게, 불투명한 것이 맞았으면 그것을 고른다
        let solid = hits.iter().find(|(_, i)| !(KINDS[self.parts[*i].kind].soft && soft_hidden));
        self.picked = solid.or(hits.first()).map(|h| h.1);
    }
}

/// DICOM과 AI 분할 결과로 구조별 메시를 만든다
#[cfg(not(target_arch = "wasm32"))]
fn segmented_meshes(dicom: &std::path::Path, o: &SegOptions, log: ctproc::Log) -> Result<Vec<Part>, String> {
    let (ct, s, cands) = load_segmentation(dicom, o, log)?;
    let mut parts = seg::part_meshes(&ct, &s, &cands, o.ai_jaws, log);
    if let Some(dir) = &o.scan {
        parts.extend(scan_parts(dir, &parts, log)?);
    }
    Ok(parts)
}

/// 구강 스캔을 CT 치아(뷰어 좌표)에 맞춰 upper_scan, lower_scan으로 만든다 (scan_to_ct.py와 같은 순서)
#[cfg(not(target_arch = "wasm32"))]
fn scan_parts(dir: &std::path::Path, parts: &[Part], log: ctproc::Log) -> Result<Vec<Part>, String> {
    let teeth: Vec<(u32, Vec<glam::DVec3>)> = parts
        .iter()
        .filter(|p| KINDS[p.kind].key == "tooth")
        .filter_map(|p| Some((seg::fdi_of(&p.name)?, p.positions.iter().map(|v| v.as_dvec3()).collect())))
        .collect();
    let scans = scan::load_and_register(dir, &teeth, log)?;
    let mut out = Vec::new();
    for ((surface, _), name) in scans.into_iter().zip(["upper_scan", "lower_scan"]) {
        let mesh = ctproc::Mesh { pos: surface.pos.iter().map(|p| p.as_vec3()).collect(), idx: surface.faces.iter().flatten().copied().collect() };
        let m = ctproc::simplify(&mesh, 150_000);
        out.push(ctproc::view_part(name, m.pos, m.idx));
    }
    Ok(out)
}

/// DICOM과 AI 분할 결과를 읽고 뿌리 조각 후보를 찾는다
#[cfg(not(target_arch = "wasm32"))]
fn load_segmentation(dicom: &std::path::Path, o: &SegOptions, log: ctproc::Log) -> Result<(ctproc::Volume, seg::Seg, Vec<seg::Candidate>), String> {
    let t = std::time::Instant::now();
    let ct = ctproc::read_dicom_dir(dicom)?;
    log(format!("DICOM 읽음 ({:.1}초)", t.elapsed().as_secs_f32()));
    let s = seg::read_seg(&o.dir, &ct)?;
    // 격자가 맞는지 확인: AI가 치아로 잡은 칸은 CT에서 밝아야 한다
    let tooth_ids: std::collections::HashSet<u8> = s
        .names
        .iter()
        .filter(|(_, n)| KINDS[scene::kind_of(n)].key == "tooth")
        .map(|(l, _)| *l)
        .collect();
    let (sum, n) = s.labels.iter().zip(&ct.data).filter(|(l, _)| tooth_ids.contains(l)).fold((0f64, 0usize), |(a, b), (_, &v)| (a + v as f64, b + 1));
    log(format!("분할 결과 읽음: 구조 {}개, 치아로 잡힌 칸의 평균 밝기 {:.0} ({:.1}초)", s.names.len(), sum / n.max(1) as f64, t.elapsed().as_secs_f32()));
    let cands = seg::find_remnants(&ct, &s, &o.assign, log);
    Ok((ct, s, cands))
}

/// 창 없이 치열 모형용 ct-teeth.js를 만든다
#[cfg(not(target_arch = "wasm32"))]
fn model_js(args: &Args, out: &std::path::Path) -> Result<(), String> {
    let (Some(dicom), Some(dir)) = (args.file.as_ref().filter(|p| p.is_dir()), args.seg.clone()) else {
        return Err("--model-js에는 DICOM 폴더와 --seg 분할폴더가 필요합니다".into());
    };
    let log = |s: String| println!("{s}");
    let o = SegOptions { dir, assign: args.assign.clone(), ai_jaws: args.ai_jaws, scan: None };
    let (ct, s, cands) = load_segmentation(dicom, &o, &log)?;
    let opts = model::Options { remap: args.remap.clone(), scan: args.scan.clone(), ..Default::default() };
    model::write_model_js(&ct, &s, &cands, &opts, out, &log)
}

fn install_korean_font(ctx: &egui::Context) {
    // 데스크톱에서는 시스템 글꼴을 쓴다. 웹에서는 페이지 옆의 글꼴 파일을 받아 쓴다(web_main)
    for path in [r"C:\Windows\Fonts\malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"] {
        if let Ok(bytes) = std::fs::read(path) {
            set_korean_font(ctx, bytes);
            break;
        }
    }
}

fn set_korean_font(ctx: &egui::Context, bytes: Vec<u8>) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert("korean".into(), std::sync::Arc::new(egui::FontData::from_owned(bytes)));
    for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(fam).or_default().insert(0, "korean".into());
    }
    ctx.set_fonts(fonts);
}

fn swatch(ui: &mut egui::Ui, color: [u8; 3], text: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(11.0), Sense::hover());
    ui.painter().rect_filled(rect, 3.0, Color32::from_rgb(color[0], color[1], color[2]));
    ui.label(RichText::new(text).size(12.0).color(Color32::from_rgb(0x55, 0x67, 0x6a)));
}

/// 확인용 마우스 흉내: 이 프레임에서 가운데를 누르고, 10프레임에 걸쳐 끈 뒤 놓고, 휠을 굴린다
const DRAG_FRAME: u32 = 5;

impl eframe::App for App {
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        let Some(rect) = self.scene_rect.filter(|_| !self.parts.is_empty()) else { return };
        let c = rect.center();
        let f = self.frames + 1; // 이번에 그릴 프레임 번호
        let none = egui::Modifiers::NONE;
        let primary = egui::PointerButton::Primary;
        if let Some((dx, dy)) = self.pending_drag {
            let at = |k: u32| c + Vec2::new(dx, dy) * (k as f32 / 10.0);
            match f {
                _ if f == DRAG_FRAME => {
                    raw.events.push(egui::Event::PointerMoved(c));
                    raw.events.push(egui::Event::PointerButton { pos: c, button: primary, pressed: true, modifiers: none });
                }
                _ if f > DRAG_FRAME && f <= DRAG_FRAME + 10 => raw.events.push(egui::Event::PointerMoved(at(f - DRAG_FRAME))),
                _ if f == DRAG_FRAME + 11 => {
                    raw.events.push(egui::Event::PointerButton { pos: at(10), button: primary, pressed: false, modifiers: none });
                    self.pending_drag = None;
                    println!("끌기 뒤: yaw {:.3} pitch {:.3}", self.yaw, self.pitch);
                }
                _ => {}
            }
        } else if let Some(lines) = self.pending_wheel.filter(|_| f >= DRAG_FRAME + 13) {
            if f == DRAG_FRAME + 13 {
                raw.events.push(egui::Event::PointerMoved(c)); // 휠은 3D 화면 위에 있을 때만 듣는다
                raw.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: Vec2::new(0.0, lines),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: none,
                });
            } else if f == DRAG_FRAME + 40 {
                self.pending_wheel = None;
                println!("휠 뒤: 거리 배율 {:.3}", self.dist_scale);
            }
        }
    }

    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &root.ctx().clone();
        if let Some(result) = self.job.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.job = None;
            self.set_parts(result);
        }
        if self.job.is_some() {
            let lines = self.progress.lock().unwrap();
            self.message = std::iter::once("CT를 처리하는 중입니다.".to_string()).chain(lines.iter().cloned()).collect::<Vec<_>>().join("\n");
        }
        if !self.parts.is_empty() {
            self.frames += 1; // 확인용 프레임 수는 장면이 생긴 뒤부터 센다
        }
        if self.closing.load(std::sync::atomic::Ordering::Relaxed) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        // 끌어다 놓은 파일 열기
        // (웹에서는 경로 없이 내용만 비동기로 읽을 수 있다)
        let dropped = ctx.input(|i| i.raw.dropped_files.first().cloned());
        if let Some(f) = dropped {
            #[cfg(not(target_arch = "wasm32"))]
            self.open(f.path().to_path_buf());
            #[cfg(target_arch = "wasm32")]
            {
                self.file_name = f.path().to_string_lossy().into_owned();
                let (tx, rx) = std::sync::mpsc::channel();
                self.job = Some(rx);
                wasm_bindgen_futures::spawn_local(async move {
                    let r = f.bytes_async().await.and_then(|b| scene::load_glb_bytes(&b, None));
                    let _ = tx.send(r);
                });
            }
        }
        for (i, p) in self.parts.iter().enumerate() {
            self.visible[i] = self.kind_on[p.kind];
        }

        egui::Panel::top("head").show(root, |ui| {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("CT 3D 보기").strong());
                ui.label(RichText::new(self.stats()).color(Color32::from_rgb(0x55, 0x67, 0x6a)));
            });
        });

        egui::Panel::bottom("panel").show(root, |ui| {
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                ui.label("방향");
                for i in 0..VIEWS.len() {
                    if ui.selectable_label(self.view_name == VIEWS[i].0, VIEWS[i].1).clicked() {
                        self.set_view(i);
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("표시");
                for k in 0..KINDS.len() {
                    if self.parts.iter().any(|p| p.kind == k) {
                        let mut on = self.kind_on[k];
                        if ui.toggle_value(&mut on, KINDS[k].label).changed() {
                            self.kind_on[k] = on;
                            if !on && self.picked.map_or(false, |p| self.parts[p].kind == k) {
                                self.picked = None;
                            }
                        }
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("뼈 농도");
                ui.add(egui::Slider::new(&mut self.bone, 5.0..=100.0).show_value(false));
                ui.label("자르기");
                ui.add(egui::Slider::new(&mut self.cut, 0.0..=100.0).show_value(false));
            });
            ui.label(
                RichText::new("CT에서 뽑은 표면입니다. 설명용이며 진단 자료가 아닙니다. 좌우는 본인 기준입니다.")
                    .size(12.0)
                    .color(Color32::from_rgb(0x55, 0x67, 0x6a)),
            );
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(Color32::from_rgb(0xee, 0xf2, 0xf1))).show(root, |ui| {
            let rect = ui.max_rect();
            self.scene_rect = Some(rect);
            let response = ui.allocate_rect(rect, Sense::click_and_drag());
            if self.parts.is_empty() || self.renderer.is_none() {
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, &self.message, egui::FontId::proportional(15.0), Color32::GRAY);
                return;
            }
            // 돌리기와 확대
            if response.dragged() {
                let d = response.drag_delta();
                self.yaw -= d.x * 0.01;
                self.pitch = (self.pitch + d.y * 0.01).clamp(-1.55, 1.55);
                self.view_name = "";
            }
            if response.hovered() {
                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.dist_scale = (self.dist_scale * (-scroll * 0.002).exp()).clamp(0.2, 4.0);
                }
            }

            let ppp = ctx.pixels_per_point();
            let size = [(rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32];
            let aspect = rect.width() / rect.height().max(1.0);
            let (view_proj, eye, light) = render::camera(self.target, self.yaw, self.pitch, self.dist(aspect), aspect);

            let mut click_at = if response.clicked() { response.interact_pointer_pos() } else { None };
            if self.frames == 3 {
                if let Some((fx, fy)) = self.pending_click.take() {
                    click_at = Some(egui::pos2(rect.left() + fx * rect.width(), rect.top() + fy * rect.height()));
                }
            }
            if let Some(pos) = click_at {
                let ndc = ((pos.x - rect.left()) / rect.width() * 2.0 - 1.0, 1.0 - (pos.y - rect.top()) / rect.height() * 2.0);
                self.pick_at(view_proj, ndc);
            }

            let rs = _frame.wgpu_render_state().expect("wgpu");
            let r = self.renderer.as_mut().unwrap();
            let bg = [0xee, 0xf2, 0xf1].map(|v: u8| v as f32 / 255.0); // 텍스처가 감마 공간이라 그대로 쓴다
            r.draw(
                size,
                rs,
                &self.parts,
                &render::Frame {
                    view_proj,
                    eye,
                    light_dir: light,
                    cut_z: (self.cut > 0.0).then(|| self.z_range.1 - self.cut / 100.0 * (self.z_range.1 - self.z_range.0)),
                    visible: &self.visible,
                    soft_opacity: self.bone / 100.0,
                    picked: self.picked,
                    background: bg.map(|v| v as f64),
                },
            );
            if let Some(id) = r.texture_id {
                ui.painter().image(id, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
            }

            // 좌우 표시(얼굴 앞에서 볼 때만), 고른 구조, 범례
            let muted = Color32::from_rgb(0x55, 0x67, 0x6a);
            if self.pitch.abs() < 0.6 && self.yaw.cos() > 0.8 {
                let font = egui::FontId::proportional(12.0);
                ui.painter().text(rect.left_top() + Vec2::new(12.0, 10.0), egui::Align2::LEFT_TOP, "본인 오른쪽", font.clone(), muted);
                ui.painter().text(rect.right_top() + Vec2::new(-12.0, 10.0), egui::Align2::RIGHT_TOP, "본인 왼쪽", font, muted);
            }
            if let Some(i) = self.picked {
                let p = &self.parts[i];
                let mut text = scene::label_of(&p.name, p.kind);
                if let Some(v) = p.volume_mm3 {
                    text += &format!(" · 부피 {}mm³", v.round());
                }
                let galley = ui.painter().layout_no_wrap(text, egui::FontId::proportional(14.0), Color32::from_rgb(0x16, 0x23, 0x24));
                let at = egui::pos2(rect.center().x - galley.size().x / 2.0, rect.top() + 34.0);
                let bg_rect = egui::Rect::from_min_size(at, galley.size()).expand2(Vec2::new(10.0, 6.0));
                ui.painter().rect(bg_rect, 8.0, Color32::from_rgb(0xfa, 0xfb, 0xfa), egui::Stroke::new(1.0, Color32::from_rgb(0xcd, 0xd7, 0xd5)), egui::StrokeKind::Inside);
                ui.painter().galley(at, galley, Color32::BLACK);
            }
            let mut legend = ui.new_child(egui::UiBuilder::new().max_rect(egui::Rect::from_min_max(rect.left_bottom() + Vec2::new(12.0, -26.0), rect.right_bottom())));
            legend.horizontal(|ui| {
                for k in 0..KINDS.len() {
                    if self.parts.iter().any(|p| p.kind == k) {
                        swatch(ui, KINDS[k].color, KINDS[k].label);
                        ui.add_space(6.0);
                    }
                }
            });
        });

        // 확인용 화면 저장: 몇 장 그린 뒤 찍고, 받으면 저장하고 닫는다
        if self.screenshot.is_some() && self.frames == 10 {
            self.bench_start = Some(std::time::Instant::now());
        }
        if self.frames == SHOT_FRAME {
            if let Some(t) = self.bench_start {
                println!("평균 프레임 시간 {:.1}ms", t.elapsed().as_secs_f64() * 1000.0 / f64::from(SHOT_FRAME - 10));
            }
            if let Some(path) = self.screenshot.take() {
                let closing = self.closing.clone();
                ctx.request_screenshot(move |img| {
                    let [w, h] = img.size;
                    let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                    if let Some(buf) = image::RgbaImage::from_raw(w as u32, h as u32, bytes) {
                        let _ = buf.save(&path);
                    }
                    closing.store(true, std::sync::atomic::Ordering::Relaxed);
                });
            }
        }
        ctx.request_repaint();
    }
}

/// 웹: 페이지 주소의 ?glb=파일.glb&view=front&hide=jaw 를 읽어 그 GLB를 받아 보여 준다.
/// 한글 글꼴은 페이지 옆의 font.ttf를 받아 쓴다(없으면 한글이 네모로 나온다)
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;
    let query = web_sys::window().and_then(|w| w.location().search().ok()).unwrap_or_default();
    let param = |k: &str| query.trim_start_matches('?').split('&').find_map(|kv| kv.split_once('=').filter(|(a, _)| *a == k).map(|(_, v)| v.to_string()));
    let args = Args {
        view: param("view"),
        hide: param("hide").map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default(),
        bone: param("bone").and_then(|v| v.parse().ok()),
        cut: param("cut").and_then(|v| v.parse().ok()),
        pick: param("pick"),
        ..Default::default()
    };
    let glb = param("glb");
    wasm_bindgen_futures::spawn_local(async move {
        let canvas = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("ct3d"))
            .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("id가 ct3d인 canvas가 없습니다");
        let font = fetch_bytes("font.ttf").await.ok();
        let (tx, rx) = std::sync::mpsc::channel();
        let name = glb.clone().unwrap_or_default();
        let mut options = eframe::WebOptions::default();
        // 브라우저에 WebGPU가 있다고 나와도 쓸 수 있는 장치가 없으면 wgpu가 WebGL로 넘어가지 않으므로, 미리 확인해 WebGL만 쓰게 한다
        let webgpu = has_webgpu_adapter().await;
        web_sys::console::log_1(&format!("ct3d: WebGPU {}", if webgpu { "사용" } else { "없음, WebGL 사용" }).into());
        if !webgpu {
            if let eframe::egui_wgpu::WgpuSetup::CreateNew(c) = &mut options.wgpu_options.wgpu_setup {
                c.instance_descriptor.backends = eframe::wgpu::Backends::GL;
            }
        }
        let start = eframe::WebRunner::new()
            .start(
                canvas,
                options,
                Box::new(move |cc| {
                    let mut app = App::new(cc, args);
                    if let Some(bytes) = font {
                        set_korean_font(&cc.egui_ctx, bytes);
                    }
                    if !name.is_empty() {
                        app.file_name = name.rsplit('/').next().unwrap_or_default().to_string();
                        app.job = Some(rx);
                    }
                    Ok(Box::new(app))
                }),
            )
            .await;
        if let Err(e) = start {
            web_sys::console::error_1(&e);
            return;
        }
        if let Some(url) = glb {
            let info_url = match url.rsplit_once('/') {
                Some((dir, _)) => format!("{dir}/parts_info.json"),
                None => "parts_info.json".into(),
            };
            let info = match fetch_bytes(&info_url).await {
                Ok(b) => String::from_utf8(b).ok(),
                Err(_) => None,
            };
            let result = fetch_bytes(&url).await.and_then(|b| scene::load_glb_bytes(&b, info.as_deref()));
            let _ = tx.send(result);
        }
    });
}

/// navigator.gpu.requestAdapter()가 장치를 주는지
#[cfg(target_arch = "wasm32")]
async fn has_webgpu_adapter() -> bool {
    use eframe::wasm_bindgen::{JsCast, JsValue};
    use web_sys::js_sys::{Function, Promise, Reflect};
    let Some(w) = web_sys::window() else { return false };
    let Ok(gpu) = Reflect::get(&w.navigator(), &JsValue::from_str("gpu")) else { return false };
    if gpu.is_undefined() || gpu.is_null() {
        return false;
    }
    let Some(request) = Reflect::get(&gpu, &JsValue::from_str("requestAdapter")).ok().and_then(|f| f.dyn_into::<Function>().ok()) else { return false };
    let Some(promise) = request.call0(&gpu).ok().and_then(|p| p.dyn_into::<Promise>().ok()) else { return false };
    matches!(wasm_bindgen_futures::JsFuture::from(promise).await, Ok(a) if !a.is_null() && !a.is_undefined())
}

#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    use eframe::wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    let window = web_sys::window().ok_or("창이 없습니다")?;
    let resp = JsFuture::from(window.fetch_with_str(url)).await.map_err(|e| format!("{url}: {e:?}"))?;
    let resp: web_sys::Response = resp.dyn_into().map_err(|_| format!("{url}: 응답이 아닙니다"))?;
    if !resp.ok() {
        return Err(format!("{url}: {}", resp.status()));
    }
    let buf = JsFuture::from(resp.array_buffer().map_err(|e| format!("{e:?}"))?).await.map_err(|e| format!("{e:?}"))?;
    Ok(web_sys::js_sys::Uint8Array::new(&buf).to_vec())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let args = parse_args();
    if let Some(out) = args.model_js.clone() {
        if let Err(e) = model_js(&args, &out) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 900.0]).with_title("CT 3D 보기").with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("ct3d-viewer", options, Box::new(|cc| Ok(Box::new(App::new(cc, args)))))
}
