//! CT 3D 보기 (Rust·egui 판)
//!
//! tools/ct3d의 Python 스크립트가 만든 GLB(ct_surface.glb, teeth_parts.glb)를 연다.
//! 웹 뷰어(viewer_template.html)와 같은 조작을 한다: 보는 방향, 종류별 켜고 끄기, 뼈 농도, 자르기, 누르면 이름.
//!
//! 사용법:
//!     ct3d-viewer [파일.glb | DICOM 폴더] [--view front|right|left|top|bottom] [--hide jaw,sinus]
//!                 [--bone 0..100] [--cut 0..100] [--pick 구조이름] [--export 저장.glb] [--screenshot 저장.png]
//!                 [--seg 분할폴더 [--assign 1:47,2:46] [--ai-jaws]]
//! DICOM 폴더를 주면 뼈와 치아 표면을 그 자리에서 만든다(tools/ct3d/volume_to_mesh.py와 같은 순서).
//! --seg로 segment_teeth.py의 출력 폴더를 함께 주면 구조별 메시를 만든다(find_remnants.py, labels_to_mesh.py와 같은 순서).
//! 파일이나 폴더는 창에 끌어다 놓아도 열린다. --screenshot을 주면 화면을 PNG로 저장하고 닫는다(확인용).

mod ctproc;
mod render;
mod scene;
mod seg;

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
    export: Option<PathBuf>,   // DICOM에서 만든 메시를 GLB로 저장
    seg: Option<PathBuf>,      // segment_teeth.py의 출력 폴더
    assign: std::collections::HashMap<usize, u32>, // 뿌리 조각 후보 번호 → FDI 번호
    ai_jaws: bool,
}

#[derive(Clone)]
struct SegOptions {
    dir: PathBuf,
    assign: std::collections::HashMap<usize, u32>,
    ai_jaws: bool,
}

type Job = std::sync::mpsc::Receiver<Result<Vec<Part>, String>>;

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
            "--assign" => {
                a.assign = it
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter_map(|kv| {
                        let (k, v) = kv.split_once(':')?;
                        Some((k.trim().parse().ok()?, v.trim().parse().ok()?))
                    })
                    .collect()
            }
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
            export: args.export,
            seg: args.seg.map(|dir| SegOptions { dir, assign: args.assign, ai_jaws: args.ai_jaws }),
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
fn segmented_meshes(dicom: &std::path::Path, o: &SegOptions, log: ctproc::Log) -> Result<Vec<Part>, String> {
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
    Ok(seg::part_meshes(&ct, &s, &cands, o.ai_jaws, log))
}

fn install_korean_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    // 데스크톱에서는 시스템 글꼴을 쓴다. 웹 빌드에서는 글꼴 파일을 함께 넣어야 한다.
    for path in [r"C:\Windows\Fonts\malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert("korean".into(), std::sync::Arc::new(egui::FontData::from_owned(bytes)));
            for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.entry(fam).or_default().insert(0, "korean".into());
            }
            break;
        }
    }
    ctx.set_fonts(fonts);
}

fn swatch(ui: &mut egui::Ui, color: [u8; 3], text: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(11.0), Sense::hover());
    ui.painter().rect_filled(rect, 3.0, Color32::from_rgb(color[0], color[1], color[2]));
    ui.label(RichText::new(text).size(12.0).color(Color32::from_rgb(0x55, 0x67, 0x6a)));
}

impl eframe::App for App {
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
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if let Some(p) = dropped.into_iter().next() {
            self.open(p);
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

fn main() -> eframe::Result {
    let args = parse_args();
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 900.0]).with_title("CT 3D 보기").with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("ct3d-viewer", options, Box::new(|cc| Ok(Box::new(App::new(cc, args)))))
}
