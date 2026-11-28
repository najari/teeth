//! 3D 장면을 별도 텍스처에 그린다. 그 텍스처를 egui 이미지로 화면에 띄운다.
//! 깊이 버퍼와 안티에일리어싱(4배 MSAA)을 이 모듈이 직접 관리한다.

use crate::scene::{Part, KINDS};
use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu, wgpu};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

// egui는 텍스처 값을 감마(sRGB) 공간 값으로 그대로 화면에 낸다. 그래서 sRGB 형식 대신
// 일반 형식에 셰이더가 감마를 입힌 값을 쓴다(sRGB 형식을 쓰면 화면이 어둡게 나온다).
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SAMPLES: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    light_dir: [f32; 4],
    cut: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PartUniform {
    color: [f32; 4],
    highlight: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nrm: [f32; 3],
}

struct GpuPart {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    count: u32,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

struct Targets {
    size: [u32; 2],
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolve: wgpu::Texture,
    resolve_view: wgpu::TextureView,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    opaque: wgpu::RenderPipeline,
    blended: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    part_layout: wgpu::BindGroupLayout,
    parts: Vec<GpuPart>,
    targets: Option<Targets>,
    pub texture_id: Option<egui::TextureId>,
}

/// 그릴 때 필요한 값
pub struct Frame<'a> {
    pub view_proj: Mat4,
    pub eye: Vec3,
    pub light_dir: Vec3,
    pub cut_z: Option<f32>,
    pub visible: &'a [bool],
    pub soft_opacity: f32,
    pub picked: Option<usize>,
    pub background: [f64; 3],
}

pub fn srgb_to_linear(c: [u8; 3]) -> [f32; 3] {
    c.map(|v| {
        let x = v as f32 / 255.0;
        if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
    })
}

impl Renderer {
    pub fn new(rs: &egui_wgpu::RenderState) -> Self {
        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ct3d shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let uniform_entry = |vis| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let global_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[uniform_entry(wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        let part_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("part"),
            entries: &[uniform_entry(wgpu::ShaderStages::FRAGMENT)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ct3d layout"),
            bind_group_layouts: &[Some(&global_layout), Some(&part_layout)],
            immediate_size: 0,
        });
        let make = |blend: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if blend { "blended" } else { "opaque" }),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(if blend { wgpu::BlendState::ALPHA_BLENDING } else { wgpu::BlendState::REPLACE }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(!blend), // 반투명은 깊이를 쓰지 않아 뒤에 있는 것이 비친다
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let opaque = make(false);
        let blended = make(true);
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &global_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }],
        });
        Self { device, queue, opaque, blended, globals, globals_bind, part_layout, parts: Vec::new(), targets: None, texture_id: None }
    }

    /// 메시를 GPU로 올린다. 예전 메시는 버린다.
    pub fn upload(&mut self, parts: &[Part]) {
        self.parts = parts
            .iter()
            .map(|p| {
                let verts: Vec<Vertex> = p.positions.iter().zip(&p.normals).map(|(a, n)| Vertex { pos: a.to_array(), nrm: n.to_array() }).collect();
                let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&p.name),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&p.name),
                    contents: bytemuck::cast_slice(&p.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&p.name),
                    size: std::mem::size_of::<PartUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&p.name),
                    layout: &self.part_layout,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
                });
                GpuPart { vbuf, ibuf, count: p.indices.len() as u32, ubuf, bind }
            })
            .collect();
    }

    fn ensure_targets(&mut self, size: [u32; 2], egui_renderer: &mut egui_wgpu::Renderer) {
        if self.targets.as_ref().map_or(false, |t| t.size == size) {
            return;
        }
        let extent = wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 };
        let make = |format, samples, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ct3d target"),
                size: extent,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa = make(COLOR_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let depth = make(DEPTH_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let resolve = make(COLOR_FORMAT, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let resolve_view = resolve.create_view(&Default::default());
        match self.texture_id {
            Some(id) => egui_renderer.update_egui_texture_from_wgpu_texture(&self.device, &resolve_view, wgpu::FilterMode::Linear, id),
            None => self.texture_id = Some(egui_renderer.register_native_texture(&self.device, &resolve_view, wgpu::FilterMode::Linear)),
        }
        self.targets = Some(Targets { size, msaa, depth, resolve, resolve_view });
    }

    /// 장면을 텍스처에 그린다. 불투명한 것을 먼저, 반투명한 것은 먼 것부터 그린다.
    pub fn draw(&mut self, size: [u32; 2], rs: &egui_wgpu::RenderState, parts: &[Part], f: &Frame) {
        self.ensure_targets(size, &mut rs.renderer.write());
        let g = Globals {
            view_proj: f.view_proj.to_cols_array_2d(),
            eye: f.eye.extend(1.0).to_array(),
            light_dir: f.light_dir.extend(0.0).to_array(),
            cut: [f.cut_z.unwrap_or(0.0), if f.cut_z.is_some() { 1.0 } else { 0.0 }, 0.0, 0.0],
        };
        self.queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&g));
        let pick = srgb_to_linear([0x1f, 0xa6, 0x97]);
        for (i, (p, gp)) in parts.iter().zip(&self.parts).enumerate() {
            let k = &KINDS[p.kind];
            let c = srgb_to_linear(k.color);
            let alpha = if k.soft { f.soft_opacity } else { 1.0 };
            let u = PartUniform {
                color: [c[0], c[1], c[2], alpha],
                highlight: [pick[0], pick[1], pick[2], if f.picked == Some(i) { 1.0 } else { 0.0 }],
            };
            self.queue.write_buffer(&gp.ubuf, 0, bytemuck::bytes_of(&u));
        }

        let (mut solid, mut glass): (Vec<usize>, Vec<usize>) = (0..parts.len())
            .filter(|&i| f.visible[i])
            .partition(|&i| !KINDS[parts[i].kind].soft || f.soft_opacity >= 1.0);
        solid.sort_by_key(|&i| parts[i].kind);
        glass.sort_by(|&a, &b| {
            let (da, db) = (parts[a].centre().distance(f.eye), parts[b].centre().distance(f.eye));
            db.total_cmp(&da)
        });

        let t = self.targets.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ct3d") });
        {
            let bg = f.background;
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ct3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.msaa,
                    resolve_target: Some(&t.resolve_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0], g: bg[1], b: bg[2], a: 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.globals_bind, &[]);
            for (list, pipe) in [(&solid, &self.opaque), (&glass, &self.blended)] {
                pass.set_pipeline(pipe);
                for &i in list.iter() {
                    let gp = &self.parts[i];
                    pass.set_bind_group(1, &gp.bind, &[]);
                    pass.set_vertex_buffer(0, gp.vbuf.slice(..));
                    pass.set_index_buffer(gp.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..gp.count, 0, 0..1);
                }
            }
        }
        self.queue.submit([enc.finish()]);
        let _ = &t.resolve; // 텍스처를 살려 둔다
    }
}

/// 투영 행렬과 카메라 위치. 화면 비율이 세로로 길면 가로가 잘리지 않게 물러난다.
pub fn camera(target: Vec3, yaw: f32, pitch: f32, dist: f32, aspect: f32) -> (Mat4, Vec3, Vec3) {
    let dir = Vec3::new(pitch.cos() * yaw.sin(), pitch.sin(), pitch.cos() * yaw.cos());
    let eye = target + dir * dist;
    let view = look_at(eye, target, Vec3::Y);
    let proj = perspective(28f32.to_radians(), aspect, dist / 40.0, dist * 4.0);
    // 빛은 카메라에 붙어 왼쪽 위 앞에서 비춘다
    let right = Vec3::Y.cross(dir).normalize_or(Vec3::X);
    let up = dir.cross(right);
    let light = (-0.4 * right + 0.6 * up + dir).normalize();
    (proj * view, eye, light)
}

/// 오른손 좌표계의 시점 행렬 (카메라가 -z를 본다)
fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize_or(Vec3::X);
    let u = s.cross(f);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// 오른손 좌표계의 원근 투영. wgpu의 깊이 범위(0..1)에 맞춘다.
fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let h = 1.0 / (fov_y * 0.5).tan();
    let r = far / (near - far);
    Mat4::from_cols(
        Vec4::new(h / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, h, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, -1.0),
        Vec4::new(0.0, 0.0, r * near, 0.0),
    )
}

/// 화면 좌표(-1..1)를 월드 광선으로 바꾼다.
pub fn unproject(view_proj: Mat4, ndc_x: f32, ndc_y: f32) -> (Vec3, Vec3) {
    let inv = view_proj.inverse();
    let near = inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let (a, b) = (near.truncate() / near.w, far.truncate() / far.w);
    (a, (b - a).normalize())
}
