// 메시 하나를 그리는 셰이더. 양면을 같은 밝기로 칠하고, 자르기 평면 앞쪽은 버린다.

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,        // 카메라 위치
    light_dir: vec4<f32>,  // 빛이 오는 방향 (카메라에 붙어 있음)
    cut: vec4<f32>,        // x: 이 z보다 앞은 버림, y: 1이면 자르기 켜짐
};

struct Part {
    color: vec4<f32>,      // 선형 색, a는 불투명도
    highlight: vec4<f32>,  // 고른 구조의 색, a는 섞는 비율
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(1) @binding(0) var<uniform> part: Part;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) nrm: vec3<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    o.clip = g.view_proj * vec4<f32>(v.pos, 1.0);
    o.world = v.pos;
    o.nrm = v.nrm;
    return o;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    if (g.cut.y > 0.5 && i.world.z > g.cut.x) {
        discard;
    }
    var n = normalize(i.nrm);
    let to_eye = normalize(g.eye.xyz - i.world);
    if (dot(n, to_eye) < 0.0) {
        n = -n;                     // 안쪽 면도 바깥처럼 칠한다
    }
    let l = normalize(g.light_dir.xyz);
    let diffuse = max(dot(n, l), 0.0);
    let sky = mix(vec3<f32>(0.33, 0.37, 0.38), vec3<f32>(1.0, 1.0, 1.0), n.y * 0.5 + 0.5);
    let h = normalize(l + to_eye);
    let spec = pow(max(dot(n, h), 0.0), 40.0) * 0.12;
    let base = mix(part.color.rgb, part.highlight.rgb, part.highlight.a);
    let rgb = base * (0.42 * sky + 0.62 * diffuse) + vec3<f32>(spec, spec, spec);
    // 조명은 선형 공간에서 계산하고, 내보낼 때 sRGB 감마를 입힌다
    return vec4<f32>(linear_to_srgb(rgb), part.color.a);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}
