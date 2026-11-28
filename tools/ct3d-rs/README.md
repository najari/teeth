# CT 3D 보기 (Rust·egui)

CT를 3D로 보는 데스크톱 뷰어입니다. [Rust·egui 전환 검토](../../docs/Rust-egui-전환-검토.md)의 1·2단계로 만들었고, 웹 뷰어(`viewer_template.html`)와 같은 조작을 합니다.

- **DICOM 폴더를 바로 열기:** 뼈와 치아 표면을 그 자리에서 만듭니다. [`volume_to_mesh.py`](../ct3d/volume_to_mesh.py)와 같은 순서이고, Python 없이 돌아갑니다.
- **GLB 파일 열기:** [`tools/ct3d`](../ct3d/)의 Python 스크립트가 만든 `ct_surface.glb`, `teeth_parts.glb`를 엽니다.

- 보는 방향 5가지, 끌어서 돌리기, 휠로 확대
- 종류별 켜고 끄기, 뼈 농도, 앞에서부터 자르기
- 구조를 누르면 한글 이름과 부피(`parts_info.json`이 옆에 있을 때)
- GLB 파일을 창에 끌어다 놓아 열기

## 빌드와 실행

Rust 1.95 이상이 필요합니다(eframe 0.37의 최소 버전). Windows(Rust 1.100, Intel Iris Xe)에서 만들고 확인했습니다.

```
cargo build --release
target\release\ct3d-viewer.exe <DICOM 폴더>
target\release\ct3d-viewer.exe <출력 폴더>\teeth_parts.glb
```

meshopt가 C++ 라이브러리를 빌드하므로 C++ 컴파일러(Windows는 Visual Studio의 C++ 도구)가 필요합니다.

| 인자 | 뜻 |
|---|---|
| `--view front\|right\|left\|top\|bottom` | 처음 보는 방향 |
| `--hide jaw,sinus` | 처음에 끌 종류 |
| `--bone 0..100`, `--cut 0..100` | 뼈 농도, 자르기 |
| `--pick 구조이름` | 처음에 고를 구조 |
| `--click 0.5,0.47` | 3D 화면의 그 비율 위치를 누른 것처럼 함(확인용) |
| `--export 저장.glb` | DICOM에서 만든 메시를 GLB로 저장 |
| `--screenshot 저장.png` | 화면을 저장하고 닫음(확인용). 평균 프레임 시간도 출력 |

DICOM 폴더를 열면 단계별 진행과 걸린 시간을 콘솔과 화면에 보여 줍니다. 임계값(뼈 1000, 치아 2800, 법랑질 3300)은 Python 판과 같고, VATECH 콘빔 CT 한 건에 맞춘 값입니다.

## 구조

| 파일 | 내용 |
|---|---|
| `src/main.rs` | 창, 패널, 카메라 조작, 고르기 |
| `src/ctproc.rs` | DICOM 읽기, 축소·흐리기·덩어리 나누기·넓히기, 등가면(Surface Nets), Taubin, 삼각형 줄이기(meshopt), GLB 저장 |
| `src/scene.rs` | GLB 읽기, 이름으로 종류와 한글 이름 정하기, 광선-삼각형 검사 |
| `src/render.rs` | wgpu로 별도 텍스처에 그리기(4배 MSAA, 깊이 버퍼, 불투명 먼저·반투명은 먼 것부터) |
| `src/shader.wgsl` | 조명, 양면 칠하기, 자르기 평면 |

- 3D 장면은 별도 텍스처에 그려 egui 이미지로 띄웁니다. egui는 텍스처 값을 감마(sRGB) 공간 값으로 그대로 쓰므로, 텍스처는 일반 형식으로 만들고 셰이더가 감마를 입힙니다.
- 한글은 시스템 글꼴(Windows는 맑은 고딕)을 읽어 씁니다. 웹 빌드에서는 글꼴 파일을 함께 넣어야 합니다.
- 메시 파일에는 환자의 형상이 들어 있습니다. 출력 폴더는 저장소 밖에 둡니다.
