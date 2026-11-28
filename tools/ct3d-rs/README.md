# CT 3D 보기 (Rust·egui)

CT를 3D로 보는 데스크톱 뷰어입니다. [Rust·egui 전환 검토](../../docs/Rust-egui-전환-검토.md)의 1~4단계로 만들었고, 웹 뷰어(`viewer_template.html`)와 같은 조작을 합니다.

- **DICOM 폴더를 바로 열기:** 뼈와 치아 표면을 그 자리에서 만듭니다. [`volume_to_mesh.py`](../ct3d/volume_to_mesh.py)와 같은 순서이고, Python 없이 돌아갑니다.
- **AI 분할 결과로 구조별 메시:** DICOM 폴더와 함께 `segment_teeth.py`의 출력 폴더를 주면 치아·뿌리 조각·턱뼈 등을 구조별로 만듭니다. [`find_remnants.py`](../ct3d/find_remnants.py), [`labels_to_mesh.py`](../ct3d/labels_to_mesh.py)와 같은 순서입니다. AI 분할 자체는 Python으로 돌립니다.
- **치열 모형 데이터 만들기:** 같은 입력으로 치열 모형(`index.html`)이 읽는 `ct-teeth.js`를 만듭니다. [`teeth_for_model.py`](../ct3d/teeth_for_model.py)와 같은 순서입니다.
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
| `--drag 120,40` | 3D 화면 가운데에서 그만큼(점) 끈 것처럼 함(확인용) |
| `--wheel 3` | 휠을 그만큼(줄) 굴린 것처럼 함. 양수가 확대(확인용) |
| `--export 저장.glb` | DICOM에서 만든 메시를 GLB로 저장 |
| `--seg 분할폴더` | AI 분할 결과(`teeth_labels.nii.gz`, `segmentation_info.json`)로 구조별 메시를 만듦 |
| `--assign 1:47,2:46` | 뿌리 조각 후보 번호에 치아 번호를 붙임. 후보 목록은 콘솔에 나옴 |
| `--model-js ct-teeth.js` | 창을 열지 않고 치열 모형용 파일만 만듦. DICOM 폴더와 `--seg`가 필요함 |
| `--remap 42:41` | 분할 번호를 모형 번호로 바꿈(`--model-js`와 함께) |
| `--scan 스캔폴더` | 구강 스캔(이름에 UpperJaw, LowerJaw가 들어간 STL)을 CT 치아에 맞춤. `--seg`와 함께면 뷰어에 "구강 스캔"으로 겹치고, `--model-js`와 함께면 치아 머리를 스캔으로 다듬고 잇몸을 넣음 |
| `--ai-jaws` | 턱뼈를 CT 밝기로 다시 뽑지 않고 AI 결과를 그대로 씀 |
| `--screenshot 저장.png` | 화면을 저장하고 닫음(확인용). 평균 프레임 시간도 출력 |

## 웹 빌드

GLB를 보는 기능만 웹으로 빌드됩니다. DICOM 처리와 `--model-js`는 데스크톱에서만 됩니다.

Windows에서는 [`view_web.bat`](view_web.bat) 하나로 끝납니다. GLB 파일을 이 파일 위에 끌어다 놓거나 다음처럼 실행합니다.

```
view_web.bat <출력 폴더>\teeth_parts.glb "view=right&hide=jaw"
```

빌드에 필요한 도구가 없으면 설치하고, 빌드한 다음 페이지·GLB·글꼴을 `target\web-site`(git이 추적하지 않음)에 모읍니다. 이어서 이 PC에서만 접속되는 서버(127.0.0.1:8000)를 띄우고 브라우저를 엽니다. 서버 창을 닫으면 끝납니다. 첫 빌드는 몇 분 걸립니다. 직접 하려면 아래 순서를 따릅니다.

```
cargo install wasm-bindgen-cli --version 0.2.130   # Cargo.lock의 wasm-bindgen과 같은 버전
cargo build --profile web --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir <웹 폴더> target\wasm32-unknown-unknown\web\ct3d-viewer.wasm
```

`<웹 폴더>`에 [`web/index.html`](web/index.html), GLB, 한글 글꼴 `font.ttf`를 함께 두고 `index.html?glb=teeth_parts.glb&view=front&hide=jaw`처럼 엽니다. GLB 옆에 `parts_info.json`이 있으면 부피도 나옵니다. `bone`, `cut`, `pick`도 같은 이름으로 받습니다. GLB 파일을 창에 끌어다 놓아도 열립니다.

- WebGPU를 쓸 수 있으면 쓰고, 없으면 WebGL2로 그립니다.
- 크기는 wasm 7.1MB(gzip 2.4MB)입니다. 글꼴은 따로 받으므로 한글만 남긴 작은 글꼴을 쓰는 것이 좋습니다. 맑은 고딕은 다시 배포할 수 없습니다.

## 처리 순서

DICOM 폴더를 열면 단계별 진행과 걸린 시간을 콘솔과 화면에 보여 줍니다. 임계값(뼈 1000, 치아 2800, 법랑질 3300)은 Python 판과 같고, VATECH 콘빔 CT 한 건에 맞춘 값입니다.

## 구조

| 파일 | 내용 |
|---|---|
| `src/main.rs` | 창, 패널, 카메라 조작, 고르기 |
| `src/ctproc.rs` | DICOM 읽기, 축소·흐리기·덩어리 나누기·넓히기, 등가면(Surface Nets), Taubin, 삼각형 줄이기(meshopt), GLB 저장 |
| `src/seg.rs` | AI 분할 결과 읽기, 뿌리 조각 찾기, 구조별 메시, 턱뼈 전체 |
| `src/model.rs` | 치아마다 자기 좌표, 치아 목 높이, 머리·뿌리 자르기, 모형 좌표로 옮기기, 치열궁 맞추기, `ct-teeth.js` 쓰기 |
| `src/scan.rs` | STL 읽기, 가까운 점 찾기(칸 나누기), 정합(일부만 쓰는 ICP, Horn 사원수), 치아 머리를 스캔에 붙이기, 잇몸 가르기(표면을 따른 최단 거리) |
| `src/scene.rs` | GLB 읽기, 이름으로 종류와 한글 이름 정하기, 광선-삼각형 검사 |
| `src/render.rs` | wgpu로 별도 텍스처에 그리기(4배 MSAA, 깊이 버퍼, 불투명 먼저·반투명은 먼 것부터) |
| `src/shader.wgsl` | 조명, 양면 칠하기, 자르기 평면 |

- 3D 장면은 별도 텍스처에 그려 egui 이미지로 띄웁니다. egui는 텍스처 값을 감마(sRGB) 공간 값으로 그대로 쓰므로, 텍스처는 일반 형식으로 만들고 셰이더가 감마를 입힙니다.
- 한글은 시스템 글꼴(Windows는 맑은 고딕)을 읽어 씁니다. 웹에서는 페이지 옆의 `font.ttf`를 받아 씁니다.
- 메시 파일에는 환자의 형상이 들어 있습니다. 출력 폴더는 저장소 밖에 둡니다.
