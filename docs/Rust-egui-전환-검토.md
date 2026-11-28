# Rust·egui 전환 검토

작성일: 2026-11-28

지금까지 Python 스크립트와 웹 페이지(three.js)로 만든 두 가지를 Rust와 egui로 다시 구현할 수 있는지 검토한 문서입니다.

- **CT 뷰어:** CT에서 뼈·치아·신경관을 뽑아 돌려 보는 화면
- **치열 모형:** 표준 치열에 CT 치아 모양과 자리를 올려 치료 단계를 보여 주는 화면

**조사 범위**

- 라이브러리 버전과 라이선스는 2026-11-28에 docs.rs와 GitHub에서 확인한 값입니다.
- 가장 큰 위험인 DICOM 읽기와 표면 추출은 이 PC에서 Rust 시험 프로그램을 만들어 실제 CT로 돌려 봤습니다(4장).
- egui 화면은 1단계 뷰어를 만들어 확인했습니다(8장). DICOM에서 임계값 메시를 만드는 2단계(9장)와, AI 분할 결과로 구조별 메시를 만드는 3단계(10장), 치열 모형용 `ct-teeth.js`를 만드는 4단계(11장)도 Rust로 옮겨 Python 결과와 대조했습니다. 웹 빌드는 해 보지 않았습니다.

---

## 1. 결론

1. **가능합니다.** AI 자동 분할을 뺀 모든 단계에 쓸 수 있는 Rust 라이브러리가 있습니다. 시험 프로그램에서 이 CT의 무손실 JPEG 425장을 1.8초에 풀었고, 화소 값이 Python 결과와 정확히 같았습니다.
2. **AI 자동 분할은 당분간 Python으로 둡니다.** TotalSegmentator `teeth` 모델의 ONNX 변환본이 공개돼 있지 않고, nnU-Net의 추론 절차(리샘플링, 슬라이딩 윈도, 후처리)를 Rust로 다시 짜야 합니다. Rust 앱이 Python을 별도 프로세스로 부르는 구조가 현실적입니다.
3. **얻는 것:** 설치 없이 실행되는 프로그램 하나, CT 원본을 직접 열어 그 자리에서 처리하는 속도, 웹과 같은 코드로 데스크톱 앱까지 만들 수 있는 점입니다.
4. **치르는 것:** AI 분할을 뺀 지금 코드 약 2,100줄을 다시 짜야 합니다. three.js가 해 주던 재질, 투명도 정렬, 자르기 평면, 클릭 선택, glTF 읽기를 직접 만들거나 3D 크레이트에 맡겨야 합니다. 한글 글꼴도 직접 넣어야 합니다.
5. **권장 순서:** CT 뷰어를 먼저 데스크톱 앱으로 옮기고, 치열 모형은 웹 페이지를 유지합니다. 치열 모형을 옮기는 것은 CT 뷰어가 자리를 잡은 뒤에 판단합니다(7장).

---

## 2. 지금 구현의 구성

| 구분 | 파일 | 줄 수 | 하는 일 | 쓰는 라이브러리 |
|---|---|---|---|---|
| 변환 | `dicom_to_volume.py` | 63 | DICOM 425장을 볼륨 하나로 | SimpleITK |
| 임계값 메시 | `volume_to_mesh.py` | 151 | 흐리기, 등가면, 연결 성분, 매끄럽게, 삼각형 줄이기 | SimpleITK, VTK(PyVista) |
| AI 분할 | `segment_teeth.py`, `setup_segmentation.py` | 122 | 치아 번호별 분할 | TotalSegmentator, PyTorch |
| 뿌리 조각 | `find_remnants.py` | 144 | 뼈에 붙지 않은 밝은 덩어리 찾기 | scipy.ndimage |
| 구조별 메시 | `labels_to_mesh.py` | 288 | 구조마다 메시, 턱뼈 전체 | SimpleITK, VTK, scipy |
| 모형 데이터 | `teeth_for_model.py` | 279 | 치아별 좌표축, 교합면 맞추기, 치열궁 맞추기 | numpy, scipy.optimize |
| CT 뷰어 | `viewer_template.html`, `build_viewer.py` | 467 | 메시 보기, 켜고 끄기, 자르기, 클릭 | three.js r128 |
| 치열 모형 | `index.html` 스크립트 | 733 | 표준 치아 생성, 잇몸·뼈 띠, 치료 단계 | three.js r128 |

AI 분할을 빼면 옮길 코드는 Python 약 940줄과 JavaScript 약 1,200줄입니다.

---

## 3. 단계별 Rust 대응

| 단계 | Rust 크레이트 | 버전, 라이선스 | 판단 |
|---|---|---|---|
| DICOM 읽기 | [dicom-object](https://docs.rs/crate/dicom-object/latest), [dicom-pixeldata](https://docs.rs/crate/dicom-pixeldata/latest) | 0.10.1, MIT/Apache-2.0 | **실측으로 확인.** 무손실 JPEG(.70)를 순수 Rust `jpeg-decoder`로 풂 |
| NIfTI 읽기·쓰기 | [nifti](https://docs.rs/crate/nifti/latest) | 0.18.0, MIT/Apache-2.0 | `.nii.gz`를 ndarray로 읽음. AI 분할 결과를 받을 때 필요 |
| 흐리기, 연결 성분, 팽창, 구멍 메우기 | [ndarray-ndimage](https://github.com/imeka/ndarray-ndimage) | 0.6.0 | `gaussian_filter`, `label`, `binary_dilation` 있음. 일부 함수는 3D·자료형 제한이 있어 확인 필요 |
| 등가면 추출 | [fast-surface-nets](https://docs.rs/fast-surface-nets), [mcubes](https://docs.rs/crate/mcubes/latest), [isosurface](https://docs.rs/crate/isosurface/latest) | 0.2.1(MIT/Apache-2.0) 등 | **실측으로 확인.** Surface Nets로 뼈 표면 0.55초 |
| 삼각형 줄이기 | [meshopt](https://docs.rs/crate/meshopt/latest) | 0.7.0, MIT/Apache-2.0 | C++ meshoptimizer를 감싼 것이라 빌드에 C++ 컴파일러 필요 |
| 매끄럽게(Taubin) | 없음 | | 직접 구현. 이웃 꼭짓점 평균을 두 번 번갈아 하는 짧은 코드 |
| 주축, 평면 맞추기 | [nalgebra](https://nalgebra.org/) | | SVD로 numpy와 같은 계산 |
| 치열궁 맞추기 | [argmin](https://argmin-rs.org/) 또는 직접 | | Nelder-Mead 있음. 변수 3개라 직접 짜도 됨 |
| glTF 읽기·쓰기 | [gltf](https://docs.rs/crate/gltf/latest) | 1.4.1, MIT/Apache-2.0 | 지금 만든 GLB 파일을 그대로 읽을 수 있음 |
| AI 분할 | [ort](https://docs.rs/crate/ort/latest)(ONNX Runtime) | 2.0.0-rc, MIT/Apache-2.0 | 모델의 ONNX 변환과 nnU-Net 추론 절차를 직접 옮겨야 함(5장) |
| 화면 | [eframe/egui](https://github.com/emilk/egui) | 0.36, MIT/Apache-2.0 | wgpu가 기본 렌더러. 웹에서는 WebGPU, 안 되면 WebGL |
| 3D 그리기 | `egui_wgpu` 콜백 + WGSL 셰이더 | | egui 공식 예제(`custom3d_wgpu`)의 방식. 재질과 조명은 직접 작성 |

SimpleITK와 VTK는 Rust에 같은 수준의 대체재가 없습니다. 이번 작업에 쓴 기능(가우스 흐리기, 연결 성분, 팽창, 구멍 메우기, 등가면)은 위 크레이트와 짧은 직접 구현으로 채울 수 있습니다.

---

## 4. 시험 결과

`dicom-object`, `dicom-pixeldata`, `fast-surface-nets`로 약 70줄짜리 시험 프로그램을 만들어 이 PC(i5-1240P, RAM 16GB)에서 실행했습니다. Rust 1.100.0, 릴리스 빌드입니다.

| 항목 | 결과 |
|---|---|
| 첫 빌드 | 57초(크레이트 내려받기와 컴파일 포함) |
| 전송 구문 | 1.2.840.10008.1.2.4.70(무손실 JPEG)으로 읽힘 |
| 425장 디코딩 | 1.83초(16스레드 병렬) |
| 화소 값 범위 | -1000~6127. Python(SimpleITK) 결과와 같음 |
| 뼈 표면(0.4mm, 값 1000) | 0.55초, 삼각형 157만 개 |

- Python 쪽은 임계값 메시 단계 전체에 43초 걸렸습니다. 이 단계에는 흐리기, 연결 성분, 매끄럽게, 줄이기까지 들어 있어 바로 비교할 수는 없습니다. 읽기와 등가면 추출만 놓고 보면 Rust 쪽이 충분히 빠릅니다.
- 삼각형 157만 개는 줄이기 전 값입니다. 화면에 올리려면 meshopt로 20만 개 안팎까지 줄여야 합니다. 이 단계는 시험하지 않았습니다.
- C++ 컴파일러(Visual Studio 2022의 C++ 도구)는 이 PC에 설치돼 있어 meshopt 빌드에 문제는 없어 보입니다. 시험하지 않았습니다.

---

## 5. AI 자동 분할

지금 쓰는 TotalSegmentator `teeth` 작업은 nnU-Net 모델 3개를 잇달아 돌립니다(머리뼈 범위 찾기, 치아 덩어리 찾기, 치아 번호별 분할). Rust로 옮기는 길은 두 가지입니다.

| 방법 | 내용 | 판단 |
|---|---|---|
| A. Python을 별도 프로세스로 호출 | Rust 앱이 `segment_teeth.py`를 실행하고 결과 NIfTI를 읽음 | **권장.** 지금 검증된 결과를 그대로 씀 |
| B. ONNX로 바꿔 Rust에서 추론 | 모델을 ONNX로 바꾸고 `ort`로 추론 | 큰 작업. 아래 이유 |

B가 어려운 이유는 다음과 같습니다.

- TotalSegmentator의 `total` 작업은 ONNX 변환본이 공개돼 있고([TS-Portable](https://huggingface.co/sigjhl/ts-portable)), 원본과의 차이가 평균 Dice 0.00012 이내라고 밝히고 있습니다. `teeth` 작업의 변환본은 찾지 못했습니다. 직접 변환해야 합니다.
- ONNX 파일은 신경망 하나만 담습니다. 리샘플링, 패치를 겹쳐 가며 돌리는 슬라이딩 윈도와 가우스 가중치, 세 모델을 잇는 자르기, 후처리는 Rust로 다시 짜야 합니다.
- 결과가 Python과 같은지 확인하는 검증 작업이 따로 필요합니다.
- `ort`는 CUDA와 DirectML(Windows GPU)을 지원합니다. 장치가 없으면 CPU로 조용히 넘어갈 수 있어, 3D 모델이 예상보다 훨씬 느려질 수 있습니다.
- 모델 가중치의 라이선스를 따로 확인해야 합니다. 위 `total_fast` 변환본 하나는 가중치를 비상업용(CC-BY-NC-4.0)으로 공개했습니다.

---

## 6. 화면(egui)

### 6.1 구조

- **eframe**이 창, 입력, 렌더러(wgpu)를 맡고, **egui**로 버튼, 슬라이더, 범례, 정보 상자를 그립니다. 지금 화면의 패널 구성을 그대로 옮길 수 있습니다.
- 3D 장면은 egui 공식 예제 `custom3d_wgpu`의 방식으로 그립니다. 화면 영역을 잡고 `egui_wgpu::Callback::new_paint_callback`에 `CallbackTrait`(준비 `prepare`, 그리기 `paint`)을 구현한 객체를 넘깁니다. 파이프라인과 버퍼는 `callback_resources`에 둡니다.
- 대안으로 [three-d](https://docs.rs/crate/three-d/latest)나 Bevy(+bevy_egui) 같은 3D 크레이트를 쓸 수 있습니다. three-d의 egui 연결 크레이트는 egui 0.26에 묶여 있어 최신 egui와 버전이 맞지 않습니다.

### 6.2 three.js가 해 주던 것 가운데 직접 만들어야 하는 것

| 기능 | 지금 | Rust에서 |
|---|---|---|
| 조명과 재질 | `MeshStandardMaterial` | WGSL 셰이더에 간단한 조명 작성 |
| 반투명 뼈 | 투명도와 그리기 순서 | 불투명 먼저, 반투명은 깊이 쓰기 끄고 나중에 |
| 자르기 | `clippingPlanes` | 셰이더에서 평면 뒤 조각 버리기 |
| 돌리기·확대 | `OrbitControls` | egui 끌기·휠 입력으로 카메라 계산 |
| 치아 누르기 | `Raycaster` | CPU에서 광선-삼각형 검사(메시당 경계 상자로 먼저 거름) |
| 메시 읽기 | `GLTFLoader` | `gltf` 크레이트 |
| 표준 치아 생성 | `SphereGeometry`를 변형 | 구 격자를 직접 만들어 같은 함수로 변형 |

### 6.3 주의할 점

- **한글 글꼴:** egui 기본 글꼴에는 한글이 없습니다. 글꼴 파일을 프로그램에 넣어 시작할 때 등록해야 합니다. 데스크톱은 시스템 글꼴(맑은 고딕)을 쓸 수 있지만, 웹에서는 시스템 글꼴을 읽지 못해 글꼴을 함께 내려보내야 합니다. Noto Sans KR(SIL OFL 1.1)을 한글 범위만 남겨 줄이는 방법이 있습니다.
- **웹 빌드:** eframe은 WebAssembly로 빌드되고 GitHub Pages에 올릴 수 있습니다. 다만 wasm 파일, 글꼴, 메시를 합친 첫 로딩이 지금 페이지(약 50KB + 0.9MB)보다 무거워집니다. 크기는 재 보지 않았습니다.
- **웹에서 CT 처리:** `dicom-pixeldata`는 wasm에서도 돌아간다고 문서에 적혀 있습니다(병렬 처리 기능을 끄거나 따로 설정). 브라우저에서 CT를 직접 여는 것도 가능하지만, 의료 데이터를 웹에 올리는 구조가 되므로 권하지 않습니다.
- **내장 그래픽:** 이 PC는 Intel Iris Xe입니다. wgpu는 DirectX 12와 Vulkan을 지원해 문제는 없어 보이지만 실행해 보지 않았습니다.

---

## 7. 권장 구성과 순서

### 7.1 구성

```
Rust 앱 (eframe/egui, 데스크톱)
├─ DICOM 폴더 열기        dicom-rs                (Rust)
├─ 임계값 메시            ndarray-ndimage, fast-surface-nets, meshopt (Rust)
├─ 뿌리 조각 찾기          ndarray-ndimage        (Rust)
├─ AI 분할               segment_teeth.py 호출   (Python, 별도 프로세스)
├─ 구조별 메시·모형 데이터  nalgebra 등            (Rust)
└─ 3D 화면               egui_wgpu + WGSL       (Rust)

웹 페이지 (index.html, three.js)  ← 당분간 유지, Rust 앱이 만든 ct-teeth.js를 읽음
```

### 7.2 순서

| 단계 | 내용 | 끝났다고 보는 기준 |
|---|---|---|
| 1 | 3D 화면 뼈대: GLB 메시를 egui 창에 띄우고 돌리기, 켜고 끄기, 자르기, 누르기 | 지금 만든 `teeth_parts.glb`가 CT 뷰어와 같게 보임 |
| 2 | DICOM 열기와 임계값 메시를 Rust로 | Python의 `ct_surface.glb`와 삼각형 수·모양이 비슷함 |
| 3 | AI 분할 결과 읽기, 뿌리 조각, 구조별 메시를 Rust로 | Python 결과와 구조 수·부피가 같음 |
| 4 | `ct-teeth.js` 만들기를 Rust로 | Python이 만든 파일과 자리·축이 소수점 둘째 자리까지 같음 |
| 5 | 치열 모형을 egui로 옮길지 판단 | 웹 공개가 계속 필요하면 웹 페이지 유지 |
| 6 | AI 분할을 ONNX로 옮길지 판단 | Python 설치 없이 돌려야 할 이유가 생기면 |

1단계를 먼저 하는 이유는 화면 쪽 위험(글꼴, 투명도, 선택, 내장 그래픽)이 가장 크고 아직 확인하지 않았기 때문입니다. 2~4단계는 Python 결과와 숫자로 비교할 수 있어 검증이 쉽습니다.

### 7.3 치열 모형을 웹으로 두는 이유

- 지금 공개 사이트로 누구나 링크만으로 볼 수 있습니다. egui로 옮기면 같은 일을 wasm 빌드로 해야 하고, 로딩이 무거워집니다.
- 치열 모형은 CT를 직접 다루지 않고 `ct-teeth.js` 하나만 읽습니다. Rust 앱이 이 파일만 만들어 주면 됩니다.
- 표준 치아 생성, 잇몸·뼈 띠, 치료 단계 전환은 three.js 위에서 이미 동작하고 확인돼 있습니다.

---

## 8. 1단계 결과 (2026-11-28)

7.2의 1단계로 GLB 메시를 egui 창에 띄우는 뷰어를 만들었습니다([`tools/ct3d-rs`](../tools/ct3d-rs/)). Rust 약 1,000줄(셰이더 56줄 포함)이고, 웹 CT 뷰어와 같은 조작을 합니다. 웹 뷰어는 약 450줄이었습니다. three.js가 해 주던 그리기, 카메라, 고르기를 직접 짠 만큼 늘었습니다.

| 확인한 것 | 결과 |
|---|---|
| 빌드 | eframe 0.37, wgpu 30. 첫 빌드 약 3분, 실행 파일 15MB |
| 3D 그리기 | 별도 텍스처에 4배 MSAA로 그려 egui 이미지로 띄움. 이 PC(Intel Iris Xe)에서 동작 |
| 프레임 시간 | 약 11ms(구조 45개, 삼각형 44만 개, 1280×900 창) |
| 한글 | 맑은 고딕을 읽어 제목, 버튼, 범례, 구조 이름 모두 표시 |
| 반투명 턱뼈 | 불투명 먼저, 반투명은 먼 것부터 그려 치아가 비쳐 보임 |
| 자르기 | 셰이더에서 평면 앞쪽을 버려 동작 |
| 누르기 | 화면 위치를 누른 것처럼 흉내 내 확인. 위 앞니 쪽에서 11번, 턱 아래쪽에서 아래턱뼈가 골라짐 |
| 두 종류 GLB | `teeth_parts.glb`, `ct_surface.glb` 모두 열림 |

**만들면서 알게 된 것**

- **최신 API로 바뀐 곳이 많았습니다.** eframe 0.37은 `App::update` 대신 `App::ui`를 쓰고, 패널이 `egui::Panel::top` 등으로 바뀌었습니다. wgpu 30은 파이프라인 설정의 여러 항목이 `Option`이 됐습니다. 인터넷에 있는 예제 대부분이 옛 API라 그대로는 컴파일되지 않습니다.
- **glam 0.34에는 카메라 행렬 함수(`look_at_rh`, `perspective_rh`)가 없습니다.** 20줄 정도로 직접 작성했습니다.
- **색이 어둡게 나오는 문제가 있었습니다.** egui는 텍스처 값을 감마 공간 값으로 그대로 쓰므로, sRGB 형식 텍스처에 그리면 한 번 더 어두워집니다. 일반 형식 텍스처에 셰이더가 감마를 입혀 해결했고, 배경 픽셀 값이 지정한 색과 정확히 같아졌습니다.
- **확인용 장치를 넣었습니다.** `--screenshot`으로 화면을 저장하고 닫는 기능을 넣어, 사람이 없어도 화면을 확인할 수 있습니다.

**아직 확인하지 못한 것**

- 실제 마우스로 끌어 돌리기와 휠 확대는 코드만 있고, 자동 확인은 하지 않았습니다. 이 PC에서 직접 실행해 보면 됩니다.
- 웹(wasm) 빌드는 해 보지 않았습니다.

## 9. 2단계 결과 (2026-11-28)

뷰어가 DICOM 폴더를 직접 열어 뼈와 치아 표면을 만들게 했습니다(`src/ctproc.rs`, 약 450줄). Python `volume_to_mesh.py`와 같은 순서이고 같은 임계값을 씁니다. AI 분할 없이 임계값만 쓰는 단계라 Python이 필요 없습니다.

**옮긴 방법**

| 단계 | Python | Rust |
|---|---|---|
| DICOM 읽기 | SimpleITK | dicom-rs, 슬라이스 병렬 |
| 2배 축소, 가우스 흐리기 | SimpleITK | 직접 구현(축마다 따로, 병렬) |
| 덩어리 나누기, 넓히기 | SimpleITK | 직접 구현(면 이웃 6방향, 공 모양) |
| 작은 덩어리 버리기 | 메시에서 | 볼륨에서(등가면 전에) |
| 등가면 | VTK flying edges | fast-surface-nets |
| 매끄럽게 | VTK windowed sinc 30회 | Taubin 10쌍(직접 구현) |
| 삼각형 줄이기 | VTK decimate | meshopt simplify |

ndarray-ndimage는 쓰지 않았습니다. 필요한 연산이 몇 개 안 되고 짧아서, 직접 짜는 쪽이 자료형과 경계 처리를 맞추기 쉬웠습니다.

**시간 (이 PC, CPU)**

| 단계 | 누적 시간 |
|---|---|
| DICOM 425장 읽기 | 1.6초 |
| 축소, 흐리기 | 2.0초 |
| 치아 덩어리 찾기(16개) | 2.1초 |
| 뼈 표면(삼각형 112만 → 22만) | 4.0초 |
| 치아 표면(삼각형 43만 → 17만) | 4.9초 |

Python은 메시 단계만 43초, DICOM 변환은 따로 걸렸습니다. Rust는 DICOM 폴더에서 화면까지 4.9초입니다.

**Python 결과와 대조**

두 GLB의 표면에서 점을 20만 개씩 뽑아, 상대 표면의 가장 가까운 점까지 거리를 쟀습니다. 같은 Python 메시를 다른 표본으로 두 번 뽑아 잰 값(표본 잡음)을 기준으로 함께 적었습니다.

| 구조 | 삼각형 수(Python / Rust) | 면적 차이 | 거리 중앙값 | 거리 95% | 1mm 넘게 다른 곳 |
|---|---|---|---|---|---|
| 뼈 | 220,000 / 220,000 | +1.7% | 0.26mm (잡음 0.25) | 0.54~0.56mm (잡음 0.51) | 0.6~1.4% |
| 치아 | 180,000 / 179,996 | -4.0% | 0.09mm (잡음 0.08) | 0.18~0.20mm (잡음 0.17) | 0% |

- 중앙값과 95% 값이 표본 잡음과 거의 같아서, 표면 위치는 사실상 같습니다. 경계 상자도 0.2mm 안에서 같습니다.
- 뼈에서 1mm 넘게 다른 1% 안팎은 잘게 부서진 얇은 뼈 부분입니다. 작은 덩어리를 메시에서 버리던 것을 볼륨에서 버리게 바꾼 차이로 보입니다.
- 치아 면적이 4% 작은 것은 매끄럽게 하는 방법(VTK windowed sinc와 Taubin)의 차이로 보입니다.

## 10. 3단계 결과 (2026-11-28)

뷰어가 DICOM 폴더와 AI 분할 결과(`segment_teeth.py`의 출력)를 함께 받아 구조별 메시를 만들게 했습니다(`src/seg.rs`, 약 540줄). `find_remnants.py`와 `labels_to_mesh.py`를 옮긴 것이고, AI 분할 자체는 그대로 Python입니다.

```
ct3d-viewer <DICOM 폴더> --seg <분할 폴더> --assign 1:47,2:46,3:45
```

**옮긴 방법**

- 분할 결과(`teeth_labels.nii.gz`)는 [nifti](https://docs.rs/crate/nifti/latest) 크레이트로 읽습니다. 격자가 CT와 같은지 크기로 확인하고, AI가 치아로 잡은 칸의 CT 평균 밝기(2503)로 겹침을 한 번 더 확인합니다.
- 2단계에서 만든 연산(축소, 흐리기, 덩어리 나누기, 넓히기, 등가면, 매끄럽게, 줄이기)을 그대로 쓰고, 깎기와 구멍 메우기를 더했습니다. 넓히기는 scipy와 같은 마름모 모양과 SimpleITK와 같은 공 모양을 Python 코드와 똑같이 골라 썼습니다.
- 턱뼈를 가르는 단계에서 "가까운 씨앗에 붙이기"는 Python의 거리 변환 대신 씨앗에서 퍼져 나가는 너비 우선 탐색으로 했습니다.

**Python 결과와 대조**

| 항목 | Rust | Python |
|---|---|---|
| 걸린 시간 | 약 7.5초(DICOM 읽기 포함) | 메시 단계만 36초 |
| 뿌리 조각 후보 부피 | 83.0 / 88.3 / 64.6 / 146.6 / 29.9mm³ | 83.1 / 88.3 / 64.7 / 146.8 / 29.9mm³ |
| 뺀 구조 | 크라운·46번 조각(뿌리 조각과 겹침), 41번(너무 작음) | 같음 |
| 턱뼈 가르기 | 밝기 1100 | 같음 |
| 구조 이름 | 45개, 모두 같음 | 45개 |

구조마다 표면에서 점을 뽑아 서로의 거리를 쟀습니다.

| 구조 | 거리 중앙값 | 표본 잡음 | 1mm 넘게 다른 곳 |
|---|---|---|---|
| 치아와 뿌리 조각 25개 | 0.12mm(가장 큰 것 0.14) | 0.07mm | 0% |
| 아래턱뼈 | 0.49mm | 0.49mm | 5.7%(잡음 기준 95%가 1.01mm) |
| 위턱뼈 | 0.30mm | 0.30mm | 0.4% |

- 치아는 중앙값이 표본 잡음보다 0.05mm쯤 큽니다. 매끄럽게 하는 방법(VTK windowed sinc와 Taubin)의 차이로 보이며, 면적은 중앙값 -0.3%로 같습니다.
- 턱뼈는 표본 잡음과 같습니다.

## 11. 4단계 결과 (2026-11-28)

치열 모형(`index.html`)이 읽는 `ct-teeth.js`를 Rust로 만들게 했습니다(`src/model.rs`, 약 620줄). `teeth_for_model.py`를 옮긴 것이고, 창을 열지 않고 파일만 만들고 끝납니다.

```
ct3d-viewer <DICOM 폴더> --seg <분할 폴더> --assign 1:47,2:46,3:45 --remap 42:41 --model-js ct-teeth.js
```

**옮긴 방법**

| 단계 | Python | Rust |
|---|---|---|
| 치아의 긴 축, 교합면 | numpy SVD | 3×3 공분산의 고유벡터(야코비 회전, 직접 구현) |
| 치아 표면 | 3단계와 같은 VTK 순서, 삼각형 2,600개 | 3단계 함수에 삼각형 수만 넘김 |
| 머리와 뿌리 자르기 | pyvista `clip` | 삼각형을 평면으로 잘라 자른 자리 꼭짓점을 이웃과 함께 씀(직접 구현) |
| 치열궁 맞추기 | scipy Nelder-Mead | scipy와 같은 시작 단순체, 계수, 멈춤 조건으로 직접 구현 |
| 파일 쓰기 | json, base64 | 직접 씀(필드 순서를 Python과 같게) |

**Python 결과와 대조**

Python으로 다시 만든 파일이 저장소의 `ct-teeth.js`와 해시까지 같아서, 그것을 기준으로 비교했습니다.

| 항목 | 결과 |
|---|---|
| 걸린 시간 | Rust 0.8초(DICOM 읽기와 뿌리 조각 찾기를 합치면 7.7초), Python 9.7초(`ct.nii.gz`와 분할 결과에서) |
| 치아 | 25개(치아 22, 뿌리 조각 3), 번호와 뿌리 조각 표시 모두 같음 |
| 자리(`pos`) | 차이 최대 0.014mm, 중앙값 0.011mm |
| 축(`axes`) | 차이 최대 0.019°, 중앙값 0.013° |
| 폭·두께·머리 높이·뿌리 길이 | 차이 최대 0.05mm |
| 위턱 치열궁 | `[29.203, 48.51, -0.0]`, 같음 |
| 아래턱 치열궁 | Python `[27.553, 38.784, 3.26]`, Rust `[27.548, 38.775, 3.257]` |
| 모양(머리·뿌리 따로) | 거리 중앙값과 95% 값이 25개 모두 표본 잡음과 같음(차이 0.002mm 이하) |
| 치열 모형 화면 | 두 파일로 정면 화면을 찍어 비교. 화소 126만 개 가운데 0.05%(682개)만 눈에 띄게 다름 |

- 자리와 축이 소수 둘째 자리까지 같아서 7.2의 4단계 기준을 채웠습니다. 아래턱 치열궁의 0.01mm 차이는 치아 목 위치가 0.01mm쯤 달라서 생긴 것입니다.
- 3단계에서는 치아 표면 거리가 표본 잡음보다 0.05mm 컸는데, 여기서는 같습니다. 삼각형을 2,600개로 줄이면 줄이는 과정의 차이가 매끄럽게 하는 방법의 차이보다 커지기 때문으로 보입니다.
- **치수 자리 안쪽 면의 방향이 다릅니다.** 큰어금니처럼 치수 자리가 큰 치아는 안쪽에 닫힌 빈 면이 있습니다. Python 판은 VTK가 법선을 맞추면서 이 면도 바깥을 보게 뒤집어, 부피를 재면 빈 공간만큼(16번 약 25mm³) 더해집니다. Rust 판은 빈 공간 쪽을 봅니다. 겉면 부피는 0.4% 안에서 같고, 모형은 안쪽 면을 보여 주지 않아 화면에는 차이가 없습니다.

## 12. 확인하지 못한 부분

- **egui 화면:** 8장에서 대부분 확인했습니다. 마우스 조작과 웹 빌드는 남아 있습니다.
- **다른 CT:** 이 CT 한 건으로만 확인했습니다. 다른 장비의 CT는 임계값을 다시 맞춰야 합니다.
- **wasm 크기와 웹 성능:** 재 보지 않았습니다.
- **eframe 최신 버전:** 0.37로 만들었습니다. 새 버전이 나오면 API가 다시 바뀔 수 있습니다.
- **5·6단계:** 치열 모형을 egui로 옮길지, AI 분할을 ONNX로 옮길지는 아직 판단하지 않았습니다(7.2).

## 참고한 곳

- [egui 저장소와 릴리스](https://github.com/emilk/egui/releases), [custom3d_wgpu 예제](https://github.com/emilk/egui/blob/main/crates/egui_demo_app/src/apps/custom3d_wgpu.rs)
- [dicom-pixeldata 문서](https://docs.rs/dicom-pixeldata/latest/dicom_pixeldata/), [전송 구문 지원 표](https://docs.rs/dicom-transfer-syntax-registry/latest/dicom_transfer_syntax_registry/)
- [fast-surface-nets](https://github.com/bonsairobo/fast-surface-nets-rs), [meshopt](https://docs.rs/crate/meshopt/latest), [ndarray-ndimage](https://github.com/imeka/ndarray-ndimage), [nifti](https://docs.rs/crate/nifti/latest), [gltf](https://docs.rs/crate/gltf/latest)
- [ort](https://docs.rs/crate/ort/latest), [TS-Portable](https://huggingface.co/sigjhl/ts-portable), [TotalSegmentator ONNX total_fast](https://huggingface.co/Aralario/totalsegmentator-onnx-total-fast)
- [three-d egui 연결 크레이트](https://docs.rs/crate/egui_render_three_d/latest)
- 참고할 만한 Rust 의료 영상 프로젝트: [CDViewer](https://github.com/pejmanS21/CDViewer)(egui DICOM 뷰어), [volren-gpu](https://docs.rs/crate/volren-gpu/latest)(wgpu 볼륨 렌더러). 둘 다 코드를 읽어 보지는 않았습니다.
