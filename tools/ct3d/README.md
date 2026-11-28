# CT를 3D로 만드는 도구

치과 CT(DICOM)를 읽어 브라우저에서 돌려 볼 수 있는 3D 메시와 뷰어를 만드는 스크립트입니다. 조사 배경과 실행 결과는 [CT 데이터 3D 시각화 오픈소스 조사 결과](../../docs/CT-3D-시각화-오픈소스-조사결과.md)에 있습니다.

CT 원본과 여기서 만든 결과물은 개인 의료정보입니다. 출력 폴더는 저장소 밖에 두고 커밋하지 않습니다.

모든 단계가 GPU 없이 돌아갑니다. Intel i5-1240P, RAM 16GB인 노트북에서 전부 실행해 확인했습니다.

## 1. 변환과 임계값 메시

Python 3.10 이상에 다음 패키지가 필요합니다.

```
pip install SimpleITK numpy scipy pyvista trimesh pillow
```

| 순서 | 명령 | 결과 |
|---|---|---|
| 1 | `python dicom_to_volume.py <DICOM 폴더> <출력 폴더>` | `ct.nii.gz`, 단면과 투영 그림, `volume_info.json` |
| 2 | `python volume_to_mesh.py <출력 폴더>/ct.nii.gz <출력 폴더>` | `ct_surface.glb`(뼈와 치아 메시), 확인용 렌더링 |
| 3 | `python build_viewer.py <출력 폴더>/ct_surface.glb <출력 폴더>/ct_viewer.html <출력 폴더>/mesh_info.json` | 파일 하나로 열리는 뷰어 |

- 2번의 임계값은 `--bone`, `--teeth`, `--enamel`로 바꿉니다. 기본값 1000, 2800, 3300은 VATECH 콘빔 CT 한 건에 맞춘 값이라 다른 장비의 CT에는 다시 맞춰야 합니다.
- 임계값 분할은 치아 뿌리와 턱뼈를 나누지 못합니다. 치아 메시에는 밝게 찍힌 머리 부분만 들어갑니다.

## 2. 자동 분할과 치아별 메시

TotalSegmentator의 `teeth` 작업으로 치아를 번호별로 나눕니다. 별도 가상환경을 씁니다.

| 순서 | 명령 | 결과 |
|---|---|---|
| 1 | `python setup_segmentation.py` | 저장소 옆 `.venv-seg`에 PyTorch와 TotalSegmentator 설치 |
| 2 | `<.venv-seg의 python> segment_teeth.py <ct.nii.gz> <분할 폴더>` | `teeth_labels.nii.gz`, `segmentation_info.json` |
| 3 | `python labels_to_mesh.py <분할 폴더> <ct.nii.gz> <출력 폴더>` | `teeth_parts.glb`(구조마다 메시 하나), `parts_info.json` |
| 4 | `python build_viewer.py <출력 폴더>/teeth_parts.glb <출력 폴더>/teeth_viewer.html <출력 폴더>/parts_info.json` | 치아를 누르면 이름이 나오는 뷰어 |

- 2번은 CPU로 약 7분 걸렸고 RAM을 9GB쯤 썼습니다. 3번과 4번은 1절의 가상환경으로 실행합니다.
- `setup_segmentation.py`는 NVIDIA GPU가 있으면 CUDA용 PyTorch를 설치합니다. GPU에서 돌리는 경로는 아직 실행해 보지 않았습니다.
- `run_on_gpu_pc.bat`은 1번과 2번을 다른 PC에서 한 번에 실행하는 파일입니다. `setup_segmentation.py`, `segment_teeth.py`, `ct.nii.gz`와 같은 폴더에 두고 실행합니다.
- 계산은 모두 PC 안에서 합니다. 처음 실행할 때 모델 가중치 세 개(합쳐서 약 640MB)만 내려받습니다.
- TotalSegmentator는 기본으로 실행 통계를 자기 서버로 보냅니다. 영상은 보내지 않지만 `segment_teeth.py`는 이 기능을 끄고 실행합니다.
- `ct.nii.gz`에는 형상과 화소 값만 있고 이름 같은 식별 정보는 없습니다.

## 3. 치열 모형에 넣기

```
python teeth_for_model.py <분할 폴더> <ct.nii.gz> <저장소>/ct-teeth.js [--remap 42:41]
```

- 자동 분할한 치아를 치아마다 자기 좌표(원점은 치아 목, +y는 머리 쪽, +z는 볼·입술 쪽)로 바꾸고 머리와 뿌리로 잘라 `ct-teeth.js`에 담습니다.
- `index.html`은 옆에 `ct-teeth.js`가 있으면 표준형 대신 이 모양을 씁니다.
- `--remap`은 분할 모델이 붙인 번호를 모형의 번호로 바꿀 때 씁니다.
- `ct-teeth.js`에는 환자의 치아 모양이 들어 있습니다. 공개 저장소에 올릴지는 따로 정합니다.

## 4. 뷰어

- `viewer_template.html`은 데이터 없이도 열립니다. GLB 파일을 화면에 끌어다 놓으면 브라우저 안에서만 읽어서 보여 줍니다.
- 메시 이름으로 종류를 정합니다. `bone`, `teeth`는 임계값 메시이고, `..._fdi11` 같은 이름은 자동 분할 결과입니다.
- 주소 끝에 `#view=right&bone=100&cut=40&hide=jaw,sinus&pick=implant`처럼 붙이면 그 상태로 열립니다.
