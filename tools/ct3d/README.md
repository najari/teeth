# CT를 3D로 만드는 도구

치과 CT(DICOM)를 읽어 브라우저에서 돌려 볼 수 있는 3D 메시와 뷰어를 만드는 스크립트입니다. 조사 배경과 실행 결과는 [CT 데이터 3D 시각화 오픈소스 조사 결과](../../docs/CT-3D-시각화-오픈소스-조사결과.md)에 있습니다.

CT 원본과 여기서 만든 결과물은 개인 의료정보입니다. 출력 폴더는 저장소 밖에 두고 커밋하지 않습니다.

## 1. GPU 없이 되는 단계

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
- `viewer_template.html`은 데이터 없이도 열립니다. GLB 파일을 화면에 끌어다 놓으면 브라우저 안에서만 읽어서 보여 줍니다.
- 뷰어 주소 끝에 `#view=right&bone=100&cut=40`처럼 붙이면 그 상태로 열립니다.

## 2. GPU가 있는 PC에서 하는 단계

치아를 번호별로 나누는 자동 분할입니다. TotalSegmentator의 `teeth` 작업을 씁니다.

```
python setup_segmentation.py
<가상환경>\Scripts\python.exe segment_teeth.py <ct.nii.gz> <출력 폴더>
```

- `setup_segmentation.py`는 NVIDIA GPU가 있으면 CUDA용 PyTorch를, 없으면 CPU용을 설치합니다.
- `segment_teeth.py`는 구조마다 NIfTI 파일을 하나씩 만들고 걸린 시간을 `segmentation_info.json`에 적습니다.
- 계산은 모두 그 PC 안에서 합니다. 처음 실행할 때 모델 가중치만 내려받습니다.
- 필요한 입력은 `ct.nii.gz` 하나입니다. 이 파일에는 형상과 화소 값만 있고 이름 같은 식별 정보는 없습니다.
