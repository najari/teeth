"""자동 분할(TotalSegmentator) 실행 환경을 만든다.

NVIDIA GPU가 있으면 CUDA용 PyTorch를, 없으면 CPU용 PyTorch를 설치한다.
GPU가 있는 PC에서 한 번만 실행하면 된다.

사용법:
    python setup_segmentation.py [가상환경 폴더]

가상환경 폴더를 생략하면 저장소 옆의 .venv-seg에 만든다.
"""
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# 최신 것부터 시도한다. 그래픽 드라이버가 오래돼 GPU를 못 잡으면 다음 것으로 넘어간다.
CUDA_INDEXES = ["cu126", "cu124", "cu121"]


def run(*cmd) -> int:
    print(">", *[str(c) for c in cmd], flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def cuda_works(py: Path) -> bool:
    code = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    return subprocess.run([str(py), "-c", code]).returncode == 0


def main() -> None:
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 이상이 필요합니다. 지금 버전: " + sys.version.split()[0])

    env_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / ".venv-seg"
    py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        print("가상환경을 만듭니다:", env_dir, flush=True)
        venv.create(env_dir, with_pip=True)

    if shutil.which("nvidia-smi"):
        for cu in CUDA_INDEXES:
            run(py, "-m", "pip", "install", "torch", "--index-url", f"https://download.pytorch.org/whl/{cu}")
            if cuda_works(py):
                break
            run(py, "-m", "pip", "uninstall", "-y", "torch")
        else:
            print("CUDA용 PyTorch가 GPU를 잡지 못했습니다. 그래픽 드라이버를 업데이트한 뒤 다시 실행하세요.")
            print("지금은 CPU용으로 설치합니다.", flush=True)
    else:
        print("nvidia-smi가 없어 CPU용 PyTorch를 설치합니다.", flush=True)

    if run(py, "-m", "pip", "install", "TotalSegmentator") != 0:
        sys.exit("TotalSegmentator 설치에 실패했습니다.")

    report = (
        "import torch, importlib.metadata as m;"
        "print('PyTorch', torch.__version__);"
        "print('TotalSegmentator', m.version('TotalSegmentator'));"
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음 (CPU로 실행)')"
    )
    print("\n설치 결과", flush=True)
    run(py, "-c", report)
    print("\n다음 단계:")
    print(f'  "{py}" "{Path(__file__).with_name("segment_teeth.py")}" <ct.nii.gz> <출력 폴더>')


if __name__ == "__main__":
    main()
