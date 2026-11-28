"""GLB 메시를 viewer_template.html에 넣어 파일 하나로 열리는 뷰어를 만든다.

사용법:
    python build_viewer.py <ct_surface.glb> <출력.html> [mesh_info.json]

만들어진 HTML에는 환자의 CT에서 뽑은 형상이 들어 있다. 공개 저장소에 올리지 않는다.
"""
import base64
import sys
from pathlib import Path


def main() -> None:
    glb, out = Path(sys.argv[1]), Path(sys.argv[2])
    info = Path(sys.argv[3]).read_text(encoding="utf-8") if len(sys.argv) > 3 else "{}"
    template = (Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8")
    html = template.replace("__GLB_BASE64__", base64.b64encode(glb.read_bytes()).decode("ascii"))
    html = html.replace("__MESH_INFO__", info.replace("</", "<\\/"))
    out.write_text(html, encoding="utf-8")
    print(f"저장: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
