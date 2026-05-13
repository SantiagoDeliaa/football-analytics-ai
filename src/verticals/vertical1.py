from pathlib import Path
import runpy


def render_vertical1() -> None:
    legacy_path = Path(__file__).with_name("vertical1_legacy.py")
    runpy.run_path(str(legacy_path), run_name="__main__")
