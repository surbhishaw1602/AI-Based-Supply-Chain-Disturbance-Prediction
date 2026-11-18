import streamlit.components.v1 as components
from pathlib import Path


COMPONENT_DIR = Path(__file__).resolve().parent / "gps_component"

if not (COMPONENT_DIR / "index.html").is_file():
    raise FileNotFoundError(
        f"GPS component HTML was not found: {COMPONENT_DIR / 'index.html'}"
    )


gps_component = components.declare_component(
    "automatic_gps_tracker",
    path=str(COMPONENT_DIR),
)


def automatic_gps_tracker(enabled=True, key=None):
    """
    Capture browser GPS automatically while tracking is enabled.
    Request high-accuracy readings and accept accuracy up to 100 metres.
    """

    return gps_component(
        enabled=bool(enabled),
        high_accuracy=True,
        maximum_age=0,
        timeout=30000,
        update_interval=5000,
        max_accuracy=100,
        default=None,
        key=key,
    )
