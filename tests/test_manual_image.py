import base64
from pathlib import Path
from unittest.mock import patch

from websight import websight_call


def load_image_b64(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


@patch("websight.model.websight._get_websight_pipe")
def test_manual_websight_on_sample_image(mock_factory):
    # This manual test fakes the model output but exercises the end-to-end call path
    def fake_pipe(**kwargs):
        return [
            {
                "generated_text": [
                    {"content": "Thought: do it\nAction: click(point='(100, 200)')"}
                ]
            }
        ]

    mock_factory.return_value = fake_pipe

    image_path = Path("data/showdown_clicks/images/0b1c958b929acdbf.png")
    assert image_path.exists(), "Sample image not found"
    image_b64 = load_image_b64(image_path)

    action = websight_call(
        prompt="Click the target area",
        history=[],
        image_base64=image_b64,
    )
    assert action.action == "click"
    assert action.args == {"x": "100", "y": "200"}


if __name__ == "__main__":
    test_manual_websight_on_sample_image(None)
