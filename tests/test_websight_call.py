from unittest.mock import patch

from websight.model.websight import websight_call


def make_mock_response(action_str: str):
    return [{"generated_text": [{"content": f"Thought: do it\nAction: {action_str}"}]}]


@patch("websight.model.websight._get_websight_pipe")
def test_websight_call_click(mock_factory):
    def fake_pipe(**kwargs):
        return make_mock_response("click(point='(10, 20)')")

    mock_factory.return_value = fake_pipe
    action = websight_call(
        prompt="Click the button",
        history=[],
        image_base64="data:image/png;base64,abcd",
    )
    assert action.action == "click"
    assert action.args == {"x": "10", "y": "20"}


@patch("websight.model.websight._get_websight_pipe")
def test_websight_call_scroll(mock_factory):
    def fake_pipe(**kwargs):
        return make_mock_response("scroll(point='(400, 500)', direction='down')")

    mock_factory.return_value = fake_pipe
    action = websight_call(
        prompt="Scroll down",
        history=[],
        image_base64="data:image/png;base64,abcd",
    )
    assert action.action == "scroll"
    assert action.args == {"x": "400", "y": "500", "direction": "down"}
