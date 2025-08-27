from websight.model.actions import parse_action


def test_parse_click_point():
    action = parse_action("click(point='(100, 200)')", "Clicking the button")
    assert action.action == "click"
    assert action.args == {"x": "100", "y": "200"}
    assert action.reasoning == "Clicking the button"


def test_parse_click_start_box():
    action = parse_action("click(start_box='(10, 20)')", "Clicking start box")
    assert action.action == "click"
    assert action.args == {"x": "10", "y": "20"}


def test_parse_drag():
    action = parse_action(
        "drag(start_box='(1, 2)', end_box='(3, 4)')", "Dragging to select"
    )
    assert action.action == "drag"
    assert action.args == {
        "start_x": "1",
        "start_y": "2",
        "end_x": "3",
        "end_y": "4",
    }


def test_parse_scroll_point():
    action = parse_action(
        "scroll(point='(400, 500)', direction='down')", "Scrolling down"
    )
    assert action.action == "scroll"
    assert action.args == {"x": "400", "y": "500", "direction": "down"}


def test_parse_scroll_start_box():
    action = parse_action(
        "scroll(start_box='(40, 50)', direction='up')", "Scrolling up"
    )
    assert action.action == "scroll"
    assert action.args == {"x": "40", "y": "50", "direction": "up"}


def test_parse_hotkey():
    action = parse_action("hotkey(key='ctrl c')", "Copying")
    assert action.action == "hotkey"
    assert action.args == {"key": "ctrl c"}


def test_parse_type():
    action = parse_action("type(content='hello world')", "Typing")
    assert action.action == "type"
    assert action.args == {"content": "hello world"}


def test_parse_wait():
    action = parse_action("wait()", "Waiting")
    assert action.action == "wait"
    assert action.args == {}


def test_parse_finished():
    action = parse_action("finished(content='done')", "Done")
    assert action.action == "finished"
    assert action.args == {"content": "done"}


def test_parse_goto_url():
    action = parse_action("goto_url(url='https://example.com')", "Navigate")
    assert action.action == "goto_url"
    assert action.args == {"url": "https://example.com"}
