from clip_bbox import hello, print_hello
from unittest.mock import patch


def test_hello():
    assert hello() == "Hello, world!"


@patch('builtins.print')
def test_print_hello(mock_print):
    print_hello()
    assert mock_print.call_args.args == ("Hello, world!",)
