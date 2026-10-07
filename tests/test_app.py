from app import render, summarise
from clumsygpt.generate import Result


def test_render_marks_slips_and_escapes_text():
    r = Result(
        segments=[(' "a <b> cat', None), (" dog", "slip"), (" ran.", None)],
        slips=[("cat", "dog")],
        trials=3,
        tokens=5,
    )
    html_text = render(r)
    assert '<mark title="intended: cat"> dog</mark>' in html_text
    assert "&lt;b&gt;" in html_text and "<b>" not in html_text
    assert "&quot;a" in html_text or '"a' in html_text


def test_summary_lists_slips():
    r = Result(segments=[], slips=[("cat", "dog")], trials=3, tokens=5)
    assert "1 slips" in summarise(r) and "dog (for cat)" in summarise(r)
