from app import join_text, layout, render_marked, summarise, to_marked
from clumsygpt.generate import Result

RESULT = Result(
    segments=[(' "a <b> cat', None), (" dog", "slip"), (" ran.", None)],
    slips=[("cat", "dog")],
    trials=3,
    tokens=5,
)


def test_to_marked_keeps_the_intended_word_of_each_slip():
    assert to_marked(RESULT) == [(' "a <b> cat', None), (" dog", "cat"), (" ran.", None)]


def test_join_text_rebuilds_the_text():
    assert join_text(to_marked(RESULT)) == ' "a <b> cat dog ran.'


def test_render_marks_slips_and_escapes_text():
    html_text = render_marked(to_marked(RESULT))
    assert '<span class="clumsy-slip" title="intended: cat"> dog</span>' in html_text
    assert "&lt;b&gt;" in html_text and "<b>" not in html_text


def test_slips_are_coloured_text_without_a_background():
    html_text = render_marked(to_marked(RESULT))
    slip_rule = html_text.split(".clumsy-slip{")[1].split("}")[0]
    assert "color:" in slip_rule and "background" not in slip_rule


def visible(update):
    return update["visible"]


def test_highlighted_view_shows_when_highlighting_is_on_and_text_is_unedited():
    marked = to_marked(RESULT)
    box, view = layout(join_text(marked), marked, True)
    assert not visible(box) and visible(view)


def test_plain_text_box_shows_when_highlighting_is_off():
    marked = to_marked(RESULT)
    box, view = layout(join_text(marked), marked, False)
    assert visible(box) and not visible(view)


def test_plain_text_box_shows_when_the_text_was_edited():
    marked = to_marked(RESULT)
    box, view = layout(join_text(marked) + " edited", marked, True)
    assert visible(box) and not visible(view)


def test_plain_text_box_shows_before_anything_is_generated():
    box, view = layout("an opening", [], True)
    assert visible(box) and not visible(view)


def test_summary_lists_slips():
    assert "1 slips" in summarise(RESULT) and "dog (for cat)" in summarise(RESULT)
