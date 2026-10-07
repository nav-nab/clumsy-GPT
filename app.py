"""The ClumsyGPT inference GUI. Run with: python app.py"""

import html
from pathlib import Path

import gradio as gr

from clumsygpt import config
from clumsygpt.generate import ClumsyGenerator

OPENINGS_FILE = Path(__file__).parent / "openings.txt"

# A slip is shown by colouring its text. The selectors are two classes deep so that they
# win over the theme's own text colour.
SLIP_STYLE = (
    "<style>.clumsy-view .clumsy-slip{color:#b0583a;cursor:help}"
    ".dark .clumsy-view .clumsy-slip{color:#e8a07a}</style>"
)

# Hovering over (or focusing) a slip shows a box saying what the model meant to say. One box is
# shared by all slips and is placed with a little script, so that it can be kept inside the window.
# It sits below the word, or above it when there is no room below.
TOOLTIP_JS = """
() => {
  if (window.__clumsyTip) return;
  const tip = document.createElement('div');
  tip.style.cssText = 'position:fixed;display:none;z-index:1000;pointer-events:none;' +
    'padding:4px 10px;font-size:0.85em;line-height:1.4;border-radius:var(--radius-lg);' +
    'color:var(--body-text-color);background:var(--background-fill-primary);' +
    'border:1px solid var(--border-color-primary);box-shadow:0 2px 8px rgba(0,0,0,0.2);';
  (document.querySelector('.gradio-container') || document.body).appendChild(tip);
  window.__clumsyTip = tip;
  const slipOf = (e) => e.target.closest && e.target.closest('.clumsy-slip');
  const show = (el) => {
    tip.textContent = el.dataset.tip;
    tip.style.display = 'block';
    const r = el.getBoundingClientRect();
    const left = Math.max(8, Math.min(r.left, window.innerWidth - tip.offsetWidth - 8));
    let top = r.bottom + 6;
    if (top + tip.offsetHeight > window.innerHeight - 8) top = r.top - tip.offsetHeight - 6;
    tip.style.left = left + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  };
  const hide = () => { tip.style.display = 'none'; };
  for (const [on, off] of [['mouseover', 'mouseout'], ['focusin', 'focusout']]) {
    document.addEventListener(on, (e) => { const el = slipOf(e); if (el) show(el); });
    document.addEventListener(off, (e) => { if (slipOf(e)) hide(); });
  }
}
"""

# The read-only view copies the look of the text box so that switching between them is not jarring.
VIEW_STYLE = (
    "white-space: pre-wrap; line-height: 1.7; min-height: 18em; padding: 12px; "
    "color: var(--body-text-color); background: var(--input-background-fill); "
    "border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg);"
)


def load_openings():
    """One opening per line. Trailing spaces are kept: the model supplies the next space."""
    text = OPENINGS_FILE.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def to_marked(result):
    """The generated text as (text, intended word) pairs. The intended word is None unless the text is a slip."""
    intended = iter(result.slips)
    return [(text, next(intended)[0] if label == "slip" else None) for text, label in result.segments]


def join_text(marked):
    return "".join(text for text, _ in marked)


def render_marked(marked):
    """The text as one paragraph, with slips coloured. Each carries what the model meant to say, for the hover box."""
    parts = []
    for text, intended in marked:
        escaped = html.escape(text)
        if intended is None:
            parts.append(escaped)
        else:
            tip = html.escape(f"What I meant to say was: {intended}")
            parts.append(f'<span class="clumsy-slip" tabindex="0" data-tip="{tip}">{escaped}</span>')
    return f'{SLIP_STYLE}<div class="clumsy-view" style="{VIEW_STYLE}">{"".join(parts)}</div>'


def layout(text, marked, highlight):
    """Updates for the text box and the highlighted view, of which only one is visible.

    The highlighted view is shown when highlighting is on and the text is still
    exactly what was generated. If the text was edited, the highlights no longer
    line up with it, so the plain text box is shown.
    """
    show_view = bool(highlight and marked and join_text(marked) == text)
    return (
        gr.update(value=text, visible=not show_view),
        gr.update(value=render_marked(marked) if show_view else "", visible=show_view),
    )


def summarise(result):
    line = f"**{len(result.slips)} slips** in {result.trials} eligible words ({result.tokens} tokens)."
    if result.slips:
        line += " " + ", ".join(f"{new} (for {old})" for old, new in result.slips) + "."
    return line


def build_app(generator):
    openings = load_openings()
    choices = [
        (f"{i}. {line[:80]}{'…' if len(line) > 80 else ''}", line)
        for i, line in enumerate(openings, start=1)
    ]

    def run(text, clumsiness, looseness, top_k, slip_type, marked, highlight):
        if not text or not text.strip():
            raise gr.Error("Type an opening or pick one from the list.")
        result = generator.generate(text, clumsiness, looseness, int(top_k), slip_type)
        # Keep the highlights from earlier runs when this continues text that was generated before.
        earlier = marked if marked and join_text(marked) == text else [(text, None)]
        marked = earlier + to_marked(result)[1:]  # the first segment is the opening itself
        return (marked, summarise(result)) + layout(join_text(marked), marked, highlight)

    def on_toggle(text, marked, highlight):
        return layout(text, marked, highlight)

    def on_pick(opening):
        if not opening:
            return gr.update(), gr.update(), gr.update()
        return gr.update(value=opening, visible=True), [], gr.update(value="", visible=False)

    with gr.Blocks(title="ClumsyGPT") as demo:
        marked = gr.State([])
        gr.Markdown("# ClumsyGPT\nA language model that sometimes says the wrong, but related, word.")

        text = gr.Textbox(
            label="Text", lines=12, max_lines=40,
            placeholder="Type an opening, or pick one below. Each Generate continues the text.",
        )
        view = gr.HTML(visible=False)

        with gr.Row():
            picker = gr.Dropdown(choices=choices, value=None, label="Load an opening", scale=4)
            highlight = gr.Checkbox(
                value=True, label="Highlight slips", scale=1,
                info="Shows the text read-only. Untick to edit it.",
            )
        button = gr.Button("Generate", variant="primary")
        stats = gr.Markdown()

        with gr.Row():
            clumsiness = gr.Slider(
                config.CLUMSINESS[1], config.CLUMSINESS[2], value=config.CLUMSINESS[0],
                step=0.005, label="Clumsiness",
                info="Chance that each eligible word is replaced by a slip",
            )
            looseness = gr.Slider(
                config.LOOSENESS[1], config.LOOSENESS[2], value=config.LOOSENESS[0],
                step=0.01, label="Looseness",
                info="Low: the closest associate. High: more distant ones",
            )
        with gr.Row():
            top_k = gr.Slider(
                config.TOP_K[1], config.TOP_K[2], value=config.TOP_K[0],
                step=10, label="Top-K",
                info="How many of the model's own candidates a slip is drawn from",
            )
            slip_type = gr.Slider(
                config.SLIP_TYPE[1], config.SLIP_TYPE[2], value=config.SLIP_TYPE[0],
                step=0.05, label="Slip type",
                info="0: substitution (the intended word). 1: intrusion (the previous word)",
            )

        demo.load(None, None, None, js=TOOLTIP_JS)
        picker.change(on_pick, picker, [text, marked, view])
        highlight.change(on_toggle, [text, marked, highlight], [text, view])
        button.click(
            run,
            [text, clumsiness, looseness, top_k, slip_type, marked, highlight],
            [marked, stats, text, view],
        )
    return demo


if __name__ == "__main__":
    print(f"Loading {config.MODEL_NAME} ...")
    build_app(ClumsyGenerator.load()).launch()
