"""The ClumsyGPT inference GUI. Run with: python app.py"""

import html
from pathlib import Path

import gradio as gr

from clumsygpt import config
from clumsygpt.generate import ClumsyGenerator

OPENINGS_FILE = Path(__file__).parent / "openings.txt"


def load_openings():
    """One opening per line. Trailing spaces are kept: the model supplies the next space."""
    text = OPENINGS_FILE.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def render(result):
    """The generated text as one paragraph, with each slip marked and its intended word on hover."""
    intended = iter(result.slips)
    parts = []
    for text, label in result.segments:
        if label == "slip":
            old, _ = next(intended)
            parts.append(f'<mark title="intended: {html.escape(old)}">{html.escape(text)}</mark>')
        else:
            parts.append(html.escape(text))
    style = "white-space: pre-wrap; line-height: 1.8; font-size: 1.05em;"
    return f'<div style="{style}">{"".join(parts)}</div>'


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

    def run(opening, clumsiness, looseness, top_k, slip_type):
        if not opening or not opening.strip():
            raise gr.Error("Type an opening or pick one from the list.")
        result = generator.generate(opening, clumsiness, looseness, int(top_k), slip_type)
        return render(result), summarise(result)

    with gr.Blocks(title="ClumsyGPT") as demo:
        gr.Markdown("# ClumsyGPT\nA language model that sometimes says the wrong, but related, word.")
        picker = gr.Dropdown(choices=choices, value=None, label="Pick an opening")
        opening = gr.Textbox(label="Opening (editable)", lines=4)
        picker.change(lambda text: text if text else gr.update(), picker, opening)

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

        button = gr.Button("Generate", variant="primary")
        output = gr.HTML(label="Output")
        stats = gr.Markdown()
        button.click(run, [opening, clumsiness, looseness, top_k, slip_type], [output, stats])
    return demo


if __name__ == "__main__":
    print(f"Loading {config.MODEL_NAME} ...")
    build_app(ClumsyGenerator.load()).launch()
