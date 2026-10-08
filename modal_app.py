"""Serve the ClumsyGPT GUI on Modal. Deploy with: modal deploy modal_app.py

The app itself (app.py) is unchanged. This file only describes the container it runs in:
one CPU core, one container at a time, and asleep when nobody is using it.
"""

import modal

from clumsygpt import config

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # The CPU build of torch is much smaller than the default one, which bundles CUDA.
    .pip_install("torch", index_url=CPU_TORCH_INDEX)
    .pip_install_from_requirements("requirements.txt")
    # The model is stored in the image, so a cold start does not have to download it.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{config.MODEL_NAME}')\""
    )
    .add_local_python_source("app", "clumsygpt")
    .add_local_file("openings.txt", "/root/openings.txt")
    .add_local_file("assets/avatar.svg", "/root/assets/avatar.svg")
)

app = modal.App("clumsygpt", image=image)


@app.function(
    cpu=1.0,  # one physical core (two vCPUs)
    memory=(3072, 4096),  # MiB: reserved, and the most it may use
    max_containers=1,  # Gradio keeps its queue in memory, so there must be one container
    scaledown_window=120,  # seconds idle before the container is shut down
    timeout=600,
)
@modal.concurrent(max_inputs=20)  # many browser tabs can share the container; Gradio queues the generations
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI

    from app import build_app
    from clumsygpt.generate import ClumsyGenerator

    demo = build_app(ClumsyGenerator.load())
    return gr.mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
