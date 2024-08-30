import random

import gradio as gr
import numpy as np
import torch
import spaces
from diffusers import FluxPipeline
from PIL import Image
from diffusers.utils import export_to_gif

HEIGHT = 256
WIDTH = 1024
MAX_SEED = np.iinfo(np.int32).max

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to(device)

def split_image(input_image, num_splits=4):
    # Create a list to store the output images
    output_images = []

    # Split the image into four 256x256 sections
    for i in range(num_splits):
        left = i * 256
        right = (i + 1) * 256
        box = (left, 0, right, 256)
        output_images.append(input_image.crop(box))

    return output_images

@spaces.GPU(duration=190)
def predict(prompt, seed=42, randomize_seed=False, guidance_scale=5.0, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    prompt_template = f"""
    A  side by side 4 frame image showing consecutive stills from a looped gif moving from left to right.
    The gif is of {prompt}.
    """

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    image = pipe(
        prompt=prompt_template,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(seed),
        height=HEIGHT,
        width=WIDTH
    ).images[0]

    return export_to_gif(split_image(image, 4), "flux.gif", fps=4), image, seed

demo = gr.Interface(fn=predict, inputs="text", outputs="image")

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
#stills{max-height:160px}
"""
examples = [
    "a cat waving its paws in the air",
    "a panda moving their hips from side to side",
    "a flower going through the process of blooming"
]

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# FLUX Gif Generator")
        gr.Markdown("Create GIFs with Flux-dev. Based on @fofr's [tweet](https://x.com/fofrAI/status/1828910395962343561).")
        gr.Markdown("For better results include a description of the motion in your prompt")
        with gr.Row():
            prompt = gr.Text(label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
            submit = gr.Button("Submit", scale=0)

        output = gr.Image(label="GIF", show_label=False)
        output_stills = gr.Image(label="stills", show_label=False, elem_id="stills")
        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=3.5,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )

        gr.Examples(
            examples=examples,
            fn=predict,
            inputs=[prompt],
            outputs=[output, output_stills, seed],
            cache_examples="lazy"
        )
        gr.on(
            triggers=[submit.click, prompt.submit],
            fn=predict,
            inputs=[prompt, seed, randomize_seed, guidance_scale, num_inference_steps],
            outputs = [output, output_stills, seed]
        )

demo.launch()