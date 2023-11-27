import diffusers
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import transformers
import cv2
import io
import os

import sys
import os
import shutil
import time

import torch
import numpy as np

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Response
import uvicorn
import datetime

if torch.cuda.is_available():
    device_name = torch.device("cuda")
    torch_dtype = torch.float16
else:
    device_name = torch.device("cpu")
    torch_dtype = torch.float32

clip_skip = 2

if clip_skip > 1:
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder",
        num_hidden_layers=12 - (clip_skip - 1),
        torch_dtype=torch_dtype
    )

# Load the pipeline.

# model_path = "disneyPixarCartoon_v10"
model_path = "dreamlike-art/dreamlike-anime-1.0"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)

if clip_skip > 1:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        text_encoder=text_encoder,
    )
else:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

pipe = pipe.to(device_name)

pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

def get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character=",",
    device=torch.device("cpu")
):
    max_length = pipe.tokenizer.model_max_length
    # Simple method of checking if the prompt is longer than the negative
    # prompt - split the input strings using `split_character`.
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    # If prompt is longer than negative prompt.
    if count_prompt >= count_negative_prompt:
        input_ids = pipe.tokenizer(
            prompt, return_tensors="pt", truncation=False
        ).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(
            negative_prompt,
            truncation=False,
            padding="max_length",
            max_length=shape_max_length,
            return_tensors="pt"
        ).input_ids.to(device)

    # If negative prompt is longer than prompt.
    else:
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors="pt", truncation=False
        ).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            padding="max_length",
            max_length=shape_max_length
        ).input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(
            pipe.text_encoder(input_ids[:, i: i + max_length])[0]
        )
        neg_embeds.append(
            pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
        )

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

prompt = """((Crying)), Tear, Sad ,masterpiece, high quality best quality, bangs, beach, blue_sky, blush, bow, cloud, cloudy_sky, collarbone, day, grass, hair_bow, holding, holding_letter, horizon, incoming_gift, kazami_yuuka, leaning_forward, lens_flare, letter, light_rays, long_hair, looking_at_viewer, love_letter, mountain, """

negative_prompt = """drawn by bad-artist, sketch by bad-artist-anime, (bad_prompt:0.8), (artist name, signature, watermark:1.4), (ugly:1.2), (worst quality, poor details:1.4), bad-hands-5, badhandv4, blurry"""

prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character=",",
    device=device_name
)

use_prompt_embeddings = True

start_idx = 1111
batch_size = 10
# seed = start_idx + batch_size
seed = 170953175

num_inference_steps = 25

guidance_scale = 6.75

width = 768
height = 512

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

async def saved_generated_image(image: File):
    output_folder = "out"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(
        output_folder, f"output-{int(datetime.datetime.now().timestamp())}.png"
    )
    image.save(output_filename)
    return output_filename.split("/")[-1]

async def process_image_sceleton(raw_image: bytes):
    pillow_image = Image.open(io.BytesIO(raw_image)).convert("RGB")
    pillow_resized_image = pillow_image.resize((768, 512))
    # image = np.array(Image.open(io.BytesIO(raw_image)))
    image = np.array(pillow_resized_image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

@app.post("/image")
async def generate(image: bytes = File(...)):

    image_skeleton = await process_image_sceleton(image)

    if use_prompt_embeddings is False:
        output = pipe(
            prompt=prompt,
            image=image_skeleton,
            negative_prompt=negative_prompt,
            # width=width,
            # height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=torch.manual_seed(seed),
        )
    else:
        output = pipe(
            prompt_embeds=prompt_embeds,
            image=image_skeleton,
            negative_prompt_embeds=negative_prompt_embeds,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=torch.manual_seed(seed),
        )

    output_filename = await saved_generated_image(output.images[0])
    return {"file_name": output_filename}


@app.get("/get/{filename}")
async def get_image(filename: str):
    with open(f"out/{filename}", "rb") as f:
        return Response(f.read(), media_type="image/png")

if __name__ == "__main__":
    import os
    print(f"Running on PID: {os.getpid()}")
    uvicorn.run("main:app", host="0.0.0.0", port=9090)
