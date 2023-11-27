import cv2
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Response

import uvicorn
from diffusers import UniPCMultistepScheduler
import io
import os
import inspect
import datetime
import warnings
from torch import autocast
import requests
from io import BytesIO
from huggingface_hub import login
login(token="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")

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
    pillow_resized_image = pillow_image.resize((512, 512))
    # image = np.array(Image.open(io.BytesIO(raw_image)))
    image = np.array(pillow_resized_image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

# model_id = "sd-dreambooth-library/mr-potato-head"
model_id = "dreamlike-art/dreamlike-anime-1.0"

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
generator = torch.manual_seed(2) # Consistent Output
# generator = torch.Generator(device="cuda")
prompt = "a photo of sks mr potato head, best quality, extremely detailed"

@app.post("/image")
async def generate(image: bytes = File(...)):

    image_skeleton = await process_image_sceleton(image)
    image_skeleton
    output = pipe(
        prompt,
        image_skeleton,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        generator=generator,
        num_inference_steps=20,
    )
    output_filename = await saved_generated_image(output.images[0])
    return {"file_name": output_filename}

@app.get("/get/{filename}")
async def get_image(filename: str):
    with open(f"out/{filename}", "rb") as f:
        return Response(f.read(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("main:app",host="0.0.0.0", port=9090)