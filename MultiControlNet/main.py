# System
import io
import os
import datetime
# Pytorch
import torch
# Popolar
import cv2
import numpy as np
from PIL import Image
# Diffusers
from transformers import pipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import (
    OpenposeDetector,
    HEDdetector,
)
from huggingface_hub import login
# FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Response

login(token="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")
# model_id = "disneyPixarCartoon_v10"
model_id = "dreamlike-art/dreamlike-anime-1.0"
# model_id = "nitrosocke/mo-di-diffusion"

depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")

controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
).to("cuda")
controlnet_pose = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
).to("cuda")
controlnet_hed = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-hed",
    torch_dtype=torch.float16
)


to_hed = HEDdetector.from_pretrained(
    'lllyasviel/Annotators'
).to("cuda")
to_pose = OpenposeDetector.from_pretrained(
    "lllyasviel/ControlNet"
).to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    controlnet=[
        controlnet_pose,
        # controlnet_canny,
        # controlnet_hed,
    ],
).to("cuda")


async def process_image_sceleton(raw_image: bytes):
    pillow_image = Image.open(io.BytesIO(raw_image)).convert("RGB")
    pillow_resized_image = pillow_image.resize((512, 512))

    image1 = np.array(pillow_resized_image)
    image = cv2.Canny(image1, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    image2 = np.array(pillow_resized_image)
    pose_image = to_pose(image2)

    image3 = np.array(pillow_resized_image)
    hed_image = to_hed(image3)

    pose_image = pose_image.resize((512, 512))
    canny_image = canny_image.resize((512, 512))

    return pose_image, canny_image, hed_image


async def saved_generated_image(image: File):
    output_folder = "out"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(
        output_folder, f"output-{int(datetime.datetime.now().timestamp())}.png"
    )
    image.save(output_filename)
    return output_filename.split("/")[-1]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.prompt_female = "cute girl,anime, masterpiece, high quality, looking at viewer, blush, smile, cute eyes, iridescent, KIMONO"
app.prompt_male = "handsome boy,anime, masterpiece, high quality, looking at viewer, blush, smile, cute eyes, iridescent, KIMONO"

@app.post("/male")
async def change_male_prompt(prompt:str= "handsome boy,anime, masterpiece, high quality, looking at viewer, smile, cute eyes, iridescent, KIMONO"):
    app.prompt_male = prompt
    return "OK"

@app.post("/female")
async def change_female_prompt(prompt:str= "cute girl,anime, masterpiece, high quality, looking at viewer, blush, smile, cute eyes, iridescent, KIMONO"):
    app.prompt_female = prompt
    return "OK"
    

@app.post("/image")
async def generate(raw_image: bytes = File(...), is_boy: str = "female",prompt:str = "handsome"):
    print(is_boy)
    pose_image, canny_image, hed_image = await process_image_sceleton(raw_image)
    if is_boy == "male":
        output = pipe(
            prompt=prompt + app.prompt_male,
            negative_prompt="simple background,no breast,no big breast, duplicate, retro style, low quality, lowest quality, bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, watermark, signature, text, extra digit, worst quality,jpeg artifacts, blurry, bad hands,bad hand, blurry",
            image=[
                pose_image,
                # canny_image,
                # hed_image
            ],
            width=512,
            height=512,
            # guidance_scale=6.75,
            # num_inference_steps=50,
            generator= torch.Generator(device="cuda"),
        ).images[0]
        print("male")
    else:
        output = pipe(
            prompt=prompt +  app.prompt_female,
            negative_prompt="simple background, duplicate, retro style, low quality, lowest quality, bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, watermark, signature, text, extra digit, worst quality,jpeg artifacts, blurry, bad hands,bad hand, blurry",
            image=[
                pose_image,
                # canny_image,
                # hed_image
            ],
            width=512,
            height=512,
            # guidance_scale=6.75,
            # num_inference_steps=50,
            generator= torch.Generator(device="cuda"),
        ).images[0]
        print("female")
    output_filename = await saved_generated_image(output)
    return {"file_name": output_filename}


@app.get("/get/{filename}")
async def get_image(filename: str):
    with open(f"out/{filename}", "rb") as f:
        return Response(f.read(), media_type="image/png")

if __name__ == "__main__":
    import os
    print(f"Running on PID: {os.getpid()}")
    uvicorn.run("main:app", host="0.0.0.0", port=9090)
