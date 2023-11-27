#!/bin/sh
scp -r -P 23410 ~/Projects/Hobbies/Remote-GPU/MultiControlNet/ root@ssh4.vast.ai:./
ssh -p 23410 root@ssh4.vast.ai -L 8080:localhost:8080 -o LogLevel=QUIET << EOF

apt-get update
apt-get upgrade -y
apt install python3.8-venv -y
cd MultiControlNet

# wget https://civitai.com/api/download/models/69832 --content-disposition
# wget https://raw.githubusercontent.com/huggingface/diffusers/v0.20.0/scripts/convert_original_stable_diffusion_to_diffusers.py

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

pip install -r requirements.txt

# python3 convert_original_stable_diffusion_to_diffusers.py --checkpoint_path disneyPixarCartoon_v10.safetensors --dump_path disneyPixarCartoon_v10/ --from_safetensors

python3 main.py
exit


