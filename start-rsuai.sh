#!/bin/sh
scp -r -P 12710 ~/Projects/Hobbies/Remote-GPU/RSUAI/ root@ssh5.vast.ai:./
ssh -p 12710 root@ssh5.vast.ai -L 8080:localhost:8080 -o LogLevel=QUIET << EOF
apt-get update
apt-get upgrade -y
apt install python3.8-venv -y
cd RSUAI
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install -q diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git
pip install -q opencv-contrib-python
pip install -q controlnet_aux
pip install fastapi uvicorn
pip install python-multipart
ngrok config add-authtoken 21s99trxbHP4KdmM8hgt6dsBC7s_7dGRox793V4ZqqpotRxCw
python3 main.py
exit


