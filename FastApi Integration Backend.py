from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from unsloth import FastVisionModel
import string
import re
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import BitsAndBytesConfig
import os
import nest_asyncio
from pyngrok import ngrok, conf
import uvicorn

# === Optional text preprocessing ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', 'number', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# === Load fine-tuned FastVision model ===
print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model, tokenizer = FastVisionModel.from_pretrained(
    "/kaggle/input/xperay/pytorch/default/1/Llama3.2",
    quantization_config=bnb_config,
    device_map="auto"
)
FastVisionModel.for_inference(model)
print("Model loaded.")

# === Load image from URL ===
def load_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

# === Generate caption using chat-based format ===
def generate_caption_from_url(image_url, model, tokenizer):
    image = load_image_from_url(image_url)

    instruction = (
        "You are an expert radiologist. Carefully analyze the provided medical image "
        "and provide a detailed, accurate, and professional radiology report. "
        "Describe any abnormalities, anatomical structures, and relevant findings using precise medical terminology."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, use_cache=True)

    predicted_caption = tokenizer.decode(output[0], skip_special_tokens=True).split("assistant")[-1].strip()
    return predicted_caption

# === FastAPI app setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === POST endpoint to generate caption ===
@app.post("/generate_caption")
async def get_caption(request: Request):
    try:
        data = await request.json()
        image_url = data.get("image_url")

        if not image_url:
            return JSONResponse(content={"error": "Image URL is required."}, status_code=400)

        caption = generate_caption_from_url(image_url, model, tokenizer)
        return JSONResponse(content={"generated_caption": caption})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
# Apply async loop fix for Jupyter/Kaggle
nest_asyncio.apply()

# Set your ngrok auth token (manually or via Kaggle secrets)
os.environ['ngrok_authToken']='2pqNMvvvTDN1B4RL6beU7Ccbwu1_6RYQsJQhg7S1oARTDefwo'
conf.get_default().auth_token = os.environ["ngrok_authToken"]

# Expose FastAPI app via ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Start FastAPI
uvicorn.run(app, host="0.0.0.0", port=8000)
