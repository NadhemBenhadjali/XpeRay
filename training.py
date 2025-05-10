import os
import torch
from unsloth import FastVisionModel 
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm


# Set GPU 3 as the only visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Ensure PyTorch uses GPU 3
device = torch.device("cuda:0")  
# Replace with your Hugging Face token
login("yourtokenhere")  # Replace with your Hugging Face token

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
if torch.cuda.get_device_capability()[0] >= 8:
    print("up")# Ampere or later (e.g., A100, RTX 3090)
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attn_2"

else:
    print("down")
    torch_dtype = torch.float16
    attn_implementation = "eager"

if True :
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",
        attn_implementation=attn_implementation,
    )
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules      = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

### Data Prep
dataset = load_dataset("eltorio/ROCOv2-radiology", split="train[:100%]")
dataset_val= load_dataset("eltorio/ROCOv2-radiology", split="validation[:100%]")


def convert_to_conversation(sample, target_size=(256, 256)):
    """
    Convert an image/caption sample into a conversation with (user, assistant).
    Also resize the PIL image to prevent giant resolution blow-ups.
    """
    instruction = (
        "You are an expert radiologist. Carefully analyze the provided medical image "
        "and provide a detailed, accurate, and professional radiology report. "
        "Describe any abnormalities, anatomical structures, and relevant findings using precise medical terminology."
    )

    # Safely convert/resize the image
    img = sample["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.resize(target_size)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": img},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["caption"]}],
        },
    ]
    return {"messages": conversation}

converted_dataset = [convert_to_conversation(s) for s in tqdm(dataset, desc="Converting training dataset")]
converted_dataset_val = [convert_to_conversation(s) for s in tqdm(dataset_val, desc="Converting validation dataset")]

FastVisionModel.for_training(model)  # Enable training mode

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    eval_dataset=converted_dataset_val,  # Add evaluation dataset
    args=SFTConfig(
        per_device_train_batch_size=4,  # Reduce batch size
        gradient_accumulation_steps=8,  # Helps simulate larger batch sizes
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,  # Force fp16 for lower memory usage
        bf16=True,  # Disable bf16 to avoid mixed precision conflicts
        logging_steps=10,
        optim="paged_adamw_8bit",  # More memory-efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",

        # ✅ Enable gradient checkpointing
        gradient_checkpointing=True,

        # Vision-specific tuning
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,  # Reduce sequence length

        # Save checkpoint every 500 steps
        save_steps=500,
        evaluation_strategy="steps",  # Enable evaluation during training
        eval_steps=300,  # Evaluate every 100 steps
    ),
)

image = dataset[0]["image"]
instruction = (
        "You are an expert radiologist. Carefully analyze the provided medical image "
        "and provide a detailed, accurate, and professional radiology report. "
        "Describe any abnormalities, anatomical structures, and relevant findings using precise medical terminology."
    )

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

# Set the environment variable to use GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Ensure PyTorch uses the GPU
device = torch.device("cuda:0")  # After masking, this refers to GPU 3

# Move the model to GPU 3
model.to(device)
inputs = inputs.to(device)  # Move input tensors to GPU 3

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs = {num_gpus}")
# Get GPU properties
gpu_stats = torch.cuda.get_device_properties(0)

# Get memory usage specific to the selected GPU
start_gpu_memory = round(torch.cuda.max_memory_reserved(device=0) / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

model.save_pretrained("XpeRay") # Local saving
tokenizer.save_pretrained("XpeRay") # Local saving
