# XpeRay: Vision-Language Radiology Report Generation

**XpeRay** is a vision-language model fine-tuned using Unsloth's `FastVisionModel` based on the `LLaMA-3.2-11B-Vision-Instruct` (4-bit) architecture. It is designed to generate detailed, accurate radiology reports from medical images.

## Features

- Vision-language fine-tuning for clinical image-to-text generation
- Uses LoRA with 4-bit quantization for efficient training
- Compatible with gradient checkpointing and Flash Attention 2
- Evaluation using K-Fold cross-validation with BLEU, ROUGE, and BERTScore
- Efficient model deployment and local inference

## Directory Structure

```
XpeRay/
│
├── XpeRay/                # Directory containing saved fine-tuned model
├── training.py            # Fine-tuning script using Unsloth and Hugging Face
├── inference.py           # Inference script to generate radiology reports
├── test.py                # Evaluation script with K-Fold validation
├── results.txt            # Evaluation results summary
````

## Dataset

- **ROCO v2 Radiology**  
  Hugging Face Dataset: [`eltorio/ROCOv2-radiology`](https://huggingface.co/datasets/eltorio/ROCOv2-radiology)

## Training Configuration

- Batch size: 4  
- Accumulation steps: 8  
- Precision: bfloat16 (BF16)  
- Optimizer: `paged_adamw_8bit`  
- LoRA Config: `r=16`, `alpha=16`, no dropout  
- Sequence length: 2048  
- Checkpoints saved every 500 steps

Training is conducted on a single specified GPU with memory monitoring and device allocation handled via environment variables.

## Evaluation

- K-Fold cross-validation implemented in `test.py`
- Evaluation metrics:
  - BLEU
  - ROUGE-L
  - BERTScore
- Results logged in `results.txt`

## Example Usage

**Training**
```bash
python training.py
````

**Inference**

```bash
python inference.py --image_path path/to/image.png
```

## Model Loading

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained("XpeRay")
```

## Requirements

* Python ≥ 3.8
* PyTorch
* transformers
* datasets
* unsloth
* trl
* tqdm
* Pillow

## License

This project is for research and educational purposes only. Please verify your use case complies with any dataset and model licensing terms.

```
