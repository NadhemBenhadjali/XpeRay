from unsloth import FastVisionModel
import warnings
import string
import re
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import KFold
from datasets import load_dataset
import evaluate

# Load Metrics from Hugging Face
eval_metrics = {
    "rouge": evaluate.load("rouge"),
    "bert_scorer": evaluate.load("bertscore"),
}

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', 'number', text)  # Replace numbers with 'number'
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Function to calculate BLEU score with smoothing
# Function to calculate BLEU-1 score with smoothing
def calculate_bleu_score(reference, candidate):
    reference_tokens = [nltk.word_tokenize(preprocess_text(reference))]
    candidate_tokens = nltk.word_tokenize(preprocess_text(candidate))
    smoother = SmoothingFunction().method1  # Use smoothing to handle short captions
    return sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(1, 0, 0, 0),  # BLEU-1
        smoothing_function=smoother
    )

# Function to calculate BERTScore
def calculate_bertscore(reference, candidate):
    return eval_metrics["bert_scorer"].compute(predictions=[candidate], references=[reference], model_type='microsoft/deberta-xlarge-mnli')["f1"][0]

# Function to calculate ROUGE score
def calculate_rouge(reference, candidate):
    return eval_metrics["rouge"].compute(predictions=[candidate], references=[reference])["rouge1"]

# Load model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained("XpeRay", load_in_4bit=True)
FastVisionModel.for_inference(model)

dataset = load_dataset("eltorio/ROCOv2-radiology", split="test[:100%]")
dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.3)))


kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_scores = []

# Open a file to save results
with open("results.txt", "w") as f:
    for fold_num, (train_index, test_index) in enumerate(kf.split(dataset), start=1):
        test_dataset = dataset.select(test_index)
        fold_scores = []

        fold_header = f"\n================== Fold {fold_num} ==================\n"
        print(fold_header)
        f.write(fold_header)

        for i, sample in enumerate(test_dataset, start=1):
            image, ground_truth_caption = sample["image"], sample["caption"]

            # Prepare model input
            instruction = (
                "You are an expert radiologist. Carefully analyze the provided medical image "
                "and provide a detailed, accurate, and professional radiology report. "
                "Describe any abnormalities, anatomical structures, and relevant findings using precise medical terminology.")
            messages = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": image}]}]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

            # Generate prediction
            output = model.generate(**inputs, max_new_tokens=128, use_cache=True)
            predicted_caption = tokenizer.decode(output[0], skip_special_tokens=True).split("assistant")[-1].strip()

            # Compute evaluation metrics
            bleu = calculate_bleu_score(ground_truth_caption, predicted_caption)
            bert = calculate_bertscore(ground_truth_caption, predicted_caption)
            rouge = calculate_rouge(ground_truth_caption, predicted_caption)

            fold_scores.append((bleu, bert, rouge))

            row_output = (
                f"[Sample {i}]\n"
                f"Ground Truth: {ground_truth_caption}\n"
                f"Predicted:    {predicted_caption}\n"
                f"BLEU: {bleu:.4f} | BERT: {bert:.4f} | ROUGE: {rouge:.4f}\n"
                f"{'-'*50}\n"
            )
            print(row_output)
            f.write(row_output)

        # Save fold average
        fold_avg = np.mean(fold_scores, axis=0)
        all_scores.append(fold_avg)
        fold_summary = f"Fold {fold_num} Average - BLEU: {fold_avg[0]:.4f}, BERT: {fold_avg[1]:.4f}, ROUGE: {fold_avg[2]:.4f}\n"
        print(fold_summary)
        f.write(fold_summary)

    # Compute and save overall averages
    average_scores = np.mean(all_scores, axis=0)
    final_summary = f"\nOverall Average Scores - BLEU: {average_scores[0]:.4f}, BERT: {average_scores[1]:.4f}, ROUGE: {average_scores[2]:.4f}\n"
    print(final_summary)
    f.write(final_summary)

print("✅ Results saved to results.txt")
