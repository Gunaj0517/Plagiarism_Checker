from transformers import AutoTokenizer, AutoModel
import torch

# GraphCodeBERT Model Load Kar
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

print("GraphCodeBERT Model Loaded Successfully!")
