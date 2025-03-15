from transformers import AutoTokenizer, AutoModel

# Load CodeT5 Model and Tokenizer
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
