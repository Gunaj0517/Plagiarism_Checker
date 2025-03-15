from transformers import AutoTokenizer, AutoModel
import torch

# ✅ Load CodeT5 tokenizer and model
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_code_embedding(code_snippet):
    tokens = tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True)
    
    # ✅ Explicitly create decoder_input_ids (important fix!)
    decoder_input_ids = torch.zeros_like(tokens["input_ids"])  # Dummy decoder input

    with torch.no_grad():
        output = model(**tokens, decoder_input_ids=decoder_input_ids, past_key_values=None)

    return output.last_hidden_state.mean(dim=1)  # Take mean embedding

# Example Code Snippets
code1 = "def add(a, b): return a + b"
code2 = "def sum(x, y): return x + y"

# ✅ Get embeddings
embedding1 = get_code_embedding(code1)
embedding2 = get_code_embedding(code2)

print(embedding1.shape)  # Expected: torch.Size([1, 768])
