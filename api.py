from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

# Load GraphCodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

# Function to Get Code Embedding
def get_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return embedding

# API Route to Compare Code
@app.route('/compare', methods=['POST'])
def compare_code():
    data = request.json
    code1, code2 = data["code1"], data["code2"]

    embed1 = get_embedding(code1)
    embed2 = get_embedding(code2)
    
    similarity = torch.nn.functional.cosine_similarity(embed1, embed2).item()
    
    return jsonify({"similarity_score": similarity})

if __name__ == '__main__':
    app.run(port=5000)
