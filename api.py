from flask import Flask, request, jsonify
from transformers import RobertaModel, RobertaTokenizer
import torch

app = Flask(__name__)

# Load GraphCodeBERT model and tokenizer
MODEL_NAME = "microsoft/graphcodebert-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME)

def get_code_embedding(code):
    tokens = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1)  # Get the embedding

def compute_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

@app.route("/compare", methods=["POST"])
def compare_code():
    data = request.get_json()
    code1 = data.get("code1", "")
    code2 = data.get("code2", "")

    if not code1 or not code2:
        return jsonify({"error": "Both code snippets are required"}), 400

    # Convert C++ code to embeddings
    embedding1 = get_code_embedding(code1)
    embedding2 = get_code_embedding(code2)

    # Compute similarity score (1 = identical, 0 = completely different)
    similarity_score = compute_similarity(embedding1, embedding2)

    return jsonify({"similarity": similarity_score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


""" curl -X POST http://127.0.0.1:5000/compare \
     -H "Content-Type: application/json" \
     -d "{\"code1\": \"#include <iostream>\\nusing namespace std;\\nvoid bubbleSort(int arr[], int n) { for (int i = 0; i < n - 1; i++) { for (int j = 0; j < n - i - 1; j++) { if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]); } } }\\nint main() { int arr[] = {5, 3, 8, 6, 2}; int n = 5; bubbleSort(arr, n); for (int i = 0; i < n; i++) cout << arr[i] << \\\" \\\"; return 0; }\", \"code2\": \"#include <iostream>\\nusing namespace std;\\nbool isPrime(int n) { if (n <= 1) return false; for (int i = 2; i * i <= n; i++) { if (n % i == 0) return false; } return true; }\\nint main() { int num = 29; if (isPrime(num)) cout << \\\"Prime\\\"; else cout << \\\"Not Prime\\\"; return 0; }\"}"
"""

""" curl -X POST http://127.0.0.1:5000/compare \
     -H "Content-Type: application/json" \
     -d "{\"code1\": \"#include <iostream>\\nusing namespace std;\\nint gcd(int a, int b) { while (b != 0) { int temp = b; b = a % b; a = temp; } return a; }\\nint main() { int a = 36, b = 60; cout << gcd(a, b); return 0; }\", \"code2\": \"#include <iostream>\\nusing namespace std;\\nint findGCD(int x, int y) { if (y == 0) return x; return findGCD(y, x % y); }\\nint main() { int x = 36, y = 60; cout << findGCD(x, y); return 0; }\"}"
"""