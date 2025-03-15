from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# Load the correct CodeBERT model
model = SentenceTransformer("microsoft/codebert-base")

def get_code_embedding(code_snippet):
    """
    Converts a code snippet into an embedding.
    """
    embedding = model.encode(code_snippet, convert_to_tensor=True)
    return embedding

# Sample C++ code snippets
code1 = """ 
#include <iostream>
using namespace std;
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}
int main() {
    int arr[] = {5, 3, 8, 6, 2};
    int n = 5;
    bubbleSort(arr, n);
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    return 0;
}
"""

code2 = """ 
#include <iostream>
using namespace std;
bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0)
            return false;
    }
    return true;
}
int main() {
    int num = 29;
    if (isPrime(num))
        cout << "Prime";
    else
        cout << "Not Prime";
    return 0;
}
"""

# Generate embeddings
embedding1 = get_code_embedding(code1)
embedding2 = get_code_embedding(code2)

# Compute Cosine Similarity
cosine_similarity = F.cosine_similarity(embedding1, embedding2, dim=0)
print(f"Cosine Similarity: {cosine_similarity.item()}")

# Compute Euclidean Distance
euclidean_distance = torch.dist(embedding1, embedding2, p=2)
print(f"Euclidean Distance: {euclidean_distance.item()}")
