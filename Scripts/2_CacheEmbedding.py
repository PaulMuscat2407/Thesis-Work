from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your dataset of reference texts
df = pd.read_csv('Semantic_Similarity/assistant_messages.csv')
references = df['content'].tolist()

print("Encoding")
# Compute embeddings for all references
reference_embeddings = model.encode(references)
print("Finished")

# Save the embeddings to a file
np.save('reference_embeddings.npy', reference_embeddings)