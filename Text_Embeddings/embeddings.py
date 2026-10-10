from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

input_file = DATASET_DIR / "generated_text.csv"
if not input_file.exists():
    raise FileNotFoundError(
        f"Could not find {input_file}. Add a CSV with columns 'sentence_id' and 'text'."
    )

df = pd.read_csv(input_file)
if "text" not in df.columns:
    raise ValueError("generated_text.csv must contain a column named 'text'.")

df = df.dropna(subset=["text"]).copy()
sentences = df["text"].astype(str).tolist()
if not sentences:
    raise ValueError("No sentences were found in generated_text.csv.")

print("Loading Sentence Transformer model: all-MiniLM-L6-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    sentences, convert_to_numpy=True, normalize_embeddings=True
)

np.save(DATASET_DIR / "embeddings.npy", embeddings)
print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)
print("Embedding dimension:", embeddings.shape[1])

# Because embeddings are normalized, their dot product equals cosine similarity.
similarity_matrix = embeddings @ embeddings.T
similarity_rows = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        similarity_rows.append({
            "sentence_id_1": int(df.iloc[i]["sentence_id"]) if "sentence_id" in df.columns else i + 1,
            "sentence_id_2": int(df.iloc[j]["sentence_id"]) if "sentence_id" in df.columns else j + 1,
            "sentence_1": sentences[i],
            "sentence_2": sentences[j],
            "cosine_similarity": float(similarity_matrix[i, j])
        })

scores = pd.DataFrame(similarity_rows).sort_values(
    "cosine_similarity", ascending=False
)
scores.to_csv(DATASET_DIR / "similarity_scores.csv", index=False)

pairs_file = DATASET_DIR / "sentence_pairs.csv"
if pairs_file.exists():
    pairs = pd.read_csv(pairs_file)
    results = []
    id_to_position = {
        int(row["sentence_id"]): pos
        for pos, (_, row) in enumerate(df.iterrows())
    } if "sentence_id" in df.columns else {i + 1: i for i in range(len(sentences))}
    for _, row in pairs.iterrows():
        id1, id2 = int(row["sentence_id_1"]), int(row["sentence_id_2"])
        if id1 in id_to_position and id2 in id_to_position:
            i, j = id_to_position[id1], id_to_position[id2]
            results.append({
                "sentence_id_1": id1,
                "sentence_id_2": id2,
                "sentence_1": sentences[i],
                "sentence_2": sentences[j],
                "cosine_similarity": float(similarity_matrix[i, j])
            })
    pd.DataFrame(results).to_csv(pairs_file, index=False)

print("\nTop 5 most similar sentence pairs:")
print(scores.head(5)[["sentence_1", "sentence_2", "cosine_similarity"]].to_string(index=False))
print("\nSaved output files:")
print("- dataset/embeddings.npy")
print("- dataset/similarity_scores.csv")
print("- dataset/sentence_pairs.csv")
