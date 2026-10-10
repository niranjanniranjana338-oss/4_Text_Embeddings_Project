# Week 4 – Text Embeddings

**Task:** Generating and Comparing Text Embeddings using Sentence Transformers

## Objective
- Generate sentence embeddings using a pre-trained Sentence Transformer.
- Compare sentence meaning using cosine similarity.
- Understand embedding dimensions and prepare data for semantic search and RAG.

## Model
`all-MiniLM-L6-v2`

## Folder Structure
```text
Text_Embeddings/
├── dataset/
│   ├── generated_text.csv
│   ├── embeddings.npy          # generated after running the program
│   ├── similarity_scores.csv   # generated after running the program
│   └── sentence_pairs.csv
├── notebook/
│   └── Week4_Text_Embeddings.ipynb
├── embeddings.py
├── README.md
└── requirements.txt
```

## Installation
Use Python 3.10 or 3.11 if possible. Open a terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
```

## Run the Python program
```bash
python embeddings.py
```

The first run downloads the `all-MiniLM-L6-v2` model and requires internet access.

## Dataset and outputs
`dataset/generated_text.csv` contains 20 practice sentences. The dataset folder includes all four expected files. `embeddings.npy`, `similarity_scores.csv`, and the results in `sentence_pairs.csv` are included as sample artifacts so the folder structure is complete. When you run the notebook or `embeddings.py`, these output files are regenerated using the selected Sentence Transformer model.

## Expected observation
For 20 sentences, the embedding array should have 20 rows and 384 columns, so its shape should be `(20, 384)`. Similar sentences generally have higher cosine similarity scores than unrelated sentences. Exact scores can vary with model/library versions.

## Week 4 Observation
Text embeddings convert sentences into numerical vectors that capture semantic meaning. Comparing these vectors helps identify related content and is useful for semantic search, recommendation systems, and future Retrieval-Augmented Generation (RAG) applications.
