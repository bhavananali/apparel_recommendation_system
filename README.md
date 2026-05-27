# Apparel Recommendation System

**ML · NLP · Computer Vision · Amazon Fashion Dataset**

> Similarity-based product retrieval across 16K+ apparel items using text and image embeddings — benchmarked across 7 models.

---

## Overview

Built a hybrid recommendation system for fashion products using the Amazon Tops dataset. The project progressively compares text-based retrieval (Bag of Words → TF-IDF → Word2Vec) with visual similarity (VGG16 CNN), ultimately combining both signals for a stronger recommender.

---

## Models Compared

| Model | Approach | Features Used |
|---|---|---|
| Bag of Words | Count-based similarity | Product title |
| TF-IDF | Weighted term frequency | Product title |
| IDF-weighted BoW | IDF-scaled counts | Product title |
| Avg Word2Vec | Dense word embeddings | Product title |
| IDF-weighted Word2Vec | Weighted W2V vectors | Product title |
| Brand + Colour + W2V | Hybrid text + metadata | Title, brand, color |
| **VGG16 CNN** | Visual feature extraction | Product images (25088-dim) |

Model quality evaluated using **average euclidean distance** across top-20 recommendations — lower is better.

---

## Pipeline

```
Raw JSON (180K products)
    ↓ Filter to 6 features (asin, brand, color, title, image_url, price)
    ↓ Remove nulls → 28K products
    ↓ Remove short titles
    ↓ Two-stage deduplication → 16K products
    ↓ NLP preprocessing (stopword removal, lowercasing)
    ↓ Feature extraction (TF-IDF / Word2Vec / VGG16)
    ↓ Cosine / Euclidean similarity → Top-N recommendations
```

---

## Dataset

- **Source:** Amazon Tops Fashion dataset (`tops_fashion.json`)
- **Raw size:** 180K products, 19 features
- **After cleaning:** ~16K unique products
- **Features used:** `asin`, `brand`, `color`, `product_type_name`, `medium_image_url`, `title`, `formatted_price`

---

## Key Techniques

- Two-stage deduplication using word-overlap heuristics
- Cyclical NLP preprocessing pipeline (stopword removal, lowercasing, special character stripping)
- TF-IDF and IDF-weighted Word2Vec for semantic similarity
- VGG16 (pretrained on ImageNet) for visual feature extraction — each image encoded as a 25,088-dim vector
- Cosine and euclidean pairwise distance for retrieval
- Heatmap visualisations to interpret word-level similarity between recommended pairs

---

## Tech Stack

- **NLP:** NLTK, scikit-learn (TF-IDF, CountVectorizer), Gensim (Word2Vec)
- **Vision:** Keras / TensorFlow, VGG16 (pretrained)
- **Data:** Pandas, NumPy, SciPy
- **Visualisation:** Matplotlib, Seaborn, Plotly

---

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset and pickle files from the links in the notebook, place them in a `pickels/` folder, then run `Apparel_recommendation.ipynb` cell by cell.

> Note: VGG16 feature extraction is compute-heavy (~few hours). Pre-extracted `.npy` files are linked inside the notebook.

---

## Results

Final model comparison plotted as average euclidean distance across all 7 approaches — the hybrid Brand + Colour + IDF-W2V and CNN models produce the most visually and semantically coherent recommendations.
