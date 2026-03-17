# ⟨vec⟩ Text Embeddings

An interactive, beginner-friendly web application designed to teach **Text Embeddings** from first principles.

Learn how text gets transformed into numbers (vectors) for machine learning through live demos, visualizations, and plain-English explanations of the math involved.

**Live Demo:** [https://upendra-eth.github.io/text-embedding-app/](https://upendra-eth.github.io/text-embedding-app/)

---

## 🎯 Features

- **No Server Required:** All embedding math, tokenization, and vector manipulation runs entirely in the browser using pure JavaScript.
- **Data Type Journey:** Trace the exact data transformations from `string` → `tokens` → `float32 vectors`.
- **Live Tokenizer:** Type a sentence and watch it split into tokens and integer IDs in real time.
- **Math Visualizer:** Interactive sliders to understand Dot Product, L2 Norm, and Cosine Similarity.
- **Embedding Space:** 2D PCA scatter plot showing semantic clusters of words (e.g., Animals, Food, Tech).
- **Similarity Explorer:** Compare two sentences and get a live cosine similarity score with factor breakdowns.
- **Animated Mean Pooling:** A step-by-step canvas animation showing how token vectors are averaged into a single sentence vector.

## 🚀 How to Run Locally

You don't need any build tools or dependencies. Just serve the static files:

1. Clone the repository:
   ```bash
   git clone https://github.com/upendra-eth/text-embedding-app.git
   cd text-embedding-app
   ```
2. Open `index.html` in any modern web browser.
   *(Or spin up a simple local server if you prefer: `python -m http.server 8000`)*

## 🧠 What You'll Learn

1. **What is an Embedding?** (The GPS and Colour analogies)
2. **The Data Pipeline** (Strings to Vectors)
3. **Tokenization & Vocabularies**
4. **How Neural Networks Learn Vectors**
5. **The Math:** Cosine Similarity & Dot Products
6. **Vector Geometry and PCA**
7. **Semantic Similarity Scoring**
8. **Mean Pooling vs [CLS] Tokens**

## 🛠️ Built With

- **HTML5 & Vanilla CSS** (Custom Dark-mode Glassmorphism UI)
- **Vanilla JavaScript** (Custom Tokenizer and Embedding Engine)
- **Chart.js** (For PCA scatter plots and dimension bar charts)
- **MathJax** (For rendering LaTeX math formulas)

---

*Built as a learning project inspired by real-world embedding models like MiniLM and BERT.*
