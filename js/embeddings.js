// ============================================================
//  embeddings.js — Text Embedding Math Engine (Pure JS)
//  Covers: Tokenization → Token IDs → Vectors → Pooling → Norm
// ============================================================

// ─── 1. VOCABULARY & TOKEN IDs ─────────────────────────────
// In real models (like MiniLM), vocabulary has 30,000+ tokens.
// Here we use a curated 200-word vocabulary for demonstration.
const VOCABULARY = {
  // Common stop words
  "the": 1, "a": 2, "an": 3, "is": 4, "are": 5, "was": 6, "were": 7,
  "be": 8, "been": 9, "being": 10, "have": 11, "has": 12, "had": 13,
  "do": 14, "does": 15, "did": 16, "will": 17, "would": 18, "could": 19,
  "should": 20, "shall": 21, "may": 22, "might": 23, "must": 24,
  "can": 25, "to": 26, "of": 27, "in": 28, "for": 29, "on": 30,
  "with": 31, "at": 32, "by": 33, "from": 34, "this": 35, "that": 36,
  "it": 37, "its": 38, "i": 39, "he": 40, "she": 41, "they": 42,
  "we": 43, "you": 44, "not": 45, "no": 46, "but": 47, "if": 48,
  "or": 49, "and": 50, "as": 51, "up": 52, "out": 53, "so": 54,
  // Animals
  "cat": 100, "dog": 101, "bird": 102, "fish": 103, "lion": 104,
  "tiger": 105, "bear": 106, "wolf": 107, "fox": 108, "horse": 109,
  "rabbit": 110, "mouse": 111, "elephant": 112, "monkey": 113,
  // Food
  "pizza": 120, "burger": 121, "pasta": 122, "salad": 123, "soup": 124,
  "sushi": 125, "taco": 126, "bread": 127, "rice": 128, "cake": 129,
  "coffee": 130, "tea": 131, "juice": 132, "milk": 133, "water": 134,
  // Technology
  "computer": 140, "laptop": 141, "phone": 142, "robot": 143,
  "software": 144, "hardware": 145, "internet": 146, "network": 147,
  "data": 148, "code": 149, "program": 150, "algorithm": 151,
  "machine": 152, "learning": 153, "model": 154, "neural": 155,
  "vector": 156, "embedding": 157, "token": 158, "text": 159,
  // Emotions / adjectives
  "happy": 160, "sad": 161, "angry": 162, "excited": 163, "calm": 164,
  "good": 165, "bad": 166, "great": 167, "terrible": 168,
  "fast": 169, "slow": 170, "big": 171, "small": 172, "hot": 173,
  "cold": 174, "new": 175, "old": 176,
  // Nature
  "sun": 180, "moon": 181, "star": 182, "sky": 183, "ocean": 184,
  "river": 185, "mountain": 186, "forest": 187, "tree": 188,
  "flower": 189, "rain": 190, "snow": 191, "fire": 192, "earth": 193,
  // Actions
  "run": 200, "walk": 201, "jump": 202, "eat": 203, "drink": 204,
  "sleep": 205, "read": 206, "write": 207, "speak": 208, "think": 209,
  "love": 210, "hate": 211, "help": 212, "build": 213, "create": 214,
  "learn": 215, "teach": 216, "know": 217, "see": 218, "hear": 219,
  "[UNK]": 0  // Unknown token
};

const VOCAB_SIZE = Object.keys(VOCABULARY).length;
const EMBED_DIM = 16;  // We use 16-dim for visualization (real MiniLM = 384)

// ─── 2. PRE-SEEDED WORD VECTORS ─────────────────────────────
// In real models, these are learned by training on billions of words.
// We seed ours so that semantically similar words are close together.
// Each word gets a 16-dimensional float32 vector.
//
// MATH: A word vector v ∈ ℝ^16 means 16 numbers, each a float.
//       e.g.,  "cat" → [0.12, -0.43, 0.88, 0.21, ...]
//
// We generate them deterministically from the word ID using a
// seeded pseudo-random function, with semantic offsets for clusters.

function seedRandom(seed) {
  let x = Math.sin(seed + 1) * 10000;
  return x - Math.floor(x);
}

function generateWordVector(wordId, semanticBias = null) {
  const vec = [];
  for (let i = 0; i < EMBED_DIM; i++) {
    vec.push((seedRandom(wordId * 31 + i * 17) * 2 - 1) * 0.3);
  }
  // Apply semantic bias to cluster related words together
  if (semanticBias) {
    for (let i = 0; i < EMBED_DIM; i++) {
      vec[i] += semanticBias[i] * 0.6;
    }
  }
  return normalize(vec);
}

// Semantic cluster centers (biases that pull related words together)
const SEMANTIC_CLUSTERS = {
  // Animals cluster: strong in dim 0, 1
  animals:    [ 0.8,  0.6, -0.1,  0.0,  0.1, -0.2,  0.0,  0.1, -0.1,  0.0,  0.0,  0.0,  0.0, -0.1,  0.0,  0.0],
  // Food cluster: strong in dim 2, 3
  food:       [-0.1,  0.0,  0.8,  0.7,  0.0,  0.0, -0.1,  0.0,  0.1,  0.0,  0.0,  0.0,  0.1,  0.0, -0.1,  0.0],
  // Tech cluster: strong in dim 4, 5
  tech:       [ 0.0, -0.1,  0.0,  0.0,  0.9,  0.7,  0.0,  0.1,  0.0, -0.1,  0.0,  0.0,  0.0,  0.0,  0.1,  0.0],
  // Emotion/adj: strong in dim 6, 7
  emotion:    [ 0.0,  0.0, -0.1,  0.1,  0.0,  0.0,  0.8,  0.6,  0.0,  0.0,  0.1, -0.1,  0.0,  0.0,  0.0,  0.0],
  // Nature: strong in dim 8, 9
  nature:     [ 0.1,  0.0,  0.0, -0.1,  0.0,  0.1,  0.0,  0.0,  0.9,  0.7,  0.0,  0.0, -0.1,  0.0,  0.0,  0.0],
  // Actions: strong in dim 10, 11
  actions:    [ 0.0,  0.0,  0.0,  0.0, -0.1,  0.0,  0.1,  0.0,  0.0,  0.0,  0.8,  0.6,  0.0,  0.1,  0.0, -0.1],
};

function getSemanticCluster(wordId) {
  if (wordId >= 100 && wordId <= 113) return SEMANTIC_CLUSTERS.animals;
  if (wordId >= 120 && wordId <= 134) return SEMANTIC_CLUSTERS.food;
  if (wordId >= 140 && wordId <= 159) return SEMANTIC_CLUSTERS.tech;
  if (wordId >= 160 && wordId <= 176) return SEMANTIC_CLUSTERS.emotion;
  if (wordId >= 180 && wordId <= 193) return SEMANTIC_CLUSTERS.nature;
  if (wordId >= 200 && wordId <= 219) return SEMANTIC_CLUSTERS.actions;
  return null;
}

// Build the full embedding table
const EMBEDDING_TABLE = {};
for (const [word, id] of Object.entries(VOCABULARY)) {
  EMBEDDING_TABLE[word] = generateWordVector(id, getSemanticCluster(id));
}


// ─── 3. TOKENIZER ────────────────────────────────────────────
// Input:  string  e.g. "The cat sat on the mat!"
// Output: string[] e.g. ["the", "cat", "sat", "on", "the", "mat"]
//
// Real tokenizers (BPE, WordPiece) are more complex — they split
// unknown words into sub-word pieces. Ours uses simple word splitting.
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')  // remove punctuation
    .split(/\s+/)
    .filter(t => t.length > 0);
}

// Input:  string[]  (tokens)
// Output: number[]  (integer token IDs from vocabulary)
function tokensToIds(tokens) {
  return tokens.map(t => VOCABULARY[t] !== undefined ? VOCABULARY[t] : VOCABULARY["[UNK]"]);
}

// Combined
function encodeText(text) {
  const tokens = tokenize(text);
  const ids = tokensToIds(tokens);
  return { tokens, ids };
}


// ─── 4. LOOKUP EMBEDDINGS ─────────────────────────────────────
// Input:  string[] (tokens)
// Output: number[][] (array of float32 vectors, each EMBED_DIM long)
//
// This is the "embedding lookup" — for each token, we grab its row
// from the embedding matrix (a learned lookup table in real models).
function lookupEmbeddings(tokens) {
  return tokens.map(t => {
    const key = t in EMBEDDING_TABLE ? t : "[UNK]";
    return [...EMBEDDING_TABLE[key]];
  });
}


// ─── 5. MEAN POOLING ─────────────────────────────────────────
// Problem: a sentence has N tokens, each with its own vector.
//          We need ONE vector to represent the whole sentence.
//
// Solution: Mean Pooling
//   v_sentence = (1/N) × Σ(v_token_i)   for i in 1..N
//
// This averages element-wise across all token vectors.
// The article uses exactly this approach for MiniLM.
//
// Input:  number[][] (list of token vectors)
// Output: number[]   (single pooled sentence vector)
function meanPool(vectors) {
  if (vectors.length === 0) return new Array(EMBED_DIM).fill(0);
  const pooled = new Array(EMBED_DIM).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < EMBED_DIM; i++) {
      pooled[i] += vec[i];
    }
  }
  for (let i = 0; i < EMBED_DIM; i++) {
    pooled[i] /= vectors.length;
  }
  return pooled;
}


// ─── 6. L2 NORMALIZATION ─────────────────────────────────────
// After pooling we normalize the vector to unit length.
//
// L2 Norm:  ‖v‖ = √(Σ vᵢ²)
// Normalize: v̂ = v / ‖v‖
//
// Why? So that cosine similarity only measures DIRECTION (meaning),
// not magnitude (word count / length of sentence).
// Two sentences with the same meaning should point in the same direction.
//
// Input:  number[] (any vector)
// Output: number[] (same direction, length = 1.0)
function normalize(vec) {
  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
  if (norm === 0) return vec;
  return vec.map(x => x / norm);
}


// ─── 7. COSINE SIMILARITY ────────────────────────────────────
// Measures how "similar" two vectors are based on the angle between them.
//
// Formula:  cos(θ) = (A · B) / (‖A‖ × ‖B‖)
//
// Where A · B = Σ aᵢ × bᵢ   (dot product)
//
// Output range: -1 (opposite) → 0 (unrelated) → +1 (identical)
//
// After L2 normalization, ‖A‖ = ‖B‖ = 1, so:
//   cos(θ) = A · B   (just the dot product!)
//
// Input:  number[], number[]
// Output: number in [-1, 1]
function cosineSimilarity(a, b) {
  const normA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
  const normB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
  if (normA === 0 || normB === 0) return 0;
  const dot = a.reduce((s, x, i) => s + x * b[i], 0);
  return dot / (normA * normB);
}

function dotProduct(a, b) {
  return a.reduce((s, x, i) => s + x * b[i], 0);
}

function l2Norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}


// ─── 8. FULL PIPELINE ────────────────────────────────────────
// String → Embedding (the full journey, step by step)
//
// Step 1: string → tokens (tokenize)
// Step 2: tokens → IDs   (vocabulary lookup)
// Step 3: IDs → vectors  (embedding table lookup)
// Step 4: vectors → pooled vector (mean pooling)
// Step 5: pooled → unit vector (L2 normalize)
//
// Output: { tokens, ids, tokenVectors, pooled, normalized }
function embedText(text) {
  const { tokens, ids } = encodeText(text);
  const tokenVectors = lookupEmbeddings(tokens);
  const pooled = meanPool(tokenVectors);
  const normalized = normalize(pooled);
  return { tokens, ids, tokenVectors, pooled, normalized };
}


// ─── 9. SIMPLE 2D PCA ────────────────────────────────────────
// PCA (Principal Component Analysis) reduces high-dimensional vectors
// to 2D so we can plot them. We use a simplified version.
//
// Real PCA: finds eigenvectors of the covariance matrix.
// Our version: uses dim 0 and dim 4 as the two most semantically
// distinct axes (based on our cluster design above), plus a tiny
// amount of all other dims for spread.
function projectTo2D(vec) {
  // x = weighted combo of animal/food/tech dims (0-1, 2-3, 4-5)
  // y = weighted combo of emotion/nature/action dims (6-7, 8-9, 10-11)
  const x = vec[0] * 0.5 + vec[2] * 0.3 + vec[4] * 0.2 +
            vec[1] * 0.15 + vec[3] * 0.1;
  const y = vec[6] * 0.5 + vec[8] * 0.3 + vec[10] * 0.2 +
            vec[7] * 0.15 + vec[9] * 0.1;
  return { x, y };
}


// ─── 10. SIMILARITY SCORE LABELS ─────────────────────────────
function similarityLabel(score) {
  if (score > 0.90) return { label: "Extremely Similar 🔥", color: "#10b981" };
  if (score > 0.75) return { label: "Very Similar ✅", color: "#34d399" };
  if (score > 0.55) return { label: "Moderately Similar 🔶", color: "#fbbf24" };
  if (score > 0.35) return { label: "Weakly Related 🔷", color: "#60a5fa" };
  if (score > 0.10) return { label: "Barely Related ❄️", color: "#a78bfa" };
  return { label: "Unrelated / Opposite ❌", color: "#f87171" };
}


// ─── EXPORTS (Global) ────────────────────────────────────────
window.EmbeddingEngine = {
  VOCABULARY,
  EMBEDDING_TABLE,
  EMBED_DIM,
  tokenize,
  tokensToIds,
  encodeText,
  lookupEmbeddings,
  meanPool,
  normalize,
  cosineSimilarity,
  dotProduct,
  l2Norm,
  embedText,
  projectTo2D,
  similarityLabel,
  SEMANTIC_CLUSTERS,
};
