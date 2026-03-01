from pathlib import Path
import re
import numpy as np


TEXT_PATH = Path("resourses/artical_web_rag.txt")
WINDOW_SIZE = 5
EMBED_DIM = 16
LEARNING_RATE = 0.05
EPOCHS = 50
SEED = 7


def read_corpus(path: Path) -> str:
    """Load corpus text from file, otherwise use a small fallback corpus."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "custom word2vec workflow learns vectors by predicting nearby words. "
        "skip gram updates center and context embeddings using softmax loss. "
        "better embeddings place similar words close together in vector space."
    )


def tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits for the whole vocabulary into probabilities."""
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()


def full_vocab_logits(center_vector: np.ndarray, context_matrix: np.ndarray) -> np.ndarray:
    """Compute dot products of one center vector against the whole vocabulary.

    For skip-gram with full softmax:
      logits[i] = u_i · v_center, for every vocabulary word i.
    """
    return context_matrix @ center_vector


def build_training_pairs(tokens: list[str], window_size: int) -> list[tuple[str, str]]:
    """Create (center, context) pairs from sliding windows."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    half = window_size // 2
    pairs: list[tuple[str, str]] = []
    for i in range(half, len(tokens) - half):
        center = tokens[i]
        for j in range(i - half, i + half + 1):
            if j == i:
                continue
            pairs.append((center, tokens[j]))
    return pairs


def train_skipgram(tokens: list[str], dim: int, lr: float, epochs: int, window_size: int):
    """Train a tiny full-softmax skip-gram model with SGD."""
    vocab = sorted(set(tokens))
    w2i = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)

    rng = np.random.default_rng(SEED)
    V = rng.uniform(-0.5, 0.5, size=(vocab_size, dim))  # center embeddings
    U = rng.uniform(-0.5, 0.5, size=(vocab_size, dim))  # context embeddings

    pairs = build_training_pairs(tokens, window_size)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        rng.shuffle(pairs)

        for center_word, context_word in pairs:
            c = w2i[center_word]
            t = w2i[context_word]

            v_c = V[c]
            logits = full_vocab_logits(v_c, U)
            probs = softmax(logits)

            loss = -np.log(probs[t] + 1e-12)
            total_loss += loss

            grad = probs.copy()
            grad[t] -= 1.0

            grad_v = U.T @ grad
            grad_U = np.outer(grad, v_c)

            V[c] -= lr * grad_v
            U -= lr * grad_U

        avg_loss = total_loss / len(pairs)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch={epoch:>3} avg_loss={avg_loss:.4f}")

    return vocab, w2i, V, U


def nearest_neighbors(
    word: str,
    vocab: list[str],
    vectors: np.ndarray,
    word_to_index: dict[str, int],
    top_k: int = 5,
):
    """Return nearest words by cosine similarity from center embeddings."""
    if word not in word_to_index:
        return []
    idx = word_to_index[word]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    normalized = vectors / norms
    sims = normalized @ normalized[idx]

    best = np.argsort(-sims)
    return [(vocab[i], float(sims[i])) for i in best if i != idx][:top_k]


def main() -> None:
    text = read_corpus(TEXT_PATH)
    tokens = tokenize(text)

    if len(tokens) < WINDOW_SIZE:
        raise ValueError("Corpus is too small for the selected window size.")

    vocab, w2i, center_vectors, _ = train_skipgram(
        tokens=tokens,
        dim=EMBED_DIM,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        window_size=WINDOW_SIZE,
    )

    probe = "workflow"
    neighbors = nearest_neighbors(probe, vocab, center_vectors, w2i, top_k=5)
    if neighbors:
        print(f"\nNearest words to '{probe}':")
        for word, sim in neighbors:
            print(f"  {word:<15} cosine={sim:.3f}")


if __name__ == "__main__":
    main()
