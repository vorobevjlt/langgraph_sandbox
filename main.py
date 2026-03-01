import random
import numpy as np
with open('/Users/justlikethat/langgraph-course/resourses/artical_web_rag.txt','r') as file:
    text = " ".join(line.rstrip() for line in file)

s = text

def rand_vec(dim):
    return np.random.uniform(-3.0, 3.0, size=dim)

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
dim = 3
lr = 0.05
s = text
# d[word] = (v_w, u_w)  center vec, context 
d = {}
# s is whole corpus
for tok in s:
    if tok not in d:
        d[tok] = (rand_vec(dim), rand_vec(dim))
vocab = list(d.keys())
w2i = {w:i for i,w in enumerate(vocab)}
V = np.stack([d[w][0] for w in vocab], axis=0)  # center vectors (|V|, dim)
U = np.stack([d[w][1] for w in vocab], axis=0)  # context vectors (|V|, dim
for _ in range(100):
    for i in range(len(s) - 5 + 1): 
        win = s[i:i+5]
        c_word = win[2]
        c = w2i[c_word]
    
        for j, t_word in enumerate(win):
            if j == 2: 
                continue
            t = w2i[t_word]
    
            v_c = V[c]                 # (dim,)
            z = U @ v_c                # (|V|,)
            p = softmax(z)             # (|V|,)
    
            # g = p - y
            g = p.copy()
            g[t] -= 1.0                # (|V|,)
    
            # gradients
            grad_v = U.T @ g           # (dim,)
            grad_U = np.outer(g, v_c)  # (|V|, dim)
    
            # SGD step
            V[c] -= lr * grad_v
            U    -= lr * grad_U
    loss = 0.0
    pair_count = 0
    for i in range(len(s) - 5 + 1):
        win = s[i:i+5]
        center = win[2]
        v_center = d[center][0]     # v_w
        logits = U @ v_center       # (|V|,)  logits for all context words
        logZ = logsumexp(logits)    # log sum exp over vocab
        for j, ctx in enumerate(win):
            if j == 2:
                continue
            # log p(ctx | center) = (u_ctx·v_center) - logZ
            u_ctx = d[ctx][1]
            loss -= (u_ctx @ v_center) - logZ   # negative log-likelihood
            pair_count += 1
    
    avg_loss = loss / max(pair_count, 1)
    print("loss:", loss, "avg_loss:", avg_loss, "pairs:", pair_count)