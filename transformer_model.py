import math
import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * inv
    out = xhat * gamma + beta
    cache = (x, xhat, mean, var, inv, gamma)
    return out, cache


def layernorm_backward(dout, cache):
    x, xhat, mean, var, inv, gamma = cache
    N = x.shape[-1]
    dbeta = dout.sum(axis=(0, 1))
    dgamma = (dout * xhat).sum(axis=(0, 1))
    dxhat = dout * gamma
    dvar = (dxhat * (x - mean) * -0.5 * inv**3).sum(axis=-1, keepdims=True)
    dmean = (dxhat * -inv).sum(axis=-1, keepdims=True) + dvar * (-2.0 / N) * (x - mean).sum(axis=-1, keepdims=True)
    dx = dxhat * inv + dvar * (2.0 / N) * (x - mean) + dmean * (1.0 / N)
    return dx, dgamma, dbeta


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)


def linear_forward(x, W, b):
    out = x @ W + b
    cache = (x, W)
    return out, cache


def linear_backward(dout, cache):
    x, W = cache
    dW = x.reshape(-1, x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
    db = dout.sum(axis=(0, 1))
    dx = dout @ W.T
    return dx, dW, db


def attention_forward(x, Wq, Wk, Wv, Wo, num_heads, mask):
    B, T, C = x.shape
    H = num_heads
    D = C // H
    q, q_cache = linear_forward(x, Wq, np.zeros(C))
    k, k_cache = linear_forward(x, Wk, np.zeros(C))
    v, v_cache = linear_forward(x, Wv, np.zeros(C))

    q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

    scale = np.float32(1.0 / math.sqrt(D))
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    scores = scores + mask
    probs = softmax(scores)
    att = probs @ v
    att = att.transpose(0, 2, 1, 3).reshape(B, T, C)
    out, out_cache = linear_forward(att, Wo, np.zeros(C))
    cache = (x, q, k, v, probs, att, q_cache, k_cache, v_cache, out_cache, Wq, Wk, Wv, Wo, H, D, mask)
    return out, cache


def attention_backward(dout, cache):
    (
        x, q, k, v, probs, att, q_cache, k_cache, v_cache,
        out_cache, Wq, Wk, Wv, Wo, H, D, mask
    ) = cache
    B, T, C = x.shape

    datt, dWo, dbo = linear_backward(dout, out_cache)
    datt = datt.reshape(B, T, H, D).transpose(0, 2, 1, 3)

    dprobs = datt @ v.transpose(0, 1, 3, 2)
    dv = probs.transpose(0, 1, 3, 2) @ datt

    dscores = probs * (dprobs - (dprobs * probs).sum(axis=-1, keepdims=True))
    dscores = dscores * (mask == 0)
    dscores = dscores / math.sqrt(D)

    dq = dscores @ k
    dk = dscores.transpose(0, 1, 3, 2) @ q

    dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
    dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
    dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)

    dx_q, dWq, _ = linear_backward(dq, q_cache)
    dx_k, dWk, _ = linear_backward(dk, k_cache)
    dx_v, dWv, _ = linear_backward(dv, v_cache)

    dx = dx_q + dx_k + dx_v
    return dx, dWq, dWk, dWv, dWo, dbo


class TinyTransformer:
    def __init__(self, vocab_size=256, block_size=64, d_model=96, num_heads=2, num_layers=2, seed=1):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_emb = (np.random.randn(vocab_size, d_model) * 0.02).astype(np.float32)
        self.pos_emb = (np.random.randn(block_size, d_model) * 0.02).astype(np.float32)
        self.ln1_g = [np.ones(d_model, dtype=np.float32) for _ in range(num_layers)]
        self.ln1_b = [np.zeros(d_model, dtype=np.float32) for _ in range(num_layers)]
        self.ln2_g = [np.ones(d_model, dtype=np.float32) for _ in range(num_layers)]
        self.ln2_b = [np.zeros(d_model, dtype=np.float32) for _ in range(num_layers)]

        self.Wq = [(np.random.randn(d_model, d_model) * 0.02).astype(np.float32) for _ in range(num_layers)]
        self.Wk = [(np.random.randn(d_model, d_model) * 0.02).astype(np.float32) for _ in range(num_layers)]
        self.Wv = [(np.random.randn(d_model, d_model) * 0.02).astype(np.float32) for _ in range(num_layers)]
        self.Wo = [(np.random.randn(d_model, d_model) * 0.02).astype(np.float32) for _ in range(num_layers)]

        self.W1 = [(np.random.randn(d_model, d_model * 4) * 0.02).astype(np.float32) for _ in range(num_layers)]
        self.b1 = [np.zeros(d_model * 4, dtype=np.float32) for _ in range(num_layers)]
        self.W2 = [(np.random.randn(d_model * 4, d_model) * 0.02).astype(np.float32) for _ in range(num_layers)]
        self.b2 = [np.zeros(d_model, dtype=np.float32) for _ in range(num_layers)]

        self.Wout = (np.random.randn(d_model, vocab_size) * 0.02).astype(np.float32)
        self.bout = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, idx, mask, cache_out=False):
        B, T = idx.shape
        x = self.token_emb[idx] + self.pos_emb[:T]
        caches = []
        for i in range(self.num_layers):
            x_ln1, ln1_cache = layernorm_forward(x, self.ln1_g[i], self.ln1_b[i])
            att, att_cache = attention_forward(x_ln1, self.Wq[i], self.Wk[i], self.Wv[i], self.Wo[i], self.num_heads, mask)
            x = x + att
            x_ln2, ln2_cache = layernorm_forward(x, self.ln2_g[i], self.ln2_b[i])
            h, h_cache = linear_forward(x_ln2, self.W1[i], self.b1[i])
            h_relu = np.maximum(0, h)
            y, y_cache = linear_forward(h_relu, self.W2[i], self.b2[i])
            x = x + y
            caches.append((ln1_cache, att_cache, ln2_cache, h_cache, h_relu, y_cache))

        logits, out_cache = linear_forward(x, self.Wout, self.bout)
        if cache_out:
            return logits, (idx, x, caches, out_cache)
        return logits

    def backward(self, dlogits, cache):
        idx, x, caches, out_cache = cache
        grads = {
            "token_emb": np.zeros_like(self.token_emb),
            "pos_emb": np.zeros_like(self.pos_emb),
            "ln1_g": [np.zeros_like(g) for g in self.ln1_g],
            "ln1_b": [np.zeros_like(b) for b in self.ln1_b],
            "ln2_g": [np.zeros_like(g) for g in self.ln2_g],
            "ln2_b": [np.zeros_like(b) for b in self.ln2_b],
            "Wq": [np.zeros_like(w) for w in self.Wq],
            "Wk": [np.zeros_like(w) for w in self.Wk],
            "Wv": [np.zeros_like(w) for w in self.Wv],
            "Wo": [np.zeros_like(w) for w in self.Wo],
            "W1": [np.zeros_like(w) for w in self.W1],
            "b1": [np.zeros_like(b) for b in self.b1],
            "W2": [np.zeros_like(w) for w in self.W2],
            "b2": [np.zeros_like(b) for b in self.b2],
            "Wout": np.zeros_like(self.Wout),
            "bout": np.zeros_like(self.bout),
        }

        dx, dWout, dbout = linear_backward(dlogits, out_cache)
        grads["Wout"] = dWout
        grads["bout"] = dbout

        for i in reversed(range(self.num_layers)):
            ln1_cache, att_cache, ln2_cache, h_cache, h_relu, y_cache = caches[i]
            dy = dx
            dh_relu, dW2, db2 = linear_backward(dy, y_cache)
            dh = dh_relu.copy()
            dh[h_relu <= 0] = 0
            dx_ln2, dW1, db1 = linear_backward(dh, h_cache)
            dx_ln2, dln2_g, dln2_b = layernorm_backward(dx_ln2, ln2_cache)
            dx = dx + dx_ln2
            dx_att, dWq, dWk, dWv, dWo, _ = attention_backward(dx, att_cache)
            dx_ln1, dln1_g, dln1_b = layernorm_backward(dx_att, ln1_cache)
            dx = dx + dx_ln1

            grads["W1"][i] = dW1
            grads["b1"][i] = db1
            grads["W2"][i] = dW2
            grads["b2"][i] = db2
            grads["Wq"][i] = dWq
            grads["Wk"][i] = dWk
            grads["Wv"][i] = dWv
            grads["Wo"][i] = dWo
            grads["ln1_g"][i] = dln1_g
            grads["ln1_b"][i] = dln1_b
            grads["ln2_g"][i] = dln2_g
            grads["ln2_b"][i] = dln2_b

        B, T = idx.shape
        dtoken = dx
        for b in range(B):
            for t in range(T):
                grads["token_emb"][idx[b, t]] += dtoken[b, t]
        grads["pos_emb"][:T] += dtoken.sum(axis=0)
        return grads

    def save(self, path):
        np.savez_compressed(
            path,
            token_emb=self.token_emb,
            pos_emb=self.pos_emb,
            ln1_g=np.array(self.ln1_g, dtype=object),
            ln1_b=np.array(self.ln1_b, dtype=object),
            ln2_g=np.array(self.ln2_g, dtype=object),
            ln2_b=np.array(self.ln2_b, dtype=object),
            Wq=np.array(self.Wq, dtype=object),
            Wk=np.array(self.Wk, dtype=object),
            Wv=np.array(self.Wv, dtype=object),
            Wo=np.array(self.Wo, dtype=object),
            W1=np.array(self.W1, dtype=object),
            b1=np.array(self.b1, dtype=object),
            W2=np.array(self.W2, dtype=object),
            b2=np.array(self.b2, dtype=object),
            Wout=self.Wout,
            bout=self.bout,
            meta=np.array([self.vocab_size, self.block_size, self.d_model, self.num_heads, self.num_layers]),
        )

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=True)
        vocab_size, block_size, d_model, num_heads, num_layers = data["meta"].tolist()
        model = TinyTransformer(vocab_size, block_size, d_model, num_heads, num_layers)
        model.token_emb = data["token_emb"]
        model.pos_emb = data["pos_emb"]
        model.ln1_g = list(data["ln1_g"])
        model.ln1_b = list(data["ln1_b"])
        model.ln2_g = list(data["ln2_g"])
        model.ln2_b = list(data["ln2_b"])
        model.Wq = list(data["Wq"])
        model.Wk = list(data["Wk"])
        model.Wv = list(data["Wv"])
        model.Wo = list(data["Wo"])
        model.W1 = list(data["W1"])
        model.b1 = list(data["b1"])
        model.W2 = list(data["W2"])
        model.b2 = list(data["b2"])
        model.Wout = data["Wout"]
        model.bout = data["bout"]
        return model
