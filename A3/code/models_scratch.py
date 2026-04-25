"""NumPy autoencoder implementation with hand-written backpropagation.

Supports both:
  - Single-layer (Model 1): input -> bottleneck -> output
  - Two-hidden-layer (Model 2): input -> hidden -> bottleneck -> hidden -> output

The architecture is fully parameterised by `hidden_dims`: a list of ints
describing every layer between input and bottleneck (the decoder mirrors the
encoder). This satisfies the assignment requirement of "do not write separate
code for each architecture".

Forward pass:
  z[0] = X  (input)
  a[l] = z[l-1] @ W[l] + b[l]
  z[l] = activation(a[l])          (except output layer: linear)

Loss (MSE):
  L = (1/N) * ||z[-1] - X||^2

Backward pass (chain rule through each layer):
  delta[-1] = (2/N) * (z[-1] - X)  -- dL/da at output (linear)
  For hidden l (going backwards):
    delta[l] = (delta[l+1] @ W[l+1].T) * activation'(a[l])
  Gradients:
    dW[l] = z[l-1].T @ delta[l]
    db[l] = sum(delta[l], axis=0)
"""

import numpy as np
from typing import List, Optional, Tuple


# ── Activation functions and their derivatives ────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in exp for very negative values
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_grad(z: np.ndarray) -> np.ndarray:
    """d/dx sigmoid(x) = sigma(x) * (1 - sigma(x)), evaluated at sigma(x)=z."""
    return z * (1.0 - z)


def _tanh_grad(z: np.ndarray) -> np.ndarray:
    """d/dx tanh(x) = 1 - tanh(x)^2, evaluated at tanh(x)=z."""
    return 1.0 - z ** 2


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    """d/dx relu(x) = 1 if x>0 else 0; since z=relu(x), z>0 iff x>0."""
    return (z > 0).astype(z.dtype)


_ACTIVATIONS = {
    "sigmoid": (_sigmoid,   _sigmoid_grad),
    "tanh":    (np.tanh,    _tanh_grad),
    "relu":    (_relu,      _relu_grad),
}


# ── Weight initialisation ─────────────────────────────────────────────────────

def _init_weights(
    in_dim: int,
    out_dim: int,
    activation: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """He init for ReLU, Xavier/Glorot for sigmoid/tanh.

    He init:  W ~ N(0, sqrt(2/fan_in))       (better for ReLU)
    Glorot:   W ~ U(-sqrt(6/(fan_in+fan_out)), ...) (better for tanh/sigmoid)
    """
    if activation == "relu":
        std = np.sqrt(2.0 / in_dim)
        W = rng.normal(0, std, (in_dim, out_dim)).astype(np.float64)
    else:
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        W = rng.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float64)
    b = np.zeros(out_dim, dtype=np.float64)
    return W, b


# ── Autoencoder class ─────────────────────────────────────────────────────────

class NumpyAutoencoder:
    """Fully connected autoencoder implemented in NumPy.

    The encoder has layers: input_dim -> hidden_dims[0] -> ... -> bottleneck_dim
    The decoder mirrors: bottleneck_dim -> hidden_dims[-1] -> ... -> input_dim
    The output layer is always linear (no activation).

    Parameters
    ----------
    input_dim : int
        Flat feature dimension.
    hidden_dims : list of int
        Sizes of hidden layers between input and bottleneck (encoder side).
        - [] for Model 1 (single-layer): direct input -> bottleneck -> output
        - [256] for Model 2 (two hidden layers): adds one hidden on each side
    bottleneck_dim : int
        Size of the compressed representation.
    activation : str
        'sigmoid', 'tanh', or 'relu'.
    lr : float
        Learning rate for SGD.
    seed : int
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        bottleneck_dim: int,
        activation: str = "relu",
        lr: float = 0.01,
        grad_clip: float = 1.0,
        seed: int = 42,
    ):
        self.input_dim     = input_dim
        self.hidden_dims   = hidden_dims
        self.bottleneck_dim = bottleneck_dim
        self.activation    = activation
        self.lr            = lr
        self.grad_clip     = grad_clip

        act_fn, act_grad = _ACTIVATIONS[activation]
        self._act    = act_fn
        self._actg   = act_grad

        # Build layer dimensions:
        # encoder: [input_dim, *hidden_dims, bottleneck_dim]
        # decoder: [bottleneck_dim, *hidden_dims[::-1], input_dim]
        enc_dims = [input_dim] + hidden_dims + [bottleneck_dim]
        dec_dims = [bottleneck_dim] + hidden_dims[::-1] + [input_dim]
        all_dims = enc_dims + dec_dims[1:]  # share bottleneck node

        rng = np.random.default_rng(seed)
        self.Ws: List[np.ndarray] = []
        self.bs: List[np.ndarray] = []
        self.n_layers = len(all_dims) - 1

        for i in range(self.n_layers):
            act = activation if i < self.n_layers - 1 else "linear"
            W, b = _init_weights(all_dims[i], all_dims[i + 1], activation, rng)
            self.Ws.append(W)
            self.bs.append(b)

        # Remember which layers are hidden vs output for backprop
        self._output_layer = self.n_layers - 1

    # ── Forward ──────────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run forward pass, caching pre-activations and post-activations.

        Returns
        -------
        pre_acts : list of length n_layers
            Linear combinations a[l] = z[l-1] @ W[l] + b[l]
        post_acts : list of length n_layers + 1
            z[0] = X, z[l] = activation(a[l]) (output layer: linear)
        """
        pre_acts  = []
        post_acts = [X]

        z = X
        for l in range(self.n_layers):
            a = z @ self.Ws[l] + self.bs[l]
            pre_acts.append(a)
            if l < self._output_layer:
                z = self._act(a)
            else:
                z = a   # output layer is linear
            post_acts.append(z)

        return pre_acts, post_acts

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction for input X."""
        _, post_acts = self._forward(X)
        return post_acts[-1]

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE between input and reconstruction, shape (N,)."""
        X_hat = self.predict(X)
        return np.mean((X - X_hat) ** 2, axis=1)

    # ── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def mse_loss(X: np.ndarray, X_hat: np.ndarray) -> float:
        """Mean over samples and features: (1/N) * mean_features((x - x_hat)^2)."""
        return float(np.mean((X - X_hat) ** 2))

    # ── Backward ─────────────────────────────────────────────────────────────

    def _backward(
        self,
        pre_acts: List[np.ndarray],
        post_acts: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backpropagate gradients through all layers.

        dL/d(output) = (2/N)(X_hat - X) for MSE loss (output linear layer).

        For hidden layer l (working backwards from output):
          delta[l] = (delta[l+1] @ W[l+1].T) * activation'(z[l])

        Gradients:
          dW[l] = z[l-1].T @ delta[l]
          db[l] = sum_over_batch(delta[l])
        """
        X     = post_acts[0]
        X_hat = post_acts[-1]
        N     = X.shape[0]

        dWs = [None] * self.n_layers
        dbs = [None] * self.n_layers

        # Output layer: linear activation -> dL/da = dL/dz_out
        delta = (2.0 / N) * (X_hat - X)

        for l in reversed(range(self.n_layers)):
            z_prev = post_acts[l]      # input to layer l
            dWs[l] = z_prev.T @ delta
            dbs[l] = delta.sum(axis=0)

            if l > 0:
                # Propagate error back through layer l weights, then through
                # the activation of layer l-1 (which is stored as post_acts[l])
                delta = (delta @ self.Ws[l].T) * self._actg(post_acts[l])

        return dWs, dbs

    # ── Training ─────────────────────────────────────────────────────────────

    def train_epoch(
        self,
        X: np.ndarray,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Run one epoch of mini-batch gradient descent. Returns mean loss."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.permutation(len(X))
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(X), batch_size):
            batch = X[idx[start : start + batch_size]]
            pre_acts, post_acts = self._forward(batch)

            loss = self.mse_loss(batch, post_acts[-1])
            total_loss += loss
            n_batches  += 1

            dWs, dbs = self._backward(pre_acts, post_acts)

            for l in range(self.n_layers):
                dWs[l] = np.clip(dWs[l], -self.grad_clip, self.grad_clip)
                dbs[l] = np.clip(dbs[l], -self.grad_clip, self.grad_clip)
                self.Ws[l] -= self.lr * dWs[l]
                self.bs[l] -= self.lr * dbs[l]

        return total_loss / n_batches

    def count_parameters(self) -> int:
        """Total number of trainable scalar parameters."""
        total = 0
        for W, b in zip(self.Ws, self.bs):
            total += W.size + b.size
        return total
