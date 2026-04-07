"""
05_backpropagation.py — Backpropagation lépésről lépésre
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Számítási gráf felépítése
  - Forward pass és backward pass kézzel
  - PyTorch autograd összehasonlítás
"""
import numpy as np
import torch


print("=" * 60)
print("BACKPROPAGATION — Kézi számítás vs. PyTorch autograd")
print("=" * 60)

# ── Egyszerű háló: 2 bemenet, 2 rejtett, 1 kimenet ──
# Sigmoid aktiváció, BCE loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Fix súlyok a demonstrációhoz
w1 = np.array([[0.15, 0.20], [0.25, 0.30]])  # 2x2 (bemenet → rejtett)
b1 = np.array([0.35, 0.35])
w2 = np.array([[0.40], [0.45]])               # 2x1 (rejtett → kimenet)
b2 = np.array([0.60])

x_in = np.array([0.05, 0.10])
y_true = 0.01

print(f"\nBemenet:  x = {x_in}")
print(f"Elvárt:   y = {y_true}")
print(f"Súlyok W1:\n{w1}")
print(f"Bias b1: {b1}")
print(f"Súlyok W2:\n{w2.T}")
print(f"Bias b2: {b2}")

# ── FORWARD PASS ──
print("\n--- FORWARD PASS ---")
z1 = x_in @ w1 + b1
h1 = sigmoid(z1)
print(f"  z1 = x @ W1 + b1 = {z1}")
print(f"  h1 = sigmoid(z1) = {h1}")

z2 = h1 @ w2 + b2
y_pred = sigmoid(z2)
print(f"  z2 = h1 @ W2 + b2 = {z2}")
print(f"  y_pred = sigmoid(z2) = {y_pred}")

loss = 0.5 * (y_true - y_pred[0])**2
print(f"  Loss = 0.5 * (y - ŷ)² = {loss:.6f}")

# ── BACKWARD PASS ──
print("\n--- BACKWARD PASS (láncszabály) ---")
# dL/dy_pred
dL_dy = -(y_true - y_pred[0])
print(f"  ∂L/∂ŷ = -(y - ŷ) = {dL_dy:.6f}")

# dL/dz2 = dL/dy * dy/dz2
dL_dz2 = dL_dy * sigmoid_deriv(z2[0])
print(f"  ∂L/∂z2 = ∂L/∂ŷ · σ'(z2) = {dL_dz2:.6f}")

# dL/dW2
dL_dW2 = h1.reshape(-1, 1) * dL_dz2
print(f"  ∂L/∂W2 = h1ᵀ · ∂L/∂z2 = {dL_dW2.flatten()}")

# dL/db2
dL_db2 = dL_dz2
print(f"  ∂L/∂b2 = {dL_db2:.6f}")

# dL/dh1
dL_dh1 = w2.flatten() * dL_dz2
print(f"  ∂L/∂h1 = W2 · ∂L/∂z2 = {dL_dh1}")

# dL/dz1
dL_dz1 = dL_dh1 * sigmoid_deriv(z1)
print(f"  ∂L/∂z1 = ∂L/∂h1 · σ'(z1) = {dL_dz1}")

# dL/dW1
dL_dW1 = np.outer(x_in, dL_dz1)
print(f"  ∂L/∂W1 =\n{dL_dW1}")

# ── PyTorch ellenőrzés ──
print("\n--- PyTorch AUTOGRAD ellenőrzés ---")
W1_t = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
x_t  = torch.tensor(x_in, dtype=torch.float64)

z1_t = x_t @ W1_t + b1_t
h1_t = torch.sigmoid(z1_t)
z2_t = h1_t @ W2_t + b2_t
y_t  = torch.sigmoid(z2_t)
loss_t = 0.5 * (y_true - y_t[0])**2
loss_t.backward()

print(f"  ∂L/∂W2 (PyTorch) = {W2_t.grad.flatten().numpy()}")
print(f"  ∂L/∂W2 (kézi)    = {dL_dW2.flatten()}")
print(f"  ∂L/∂W1 (PyTorch) =\n{W1_t.grad.numpy()}")
print(f"  ∂L/∂W1 (kézi)    =\n{dL_dW1}")

match_w2 = np.allclose(W2_t.grad.numpy(), dL_dW2, atol=1e-8)
match_w1 = np.allclose(W1_t.grad.numpy(), dL_dW1, atol=1e-8)
print(f"\n  W2 gradiensek egyeznek: {'✓' if match_w2 else '✗'}")
print(f"  W1 gradiensek egyeznek: {'✓' if match_w1 else '✗'}")

print("\nKész!")
