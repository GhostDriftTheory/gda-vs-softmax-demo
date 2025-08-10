
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="GD-Attention vs Softmax — Interactive Demo", layout="wide")

st.title("GD-Attention vs Softmax — Interactive Demo")
st.markdown("""
This app contrasts **Softmax Attention** (which blends) with a **GD-Attention**-style selector (which chooses).  
Given a query **q** and keys **K = {kᵢ}**, we show:

- **Softmax output**: weighted average with weights \( w_i \propto \exp(-\beta\,\|q-k_i\|^2) \)
- **GD-Attention output**: the single key with the **lowest semantic energy**, here proxied by squared distance \(\|q-k_i\|^2\) (nearest key).

Use the sliders to move the query, change the number of keys, and adjust sharpness **β**.
""")

with st.sidebar:
    st.header("Controls")
    seed = st.slider("random seed", 0, 9999, 42, 1)
    n_keys = st.slider("number of keys", 3, 12, 6, 1)
    beta = st.slider("Softmax sharpness β", 0.1, 10.0, 2.0, 0.1)
    qx = st.slider("query q.x", -5.0, 5.0, 0.0, 0.1)
    qy = st.slider("query q.y", -5.0, 5.0, 0.0, 0.1)
    xrange = st.slider("x-range", -8.0, 8.0, (-5.0, 5.0), 0.1)
    yrange = st.slider("y-range", -8.0, 8.0, (-5.0, 5.0), 0.1)
    grid_res = st.select_slider("grid resolution (heatmap)", options=[101, 151, 201], value=151)

rng = np.random.default_rng(seed)
keys = rng.uniform(low=[xrange[0], yrange[0]], high=[xrange[1], yrange[1]], size=(n_keys, 2))
q = np.array([qx, qy], dtype=float)

# Distances & weights
d2 = np.sum((keys - q)**2, axis=1)  # squared distances
weights = np.exp(-beta * d2)
weights = weights / np.sum(weights)

softmax_out = np.sum(keys * weights[:, None], axis=0)
gd_index = int(np.argmin(d2))
gd_choice = keys[gd_index]

# --- Plot 1: 2D scatter with outputs ---
fig1 = plt.figure(figsize=(6,6))
plt.scatter(keys[:,0], keys[:,1], marker='o', label="keys")
plt.scatter([q[0]], [q[1]], marker='x', s=80, label="query q")
plt.scatter([softmax_out[0]], [softmax_out[1]], marker='^', s=120, label="Softmax output")
plt.scatter([gd_choice[0]], [gd_choice[1]], marker='*', s=160, label="GD-Attention choice")
# arrow from softmax to GD for visual difference
dx, dy = gd_choice[0]-softmax_out[0], gd_choice[1]-softmax_out[1]
plt.arrow(softmax_out[0], softmax_out[1], dx, dy, length_includes_head=True, head_width=0.1)
plt.xlim(xrange); plt.ylim(yrange)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x"); plt.ylabel("y")
plt.title("Outputs: Softmax (blends) vs GD-Attention (chooses)")
plt.legend()
plt.tight_layout()

# --- Plot 2: Heatmap of Softmax max weight over grid ---
x = np.linspace(xrange[0], xrange[1], grid_res)
y = np.linspace(yrange[0], yrange[1], grid_res)
X, Y = np.meshgrid(x, y)

def max_weight_at_point(p):
    d2g = np.sum((keys - p)**2, axis=1)
    w = np.exp(-beta * d2g)
    w = w / np.sum(w)
    return np.max(w)

Z = np.zeros((grid_res, grid_res))
for i in range(grid_res):
    for j in range(grid_res):
        Z[i, j] = max_weight_at_point(np.array([X[i,j], Y[i,j]]))

fig2 = plt.figure(figsize=(6,6))
plt.contourf(X, Y, Z, levels=20)
plt.scatter(keys[:,0], keys[:,1], marker='o', label="keys")
plt.scatter([q[0]], [q[1]], marker='x', s=80, label="query q")
plt.xlim(xrange); plt.ylim(yrange)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x"); plt.ylabel("y")
plt.title("Softmax max weight (selectivity)")
plt.legend()
plt.tight_layout()

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

st.subheader("Numerics")
st.write({
    "query q": q.tolist(),
    "keys": keys.tolist(),
    "squared distances d2": d2.tolist(),
    "softmax weights": weights.tolist(),
    "softmax output": softmax_out.tolist(),
    "GD-Attention index": gd_index,
    "GD-Attention key": gd_choice.tolist()
})
st.caption("Tip: Increase β to make Softmax more selective; decrease β to blend more. GD-Attention always selects the nearest key.")
