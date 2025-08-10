# GD‑Attention vs Softmax — Interactive Demo

This app contrasts **Softmax Attention** (weighted averaging) with a **GD‑Attention**-style selector (nearest‑key choice).  
It reveals the core behavioral difference: **blend vs choose**.

## Run locally
```bash
pip install -r requirements_gda.txt
streamlit run gda_demo_app.py
```

- Left plot: keys, query, Softmax output (triangle), GD‑Attention choice (star), and the difference arrow.
- Right plot: Softmax selectivity heatmap (max per‑key weight over a grid).
