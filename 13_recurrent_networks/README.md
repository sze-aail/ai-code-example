# 13_recurrent_networks

Lec13 mintakodok szekvenciakhoz, attentionhoz es modern generativ modellekhez.
A mappa szerkezete koveti a korabbi moduloket:

- `lecture_code/` - futtathato, oktatasi peldak
- `exercises/` - feladat-otletek

## Gyors inditas

```bash
python 13_recurrent_networks/lecture_code/01_bptt_tiny_rnn_parity_numpy.py --quick
python 13_recurrent_networks/lecture_code/03_self_attention_numpy_causal_mask.py --quick
python 13_recurrent_networks/lecture_code/05_tiny_transformer_encoder_block_torch.py --quick
```

## Interaktiv futtatas

```bash
python 13_recurrent_networks/lecture_code/01_bptt_tiny_rnn_parity_interactive.py
python 13_recurrent_networks/lecture_code/03_self_attention_numpy_causal_mask_interactive.py
python 13_recurrent_networks/lecture_code/06_vae_mnist_interactive.py --quick
python 13_recurrent_networks/lecture_code/10_mini_rag_interactive.py
```

Gyors, nem-blokkolo ellenorzeshez:

```bash
python 13_recurrent_networks/lecture_code/smoke_test_recurrent_networks_interactive.py
```

## Temak

1. BPTT pici RNN-en (NumPy)
2. LSTM cella nullarol (PyTorch)
3. Self-attention NumPy-bol (+ causal mask)
4. Multi-head attention (PyTorch)
5. Pici Transformer encoder blokk
6. VAE MNIST-en
7. Mini DCGAN MNIST-en
8. Toy 2D diffusion
9. HuggingFace pipeline minimal demo
10. Mini RAG (retrieval + valasz)

