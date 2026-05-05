# Lecture Code - Lec13

Minden script egy konkret diatema futtathato, kis meretu peldaja.
A legtobb script tamogatja a `--quick` flaget, hogy oran gyorsan fusson.

## Futtatas

```bash
python 13_recurrent_networks/lecture_code/smoke_test_recurrent_networks.py
python 13_recurrent_networks/lecture_code/smoke_test_recurrent_networks_interactive.py
```

## Interaktiv verziok

Az alabbi parok keszultek el:

- `01_bptt_tiny_rnn_parity_numpy.py` -> `01_bptt_tiny_rnn_parity_interactive.py`
- `02_lstm_cell_from_scratch_torch.py` -> `02_lstm_cell_from_scratch_interactive.py`
- `03_self_attention_numpy_causal_mask.py` -> `03_self_attention_numpy_causal_mask_interactive.py`
- `04_multihead_attention_torch.py` -> `04_multihead_attention_interactive.py`
- `05_tiny_transformer_encoder_block_torch.py` -> `05_tiny_transformer_encoder_block_interactive.py`
- `06_vae_mnist_torch.py` -> `06_vae_mnist_interactive.py`
- `07_mini_dcgan_mnist_torch.py` -> `07_mini_dcgan_mnist_interactive.py`
- `08_toy_2d_diffusion_torch.py` -> `08_toy_2d_diffusion_interactive.py`
- `09_huggingface_pipelines_demo.py` -> `09_huggingface_pipelines_interactive.py`
- `10_mini_rag_sentence_transformers.py` -> `10_mini_rag_interactive.py`

## Megjegyzesek

- Az `01-04`, `06-08` interaktiv verziok `matplotlib` widgeteket hasznalnak.
- Az `05`, `09`, `10` interaktiv verziok REPL / CLI jellegu interakciot adnak.
- A `06` es `07` scripthez `torchvision` kell (MNIST).
- A `09` es `10` script internetet igenyelhet modellletolteshez.
- Ha optional csomag hianyzik, a script jelzi es kulturaltan kilep.

