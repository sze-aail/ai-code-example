# 11_metrics - Lec13 generativ es NLP metrikak
Minimal, futtathato receptek a Lec13 metrikakhoz.
## Script lista
1. `01_fid_toy.py` - FID trend + clean-fid integracio
2. `02_is_kid_toy.py` - Inception Score es KID (torchmetrics)
3. `03_clip_score_toy.py` - CLIP score (open-clip)
4. `04_bleu_rouge_meteor_toy.py` - BLEU (sacrebleu), ROUGE, METEOR
5. `05_bertscore_toy.py` - BERTScore multilingual modelllel
6. `06_perplexity_toy.py` - stride-os perplexity (GPT-2 minta)
7. `07_lm_eval_harness_wrapper.py` - lm_eval wrapper es command preview
8. `08_human_ab_tester.py` - Streamlit A/B human eval minimal app
9. `09_llm_intrinsic_eval.py` - HuggingFace evaluate intrinsic metricak
## Gyors ellenorzes
```bash
python 13_recurrent_networks/11_metrics/smoke_test_metrics.py
```
## Javasolt notebook kiterjesztes
- `10_fid_kep_eloszlas.ipynb`
- `11_is_kid.ipynb`
- `12_clip_score_t2i.ipynb`
- `13_bleu_sacrebleu.ipynb`
- `14_rouge_meteor.ipynb`
- `15_bertscore.ipynb`
- `16_perplexity_gpt2.ipynb`
- `17_lm_eval_harness_mini.ipynb`
- `18_emberi_ertekeles_streamlit.py`
## Megjegyzesek
- A scriptek toy/adatfuggetlen mintak.
- Opcionális csomag hianyaban a script nem dob hibat, csak jelzi a hianyzo dependency-t.
- A riportokban mindig szerepeljen a metric-konfiguracio (modell, preprocess, shot-szam, harness verzio).
