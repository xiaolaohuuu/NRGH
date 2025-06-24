# NRGH
The code of Noise-Robust Generative Hashing for Cross-Modal Retrieval
## Play with NRGH
Before running the main script, you need to generate the `.mat`, correspondence noise and label noise. To do this, run `tools.py`, `generate_noisy_corr.py` and `generate_noisy_label.py`.
```bash
python ./utils/tools.py
python ./noise/generate_noisy_corr.py
python ./noise/generate_noisy_label.py
```
Once these are generated, you can run the main script `NRGH.py` to play with the model:
```bash
python NRCH.py
```
