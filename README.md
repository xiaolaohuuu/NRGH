# NRGH
The code of Noise-Robust Generative Hashing for Cross-Modal Retrieval
## Play with NRGH
Before running the main script, you need to generate the `.mat`, correspondence noise and label noise. To do this, run `tools.py`, `generate_noisy_corr.py` and `generate_noisy_label.py`.
```bash
python ./utils/tools.py
python ./noise/generate_noisy_corr.py
python ./noise/generate_noisy_label.py
```
We perform BLIP-based text generation offline by running the following script:
```bash
python ./utils/gene_txt.py
```
Once these are generated, you can run the main script `NRGH.py` to play with the model:
```bash
python NRCH.py
```
The preprocessed version of the Flickr-25K dataset used in our experiments is publicly available at:
[Baidu Netdisk Download Link](https://pan.baidu.com/s/1aCFYI71qX7BbR9GCVjSp2Q) (Extraction Code: `a7sm`)
