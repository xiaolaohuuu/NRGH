import numpy as np
import os
from scipy.io import loadmat, savemat

def batch_create_fixed_noisy_text(
    mat_path,
    save_path,
    noise_ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
    save_prefix="MIRFlickr_clean_split"
):
    data = loadmat(mat_path)
    tag_train = data["TagTrain"]
    TxtNamesTrain = data["TxtNamesTrain"].T
    n = tag_train.shape[0]

    t2i_index = np.arange(n)

    for ratio in noise_ratios:
        idx = np.arange(n)
        np.random.shuffle(idx)
        noise_len = int(ratio * n)
        noise_part = idx[:noise_len]

        noisy_text_index = t2i_index.copy()
        shuffled = t2i_index[noise_part].copy()
        np.random.shuffle(shuffled)
        noisy_text_index[noise_part] = shuffled

        tag_train_noisy = tag_train[noisy_text_index]
        TxtNamesTrain_noisy = TxtNamesTrain[noisy_text_index]
        save_dict = {
            "TagTrain_noisy": tag_train_noisy,
            "TxtNamesTrain": TxtNamesTrain_noisy,
            "NoisyTextIndex": noisy_text_index
        }
        mat_name = f"{save_prefix}_noisytext_{int(ratio * 100)}.mat"
        savemat(os.path.join(save_path, mat_name), save_dict)
        
        txt_path = os.path.join(save_path, f"noisy_text_index_{int(ratio * 100)}.txt")
        np.savetxt(txt_path, noisy_text_index, fmt="%d")

        print(f"{mat_name} and {txt_path}")
        print(f"{os.path.join(save_path, mat_name)}")
        print(f"{txt_path}")

batch_create_fixed_noisy_text(
    mat_path="/data/MIRFlickr_clean.mat",
    save_path="./noise_clip",
    noise_ratios=[0.2, 0.4, 0.6, 0.8], 
    save_prefix="MIRFlickr"
)
