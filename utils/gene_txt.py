from utils.tools import get_data   
import torch.optim as optim
import argparse
import os
import scipy.io as sio
import ast
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 参数优先级处理：如果 --bit 被显式传入，就用它覆盖 --hash_dim
num_gradual = args.num_gradual
Lambda = 0.6
bit_len = args.bit if args.bit is not None else args.hash_dim
dataset = args.flag if args.flag is not None else args.dataset
noise_rate = args.noise_rate
noise_txt_rate = args.noise_txt_rate

if dataset == 'flickr':
    train_size = 10000
    n_class = 24
elif dataset == 'ms-coco':
    train_size = 10000
    n_class = 80
elif dataset == 'nuswide21':
    train_size = 10500
    n_class = 21
elif dataset == 'iapr':
    train_size = 10000
def get_config():
    config = {
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "txt_optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size":128,
        "dataset": dataset,
        "epoch": 100,
        "device": torch.device("cuda:3"),
        "bit_len": bit_len,
        "noise_type": 'symmetric',
        "noise_rate": noise_rate,
        "noise_txt_rate": noise_txt_rate,
        "random_state": 1,
        "n_class": n_class,
        "lambda":Lambda,
        "tag_len":512,
        "train_size": train_size,
        "threshold_rate":0.3,
        "num_gradual": num_gradual
    }
    return config    
config = get_config()

device = config["device"]
   #Blip加载
blip_processor = BlipProcessor.from_pretrained("/blip-image-captioning-large/")
blip_model = BlipForConditionalGeneration.from_pretrained("/blip-image-captioning-large/").to(device)
# 加载 CLIP
clip_model = CLIPModel.from_pretrained("/clip-vit-base-patch32/")
clip_processor = CLIPProcessor.from_pretrained("/clip-vit-base-patch32/")
clip_model = clip_model.to(device)

train_loader,  test_loader, dataset_loader, num_train,  num_test, num_dataset = get_data(config)
corrected_features_dict = {}
for i in range(len(train_loader.dataset)):
    image, tag, tlabel, label, imgname, txtname, ind = train_loader.dataset[i]
    imgpath = ast.literal_eval(imgname)[0]
    txtpath = ast.literal_eval(txtname)[0]
    img = Image.open(imgpath).convert("RGB")
    
    with open(txtpath, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
        noisy_text = ', '.join(words)

    inputs = blip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs)
    caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    tokenized = clip_processor(text=[caption], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        gen_feat = clip_model.get_text_features(**tokenized).squeeze(0).cpu().numpy()

    corrected_features_dict[i] = gen_feat

# 保存成 mat 文件
torch.save(corrected_features_dict, "corrected_txt_features.pt")