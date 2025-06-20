import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from utils.tools import *
from network import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import argparse
from sklearn.mixture import GaussianMixture
import wandb
import random

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--hash_dim', type = int, default = 16)
parser.add_argument('--dataset', type = str, default = 'nuswide21')
parser.add_argument('--num_gradual', type = int, default = 100)
parser.add_argument('--wdb', action='store_true', help='Enable wandb logging')
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=None)
parser.add_argument('--bit', type=int, default=None)  # optional override
parser.add_argument('--flag', type=str, default=None)
parser.add_argument('--noise_rate', type = float, default = 0.8)
parser.add_argument('--noise_txt_rate', type=float, default= 0.8)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--txt_lr', type=float, default=1e-5)
args = parser.parse_args()

default_hyperparams = {
    'flickr':    {'alpha': 0.001, 'beta': 0.6, 'gamma': 0.6},
    'ms-coco':   {'alpha': 0.001, 'beta': 0.8, 'gamma': 0.6},
    'nuswide21': {'alpha': 0.001, 'beta': 0.6, 'gamma': 0.6}
}

dataset = args.dataset.lower()
if dataset in default_hyperparams:
    if args.alpha is None:
        args.alpha = default_hyperparams[dataset]['alpha']
    if args.beta is None:
        args.beta = default_hyperparams[dataset]['beta']
    if args.gamma is None:
        args.gamma = default_hyperparams[dataset]['gamma']

num_gradual = args.num_gradual
wdb = args.wdb
Alpha = args.alpha
Beta = args.beta
Gamma = args.gamma

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


torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": args.lr, "weight_decay": 10 ** -5}},
        "txt_optimizer": {"type": optim.RMSprop, "optim_params": {"lr": args.txt_lr, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "batch_size":128,
        "dataset": dataset,
        "epoch": 200,
        "device": torch.device("cuda:0"),
        "bit_len": bit_len,
        "noise_rate": noise_rate,
        "noise_txt_rate": noise_txt_rate,
        "n_class": n_class,
        "gamma":Gamma,
        "tag_len":512,
        "train_size": train_size,
        "threshold_rate":0.3,
        "num_gradual": num_gradual,
        "alpha": Alpha,
        "beta": Beta,
        "wandb": wdb
    }
    return config    

class Robust_Loss_DynamicMargin(nn.Module):
    def __init__(self, config, bit):
        super(Robust_Loss_DynamicMargin, self).__init__()
        self.tau = 0.5
        self.shift = 0.2
        self.bit = bit

    def forward(self, u, v, y, gmm_probs):
        u = u.tanh()
        v = v.tanh()
        
        T = self.calc_neighbor(y, y)
        T.fill_diagonal_(0)

        S = u @ v.T 
        d = S.diag().view(-1, 1)
        d1 = d.expand_as(S)
        d2 = d.T.expand_as(S)

        batch_probs = gmm_probs.view(-1, 1)
        dynamic_margin = config["beta"] * (S.mean() - S.min()).item() * (1 - batch_probs)  
        margin_matrix = dynamic_margin @ torch.ones(1, S.size(1), device=S.device)

        # === I -> T ===
        mask_te = (S >= (d1 - margin_matrix)).float().detach()
        cost_te = S * mask_te + (1. - mask_te) * (S - self.shift)
        cost_te_max = cost_te.clone()
        cost_te_max.fill_diagonal_(0)
        cost_te_max += torch.diag(torch.diag(cost_te).clamp(min=0))

        # === T -> I ===
        mask_im = (S >= (d2 - margin_matrix.T)).float().detach()
        cost_im = S * mask_im + (1. - mask_im) * (S - self.shift)
        cost_im_max = cost_im.clone()
        cost_im_max.fill_diagonal_(0)
        cost_im_max += torch.diag(torch.diag(cost_im).clamp(min=0))

        loss_r = (
            -torch.diag(cost_te) + self.tau * ((cost_te_max / self.tau) * (1 - T)).exp().sum(1).log() +
            -torch.diag(cost_im) + self.tau * ((cost_im_max / self.tau) * (1 - T)).exp().sum(1).log()
        ).mean()

        Q_loss = ((u.abs() - 1 / np.sqrt(self.bit)).pow(2).mean(dim=1) +
                  (v.abs() - 1 / np.sqrt(self.bit)).pow(2).mean(dim=1)).mean()
        return config["gamma"] * loss_r + (1-config["gamma"]) * Q_loss

    def calc_neighbor(self, label1, label2):
        label1 = label1.float()
        label2 = label2.float()
        return (label1 @ label2.T > 0).float()

def split_prob(prob, threshld):
    pred = (prob >= threshld)
    return (pred+0)

def get_loss(net, txt_net, config, data_loader, Threshold, epoch,W):
    tau = 0.05
    sample_losses = []
    for image, tag, tlabel, label, ind in data_loader:
        image = image.to('cuda')
        image = image.float()
        tag = tag.to('cuda')
        tag = tag.float()
        label = label.to('cuda')
        tlabel = tlabel.to('cuda')
        u = net(image)
        v = txt_net(tag) 
        with torch.no_grad():
            label_ = (label - 0.5) * 2 
            u_sims = u @ W.tanh().t()  
            v_sims = v @ W.tanh().t()   
            loss_ = (label_ - u_sims)**2 
            loss_ += (label_ - v_sims)**2      
            loss = (loss_ * label).max(1)[0] 
        right = ((tlabel==label).float().mean(1) == 1).float() 
        for i in range(len(loss)):
            sample_losses.append((ind[i].item(), loss[i].item(), right[i].item())) 
    sample_losses_sorted = sorted(sample_losses, key=lambda x: x[0])
    sorted_losses = [item[1] for item in sample_losses_sorted]
    sorted_losses = np.array(sorted_losses)
    sorted_losses = (sorted_losses-sorted_losses.min()+ 1e-8)/(sorted_losses.max()-sorted_losses.min() + 1e-8) 
    sorted_losses = sorted_losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=5e-1,reg_covar=5e-4)
    labels = np.array([item[2] for item in sample_losses_sorted])
    loss = np.array(sorted_losses)
    gmm.fit(sorted_losses)
    prob = gmm.predict_proba(sorted_losses) 
    prob = prob[:, gmm.means_.argmin()] 
    if epoch+1>=20:
        pred = split_prob(prob,Threshold)
    else:
        pred = split_prob(prob,0)
    clean_index = np.where(labels==1)[0] 
    smaller_mean_indices = [i for i, p in enumerate(pred) if p == 1] 
    true_positives = set(smaller_mean_indices).intersection(clean_index)
    false_positives = set(smaller_mean_indices).difference(clean_index)
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) 
    return sorted_losses, torch.Tensor(pred), precision

def get_gmm_clean_mask(net, txt_net, train_loader, Threshold, epoch,W):
    sample_losses = []
    for image, tag, tlabel, label, correct_tag, ind in train_loader:
        image = image.to('cuda')
        image = image.float()
        tag = tag.to('cuda')
        tag = tag.float()
        label = label.to('cuda')
        tlabel = tlabel.to('cuda')
        u = net(image)
        v = txt_net(tag) 
        with torch.no_grad():
            label_ = (label - 0.5) * 2  
            u_sims = u @ W.tanh().t()   
            v_sims = v @ W.tanh().t()   
            loss_ = (label_ - u_sims)**2 
            loss_ += (label_ - v_sims)**2           
            loss = (loss_ * label).max(1)[0] 
        right = ((tlabel==label).float().mean(1) == 1).float() 
        for i in range(len(loss)):
            sample_losses.append((ind[i].item(), loss[i].item(), right[i].item())) 
    sample_losses_sorted = sorted(sample_losses, key=lambda x: x[0])
    sorted_losses = [item[1] for item in sample_losses_sorted]
    sorted_losses = np.array(sorted_losses)
    sorted_losses = (sorted_losses-sorted_losses.min()+ 1e-8)/(sorted_losses.max()-sorted_losses.min() + 1e-8)
    sorted_losses = sorted_losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=5e-1,reg_covar=5e-4)
    labels = np.array([item[2] for item in sample_losses_sorted])
    loss = np.array(sorted_losses)
    gmm.fit(sorted_losses)
    prob = gmm.predict_proba(sorted_losses)
    prob = prob[:, gmm.means_.argmin()] 
    if epoch+1>=20:
        pred = split_prob(prob,Threshold)
    else:
        pred = split_prob(prob,0)
    return torch.Tensor(pred), prob

def get_dual_gmm_clean_mask(net, txt_net, train_loader, Threshold, epoch, W):
    label_losses = []
    align_losses = []
    sample_indices = []

    for image, tag, _, label, _, ind in train_loader:
        image = image.to('cuda')
        tag = tag.to('cuda')
        label = label.to('cuda')

        with torch.no_grad():
            u = net(image)
            v = txt_net(tag)
            label_ = (label - 0.5) * 2

            sim_u = u @ W.tanh().T
            sim_v = v @ W.tanh().T

            label_loss = ((label_ - sim_u)**2 + (label_ - sim_v)**2).max(dim=1)[0]
            align_loss = 1 - F.cosine_similarity(u, v, dim=1)

        label_losses.append(label_loss.cpu())
        align_losses.append(align_loss.cpu())
        sample_indices.extend(ind.cpu().numpy())

    input_loss_A = torch.cat(label_losses).view(-1, 1)
    input_loss_B = torch.cat(align_losses).view(-1, 1)

    gmm_A = GaussianMixture(n_components=2,max_iter=10,tol=5e-1,reg_covar=5e-4)
    gmm_A.fit(input_loss_A.numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]
    gmm_B = GaussianMixture(n_components=2,max_iter=10,tol=5e-1,reg_covar=5e-4)
    gmm_B.fit(input_loss_B.numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
    joint_prob = 0.7 * prob_A + (1 - 0.7) * prob_B
    clean_mask = (joint_prob >= Threshold).astype(np.float32)
    return torch.tensor(clean_mask), torch.tensor(joint_prob)

def apply_text_correction(train_loader, clean_mask, gmm_probs, device):
    corrected_features_dict = {}
    for i, (image, tag, tlabel, label, correct_tag, ind) in enumerate(train_loader.dataset):
        index = ind
        if not clean_mask[index].item():
            conf = gmm_probs[index]
            orig_feat = torch.tensor(tag, dtype=torch.float32)
            correct_tag = torch.tensor(correct_tag, dtype=torch.float32)
            fused_feat = (1 - conf) * correct_tag + conf * orig_feat
            corrected_features_dict[i] = fused_feat 
    return corrected_features_dict

def correct_labels_by_similarity(clean_mask, gmm_probs, corrected_features_dict, train_loader, label_feature, device):
    corrected_labels = {}

    label_feature = torch.tensor(label_feature, dtype=torch.float32).to(device)

    for i, (image, tag, tlabel, label, correct_tag, ind) in enumerate(train_loader.dataset):
        index = ind  
        label_vec = torch.tensor(label, dtype=torch.float32).to(device)
        positive_idx = label_vec.nonzero(as_tuple=True)[0]  
        if len(positive_idx) == 0:
            continue 
        selected_label_feat = label_feature[positive_idx]
        img_feat = F.normalize(torch.tensor(image, dtype=torch.float32).to(device).view(-1), dim=0)
        if clean_mask[index].item():
            tag_feat = F.normalize(torch.tensor(tag, dtype=torch.float32).to(device).view(-1), dim=0)
            sim_img = F.cosine_similarity(img_feat.unsqueeze(0), selected_label_feat)
            sim_tag = F.cosine_similarity(tag_feat.unsqueeze(0), selected_label_feat) 
            sims = (sim_img + sim_tag) / 2
        else:
            tag_feat = F.normalize(corrected_features_dict[index].to(device).view(-1), dim=0)
            sim_img = F.cosine_similarity(img_feat.unsqueeze(0), selected_label_feat)  
            sim_tag = F.cosine_similarity(tag_feat.unsqueeze(0), selected_label_feat)  
            conf = gmm_probs[index]
            sims = conf * sim_img + (1 - conf) * sim_tag

        sims = sims.clamp(min=0)
        sims = sims / (sims.sum() + 1e-6)

        soft_label = torch.zeros_like(label_vec)
        soft_label[positive_idx] = 0.5 * label_vec[positive_idx] + (1 - 0.5) * sims
        corrected_labels[i] = soft_label.cpu()
    return corrected_labels

def getlf(config, device):
    dataset = config["dataset"]
    if dataset=='flickr':
        label_feature = loadmat('/data/Mirflickr_clip_24label.mat')['class_features']
        lf = torch.tensor(label_feature, dtype=torch.float32).to(device)
    elif dataset=='ms-coco':
        label_feature = loadmat('/data/coco_label_features.mat')['label_features']
        lf = torch.tensor(label_feature, dtype=torch.float32).to(device)
    elif dataset=='nuswide21':
        label_feature = loadmat('/data/nus_label_features.mat')['class_features']
        lf = torch.tensor(label_feature, dtype=torch.float32).to(device)
    return lf

def train(config, bit):
    device = config["device"]
    train_loader,  test_loader, dataset_loader, num_train,  num_test, num_dataset = get_data(config)
    label_feature = getlf(config, device)
    config["num_train"] = num_train
    net = ImgModule(y_dim=512, bit=bit, hiden_layer=3).to('cuda')
    txt_net = TxtModule(y_dim=512, bit=bit, hiden_layer=2).to('cuda')
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad= True).cuda()
    W = torch.nn.Parameter(W)
    net.register_parameter('W', W) # regist W into the image net
    get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
    params_dnet = get_grad_params(net)
    optimizer = config["optimizer"]["type"](params_dnet, **(config["optimizer"]["optim_params"]))
    txt_optimizer = config["txt_optimizer"]["type"](txt_net.parameters(), **(config["txt_optimizer"]["optim_params"]))
    criterion = Robust_Loss_DynamicMargin(config, bit)
    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    precision_list = []
    bestt2i=0
    besti2t=0
    bestepoch=0
    n=0
    os.makedirs('./checkpoint', exist_ok=True)
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")  
        net.eval()
        txt_net.eval()
        net.train()
        txt_net.train()
        train_loss = 0
        if (epoch+1) %10 == 0:
            print("calculating test binary code......")
            img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
            print("calculating dataset binary code.......")
            img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
            txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
            txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
            print("calculating map.......")
            t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy())
            i2t_mAP = calc_map_k(txt_trn_binary.numpy(),img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy())
            if t2i_mAP+i2t_mAP> bestt2i+besti2t:
                bestt2i=t2i_mAP
                besti2t=i2t_mAP
                bestepoch=epoch+1
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'txt_net_state_dict': txt_net.state_dict(),
                }, './checkpoint/best_model.pth') 
            t2i_mAP_list.append(t2i_mAP.item())
            i2t_mAP_list.append(i2t_mAP.item())
            epoch_list.append(epoch)
            print("%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f, bestt2i:%.3f, besti2t:%.3f, bestepoch:%d" % (
                config["info"], epoch + 1, bit, config["dataset"], config["noise_rate"],t2i_mAP, i2t_mAP, bestt2i, besti2t, bestepoch))
        clean_mask, gmm_probs = get_gmm_clean_mask(net, txt_net, train_loader, config["threshold_rate"], epoch,W)
        corrected_features_dict = apply_text_correction(train_loader, clean_mask, gmm_probs, device)
        corrected_labels = correct_labels_by_similarity(clean_mask, gmm_probs, corrected_features_dict, train_loader, label_feature, device)
        for image, tag, tlabel, label, correct_tag, ind in train_loader:
            ind_np = ind.cpu().numpy()
            image = image.to('cuda').float()
            tag = tag.to('cuda').float()
            correct_tag = correct_tag.to('cuda').float()
            corrected_label_batch = []
            for i in ind_np:
                if i in corrected_labels:
                    corrected_label_batch.append(corrected_labels[i].to('cuda'))
                else:
                    corrected_label_batch.append(label[ind_np.tolist().index(i)].to('cuda')) 
            corrected_label_batch = torch.stack(corrected_label_batch)
            mixed_tag_batch = []
            noisy_text_batch = [] 
            corrected_text_batch = [] 
            img_contrast_batch = []
            for idx_in_batch, global_idx in enumerate(ind_np):
                if clean_mask[global_idx].item():
                    mixed_tag_batch.append(tag[idx_in_batch])
                else:
                    mixed_tag_batch.append(corrected_features_dict[global_idx].to('cuda'))
                    noisy_text_batch.append(tag[idx_in_batch])
                    corrected_text_batch.append(corrected_features_dict[global_idx].to('cuda'))
                    img_contrast_batch.append(image[idx_in_batch])  

            mixed_tag_batch = torch.stack(mixed_tag_batch)  
            optimizer.zero_grad()
            txt_optimizer.zero_grad()
            u = net(image)
            v = txt_net(mixed_tag_batch) 
            u_map = {ind_np[i]: u[i] for i in range(len(ind_np))}
            v_map = {ind_np[i]: v[i] for i in range(len(ind_np))}
            gmm_probs_tensor = torch.tensor([gmm_probs[i] for i in ind_np], dtype=torch.float32).to(u.device)
            loss = criterion(u, v, corrected_label_batch.float(), gmm_probs_tensor)
            label_ = (corrected_label_batch - 0.5) * 2  
            u_sims = u.detach() @ W.tanh().t()   
            v_sims = v.detach() @ W.tanh().t()   
            loss_ = (label_ - u_sims)**2
            loss_ += (label_ - v_sims)**2
            loss += loss_.mean()     
            train_loss += loss
            loss.backward()
            optimizer.step()
            txt_optimizer.step()
        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        print(f'{config["info"]} epoch:{epoch + 1}, bit:{bit}, dataset:{config["dataset"]}, '
            f'noise_rate:{config["noise_rate"]:.1f}, txtnoise_rate:{config["noise_txt_rate"]:.1f}')
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
    print("i2t_mAP_list:", i2t_mAP_list)
    print("t2i_mAP_list:", t2i_mAP_list)
    return bestt2i,besti2t

def test(config, bit, model_path='./checkpoint/best_model.pth'):
    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)
    net = ImgModule(y_dim=512, bit=bit, hiden_layer=3).to('cuda')
    txt_net = TxtModule(y_dim=512, bit=bit, hiden_layer=2).to('cuda')
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad= True).cuda()
    W = torch.nn.Parameter(W)
    net.register_parameter('W', W)
    # Load the saved models
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    txt_net.load_state_dict(checkpoint['txt_net_state_dict'])
    net.eval()
    txt_net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
    txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
    txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
    print("calculating map.......")
    t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy())
    i2t_mAP = calc_map_k(txt_trn_binary.numpy(),img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy())
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#bash版本
if __name__ == "__main__":
    set_seed(2025)
    config = get_config()
    print(f"\n🚀 Running {dataset} | Bit: {bit_len} | Noise: {noise_rate} | TxtNoise: {noise_txt_rate}")
    print(config)

    bestt2i, besti2t = train(config, bit_len)

    print("\n✅ 训练完成，结果：")
    print(f"  bestt2i: {bestt2i}")
    print(f"  besti2t: {besti2t}")