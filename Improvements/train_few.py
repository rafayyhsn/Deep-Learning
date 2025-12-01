import os
import argparse
import random
import math
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted
from CLIP.tokenizer import tokenize
from prompt import MEDICAL_PROMPTS
from loss import FocalLoss, BinaryDiceLoss
from utils import augment
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_INDEX = {
    'Brain': 3, 'Liver': 2, 'Retina_RESC': 1,
    'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# VLC LOSS
# ================================================================
def vlc_loss(patch_tokens, mask_small, text_normal, text_abnormal, bce):
    """
    patch_tokens : (L, C)
    mask_small   : (H, H)
    text_normal  : (C, 1)
    text_abnormal: (C, 1)
    """
    L, C = patch_tokens.shape
    H = int(math.sqrt(L))

    patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

    # (L,1) -> (H,H)
    logits_norm = (patch_tokens @ text_normal).reshape(H, H)
    logits_abn = (patch_tokens @ text_abnormal).reshape(H, H)

    # BCEWithLogitsLoss expects logits; we scale for separation
    loss = bce(100.0 * logits_abn, mask_small) + bce(100.0 * logits_norm, 1.0 - mask_small)
    return loss


# ================================================================
# MAIN TRAINING
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336')
    parser.add_argument('--pretrain', type=str, default='openai')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--features_list', type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)

    # ------------------------------------------------------------
    # Load CLIP model
    # ------------------------------------------------------------
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    clip_model.eval()

    # Disable PATCH DROPOUT so we always get a square patch grid
    clip_model.visual.patch_dropout = nn.Identity()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.train()

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Few-shot support (abnormal + normal)
    abn_img, abn_mask = augment(dataset.fewshot_abnorm_img, dataset.fewshot_abnorm_mask)
    norm_img, _ = augment(dataset.fewshot_norm_img)

    # Build training tensors
    train_img = torch.cat([abn_img, norm_img], dim=0)                    # (N,3,H,W)
    train_mask = torch.cat([abn_mask, torch.zeros_like(norm_img[:, :1])], dim=0)  # (N,1,H,W)
    train_label = torch.cat([
        torch.ones(len(abn_img)),
        torch.zeros(len(norm_img))
    ]).float()   # (N,)

    train_dataset = torch.utils.data.TensorDataset(train_img, train_mask, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Memory bank from only normal images
    support_dataset = torch.utils.data.TensorDataset(norm_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1)

    # ------------------------------------------------------------
    # Losses & Optimizers
    # ------------------------------------------------------------
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    bce_logits = torch.nn.BCEWithLogitsLoss()

    seg_opt = torch.optim.Adam(model.seg_adapters.parameters(), lr=args.learning_rate)
    det_opt = torch.optim.Adam(model.det_adapters.parameters(), lr=args.learning_rate)

    # ------------------------------------------------------------
    # TEXT FEATURES
    # ------------------------------------------------------------
    with torch.no_grad():
        normal_prompts = MEDICAL_PROMPTS[args.obj]["normal"]
        abnormal_prompts = MEDICAL_PROMPTS[args.obj]["abnormal"]

        normal_emb = []
        abnormal_emb = []

        for p in normal_prompts:
            tok = tokenize([p]).to(device)
            e = clip_model.encode_text(tok)[0]
            e = e / e.norm()
            normal_emb.append(e)

        for p in abnormal_prompts:
            tok = tokenize([p]).to(device)
            e = clip_model.encode_text(tok)[0]
            e = e / e.norm()
            abnormal_emb.append(e)

        normal_vec = torch.stack(normal_emb).mean(0)      # (C,)
        abnormal_vec = torch.stack(abnormal_emb).mean(0)  # (C,)

        # (C,2) for seg/det heads (normal, abnormal)
        text_features = torch.stack([normal_vec, abnormal_vec], dim=1)

        # (C,1) for VLC loss
        normal_vlc = normal_vec.unsqueeze(1)
        abnormal_vlc = abnormal_vec.unsqueeze(1)

    best_score = 0.0

    # =============================================================
    # TRAIN LOOP
    # =============================================================
    for epoch in range(args.epoch):
        print(f"\n========== Epoch {epoch} ==========")
        losses = []

        for cpu_img, cpu_mask, cpu_label in train_loader:

            img = cpu_img.to(device)           # (1,3,H,W)
            mask = cpu_mask.to(device)         # (1,1,H,W)
            label = cpu_label.to(device).view(1)  # ensure shape (1,)

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                _, seg_tokens, det_tokens = model(img)

                seg_tokens = [p[0, 1:, :] for p in seg_tokens]  # (L,C)
                det_tokens = [p[0, 1:, :] for p in det_tokens]  # (L,C)

                # ---------------- DETECTION LOSS ----------------
                det_loss = 0.0
                for p in det_tokens:
                    p = p / p.norm(dim=-1, keepdim=True)   # (L,C)

                    # raw sim (L,2)
                    logits_2 = p @ text_features          # (L,2)

                    # Build a single binary logit per image:
                    # abnormal_logit - normal_logit
                    bin_logit = 100.0 * (logits_2[:, 1] - logits_2[:, 0])  # (L,)

                    # average over patches -> shape (1,)
                    img_logit = bin_logit.mean().view(1)

                    # THIS WAS THE BUG: ensure same shape as label (1,)
                    det_loss += bce_logits(img_logit, label)

                # ---------------- SEGMENTATION LOSS + VLC ----------------
                seg_loss = 0.0

                if CLASS_INDEX[args.obj] > 0:
                    mask_bin = (mask > 0.5).float()    # (1,1,H,W)

                    for p in seg_tokens:
                        p = p / p.norm(dim=-1, keepdim=True)   # (L,C)
                        L = p.shape[0]
                        H_patch = int(math.sqrt(L))

                        # (1,L,2)
                        logits_2 = (p @ text_features).unsqueeze(0)

                        # -> (1,2,H_patch,H_patch)
                        anomaly = logits_2.permute(0, 2, 1).view(1, 2, H_patch, H_patch)
                        anomaly = 100.0 * anomaly

                        # upscale to image resolution
                        anomaly = F.interpolate(
                            anomaly,
                            size=args.img_size,
                            mode="bilinear",
                            align_corners=True
                        )  # (1,2,H,W)

                        anomaly = torch.softmax(anomaly, dim=1)  # probabilities

                        seg_loss += loss_focal(anomaly, mask_bin)
                        seg_loss += loss_dice(anomaly[:, 1], mask_bin)

                        # VLC mask downsample to patch grid
                        mask_small = F.interpolate(
                            mask_bin, size=(H_patch, H_patch), mode='nearest'
                        )[0, 0]  # (H_patch,H_patch)

                        seg_loss += 0.05 * vlc_loss(
                            p, mask_small,
                            normal_vlc, abnormal_vlc,
                            bce_logits
                        )

                loss = seg_loss + det_loss

            seg_opt.zero_grad()
            det_opt.zero_grad()
            loss.backward()
            seg_opt.step()
            det_opt.step()

            losses.append(loss.item())

        print("Loss:", np.mean(losses))

        # =============================================================
        # MEMORY BANK FOR TESTING (NORMAL SUPPORT)
        # =============================================================
        seg_bank = []
        det_bank = []

        for (sup,) in support_loader:
            sup = sup.to(device)
            with torch.no_grad():
                _, seg_s, det_s = model(sup)
                seg_bank.append([p[0].contiguous() for p in seg_s])
                det_bank.append([p[0].contiguous() for p in det_s])

        seg_mem = [
            torch.cat([seg_bank[j][i] for j in range(len(seg_bank))], dim=0)
            for i in range(len(seg_bank[0]))
        ]
        det_mem = [
            torch.cat([det_bank[j][i] for j in range(len(det_bank))], dim=0)
            for i in range(len(det_bank[0]))
        ]

        # --------------------------------------------------------
        # EVALUATE (same interface as original test_few)
        # --------------------------------------------------------
        from test_few import test
        score = test(
            args, model, test_loader, text_features,
            seg_mem, det_mem
        )

        if score > best_score:
            best_score = score
            print("Best updated!")

            os.makedirs(args.save_path, exist_ok=True)
            torch.save({
                "seg_adapters": model.seg_adapters.state_dict(),
                "det_adapters": model.det_adapters.state_dict()
            }, os.path.join(args.save_path, f"{args.obj}.pth"))

    print(f"\nFINAL BEST SCORE = {best_score:.4f}")


if __name__ == "__main__":
    main()
