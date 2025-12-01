import os
import torch
import numpy as np
import math
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils import cos_sim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_INDEX = {
    'Brain': 3, 'Liver': 2, 'Retina_RESC': 1,
    'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3
}


# ================================================================
# TEST FUNCTION (called by train_few)
# ================================================================
def test(args, model, test_loader, text_features, seg_mem, det_mem):
    """
    text_features : (C,2) tensor  [normal_embed , abnormal_embed]
    seg_mem       : list of memory bank tensors for segmentation
    det_mem       : list of memory bank tensors for detection
    """

    gt_labels = []
    gt_masks = []

    seg_zero_maps = []
    seg_few_maps = []

    det_zero_scores = []
    det_few_scores = []

    model.eval()

    for (image, label, mask) in tqdm(test_loader):
        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            _, seg_tokens, det_tokens = model(image)

            seg_tokens = [p[0, 1:, :] for p in seg_tokens]
            det_tokens = [p[0, 1:, :] for p in det_tokens]

            # ============================================================
            # SEGMENTATION (Brain, Liver, Retina_RESC)
            # ============================================================
            if CLASS_INDEX[args.obj] > 0:

                # ---------------- FEW-SHOT SEGMENTATION ----------------
                maps_few = []
                for mem, p in zip(seg_mem, seg_tokens):
                    p = p / p.norm(dim=-1, keepdim=True)
                    cos = cos_sim(mem, p)
                    anomaly = (1 - cos).min(dim=0)[0]

                    H = int(math.sqrt(anomaly.shape[0]))
                    anomaly = anomaly.reshape(1, 1, H, H)
                    anomaly = F.interpolate(
                        anomaly, size=args.img_size,
                        mode="bilinear", align_corners=True
                    )
                    maps_few.append(anomaly[0].cpu().numpy())

                seg_few_maps.append(np.sum(maps_few, axis=0))

                # ---------------- ZERO-SHOT SEGMENTATION ----------------
                maps_zero = []
                for p in seg_tokens:
                    p = p / p.norm(dim=-1, keepdim=True)

                    logits = (p @ text_features).unsqueeze(0)  # (1,L,2)
                    B, L, C2 = logits.shape
                    H = int(math.sqrt(L))

                    logits = logits.permute(0, 2, 1).view(1, 2, H, H)
                    logits = F.interpolate(
                        logits, size=args.img_size,
                        mode="bilinear", align_corners=True
                    )
                    logits = torch.softmax(100 * logits, dim=1)[:, 1]  # abnormal prob
                    maps_zero.append(logits.cpu().numpy())

                seg_zero_maps.append(np.sum(maps_zero, axis=0))

            # ============================================================
            # DETECTION (Chest, Histopathology, Retina_OCT2017)
            # ============================================================
            else:

                # ---------------- FEW-SHOT DETECTION ----------------
                fs = []
                for mem, p in zip(det_mem, det_tokens):
                    p = p / p.norm(dim=-1, keepdim=True)
                    cos = cos_sim(mem, p)
                    anomaly = (1 - cos).min(dim=0)[0]
                    fs.append(anomaly.mean().item())

                det_few_scores.append(np.mean(fs))

                # ---------------- ZERO-SHOT DETECTION ----------------
                zs = 0
                for p in det_tokens:
                    p = p / p.norm(dim=-1, keepdim=True)

                    logits = p @ text_features   # (L,2)
                    bin_logit = 100 * (logits[:, 1] - logits[:, 0])
                    zs += bin_logit.mean().item()

                det_zero_scores.append(zs)

        # store ground truth
        gt_labels.append(int(label.cpu()))
        gt_masks.append(mask.squeeze().cpu().numpy())

    gt_labels = np.array(gt_labels)
    gt_masks = np.array(gt_masks)
    gt_masks = (gt_masks > 0).astype(np.float32)

    # ================================================================
    # SEGMENTATION METRICS
    # ================================================================
    if CLASS_INDEX[args.obj] > 0:

        seg_zero = np.array(seg_zero_maps)
        seg_few = np.array(seg_few_maps)

        # normalize (required)
        seg_zero = (seg_zero - seg_zero.min()) / (seg_zero.max() - seg_zero.min() + 1e-8)
        seg_few  = (seg_few  - seg_few.min())  / (seg_few.max()  - seg_few.min()  + 1e-8)

        final = 0.5 * seg_zero + 0.5 * seg_few

        pAUC = roc_auc_score(gt_masks.flatten(), final.flatten())
        imgAUC = roc_auc_score(gt_labels, final.reshape(final.shape[0], -1).max(axis=1))

        print("\n================ SEGMENTATION RESULTS ================")
        print(f"{args.obj} pAUC : {pAUC:.4f}")
        print(f"{args.obj} AUC  : {imgAUC:.4f}")
        print("======================================================\n")

        return pAUC + imgAUC

    # ================================================================
    # DETECTION METRICS
    # ================================================================
    det_zero = np.array(det_zero_scores)
    det_few = np.array(det_few_scores)

    det_zero = (det_zero - det_zero.min()) / (det_zero.max() - det_zero.min() + 1e-8)
    det_few  = (det_few  - det_few.min())  / (det_few.max()  - det_few.min()  + 1e-8)

    final = 0.5 * det_zero + 0.5 * det_few

    AUC = roc_auc_score(gt_labels, final)

    print("\n================= DETECTION RESULTS ==================")
    print(f"{args.obj} AUC : {AUC:.4f}")
    print("======================================================\n")

    return AUC
