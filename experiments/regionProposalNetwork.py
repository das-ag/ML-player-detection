import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, nms, box_iou

from featureExtractor import InceptionFeatureExtractor

def generate_base_anchors(scales, aspect_ratios, down_h, down_w):
    ctr_y, ctr_x = down_h/2.0, down_w/2.0
    A = len(scales) * len(aspect_ratios)
    base = np.zeros((A,4), dtype=np.float32)
    idx = 0
    for s in scales:
        for ar in aspect_ratios:
            h = down_h * s * np.sqrt(ar)
            w = down_w * s * np.sqrt(1/ar)
            base[idx,0] = ctr_y - h/2.0  # y_min
            base[idx,1] = ctr_x - w/2.0  # x_min
            base[idx,2] = ctr_y + h/2.0  # y_max
            base[idx,3] = ctr_x + w/2.0  # x_max
            idx += 1
    return base

def generate_anchor_boxes(fmap_size, base_anchors, down_h, down_w):
    Hf, Wf = fmap_size
    shift_y = np.arange(0, Hf) * down_h
    shift_x = np.arange(0, Wf) * down_w
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.stack([shift_y.ravel(), shift_x.ravel(),
                       shift_y.ravel(), shift_x.ravel()], axis=1)
    K = shifts.shape[0]; A = base_anchors.shape[0]
    all_ = base_anchors.reshape((1,A,4)) + shifts.reshape((K,1,4))
    return all_.reshape((K*A,4))

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    """
    Projects boxes between activation‐map coords and pixel/image coords.
    
    Inputs:
      bboxes              Tensor[B, ..., 4] in [ymin, xmin, ymax, xmax] (padded boxes == -1)
      width_scale_factor  float or Tensor broadcastable to bboxes[...,0]
      height_scale_factor float or Tensor broadcastable to bboxes[...,0]
      mode                'a2p' (activation→pixel) or 'p2a' (pixel→activation)
    
    Returns:
      Tensor of same shape as `bboxes`
    """
    assert mode in ['a2p','p2a']
    B = bboxes.size(0)
    # flatten all but batch
    flat = bboxes.clone().view(B, -1, 4)
    mask = (flat == -1)  # remember padded entries

    if mode == 'a2p':
        # y dims [0,2] scale by height, x dims [1,3] by width
        flat[:,:, [0,2]] *= height_scale_factor
        flat[:,:, [1,3]] *= width_scale_factor
    else:
        flat[:,:, [0,2]] /= height_scale_factor
        flat[:,:, [1,3]] /= width_scale_factor

    # restore padding
    flat[mask] = -1
    return flat.view_as(bboxes)


def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    """
    Compute IoU between every anchor and every GT box in each image.
    
    Inputs:
      batch_size      int B
      anc_boxes_all   Tensor[B, H, W, A, 4] in [ymin, xmin, ymax, xmax]
      gt_bboxes_all   Tensor[B, N, 4] (pad boxes = -1)
    
    Returns:
      Tensor[B, H*W*A, N] of IoU values (invalid GT boxes → IoU=0)
    """
    device = anc_boxes_all.device
    # flatten anchors to (B, M, 4)
    anc_flat = anc_boxes_all.view(batch_size, -1, 4)
    M = anc_flat.size(1)
    N = gt_bboxes_all.size(1)

    ious = torch.zeros((batch_size, M, N), device=device)
    for b in range(batch_size):
        anc = anc_flat[b]               # (M,4)
        gt  = gt_bboxes_all[b]         # (N,4)
        # mask out padded GT.
        valid = (gt[:,0] >= 0)
        if valid.any():
            gt_valid = gt[valid]
            iou_b   = box_iou(anc, gt_valid)  # (M, num_valid)
            # write back only into the valid columns
            ious[b,:,valid] = iou_b
    return ious


def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    """
    Given matched positive anchors and GTs (both xyxy),
    compute Faster‑RCNN regression targets (tx,ty,tw,th).
    
    Inputs:
      pos_anc_coords   Tensor[P,4] in xyxy
      gt_bbox_mapping  Tensor[P,4] in xyxy
    
    Returns:
      Tensor[P,4] of targets
    """
    # convert to cxcywh
    anc_cxywh = box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_cxywh  = box_convert(gt_bbox_mapping,  in_fmt='xyxy', out_fmt='cxcywh')

    anc_cx, anc_cy, anc_w, anc_h = anc_cxywh.unbind(dim=1)
    gt_cx,  gt_cy,  gt_w,  gt_h    = gt_cxywh.unbind(dim=1)

    tx = (gt_cx - anc_cx) / anc_w
    ty = (gt_cy - anc_cy) / anc_h
    tw = torch.log(gt_w  / anc_w)
    th = torch.log(gt_h  / anc_h)

    return torch.stack([tx, ty, tw, th], dim=1)


def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all,
                    pos_thresh=0.7, neg_thresh=0.2):
    """
    Prepare positive/negative anchor indices and targets for RPN training.
    
    Returns:
      positive_inds, negative_inds,
      GT_conf_scores, GT_offsets, GT_class_pos,
      pos_coords, neg_coords,
      pos_inds_per_image
    """
    B, H, W, A, _ = anc_boxes_all.shape
    M = H * W * A
    N = gt_bboxes_all.size(1)

    # 1) IoU matrix [B, M, N]
    iou_mat = get_iou_mat(B, anc_boxes_all, gt_bboxes_all)

    # 2) For each GT find its best anchor → force positive
    max_iou_per_gt, _ = iou_mat.max(dim=1, keepdim=True)  # [B,1,N]
    pos_mask = (iou_mat == max_iou_per_gt) & (max_iou_per_gt > 0)
    # also any anchor > pos_thresh
    pos_mask |= (iou_mat > pos_thresh)

    # 3) flatten (B,M) → total_anchors and grab positive indices
    pos_mask_flat = pos_mask.view(-1, N)      # [(B*M), N]
    pos_any       = pos_mask_flat.any(dim=1)  # [(B*M)]
    positive_inds = torch.where(pos_any)[0]

    # keep per-image positive inds if needed
    # e.g. pos_inds_per_image = torch.where(pos_mask.view(B, M, N).any(-1))

    # 4) get conf scores for positive anchors
    # IoU of each anchor with its best GT
    max_iou_per_anchor, anchor_to_gt = iou_mat.max(dim=2)  # [B, M]
    conf_scores = max_iou_per_anchor.view(-1)[positive_inds]

    # 5) get class targets
    # expand classes & gather
    cls_expand = gt_classes_all.unsqueeze(1).expand(B, M, N)
    class_map = torch.gather(cls_expand, 2, anchor_to_gt.unsqueeze(-1)).squeeze(-1)
    class_map = class_map.view(-1)
    GT_class_pos = class_map[positive_inds]

    # 6) get GT box coords for positive anchors
    bboxes_expand = gt_bboxes_all.unsqueeze(1).expand(B, M, N, 4)
    gt_box_for_anchor = torch.gather(
        bboxes_expand,
        2,
        anchor_to_gt.view(B, M, 1, 1).repeat(1,1,1,4)
    ).squeeze(2)
    gt_box_for_anchor = gt_box_for_anchor.view(-1,4)[positive_inds]

    # 7) anchor coords
    anc_flat = anc_boxes_all.view(-1,4)
    pos_coords = anc_flat[positive_inds]

    # 8) regression targets
    offsets = calc_gt_offsets(pos_coords, gt_box_for_anchor)

    # 9) negative anchors: IoU < neg_thresh
    neg_any = max_iou_per_anchor.view(-1) < neg_thresh
    neg_inds = torch.where(neg_any)[0]
    # sample as many negatives as positives
    idx = torch.randperm(neg_inds.numel(), device=neg_inds.device)[:positive_inds.numel()]
    negative_inds = neg_inds[idx]
    neg_coords = anc_flat[negative_inds]

    return (
        positive_inds,        # long tensor [#pos]
        negative_inds,        # long tensor [#pos]
        conf_scores,          # float tensor [#pos]
        offsets,              # float tensor [#pos,4]
        GT_class_pos,         # long tensor [#pos]
        pos_coords,           # float tensor [#pos,4]
        neg_coords,           # float tensor [#pos,4]
        None                  # placeholder for per-image indexing if needed
    )


def calc_cls_loss(conf_pos, conf_neg):
    """
    Binary‐cross‐entropy on RPN objectness logits.
    conf_pos / conf_neg are raw logits for positive/negative anchors.
    """
    pos_labels = torch.ones_like(conf_pos)
    neg_labels = torch.zeros_like(conf_neg)
    loss_pos   = F.binary_cross_entropy_with_logits(conf_pos, pos_labels, reduction='mean')
    loss_neg   = F.binary_cross_entropy_with_logits(conf_neg, neg_labels, reduction='mean')
    return 0.5 * (loss_pos + loss_neg)


def calc_bbox_reg_loss(gt_offsets, pred_offsets):
    """
    Smooth‐L1 loss between predicted and GT offsets.
    """
    return F.smooth_l1_loss(pred_offsets, gt_offsets, reduction='mean')


# ---------------------------------------
# 2) ProposalModule and generate_proposals
# ---------------------------------------
class ProposalModule(nn.Module):
    def __init__(self, in_channels, hidden_dim=512, n_anchors=9, p_dropout=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1     = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.dropout   = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim,       n_anchors,   kernel_size=1)
        self.reg_head  = nn.Conv2d(hidden_dim, 4*n_anchors,   kernel_size=1)

    def forward(self, feature_map,
                      pos_anc_ind=None,
                      neg_anc_ind=None,
                      pos_anc_coords=None):
        x = F.relu(self.conv1(feature_map))
        x = self.dropout(x)

        conf_logits = self.conf_head(x)    # (B, A, H, W)
        reg_offsets = self.reg_head(x)     # (B, 4A, H, W)

        # eval/inference mode
        if pos_anc_ind is None:
            return conf_logits, reg_offsets

        # training mode
        B, A, H, W = conf_logits.shape
        M = B * A * H * W

        # flatten conf: [M]  
        cflat = conf_logits.permute(0,2,3,1).reshape(-1)
        conf_pos = cflat[pos_anc_ind]
        conf_neg = cflat[neg_anc_ind]

        # flatten offsets: [M,4]
        offlat = reg_offsets.permute(0,2,3,1).reshape(-1,4)
        offs_pos = offlat[pos_anc_ind]

        # turn positive anchors + their offsets into proposals
        proposals = generate_proposals(pos_anc_coords, offs_pos)

        return conf_pos, conf_neg, offs_pos, proposals


def generate_proposals(anchors_xyxy, offsets):
    """
    anchors_xyxy: [N,4] in (ymin,xmin,ymax,xmax)
    offsets:      [N,4] in (t_x, t_y, t_w, t_h)
    """
    # to (cx,cy,w,h)
    anc_cxywh = box_convert(anchors_xyxy, in_fmt='xyxy', out_fmt='cxcywh')
    cx, cy, w, h     = anc_cxywh.unbind(dim=1)
    dx, dy, dw, dh   = offsets.unbind(dim=1)

    pcx = cx + dx * w
    pcy = cy + dy * h
    pw  = w   * torch.exp(dw)
    ph  = h   * torch.exp(dh)

    prop_cxywh = torch.stack([pcx, pcy, pw, ph], dim=1)
    # back to xyxy
    return box_convert(prop_cxywh, in_fmt='cxcywh', out_fmt='xyxy')


# ----------------------------------------
# 3) Adapted RegionProposalNetwork
# ----------------------------------------
class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 img_size,      # (H_img, W_img)
                 fmap_size,     # (H_map, W_map)
                 feat_channels, # e.g. 2048 from InceptionFeatureExtractor
                 anc_scales= [2, 2.5, 3, 3.5, 4, 5],
                 anc_ratios =[0.5, 1.5, 2.0, 2.5, 3, 3.5, 4],
                 pos_thresh=0.5,
                 neg_thresh=0.3):
        super().__init__()
        self.H_img, self.W_img = img_size
        self.H_map, self.W_map = fmap_size

        # how many pixels per activation‐cell
        self.h_scale = self.H_img // self.H_map
        self.w_scale = self.W_img // self.W_map

        self.anc_scales = anc_scales
        self.anc_ratios = anc_ratios
        self.n_anchors  = len(anc_scales) * len(anc_ratios)

        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        self.feature_extractor = InceptionFeatureExtractor()
        self.proposal_module   = ProposalModule(feat_channels,
                                                n_anchors=self.n_anchors)

    def forward(self, images, gt_bboxes, gt_classes):
        """
        images:    [B,3,H_img,W_img]
        gt_bboxes: [B,N,4] padded, in xyxy
        gt_classes:[B,N]
        """
        device = images.device
        B = images.size(0)
        # 1) feature map
        feat_map = self.feature_extractor(images)  # [B,C,H_map,W_map]

        # 2) build anchors (pixel‐space)
        base = generate_base_anchors(self.anc_scales,
                                     self.anc_ratios,
                                     self.h_scale,
                                     self.w_scale)       # [A,4]
        flat = generate_anchor_boxes((self.H_map,self.W_map),
                                     base,
                                     self.h_scale,
                                     self.w_scale)       # [H*W*A,4]
        # reshape → [H_map,W_map,A,4] → [B,H_map,W_map,A,4]
        anc = (torch.from_numpy(flat)
                     .float()
                     .view(self.H_map, self.W_map, self.n_anchors, 4)
                     .unsqueeze(0)
                     .repeat(B,1,1,1,1)
                     .to(device)            # ← move to MPS here
             )

        # 3) project GT boxes into activation‐space
        gt_proj = project_bboxes(gt_bboxes,
                                 width_scale_factor=self.w_scale,
                                 height_scale_factor=self.h_scale,
                                 mode='p2a')

        # 4) sample positives/negatives
        (pos_inds, neg_inds,
         GT_conf, GT_offs, GT_cls,
         pos_anc_xy, neg_anc_xy,
        _sep) = get_req_anchors(anc, gt_proj, gt_classes,
                               self.pos_thresh, self.neg_thresh)

        # 5) RPN heads
        conf_pos, conf_neg, offs_pos, proposals = \
            self.proposal_module(feat_map,
                                 pos_inds, neg_inds, pos_anc_xy)

        # 6) losses
        cls_loss = calc_cls_loss(conf_pos, conf_neg)
        reg_loss = calc_bbox_reg_loss(GT_offs, offs_pos)

        total = cls_loss + 5.0*reg_loss
        return total, feat_map, proposals, _sep, GT_cls

    @torch.no_grad()
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        """
        Returns a list of (proposals, scores) per image.
        """
        device = images.device
        B = images.size(0)
        feat_map = self.feature_extractor(images)

        # rebuild the same anchors
        base = generate_base_anchors(self.anc_scales,
                                     self.anc_ratios,
                                     self.h_scale,
                                     self.w_scale)
        flat = generate_anchor_boxes((self.H_map,self.W_map),
                                     base,
                                     self.h_scale,
                                     self.w_scale)
        anc = (torch.from_numpy(flat)
                   .float()
                   .view(self.H_map, self.W_map, self.n_anchors, 4)
                   .unsqueeze(0)
                   .repeat(B,1,1,1,1)
                   .to(device)            # ← and here
              )
        anc_flat = anc.reshape(B, -1, 4)

        # RPN heads
        conf_logits, reg_offsets = self.proposal_module(feat_map)
        scores = torch.sigmoid(conf_logits).reshape(B, -1)
        offs   = reg_offsets.reshape(B, -1, 4)

        results = []
        for i in range(B):
            sc  = scores[i]
            of  = offs[i]
            ac  = anc_flat[i]
            props = generate_proposals(ac, of)
            keep1 = torch.where(sc>=conf_thresh)[0]
            props = props[keep1]; sc = sc[keep1]
            keep2 = nms(props, sc, nms_thresh)
            results.append((props[keep2], sc[keep2]))

        return results