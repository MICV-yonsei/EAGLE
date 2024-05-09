import torch
from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import cv2
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
import gc

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            state_dict = state_dict["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False, map_location="cpu")
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384 * 3
        else:
            self.n_feats = 768 * 3 
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg.projection_type
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size
            
            # get selected layer activations
            feat_all, attn_all, qkv_all = self.model.get_intermediate_feat(img, n=n)

            # high level
            feat, attn, qkv = feat_all[-1], attn_all[-1], qkv_all[-1]
            
            image_feat_high = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            image_k_high = qkv[1, :, :, 1:, :].reshape(feat.shape[0], attn.shape[1], feat_h, feat_w, -1)
            B, H, I, J, D = image_k_high.shape
            image_kk_high = image_k_high.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
                       
            # mid level
            feat_mid, attn_mid, qkv_mid = feat_all[-2], attn_all[-2], qkv_all[-2]
            
            image_feat_mid = feat_mid[:, 1:, :].reshape(feat_mid.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            image_k_mid = qkv_mid[1, :, :, 1:, :].reshape(feat_mid.shape[0], attn.shape[1], feat_h, feat_w, -1)
            B, H, I, J, D = image_k_mid.shape
            image_kk_mid = image_k_mid.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            
            # low level
            feat_low, attn_low, qkv_low = feat_all[-3], attn_all[-3], qkv_all[-3]
            
            image_feat_low = feat_low[:, 1:, :].reshape(feat_low.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            image_k_low = qkv_low[1, :, :, 1:, :].reshape(feat_low.shape[0], attn.shape[1], feat_h, feat_w, -1)
            B, H, I, J, D = image_k_low.shape
            image_kk_low = image_k_low.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            
            image_feat = torch.cat([image_feat_low, image_feat_mid, image_feat_high], dim=1)
            image_kk  = torch.cat([image_kk_low, image_kk_mid, image_kk_high], dim=1)
            
            
            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            with torch.no_grad():
                code = self.cluster1(self.dropout(image_feat))
            code_kk = self.cluster1(self.dropout(image_kk))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
                code_kk += self.cluster2(self.dropout(image_kk))
        else:
            code = image_feat
            code_kk = image_kk

        if self.cfg.dropout:
            return self.dropout(image_feat), self.dropout(image_kk), code, code_kk
        else:
            return image_feat, image_kk, code, code_kk


class ResizeAndClassify(nn.Module):

    def __init__(self, dim: int, size: int, n_classes: int):
        super(ResizeAndClassify, self).__init__()
        self.size = size
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(dim, n_classes, (1, 1)),
            torch.nn.LogSoftmax(1))

    def forward(self, x):
        return F.interpolate(self.predictor.forward(x), self.size, mode="bilinear", align_corners=False)


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])

class CorrespondenceLoss(nn.Module):

    def __init__(self, cfg, ):
        super(CorrespondenceLoss, self).__init__()
        self.cfg = cfg
        self.neg_samples = cfg.neg_samples
        self.mse_loss = nn.MSELoss()

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, POS):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0
        if POS:
            shift = torch.abs(fd.mean() - cd.mean()-self.cfg.shift_bias)
        else:
            shift = (fd.mean() + cd.mean()-self.cfg.shift_bias) * self.cfg.shift_value
            
        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)
      
        return loss, cd
    
    def id_loss(self, input_tensor):
    
        batch_size, H, W, _  = input_tensor.shape
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        downsampled_tensor = F.interpolate(input_tensor, scale_factor=0.5, mode='bilinear', align_corners=True)

        reshaped_tensor = downsampled_tensor.permute(0,2,3,1).view(batch_size, H//2 * W//2, -1)
        normalized_patches = F.normalize(reshaped_tensor,dim=-1)

        similarity_matrix_batched = torch.bmm(normalized_patches, normalized_patches.transpose(-2, -1))

        min_vals = similarity_matrix_batched.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
        max_vals = similarity_matrix_batched.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        similarity_matrix_batched = (similarity_matrix_batched - min_vals) / (max_vals - min_vals)        
    
        I = torch.eye(similarity_matrix_batched.shape[1], device=similarity_matrix_batched.device)
        loss = self.mse_loss(similarity_matrix_batched, I.unsqueeze(0).repeat(batch_size, 1, 1))
        gc.collect()
        torch.cuda.empty_cache()
        return loss


    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor, orig_feats_pos_aug: torch.Tensor, 
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor, orig_code_pos_aug: torch.Tensor, 
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords3 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)
        
        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)
        
        feats_pos_aug = sample(orig_feats_pos_aug, coords3)
        code_pos_aug = sample(orig_code_pos_aug, coords3)
        
        pos_inter_loss, pos_inter_cd = self.helper(
            feats_pos_aug, feats_pos, code_pos_aug, code_pos, POS = True)
        
        if self.neg_samples > 0:
            neg_losses = []
            neg_cds = []
            feats_neg_list = []
            code_neg_list = []
            for i in range(self.cfg.neg_samples):
                perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
                feats_neg = sample(orig_feats[perm_neg], coords2)
                feats_neg_list.append(feats_neg)
                code_neg = sample(orig_code[perm_neg], coords2)
                code_neg_list.append(code_neg)
                neg_inter_loss, neg_inter_cd = self.helper(
                    feats, feats_neg, code, code_neg, POS=False)
                
                neg_losses.append(neg_inter_loss)
                neg_cds.append(neg_inter_cd)

            
            neg_inter_loss = torch.cat(neg_losses, axis=0)
            neg_inter_cd = torch.cat(neg_cds, axis=0)
            
            return (
                    pos_inter_loss.mean(),
                    pos_inter_cd,
                    neg_inter_loss,
                    neg_inter_cd
                    )
            
        else:
            return (
                    pos_inter_loss.mean(),
                    pos_inter_cd,
                    )
            

class Decoder(nn.Module):
    def __init__(self, code_channels, feat_channels):
        super().__init__()
        self.linear = torch.nn.Conv2d(code_channels, feat_channels, (1, 1))
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, feat_channels, (1, 1)))

    def forward(self, x):
        return self.linear(x) + self.nonlinear(x)


class NetWithActivations(torch.nn.Module):
    def __init__(self, model, layer_nums):
        super(NetWithActivations, self).__init__()
        self.layers = nn.ModuleList(model.children())
        self.layer_nums = []
        for l in layer_nums:
            if l < 0:
                self.layer_nums.append(len(self.layers) + l)
            else:
                self.layer_nums.append(l)
        self.layer_nums = set(sorted(self.layer_nums))

    def forward(self, x):
        activations = {}
        for ln, l in enumerate(self.layers):
            x = l(x)
            if ln in self.layer_nums:
                activations[ln] = x
        return activations


class ContrastiveCRFLoss(nn.Module):

    def __init__(self, n_samples, alpha, beta, gamma, w1, w2, shift):
        super(ContrastiveCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift

    def forward(self, guidance, clusters):
        device = clusters.device
        assert (guidance.shape[0] == clusters.shape[0])
        assert (guidance.shape[2:] == clusters.shape[2:])
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat([
            torch.randint(0, h, size=[1, self.n_samples], device=device),
            torch.randint(0, w, size=[1, self.n_samples], device=device)], 0)

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]
        coord_diff = (coords.unsqueeze(-1) - coords.unsqueeze(1)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(2)).square().sum(1)

        sim_kernel = self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(- coord_diff / (2 * self.gamma)) - self.shift

        selected_clusters = clusters[:, :, coords[0, :], coords[1, :]]
        cluster_sims = torch.einsum("nka,nkb->nab", selected_clusters, selected_clusters)
        return -(cluster_sims * sim_kernel)



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothingCrossEntropy class.
        :param smoothing: The smoothing factor (float, default: 0.1).
                          This factor dictates how much we will smooth the labels.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        """
        Forward pass of the loss function.
        :param input: Predictions from the model (before softmax) (tensor).
        :param target: True labels (tensor).
        """
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    
class newLocalGlobalInfoNCE(nn.Module):
    def __init__(self, cfg, num_classes):
        super(newLocalGlobalInfoNCE, self).__init__()
        self.cfg = cfg
        self.learned_centroids = nn.Parameter(torch.randn(num_classes, cfg.dim))
        self.prototypes = torch.randn(num_classes + cfg.extra_clusters, cfg.dim, requires_grad=True)
        
    def compute_centroid(self, features, labels):
        unique_labels = torch.unique(labels)
        centroids = []
        
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            if self.cfg.centroid_mode == 'mean':
                centroids.append(class_features.mean(0))
            elif self.cfg.centroid_mode == 'medoid':
                pairwise_dist = torch.cdist(class_features, class_features)
                centroids.append(class_features[torch.argmin(pairwise_dist.sum(0))])
            elif self.cfg.centroid_mode == 'learned':
                centroids.append(self.learned_centroids[label])
            elif self.cfg.centroid_mode == 'prototype':
                pairwise_dist = torch.cdist(class_features, class_features)
                prototype = class_features[torch.argmin(pairwise_dist.sum(0))]
                new_prototypes = self.prototypes.clone()  
                new_prototypes[label] =  prototype 
                self.prototypes = new_prototypes
                centroids.append(prototype)
                
        return torch.stack(centroids)
    
    def forward(self, S1, S2, segmentation_map, similarity_matrix):

        label_smoothing_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        batch_size, patch_size = segmentation_map.size(0), segmentation_map.size(1)
        segmentation_map = segmentation_map.reshape(-1)
        S1_centroids = self.compute_centroid(S1, segmentation_map)

        local_logits = torch.mm(S1, S1_centroids.t()) / self.cfg.contrastive_temp
        global_logits = torch.mm(S2, S1_centroids.t()) / self.cfg.contrastive_temp

        mask = (segmentation_map.unsqueeze(1) == torch.unique(segmentation_map)) 
        labels = mask.float().argmax(dim=1)

        local_weights = (similarity_matrix.mean(dim=2).reshape(-1)) * 1.0
        global_weights = (similarity_matrix.mean(dim=2).reshape(-1)) * 1.0
        
        if self.cfg.dataset_name=='cityscapes':
            local_loss = label_smoothing_criterion(local_logits, labels)
            global_loss = label_smoothing_criterion(global_logits, labels)
        
        else:
            local_loss = F.cross_entropy(local_logits, labels, reduction='none')
            global_loss = F.cross_entropy(global_logits, labels, reduction='none')
            
        local_loss = (local_loss * local_weights).mean()
        global_loss = (global_loss * global_weights).mean()

        total_loss = ((1-self.cfg.global_loss_weight) * local_loss + self.cfg.global_loss_weight * global_loss) / 2
        
        return total_loss