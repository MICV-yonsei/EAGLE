import torch

from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits
# from sklearn.cluster import KMeans, MiniBatchKMeans
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from kmeans_pytorch import kmeans, kmeans_predict
import math
from scipy.spatial.distance import cdist
import torch.nn.functional as F

def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )
    device = image.device
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h
    r, g, b = r.to(device), g.to(device), b.to(device)
    
    x = torch.repeat_interleave(torch.linspace(0, 1, w).to(device), h)
    y = torch.cat([torch.linspace(0, 1, h)] * w).to(device)

    i, j = [], [] 

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = torch.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
             axis=1,
             out=torch.zeros((n, 5), dtype=torch.float32).to(device)
        ).to(device) 
        
        distances, neighbors = knn(f.cpu().numpy(), f.cpu().numpy(), k=k)
        
        distances = torch.tensor(distances)
        neighbors = torch.tensor(neighbors)

        i.append(torch.repeat_interleave(torch.arange(n), k))
        j.append(neighbors.view(-1))

    ij = torch.cat(i + j)
    ji = torch.cat(j + i)
    coo_data = torch.ones(2 * sum(n_neighbors) * n)

    W = scipy.sparse.csr_matrix((coo_data.cpu().numpy(), (ij.cpu().numpy(), ji.cpu().numpy())), (n, n))
    return torch.tensor(W.toarray())


def rw_affinity(image, sigma=0.033, radius=1):
    """Computes a random walk-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.laplacian.rw_laplacian import _rw_laplacian
    except:
        raise ImportError(
            'Please install pymatting to compute RW affinity matrices:\n'
            'pip3 install pymatting'
        )
    h, w = image.shape[:2]
    n = h * w
    values, i_inds, j_inds = _rw_laplacian(image, sigma, radius)
    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))
    return W

def multi_seg(img, eigenvalues, eigenvectors, adaptive = True, num_eigenvectors: int = 1_000_000):
    adaptive = False
    non_adaptive_num_segments = 27
    if adaptive: 
        indices_by_gap = np.argsort(np.diff(eigenvalues))[::-1]
        index_largest_gap = indices_by_gap[indices_by_gap != 0][0]  # remove zero and take the biggest
        n_clusters = index_largest_gap + 1
        print(f'Number of clusters: {n_clusters}')
    else: # of class
        n_clusters = non_adaptive_num_segments
        
    eigenvectors = eigenvectors[:, :, 1:] # take non-constant eigenvectors

    segmap_list = []
    
    for i in range(eigenvectors.shape[0]):
        C, H, W = img[i].shape
        H_patch, W_patch = H // 8, W // 8

        eigenvector_batch = eigenvectors[i]
        
        clusters, cluster_centers = kmeans(X=eigenvector_batch, distance = 'euclidean', num_clusters=n_clusters, device=eigenvector_batch.device)

        if clusters.cpu().numpy().size == H_patch * W_patch:
            segmap = clusters.reshape(H_patch, W_patch)
        elif clusters.cpu().numpy().size == H_patch * W_patch * 4:
            segmap = clusters.reshape(H_patch * 2, W_patch * 2)
        elif clusters.cpu().numpy().size == (H_patch * W_patch - 1):
            clusters = np.append(clusters, 0)
            segmap = clusters.reshape(H_patch, W_patch)
        else:
            raise ValueError()
        segmap_list.append(segmap)
    return torch.stack(segmap_list)
    
def visualize_segmap(segmap_list):
    for segmap in segmap_list:
        segmap_uint8 = segmap.to(torch.uint8)
        output_file = f'./img/image_segmap.png'

        colormap = [[0,0,0], [120,0,0], [0, 150, 0],[240, 230, 140],[176, 48, 96],[48, 176, 96],[103, 255, 255],[238, 186, 243],[119, 159, 176],[122, 186, 220],[96, 204, 96],[220, 247, 164],[60, 60, 60],[220, 216, 20],[196, 58, 250],[120, 18, 134],[12, 48, 255],[236, 13, 176],[0, 118, 14],[165, 42, 42],[160, 32, 240],[56, 192, 255],[184, 237, 194],[180, 231, 250],[299, 300, 0], [100, 200, 94],[39,203, 123]]
        colormap = np.array(colormap)

        out_conf = np.zeros((segmap_uint8.shape[0], segmap_uint8.shape[1],3))
        for x in range(segmap_uint8.shape[0]):
            for y in range(segmap_uint8.shape[1]):
                out_conf[x,y,:] = colormap[segmap_uint8[x,y]]
        import imageio
        imageio.imsave(output_file, out_conf.astype(np.uint8))

def attention_map(image_feat):
    ax = sns.heatmap(image_feat[1])
    plt.title('feat')
    plt.savefig(f'laplacian_1.png')
    plt.close()

    ax = sns.heatmap(image_feat[2])
    plt.title('feat')
    plt.savefig(f'laplacian_2.png')
    plt.close()
    return

def get_diagonal(W, threshold: float=1e-12):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.float32)

    D = torch.matmul(W, torch.ones(W.shape[1], dtype=W.dtype).to(W.device))
    D[D < threshold] = 1.0  # Prevent division by zero.

    D_diag = torch.diag(D)
    return D_diag

class EigenLoss(nn.Module):
    def __init__(self, cfg, ):
        super(EigenLoss, self).__init__()
        self.cfg = cfg
        self.eigen_cluster = cfg.eigen_cluster

    def normalized_laplacian(self, L, D):
        D_inv_sqrt = torch.inverse(torch.sqrt(D))
        D_inv_sqrt = D_inv_sqrt.diagonal(dim1=-2, dim2=-1)
        
        D_inv_sqrt_diag = torch.diag_embed(D_inv_sqrt)

        L_norm = torch.bmm(D_inv_sqrt_diag, torch.bmm(L, D_inv_sqrt_diag))
        
        return L_norm

    def batch_trace(self,tensor):
        diagonals = torch.diagonal(tensor, dim1=1, dim2=2)
        trace_values = torch.sum(diagonals, dim=1)
        return trace_values

    def eigen(self, lap, K):
        eigenvalues_all, eigenvectors_all = torch.linalg.eigh(lap, UPLO='U')
        eigenvalues = eigenvalues_all[:, :K]
        eigenvectors = eigenvectors_all[:, :, :K]    
        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()

        for k in range(eigenvectors.shape[0]):
            if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  
                eigenvectors[k] = 0 - eigenvectors[k]
        return eigenvalues, eigenvectors
    
    def pairwise_distances(self, x, y=None, w=None):
        x_norm = (x**2).sum(1).view(-1, 1)

        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def compute_color_affinity(self, image, sigma_c=30, radius=1):
        H, W, _ = image.shape
        N = H * W
        pixels = image.view(-1, 3).float() / 255.0 
        color_distances = self.pairwise_distances(pixels)
        W_color = torch.exp(-color_distances**2 / (2 * sigma_c**2))
        
        y, x = np.mgrid[:H, :W]
        coords = np.c_[y.ravel(), x.ravel()]
        
        spatial_distances = cdist(coords, coords, metric='euclidean')
        
        W_color[spatial_distances > radius] = 0
        return W_color
    
    def laplacian(self, adj, W):
        adj = (adj * (adj > 0))
        max_values = adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        adj = adj / max_values 
        w_combs = W.to(adj.device)
        
        max_values = w_combs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        w_combs = w_combs / max_values 

        W_comb = w_combs + adj
        D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
        L_comb = D_comb - W_comb
        lap = self.normalized_laplacian(L_comb, D_comb)
        return lap
    
    def color_affinity(self, img):
        color = []
        for img_ in img:
            normalized_image = img_ / 255.0 
            pixels = normalized_image.view(-1, 3)
            color_distances = torch.cdist(pixels, pixels, p=2.0)
            color_affinity = torch.exp(-color_distances ** 2 / (2 * (0.1 ** 2)))  
            color.append(color_affinity)
            
        aff_color = torch.stack(color)
        return aff_color
    
    def laplacian_matrix(self, img, image_feat, image_color_lambda=0, which_color_matrix='knn'):
        threshold_at_zero = True

        if threshold_at_zero:
            image_feat = (image_feat * (image_feat > 0))

        max_values = image_feat.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        image_feat = image_feat / max_values 
        
        if image_color_lambda > 0:
            img_resize = F.interpolate(img, size=(28, 28), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            max_values = img_resize.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            img_norm = img_resize / max_values
            
            affinity_matrices = []

            for img_norm_b in img_norm:
                if which_color_matrix == 'knn':
                    W_lr = knn_affinity(img_norm_b)
                elif which_color_matrix == 'rw':
                    W_lr = rw_affinity(img_norm_b)
                affinity_matrices.append(W_lr)
            W_color = torch.stack(affinity_matrices).to(image_feat.device)

            W_comb = image_feat + W_color * image_color_lambda
        else:
            W_comb = image_feat

        D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
        L_comb = D_comb - W_comb
        lap = self.normalized_laplacian(L_comb, D_comb)
        return lap

    def lalign(self, img, Y, code, adj, adj_code, code_neg_torch, neg_sample=5):
        if code_neg_torch is None:
            if Y.shape[1] == 196:
                img = F.interpolate(img, size=(14, 14), mode='bilinear', align_corners=False).permute(0,2,3,1)
            else:
                img = F.interpolate(img, size=(28, 28), mode='bilinear', align_corners=False).permute(0,2,3,1)
            
            color_W = self.color_affinity(img)
            nor_adj_lap = self.laplacian(adj_code, color_W)
                
            nor_adj_lap_detach = torch.clone(nor_adj_lap.detach()) 
            eigenvalues, eigenvectors = self.eigen(nor_adj_lap_detach, K=self.eigen_cluster) 
            return eigenvectors
        else:
            adj_lap = self.laplacian_matrix(img, adj, image_color_lambda=0.1) 

            max_values = code.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            code_norm = code / max_values 

            code_neg_torch = code_neg_torch.reshape(code_neg_torch.shape[0],code_neg_torch.shape[1],code_neg_torch.shape[2], -1).permute(0,1,3,2) # [5, B, 121, 512]

            return eigenvectors
    
    def forward(self, img, feat, code, corr_feats_pos, code_neg_torch, neg_sample=5):
        feat = F.normalize(feat, p=2, dim=-1)
        adj = torch.bmm(feat, feat.transpose(1,2))
        adj_code = torch.bmm(code, code.transpose(1,2))

        if code_neg_torch is None:
            eigenvectors = self.lalign(img, feat, code, adj, adj_code, code_neg_torch, neg_sample)
            return eigenvectors 
        else:
            eigenvectors, pos, neg = self.lalign(img, feat, code, adj, adj_code, code_neg_torch, neg_sample)
            return eigenvectors, pos, neg
