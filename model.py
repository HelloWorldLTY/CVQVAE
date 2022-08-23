import numpy as np
import pandas as pd
import scanpy as sc
import os
from skmisc.loess import loess
import sklearn.preprocessing as preprocessing

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_s 
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

import scib 
import random

from sklearn.decomposition import TruncatedSVD

import scipy

# lsi for scATAC-seq data
def lsi_transform(adata: sc.AnnData, n_comp=50, n_peaks=30000):
    top_idx = set(np.argsort(adata.X.sum(axis=0).A1)[-n_peaks:])
    features = adata.var_names.tolist()
    X = adata[:, features].layers["counts"]
    idf = X.shape[0] / X.sum(axis=0).A1
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        X = tf.multiply(idf)
        X = X.multiply(1e4 / X.sum(axis=1))
    else:
        tf = X / X.sum(axis=1)
        X = tf * idf
        X = X * (1e4 / X.sum(axis=1))
    X = np.log1p(X)
    print('yes')
    u, s, vh = scipy.sparse.linalg.svds(X, n_comp)
    X_lsi = X @ vh.T / s
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
    return adata      
 
# pca for scATAC-seq data
def pca_transform(adata: sc.AnnData, features):
    adata_new = sc.AnnData(adata_gex.X)
    sc.pp.scale(adata_new, max_value = 10)
    sc.tl.pca(adata_new, features)
    adata.obsm['X_pca'] = adata_new.obsm['X_pca']
    return adata


#EMA process
class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.
    
      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
      
      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """
    
    def __init__(self, init_value, decay):
        super().__init__()
        
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))
        
    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x*torch.tanh(F.softplus(x))
        
class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
               epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        
        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)
        
        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)
        
    def forward(self, x):
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Use index to find embeddings in the latent space
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) 
        
        #EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                      (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x) # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
              updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w
    

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
    
        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)      
        
    
class Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    
    def __init__(self, in_dim=2500, latent_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        
        self.enc = nn.Sequential(
                    nn.Linear(2504, 1024),
                    nn.BatchNorm1d(1024),
                    Mish(),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    Mish(),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    Mish(),
                    nn.Linear(256, latent_dim),
        )
        
    def forward(self, x):
        return self.enc(x)
    

class Decoder(nn.Module):
    """Decoder of VQ-VAE"""
    
    def __init__(self, out_dim=2500, latent_dim=16):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        
        self.dec= nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.BatchNorm1d(256),
                    Mish(),
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    Mish(),
                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    Mish(),
                    nn.Linear(1024, 2504),
        )
        
    def forward(self, x):
        return self.dec(x)
        
        

# default type        
class VQVAE_EMA(nn.Module):
    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance1, data_variance2,
                 commitment_cost=0.25, lambda_z = 10, decay=0.99):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance1 = data_variance1
        self.data_variance2 = data_variance2
        
        self.encoder_gexatac = Encoder(in_dim, embedding_dim)
        self.vq_layer_gexatac = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder_gexatac = Decoder(in_dim, embedding_dim)
        
        self.encoder_atacgex = Encoder(in_dim, embedding_dim)
        self.vq_layer_atacgex = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder_atacgex = Decoder(in_dim, embedding_dim)
        
        self.lambda_z = lambda_z
        
    def forward(self, x, y):
        z = self.encoder_gexatac(x)
        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e, e_q_loss = self.vq_layer_gexatac(z)
        x_recon = self.decoder_gexatac(e)
    
        recon_loss = F.mse_loss(x_recon, y) / self.data_variance1 #atac
        
        z1 = self.encoder_atacgex(y)
        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e1, e_q_loss1 = self.vq_layer_atacgex(z1)
        y_recon = self.decoder_gexatac(e1)
    
        recon_loss1 = F.mse_loss(y_recon, x) / self.data_variance2 #gex       
        
#         print(F.mse_loss(z,z1))
#         print(e_q_loss + recon_loss)
        return e_q_loss + recon_loss, e_q_loss1+ recon_loss1, F.mse_loss(z,z1), e_q_loss + recon_loss +e_q_loss1+ recon_loss1 + self.lambda_z*F.mse_loss(z,z1)

# average mode
class VQVAE_EMA_avg(nn.Module):
    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance1, data_variance2,
                 commitment_cost=0.25, lambda_z = 10, decay=0.99):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance1 = data_variance1
        self.data_variance2 = data_variance2
        
        self.encoder_gexatac = Encoder(in_dim, embedding_dim)
        self.vq_layer_gexatac = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder_gexatac = Decoder(in_dim, embedding_dim)
        
        self.encoder_atacgex = Encoder(in_dim, embedding_dim)
        self.vq_layer_atacgex = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder_atacgex = Decoder(in_dim, embedding_dim)
        
        self.lambda_z = lambda_z
        
    def forward(self, x, y):
        z = self.encoder_gexatac(x)
        z1 = self.encoder_atacgex(y)
        
        z_new = (z+z1)/2
        
        if not self.training:
            e = self.vq_layer(z_new)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e, e_q_loss = self.vq_layer_gexatac(z_new)
        x_recon = self.decoder_gexatac(e)
    
        recon_loss = F.mse_loss(x_recon, y) / self.data_variance1 #atac
        
        if not self.training:
            e = self.vq_layer(z_new)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e1, e_q_loss1 = self.vq_layer_atacgex(z_new)
        y_recon = self.decoder_gexatac(e1)
    
        recon_loss1 = F.mse_loss(y_recon, x) / self.data_variance2 #gex       
        
#         print(F.mse_loss(z,z1))
#         print(e_q_loss + recon_loss)
        return e_q_loss + recon_loss, e_q_loss1+ recon_loss1, F.mse_loss(z,z1), e_q_loss + recon_loss +e_q_loss1+ recon_loss1 + self.lambda_z*F.mse_loss(z,z1)


# linear combination mode
class VQVAE_EMA_lin(nn.Module):
    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance1, data_variance2,
                 commitment_cost=0.25, lambda_z = 10, decay=0.99):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance1 = data_variance1
        self.data_variance2 = data_variance2
        
        self.encoder_gexatac = Encoder(in_dim, embedding_dim)
        self.vq_layer_gexatac = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder_gexatac = Decoder(in_dim, embedding_dim)
        
        self.encoder_atacgex = Encoder(in_dim, embedding_dim)
        self.vq_layer_atacgex = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder_atacgex = Decoder(in_dim, embedding_dim)
        
        self.lambda_z = lambda_z
        
        self.linear_comb = nn.Linear(2*embedding_dim, embedding_dim)
        
    def forward(self, x, y):
        z = self.encoder_gexatac(x)
        z1 = self.encoder_atacgex(y)
        
        z_new = self.linear_comb(torch.cat((z,z1), 1))
        
        if not self.training:
            e = self.vq_layer(z_new)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e, e_q_loss = self.vq_layer_gexatac(z_new)
        x_recon = self.decoder_gexatac(e)
    
        recon_loss = F.mse_loss(x_recon, y) / self.data_variance1 #atac
        
        if not self.training:
            e = self.vq_layer(z_new)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e1, e_q_loss1 = self.vq_layer_atacgex(z_new)
        y_recon = self.decoder_gexatac(e1)
    
        recon_loss1 = F.mse_loss(y_recon, x) / self.data_variance2 #gex       
        
#         print(F.mse_loss(z,z1))
#         print(e_q_loss + recon_loss)
        return e_q_loss + recon_loss, e_q_loss1+ recon_loss1, F.mse_loss(z,z1), e_q_loss + recon_loss +e_q_loss1+ recon_loss1 + self.lambda_z*F.mse_loss(z,z1)