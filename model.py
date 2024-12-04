import torch
from torch import nn
from torch import optim
from torch import tensor
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.manifold import TSNE
import visualtorch as visual
from torchsummary import summary
import warnings
warnings.filterwarnings('ignore')
import math
from utils import alpha


class SelfAttention(nn.Module):
    def __init__(self, d_in : int, d_out_kq : int, d_out_v : int, bias : bool = True):
        super(SelfAttention, self).__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
        
        self.bias = bias
        if bias:
            self.b_query = nn.Parameter(torch.rand(d_out_kq))
            self.b_key = nn.Parameter(torch.rand(d_out_kq))
            self.b_value = nn.Parameter(torch.rand(d_out_v))

    def forward(self, x : tensor) -> tensor:
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        if self.bias:
            queries += self.b_query
            keys += self.b_key
            values += self.b_value
        
        attn_scores = queries @ keys.T  
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1
        )
        
        context_vec = attn_weights @ values
        
        return context_vec
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_in : int, d_out_kq : int, d_out_v : int, num_heads : int, pyt : bool = True) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        
        self.n_heads = num_heads
        
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) for _ in range(num_heads)]
        )
        
        self.pyt = pyt
        self.mlh = nn.MultiheadAttention(d_in, num_heads=num_heads, dropout=0.1, add_bias_kv=True, kdim=d_out_kq, vdim=d_out_v, batch_first=True)
    
    def forward(self, x : tensor) -> tensor:
        
        if self.pyt:
            v_out, attn_weights = self.mlh(x, x, x)
            return v_out 
        
        split_x = torch.chunk(x, self.n_heads, dim=-1)
        return torch.cat([head(x) for head, x in zip(self.heads, split_x)], dim=-1)
    
class CrossAttention(nn.Module):
    def __init__(self, d_in : int, d_out_kq : int, d_out_v : int) -> None:
        
        super(CrossAttention, self).__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
        
    def forward(self, x_1 : tensor, x_2 : tensor) -> tensor:           
        queries_1 = x_1 @ self.W_query
        
        keys_2 = x_2 @ self.W_key          
        values_2 = x_2 @ self.W_value
        
        attn_scores = queries_1 @ keys_2.T
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)
        
        context_vec = attn_weights @ values_2
        return context_vec

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_in : int, d_out_kq : int, d_out_v : int, num_heads : int, pyt : bool = True):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) 
             for _ in range(num_heads)]
        )

        self.pyt = pyt
        self.mlh = nn.MultiheadAttention(d_in, num_heads=num_heads, dropout=0.1, add_bias_kv=True, kdim=d_out_kq, vdim=d_out_v, batch_first=True)
    
    def forward(self, x_1 : tensor, x_2 : tensor) -> tensor:
        
        if self.pyt:
            v_out, atten_weights = self.mlh(x_1, x_2, x_2)
            return v_out
                    
        return torch.cat([head(x_1, x_2) for head in self.heads], dim=-1)

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
        # Ensure weight_epsilon and bias_epsilon do not require gradients
        self.weight_epsilon.requires_grad = False
        self.bias_epsilon.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
  
class T1PlatformTransformerEncoder(nn.Module):
    """
    1) Add substitution for BatchNorm
    2) Change no. of self-attention layers
    """
    def __init__(self, embedding_len : int = 7, final_embedding : int = 10, n_heads : int = 1, n_layers : int = 6, dropout : float = 0.1):
        super(T1PlatformTransformerEncoder, self).__init__()
        
        self.embedding_len = embedding_len
        
        self.sa_1 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_1_1 = nn.LayerNorm(embedding_len)
        self.mlp_1 = nn.Sequential(
            nn.Linear(embedding_len, embedding_len//2),
            nn.ReLU(),
            nn.Linear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_1_2 = nn.LayerNorm(embedding_len)
        
        self.sa_2 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_2_1 = nn.LayerNorm(embedding_len)
        self.mlp_2 = nn.Sequential(
            nn.Linear(embedding_len, embedding_len//2),
            nn.ReLU(),
            nn.Linear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_2_2 = nn.LayerNorm(embedding_len)
        
        # self.sa_3 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_3_1 = nn.LayerNorm(embedding_len)
        # self.mlp_3 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_3_2 = nn.LayerNorm(embedding_len)
        
        # self.sa_4 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_4_1 = nn.LayerNorm(embedding_len)
        # self.mlp_4 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_4_2 = nn.LayerNorm(embedding_len)
        
        # self.sa_5 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_5_1 = nn.LayerNorm(embedding_len)
        # self.mlp_5 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_5_2 = nn.LayerNorm(embedding_len)
        
        self.fin_emb_transform = nn.Linear(embedding_len, final_embedding)
    
    def forward(self, x : tensor) -> tensor:
        """
        x : (batch_sz, context_len, embedding_dim)
        """
        
        x_1 = self.sa_1(x)
        x_1 = x_1 + x
        x_1 = self.ln_1_1(x_1)
        c_1 = x_1.shape[1]
        x_t1 = x_1
        x_1 = x_1.reshape(-1, self.embedding_len)
        x_1 = self.mlp_1(x_1)
        x_1 = x_1.view(-1, c_1, self.embedding_len)
        x_1 = x_1 + x_t1
        x_1 = self.ln_1_2(x_1)
        
        x_2 = self.sa_2(x_1)
        x_2 = x_2 + x_1
        x_2 = self.ln_2_1(x_2)
        c_2 = x_2.shape[1]
        x_t2 = x_2
        x_2 = x_2.reshape(-1, self.embedding_len)
        x_2 = self.mlp_2(x_2)
        x_2 = x_2.view(-1, c_2, self.embedding_len)
        x_2 = x_2 + x_t2
        x_2 = self.ln_2_2(x_2)
        
        # x_3 = self.sa_3(x_2)
        # x_3 = x_3 + x_2
        # x_3 = self.ln_3_1(x_3)
        # c_3 = x_3.shape[1]
        # x_t3 = x_3
        # x_3 = x_3.reshape(-1, self.embedding_len)
        # x_3 = self.mlp_3(x_3)
        # x_3 = x_3.view(-1, c_3, self.embedding_len) 
        # x_3 = x_3 + x_t3
        # x_3 = self.ln_3_2(x_3)
        
        # x_4 = self.sa_4(x_3)
        # x_4 = x_4 + x_3
        # x_4 = self.ln_4_1(x_4)
        # c_4 = x_4.shape[1]
        # x_t4 = x_4
        # x_4 = x_4.reshape(-1, self.embedding_len)
        # x_4 = self.mlp_4(x_4)
        # x_4 = x_4.view(-1, c_4, self.embedding_len)
        # x_4 = x_4 + x_t4
        # x_4 = self.ln_4_2(x_4)
        
        # x_5 = self.sa_5(x_4)
        # x_5 = x_5 + x_4
        # x_5 = self.ln_5_1(x_5)
        # c_5 = x_5.shape[1]
        # x_t5 = x_5
        # x_5 = x_5.reshape(-1, self.embedding_len)
        # x_5= self.mlp_5(x_5)
        # x_5 = x_5.view(-1, c_5, self.embedding_len)
        # x_5 = x_5 + x_t5
        # x_5 = self.ln_5_2(x_5)
        
        c_fin = x_2.shape[1]
        x_2 = x_2.reshape(-1, self.embedding_len)
        final_emb = self.fin_emb_transform(x_2)
        x_2 = x_2.view(-1, c_fin, self.embedding_len)
        
        return x_2, final_emb
        

class te(nn.Module):
    """
    1) Add substitution for BatchNorm
    2) Change no. of self-attention layers
    """
    def __init__(self, embedding_len : int = 7, final_embedding : int = 10, n_heads : int = 1, n_layers : int = 6, dropout : float = 0.1):
        super(te, self).__init__()
        
        self.embedding_len = embedding_len
        
        self.sa_1 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_1_1 = nn.LayerNorm(embedding_len)
        self.mlp_1 = nn.Sequential(
            NoisyLinear(embedding_len, embedding_len//2),
            nn.ReLU(),
            NoisyLinear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_1_2 = nn.LayerNorm(embedding_len)
        
        self.sa_2 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_2_1 = nn.LayerNorm(embedding_len)
        self.mlp_2 = nn.Sequential(
            NoisyLinear(embedding_len, embedding_len//2),
            nn.ReLU(),
            NoisyLinear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_2_2 = nn.LayerNorm(embedding_len)
        
        self.sa_3 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_3_1 = nn.LayerNorm(embedding_len)
        self.mlp_3 = nn.Sequential(
            NoisyLinear(embedding_len, embedding_len//2),
            nn.ReLU(),
            NoisyLinear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_3_2 = nn.LayerNorm(embedding_len)
        
        # self.sa_4 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_4_1 = nn.LayerNorm(embedding_len)
        # self.mlp_4 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_4_2 = nn.LayerNorm(embedding_len)
        
        # self.sa_5 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_5_1 = nn.LayerNorm(embedding_len)
        # self.mlp_5 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_5_2 = nn.LayerNorm(embedding_len)
        
        self.fin_emb_transform = NoisyLinear(embedding_len, final_embedding)
    
    def forward(self, x : tensor) -> tensor:
        """
        x : (batch_sz, context_len, embedding_dim)
        """
        
        x_1 = self.sa_1(x)
        x_1 = x_1 + x
        x_1 = self.ln_1_1(x_1)
        c_1 = x_1.shape[1]
        x_t1 = x_1
        x_1 = x_1.reshape(-1, self.embedding_len)
        x_1 = self.mlp_1(x_1)
        x_1 = x_1.view(-1, c_1, self.embedding_len)
        x_1 = x_1 + x_t1
        x_1 = self.ln_1_2(x_1)
        
        x_2 = self.sa_2(x_1)
        x_2 = x_2 + x_1
        x_2 = self.ln_2_1(x_2)
        c_2 = x_2.shape[1]
        x_t2 = x_2
        x_2 = x_2.reshape(-1, self.embedding_len)
        x_2 = self.mlp_2(x_2)
        x_2 = x_2.view(-1, c_2, self.embedding_len)
        x_2 = x_2 + x_t2
        x_2 = self.ln_2_2(x_2)
        
        x_3 = self.sa_3(x_2)
        x_3 = x_3 + x_2
        x_3 = self.ln_3_1(x_3)
        c_3 = x_3.shape[1]
        x_t3 = x_3
        x_3 = x_3.reshape(-1, self.embedding_len)
        x_3 = self.mlp_3(x_3)
        x_3 = x_3.view(-1, c_3, self.embedding_len) 
        x_3 = x_3 + x_t3
        x_3 = self.ln_3_2(x_3)
        
        # x_4 = self.sa_4(x_3)
        # x_4 = x_4 + x_3
        # x_4 = self.ln_4_1(x_4)
        # c_4 = x_4.shape[1]
        # x_t4 = x_4
        # x_4 = x_4.reshape(-1, self.embedding_len)
        # x_4 = self.mlp_4(x_4)
        # x_4 = x_4.view(-1, c_4, self.embedding_len)
        # x_4 = x_4 + x_t4
        # x_4 = self.ln_4_2(x_4)
        
        # x_5 = self.sa_5(x_4)
        # x_5 = x_5 + x_4
        # x_5 = self.ln_5_1(x_5)
        # c_5 = x_5.shape[1]
        # x_t5 = x_5
        # x_5 = x_5.reshape(-1, self.embedding_len)
        # x_5= self.mlp_5(x_5)
        # x_5 = x_5.view(-1, c_5, self.embedding_len)
        # x_5 = x_5 + x_t5
        # x_5 = self.ln_5_2(x_5)
        
        c_fin = x_3.shape[1]
        x_3 = x_3.reshape(-1, self.embedding_len)
        final_emb = self.fin_emb_transform(x_3)
        x_3 = x_3.view(-1, c_fin, self.embedding_len)
        
        return x_3, final_emb
    
    def reset_noise(self):
        mlps = [self.mlp_1, self.mlp_2, self.mlp_3]
        
        for mlp in mlps:
            mlp[0].reset_noise()
            mlp[2].reset_noise()        

lin_1 = nn.Linear(576, 7).to("cuda")        
 
class T1ObjectTransformerDecoder(nn.Module):
    def __init__(self, embedding_len : int = 7, pembedding_len : int = 6, final_embedding : int = 10, n_heads : int = 1):
        super(T1ObjectTransformerDecoder, self).__init__()
        
        self.embedding_len = embedding_len
        self.pembedding_len = pembedding_len
        
        self.sa_1 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_1_1 = nn.LayerNorm(embedding_len)
        self.ca_1 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        self.ln_1_2 = nn.LayerNorm(embedding_len)
        self.mlp_1 = nn.Sequential(
            nn.Linear(embedding_len, embedding_len//2),
            nn.ReLU(),
            nn.Linear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_1_3 = nn.LayerNorm(embedding_len)
        
        self.sa_2 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_2_1 = nn.LayerNorm(embedding_len)
        self.ca_2 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        self.ln_2_2 = nn.LayerNorm(embedding_len)
        self.mlp_2 = nn.Sequential(
            nn.Linear(embedding_len, embedding_len//2),
            nn.ReLU(),
            nn.Linear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_2_3 = nn.LayerNorm(embedding_len)
        
        # self.sa_3 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_3_1 = nn.LayerNorm(embedding_len)
        # self.ca_3= MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        # self.ln_3_2 = nn.LayerNorm(embedding_len)
        # self.mlp_3 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_3_3 = nn.LayerNorm(embedding_len)
        
        # self.sa_4 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_4_1 = nn.LayerNorm(embedding_len)
        # self.ca_4 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        # self.ln_4_2 = nn.LayerNorm(embedding_len)
        # self.mlp_4 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_4_3 = nn.LayerNorm(embedding_len)
        
        # self.sa_5 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_5_1 = nn.LayerNorm(embedding_len)
        # self.ca_5 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        # self.ln_5_2 = nn.LayerNorm(embedding_len)
        # self.mlp_5 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_5_3 = nn.LayerNorm(embedding_len)
        
        self.fin_emb_transform = nn.Linear(embedding_len, final_embedding)
    
    def forward(self, x, platform_em):
        """
        x : (B, context_len, embedding_len)
        platform_em : (B, pcontext_len, pembedding_len)
        """
        
        x_1 = self.sa_1(x)
        x_1 = x_1 + x
        x_1 = self.ln_1_1(x_1)
        x_t11 = x_1
        x_1 = self.ca_1(x_1, platform_em)
        x_1 = x_1 + x_t11
        x_1 = self.ln_1_2(x_1)
        x_t12 = x_1
        c_1 = x_1.shape[1]
        x_1 = x_1.reshape(-1, self.embedding_len)
        x_1 = self.mlp_1(x_1)
        x_1 = x_1.view(-1, c_1, self.embedding_len)
        x_1 = x_1 + x_t12
        x_1 = self.ln_1_3(x_1)
        
        x_2 = self.sa_2(x_1)
        x_2 = x_2 + x_1
        x_2 = self.ln_2_1(x_2)
        x_t21 = x_2
        x_2 = self.ca_2(x_2, platform_em)
        x_2 = x_2 + x_t21
        x_2 = self.ln_2_2(x_2)
        x_t22 = x_2
        c_2 = x_2.shape[1]
        x_2 = x_2.reshape(-1, self.embedding_len)
        x_2 = self.mlp_2(x_2)
        x_2 = x_2.view(-1, c_2, self.embedding_len)
        x_2 = x_2 + x_t22
        x_2 = self.ln_2_3(x_2)
        
        # x_3 = self.sa_3(x_2)
        # x_3 = x_3 + x_2
        # x_3 = self.ln_3_1(x_3)
        # x_t31 = x_3
        # x_3 = self.ca_3(x_3, platform_em)
        # x_3 = x_3 + x_t31
        # x_3 = self.ln_3_2(x_3)
        # x_t32 = x_3
        # c_3 = x_3.shape[1]
        # x_3 = x_3.reshape(-1, self.embedding_len)
        # x_3 = self.mlp_3(x_3)
        # x_3 = x_3.view(-1, c_3, self.embedding_len)
        # x_3 = x_3 + x_t32
        # x_3 = self.ln_3_3(x_3)
        
        # x_4 = self.sa_4(x_3)
        # x_4 = x_4 + x_3
        # x_4 = self.ln_4_1(x_4)
        # x_t41 = x_4
        # x_4 = self.ca_4(x_4, platform_em)
        # x_4 = x_4 + x_t41
        # x_4 = self.ln_4_2(x_4)
        # x_t42 = x_4
        # c_4= x_4.shape[1]
        # x_4 = x_4.reshape(-1, self.embedding_len)
        # x_4 = self.mlp_4(x_4)
        # x_4 = x_4.view(-1, c_4, self.embedding_len)
        # x_4 = x_4 + x_t42
        # x_4 = self.ln_4_3(x_4)
        
        # x_5 = self.sa_5(x_4)
        # x_5 = x_5 + x_4
        # x_5 = self.ln_5_1(x_5)
        # x_t51 = x_5
        # x_5 = self.ca_5(x_5, platform_em)
        # x_5 = x_5 + x_t51
        # x_5 = self.ln_5_2(x_5)
        # x_t52 = x_5
        # c_5 = x_5.shape[1]
        # x_5 = x_5.reshape(-1, self.embedding_len)
        # x_5 = self.mlp_5(x_5)
        # x_5 = x_5.view(-1, c_5, self.embedding_len)
        # x_5 = x_5 + x_t52
        # x_5 = self.ln_5_3(x_5)
        
        c_fin = x_2.shape[1]
        x_2 = x_2.reshape(-1, self.embedding_len)
        final_emb = self.fin_emb_transform(x_2)
        x_2 = x_2.view(-1, c_fin, self.embedding_len)
        
        return x_2, final_emb
    
class T1ObjectTransformerDecoderNoisy(nn.Module):
    def __init__(self, embedding_len : int = 7, pembedding_len : int = 6, final_embedding : int = 10, n_heads : int = 1):
        super(T1ObjectTransformerDecoderNoisy, self).__init__()
        
        self.embedding_len = embedding_len
        self.pembedding_len = pembedding_len
        
        self.sa_1 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_1_1 = nn.LayerNorm(embedding_len)
        self.ca_1 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        self.ln_1_2 = nn.LayerNorm(embedding_len)
        self.mlp_1 = nn.Sequential(
            NoisyLinear(embedding_len, embedding_len//2),
            nn.ReLU(),
            NoisyLinear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_1_3 = nn.LayerNorm(embedding_len)
        
        self.sa_2 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_2_1 = nn.LayerNorm(embedding_len)
        self.ca_2 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        self.ln_2_2 = nn.LayerNorm(embedding_len)
        self.mlp_2 = nn.Sequential(
            NoisyLinear(embedding_len, embedding_len//2),
            nn.ReLU(),
            NoisyLinear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_2_3 = nn.LayerNorm(embedding_len)
        
        self.sa_3 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        self.ln_3_1 = nn.LayerNorm(embedding_len)
        self.ca_3= MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        self.ln_3_2 = nn.LayerNorm(embedding_len)
        self.mlp_3 = nn.Sequential(
            NoisyLinear(embedding_len, embedding_len//2),
            nn.ReLU(),
            NoisyLinear(embedding_len//2, embedding_len),
            nn.ReLU()
        )
        self.ln_3_3 = nn.LayerNorm(embedding_len)
        
        # self.sa_4 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_4_1 = nn.LayerNorm(embedding_len)
        # self.ca_4 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        # self.ln_4_2 = nn.LayerNorm(embedding_len)
        # self.mlp_4 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_4_3 = nn.LayerNorm(embedding_len)
        
        # self.sa_5 = MultiHeadSelfAttention(embedding_len, embedding_len, embedding_len, n_heads)
        # self.ln_5_1 = nn.LayerNorm(embedding_len)
        # self.ca_5 = MultiHeadCrossAttention(embedding_len, pembedding_len, pembedding_len, n_heads)
        # self.ln_5_2 = nn.LayerNorm(embedding_len)
        # self.mlp_5 = nn.Sequential(
        #     nn.Linear(embedding_len, embedding_len//2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_len//2, embedding_len),
        #     nn.ReLU()
        # )
        # self.ln_5_3 = nn.LayerNorm(embedding_len)
        
        self.fin_emb_transform = NoisyLinear(embedding_len, final_embedding)
    
    def forward(self, x, platform_em):
        """
        x : (B, context_len, embedding_len)
        platform_em : (B, pcontext_len, pembedding_len)
        """
        
        x_1 = self.sa_1(x)
        x_1 = x_1 + x
        x_1 = self.ln_1_1(x_1)
        x_t11 = x_1
        x_1 = self.ca_1(x_1, platform_em)
        x_1 = x_1 + x_t11
        x_1 = self.ln_1_2(x_1)
        x_t12 = x_1
        c_1 = x_1.shape[1]
        x_1 = x_1.reshape(-1, self.embedding_len)
        x_1 = self.mlp_1(x_1)
        x_1 = x_1.view(-1, c_1, self.embedding_len)
        x_1 = x_1 + x_t12
        x_1 = self.ln_1_3(x_1)
        
        x_2 = self.sa_2(x_1)
        x_2 = x_2 + x_1
        x_2 = self.ln_2_1(x_2)
        x_t21 = x_2
        x_2 = self.ca_2(x_2, platform_em)
        x_2 = x_2 + x_t21
        x_2 = self.ln_2_2(x_2)
        x_t22 = x_2
        c_2 = x_2.shape[1]
        x_2 = x_2.reshape(-1, self.embedding_len)
        x_2 = self.mlp_2(x_2)
        x_2 = x_2.view(-1, c_2, self.embedding_len)
        x_2 = x_2 + x_t22
        x_2 = self.ln_2_3(x_2)
        
        x_3 = self.sa_3(x_2)
        x_3 = x_3 + x_2
        x_3 = self.ln_3_1(x_3)
        x_t31 = x_3
        x_3 = self.ca_3(x_3, platform_em)
        x_3 = x_3 + x_t31
        x_3 = self.ln_3_2(x_3)
        x_t32 = x_3
        c_3 = x_3.shape[1]
        x_3 = x_3.reshape(-1, self.embedding_len)
        x_3 = self.mlp_3(x_3)
        x_3 = x_3.view(-1, c_3, self.embedding_len)
        x_3 = x_3 + x_t32
        x_3 = self.ln_3_3(x_3)
        
        # x_4 = self.sa_4(x_3)
        # x_4 = x_4 + x_3
        # x_4 = self.ln_4_1(x_4)
        # x_t41 = x_4
        # x_4 = self.ca_4(x_4, platform_em)
        # x_4 = x_4 + x_t41
        # x_4 = self.ln_4_2(x_4)
        # x_t42 = x_4
        # c_4= x_4.shape[1]
        # x_4 = x_4.reshape(-1, self.embedding_len)
        # x_4 = self.mlp_4(x_4)
        # x_4 = x_4.view(-1, c_4, self.embedding_len)
        # x_4 = x_4 + x_t42
        # x_4 = self.ln_4_3(x_4)
        
        # x_5 = self.sa_5(x_4)
        # x_5 = x_5 + x_4
        # x_5 = self.ln_5_1(x_5)
        # x_t51 = x_5
        # x_5 = self.ca_5(x_5, platform_em)
        # x_5 = x_5 + x_t51
        # x_5 = self.ln_5_2(x_5)
        # x_t52 = x_5
        # c_5 = x_5.shape[1]
        # x_5 = x_5.reshape(-1, self.embedding_len)
        # x_5 = self.mlp_5(x_5)
        # x_5 = x_5.view(-1, c_5, self.embedding_len)
        # x_5 = x_5 + x_t52
        # x_5 = self.ln_5_3(x_5)
        
        c_fin = x_3.shape[1]
        x_3 = x_3.reshape(-1, self.embedding_len)
        final_emb = self.fin_emb_transform(x_3)
        x_3 = x_3.view(-1, c_fin, self.embedding_len)
        
        return x_3, final_emb
    
    def reset_noise(self):
        mlps = [self.mlp_1, self.mlp_2, self.mlp_3]
        
        for mlp in mlps:
            mlp[0].reset_noise()
            mlp[2].reset_noise()
            

class OPT1(nn.Module):
    def __init__(self, pembedding_len : int = 7, pn_heads : int = 1, pn_layers : int = 6, pdropout : float = 0.1, \
        oembedding_len : int = 7, max_platforms : int = 8, on_heads : int = 1, final_embedding : int = 3, \
            temperature = 0.1):
        
        super(OPT1, self).__init__()
        
        self.encoder = T1PlatformTransformerEncoder(pembedding_len, final_embedding, pn_heads, pn_layers, pdropout)
        
        self.decoder = T1ObjectTransformerDecoder(oembedding_len, pembedding_len, final_embedding, on_heads)
        
        self.final_embedding = final_embedding
        
        self.temperature = temperature
    def forward(self, state):
        p_cross_embedding, p_final_embedding = self.encoder(state["state_platform"])
        
        o_cross_embedding, o_final_embedding = self.decoder(state["state_object"], p_cross_embedding)
                
        dot_similarity_mat = torch.matmul(p_final_embedding, o_final_embedding.t()).t()
        
        prob_mat = nn.Softmax()(dot_similarity_mat / (torch.sqrt(tensor(self.final_embedding)) * self.temperature))
        
        return prob_mat
        

class OPT1Noisy(nn.Module):
    def __init__(self, pembedding_len : int = 7, pn_heads : int = 1, pn_layers : int = 6, pdropout : float = 0.1, \
        oembedding_len : int = 7, max_platforms : int = 8, on_heads : int = 1, final_embedding : int = 3, \
            temperature = 0.1):
        
        super(OPT1Noisy, self).__init__()
        
        self.encoder = te(pembedding_len, final_embedding, pn_heads, pn_layers, pdropout)
        
        self.decoder = T1ObjectTransformerDecoderNoisy(oembedding_len, pembedding_len, final_embedding, on_heads)
        
        self.final_embedding = final_embedding
        
        self.temperature = temperature
        
    def forward(self, state, smol_state=None):
        platform_c, platform_e = self.encoder(state["state_platform"])
        
        query = state["state_object"]
        
        key = lin_1(smol_state)
        value = lin_1(smol_state)
        
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        
        attention_scores = attention_scores / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
    
        attention_output = torch.bmm(attention_weights, value)
        
        object_final_emb = alpha * state["state_object"] + (1 - alpha) * attention_output
        
        o_cross_embedding, o_final_embedding = self.decoder(object_final_emb, platform_c)
                
        dot_similarity_mat = torch.matmul(platform_e, o_final_embedding.t()).t()
        
        prob_mat = nn.Softmax()(dot_similarity_mat / (torch.sqrt(tensor(self.final_embedding)) * self.temperature))
        
        return prob_mat
    
    def reset_noise(self):
        """resets noisy layers in the model"""
        self.encoder.reset_noise()
        self.decoder.reset_noise()      
        
# class PlatformObjectTransformer(nn.Module):
#     def __init__(self, pembedding_len=6, oembedding_len=7, nhead = 1, num_encoders=1, num_decoders=1, forward_expansion=4, dropout=0.1, no_of_platforms=8, no_of_objects=16):
#         super(PlatformObjectTransformer, self).__init__()
#         self.pembedding_len = pembedding_len
#         self.oembedding_len = oembedding_len
        
        
#         # Encoder and Decoder Layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=pembedding_len,
#             nhead=nhead,
#             dim_feedforward=pembedding_len * forward_expansion,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=oembedding_len,
#             nhead=nhead,
#             dim_feedforward=oembedding_len * forward_expansion,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoders)

    # def forward(self, state_platform, state_object):
    #     # Adding positional embeddings
    #     platform_state = state_platform
    #     object_state = state_object
        
    #     # Encoding
    #     encoded_platform = self.encoder(platform_state)
        
    #     # Decoding (considering encoded platform state as memory)
    #     decoded_objects = self.decoder(object_state, encoded_platform)
        
    #     return decoded_objects
        
class PPOLoss(nn.Module):
    def __init__(self, clip_param=0.2, ent_coef=0.01):
        super(PPOLoss, self).__init__()
        
        self.clip_param = clip_param
        self.ent_coef = ent_coef
        
    def forward(self, reward, log_probs_old, log_probs_x, entropy_x):           
        advantage = reward

        log_probs_new = log_probs_x
        
        ratio = torch.exp(log_probs_new - log_probs_old)
        
        surrogate_1 = ratio * advantage
        
        surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
        
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
        
        entropy_bonus = -self.ent_coef * entropy_x.mean()
        
        loss = policy_loss + entropy_bonus
        
        return loss
    