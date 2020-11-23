import torch
import torch.nn as nn
import torch.functional as F

        
class SelfAttention(nn.Module):
    def __init__(self, model_dim, n_heads, mask=False):
        self.dim = model_dim
        self.n = n_heads
        self.mask = mask

        self.d_k = self.dim * self.n # 모두 크기가 같으므로 일괄적으로 적용

        self.W_Q = nn.Linear(self.dim, self.d_k)
        self.W_K = nn.Linear(self.dim, self.d_k)
        self.W_V = nn.Linear(self.dim, self.d_k)

        self.linear = nn.Linear(self.n*self.d_k, self.dim)


    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        X = []
        for i in range(self.n):
            X.append(self.scaled_dot_product(x, Q, K, V))

        X = torch.cat(X, dim=2) #왜 2일까..?
        X = self.linear(X)
        return X


    def scaled_dot_product(self, x, Q, K, V):
        ## bmm은 batch 단위 matmul이고, broadcasting이 지원되지 않는다
        ## 사실 matmul과 정확히 어떤 차이인지 잘 모르겠다
        tmp = torch.matmul(Q, K)
        tmp = torch.div(tmp, torch.sqrt(self.d_k))

        if self.mask:
            pass

        tmp = F.softmax(tmp)
        tmp = torch.matmul(tmp, V)

        return tmp



class Encoder(nn.Module):
    def __init__(self, model_dim, n_heads, mask=False):
        self.dim = model_dim
        self.n = n_heads
        
        self.MHA = SelfAttention(self.dim, n_heads, mask)
        self.norm = nn.LayerNorm(self.dim)
        self.FFN = nn.Sequential(
            nn.Linear(self.dim, n_heads*self.dim),
            nn.ReLU(),
            nn.Linaer(n_heads*self.dim, self.dim)
        )

    def forward(x):
        orig_x = x.copy()
        x = self.MHA(x)
        x = self.norm(x + orig_x)
        ## x = self.norm (x + self.dropout(self.MHA(Q,K,V))) 

        orig_x = x.copy()
        x = self.FFN(x)
        x = self.norm(x + orig_x)
        ##  x = self.norm (x + self.dropout(self.FFN(x)))

        return x
