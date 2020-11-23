"""
(1) 데이터 로드
(2) DataSet, Dataloader 정의
"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def load_data(source_file, target_file, encoding='utf-8'):
    # 데이터 불러오기
    with open(source_file, 'r', encoding=encoding) as source:
        source_text = source.readlines()

    with open(target_file, 'r', encoding=encoding) as target:
        target_text = target.readlines()

    print(f"{len(source_text)} sentences are downloaded")

    return source_text, target_text

# 사용자정의 Dataset 클래스 구축
class transDataset(TensorDataset):
    def __init__(self, source, target, tokenizer):
    ####################################################################
    ## source랑 target을 받아서 sequence of token indices 를 반환해야함##
    ### 받은 데이터는 단순히 텍스트가 리스트 안에 쌓여 있는 형태         ##
    ####################################################################

        assert len(source) == len(target)
        self.length = len(source)

        # 일단 tokenize
        self.src = tokenizer.tokenize(source)
        self.trg = tokenizer.tokenize(target)

        # return (self.src, self.trg)

  def __len__(self):
    # 데이터셋 사이즈
    return self.length
  
  def __getitem__(self, idx):
    # 인덱싱 기능
    return (self.src[idx], self.trg[idx])



# embedding layer 정의
# torch.nn 레이어로 정의됨

class Embedding():
  def __init__(self, size, dim):
    self.vocab_size = size
    self.dim = dim
    self.embedding = np.random.uniform(-0.25, 0.25, (size, dim))
    self.embedding_nn = nn.Embedding(num_embeddings = size, embedding_dim = dim) #padding idx

  def encode(self, seq):
    # seq of tokens
    x = [np.eye(self.vocab_size)[i] for i in seq]
    return np.dot(x, self.embedding) + self.pos(seq)

  def pos(self, seq):
    # positional encoding
    d = self.dim
    return np.array([ [np.sin( t / np.power(10000, ( (i+1)// 2 / d ) ) ) for i in range(d)] for t in range(len(seq))])
