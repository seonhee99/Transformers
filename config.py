class Config():
    self.data_folder = ",".join(f'train.{l}' for l in ['fr', 'en'])
    self.vocab_size = 3000 # English-German: 37000, English-French: 32000
    self.model_type = 'unigram' #unigram으로 일단 하자 논문엔 bpe인듯
user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK],[UNK1],[UNK2]'