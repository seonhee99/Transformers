import re
import config
import sentencepiece as spm

## 모듈화.. ##
# with open('./fr-en/train.tags.fr-en.en', 'r', encoding='utf-8') as f:
#     get = re.findall('<description>.*</description>', f.read())
#     train_en = [i[48:-14] for i in get]

# with open('train.en', 'w', encoding='utf-8') as f:
#     f.write("\n".join(train_en))

# with open('./fr-en/train.tags.fr-en.fr', 'r', encoding='utf-8') as f:
#     get = re.findall('<description>.*</description>', f.read())
#     train_fr = [i[48:-14] for i in get]

# with open('train.fr', 'w', encoding='utf-8') as f:
#     f.write("\n".join(train_fr))


## tokenizing ##
cfg = config()
model_name = 'm'

spm.SentencePieceTrainer.Train(
    f'--input={cfg.data_folder} --model_prefix={model_name} --vocab_size={cfg.vocab_size} --user_defined_symbols={cfg.user_defined_symbols} --model_type={cfg.model_type}'
    )

sp = spm.SentencePieceProcessor()
sp.Load("m.model")

## 이걸로 전체 데이터를 토크나이징, 별개 파일로 저장

class TransLoader():
    pass
    """
    ** 인코더, 디코더 관점에서 고민해보기
    ** 
    """