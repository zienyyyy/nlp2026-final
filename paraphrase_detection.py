'''
NLP 2026-1 Final Project: Paraphrase Detection 개선 코드.

본 파일은 수업에서 제공된 paraphrase_detection.py starter code를 기반으로
Paraphrase Detection 성능 개선을 위해 확장한 최종 실험 코드이다.

주요 개선 사항:
 - GPT-2 backbone을 freeze하고 attention query/key/value projection에 LoRA 적용
 - Pair Swap Augmentation을 통해 문장쌍 순서에 대한 대칭성 학습
 - Hard Negative Oversampling을 통해 lexical overlap이 높은 non-paraphrase 예시 보강
 - AdamW weight decay 적용
 - 최종 dev/test inference에서 Bidirectional Inference 적용

실행:
  python -u paraphrase_detection.py \
    --use_gpu \
    --epochs 8 \
    --model_size gpt2-medium \
    --lr 5e-5 \
    --batch_size 8 \
    --seed 11711 \
    --hard_neg_ratio 0.10 \
    --weight_decay 0.01

주의:
 - 모델 학습과 선택에는 train/dev set만 사용한다.
 - test set에는 gold label이 공개되어 있지 않으므로 test accuracy는 직접 계산하지 않는다.
 - test set은 최종 prediction file 생성에만 사용한다.
'''

import argparse
import random
import math
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW
from transformers import GPT2Tokenizer

TQDM_DISABLE = False
NO_TOKEN_ID = 3919
YES_TOKEN_ID = 8505

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

PUNCT_TOKENS = {'.', '?', ',', "'", '"'}

def get_content_tokens(sentence):
  """Hard negative mining을 위해 문장에서 구두점을 제외한 내용 토큰만 추출한다."""

  return {
    tok for tok in sentence.split()
    if tok not in PUNCT_TOKENS
  }

def jaccard_overlap(s1, s2):
  """두 질문의 lexical overlap을 Jaccard similarity로 계산한다."""
  
  tokens1 = get_content_tokens(s1)
  tokens2 = get_content_tokens(s2)

  if len(tokens1) == 0 and len(tokens2) == 0:
    return 0.0

  return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def select_hard_negatives(data, ratio=0.10):
  """Non-paraphrase 중 단어 겹침이 큰 상위 ratio 비율의 예시를 hard negative로 선택한다."""
   
  hard_negative_scores = []

  for example in data:
    s1, s2, label, sent_id = example
    if label == 0:
      score = jaccard_overlap(s1, s2)
      hard_negative_scores.append((score, example))

  hard_negative_scores.sort(key=lambda x: x[0], reverse=True)

  k = int(len(hard_negative_scores) * ratio)
  return [example for _, example in hard_negative_scores[:k]]

def augment_with_pair_swap_and_hard_negatives(data, hard_neg_ratio=0.10):    
  """
  Train set에만 적용하는 데이터 증강 함수.
  Pair swap은 paraphrase relation의 대칭성을 학습시키기 위한 것이고,
  hard negative oversampling은 단어가 많이 겹치지만 의미가 다른 non-paraphrase 예시를 보강하기 위한 것이다.
  """
  
  swapped = [
    (s2, s1, label, f"{sent_id}_swap")
    for (s1, s2, label, sent_id) in data
  ]

  hard_negatives = select_hard_negatives(data, ratio=hard_neg_ratio)

  hard_negative_augmented = []
  for s1, s2, label, sent_id in hard_negatives:
    hard_negative_augmented.append((s1, s2, label, f"{sent_id}_hardneg"))
    hard_negative_augmented.append((s2, s1, label, f"{sent_id}_hardneg_swap"))

  augmented_data = data + swapped + hard_negative_augmented
  return augmented_data, len(swapped), len(hard_negative_augmented)


class LoRALinear(nn.Module):
  """기존 Linear layer는 유지하고 low-rank adapter만 추가 학습하는 LoRA layer."""
  
  def __init__(self, base_layer, r=32, alpha=64):
    super().__init__()
    self.base_layer = base_layer
    self.r = r
    self.alpha = alpha
    self.scaling = alpha / r

    in_features = base_layer.in_features
    out_features = base_layer.out_features

    self.lora_A = nn.Linear(in_features, r, bias=False)
    self.lora_B = nn.Linear(r, out_features, bias=False)

    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)

  def forward(self, x):
    return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling

class ParaphraseGPT(nn.Module):
  """Paraphrase Detection을 위해 설계된 LoRA 기반 GPT-2 Model."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(
      model=args.model_size,
      d=args.d,
      l=args.l,
      num_heads=args.num_heads
    )
    self.paraphrase_detection_head = nn.Linear(args.d, 2)

    # 1. GPT-2 backbone freeze
    for param in self.gpt.parameters():
      param.requires_grad = False

    # 2. Attention query, key, value projection에 LoRA 적용
    for layer in self.gpt.gpt_layers:
      layer.self_attention.query = LoRALinear(
        layer.self_attention.query,
        r=32,
        alpha=64
      )
      layer.self_attention.key = LoRALinear(
        layer.self_attention.key,
        r=32,
        alpha=64
      )
      layer.self_attention.value = LoRALinear(
        layer.self_attention.value,
        r=32,
        alpha=64
      )

    # 3. classification head는 학습
    for param in self.paraphrase_detection_head.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Cloze-style prompt의 마지막 토큰 representation을 사용하여 yes/no를 예측한다.

    평가 함수가 vocabulary 전체에 대한 logits를 기대하므로, binary head의 두 출력을
    GPT-2 vocabulary의 no token과 yes token 위치에만 배치한다.
    """
    
    gpt_output = self.gpt(input_ids, attention_mask)
    last_token = gpt_output['last_token']

    binary_logits = self.paraphrase_detection_head(last_token)

    vocab_size = self.gpt.word_embedding.num_embeddings
    logits = torch.full(
      (binary_logits.size(0), vocab_size),
      -1e9,
      device=binary_logits.device,
      dtype=binary_logits.dtype
    )

    no_token_id = 3919
    yes_token_id = 8505

    logits[:, no_token_id] = binary_logits[:, 0]
    logits[:, yes_token_id] = binary_logits[:, 1]

    return logits



def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.

  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)
  
  # 데이터 증강은 train set에만 적용하고, dev/test set은 원본 분포를 유지한다.
  original_train_size = len(para_train_data)
  para_train_data, num_swapped, num_hardneg = augment_with_pair_swap_and_hard_negatives(
    para_train_data,
    hard_neg_ratio=args.hard_neg_ratio
  )

  print(
    f"Pair swap + hard negative augmentation: "
    f"{original_train_size} -> {len(para_train_data)} train examples "
    f"(swapped: {num_swapped}, hard negative augmented: {num_hardneg})"
  )
  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  print(f"Total params: {total_params:,}")
  print(f"Trainable params: {trainable_params:,}")
  print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")
  lr = args.lr
  optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=lr,
    weight_decay=args.weight_decay
    )
  best_dev_acc = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트. 
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

def build_prompt(s1, s2):
  
  """Bidirectional inference에서 forward/backward 방향에 공통으로 사용하는 prompt."""
  
  return f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '


@torch.no_grad()
def model_eval_paraphrase_bidirectional(raw_data, model, device, batch_size=8):
  """Dev set에 대해 원래 방향과 swap 방향의 yes/no logits를 평균하여 평가한다."""
  model.eval()
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token

  y_true, y_pred, sent_ids = [], [], []

  for start in tqdm(range(0, len(raw_data), batch_size), desc='bidir-eval', disable=TQDM_DISABLE):
    batch = raw_data[start:start + batch_size]

    sent1 = [x[0] for x in batch]
    sent2 = [x[1] for x in batch]
    labels = [YES_TOKEN_ID if x[2] == 1 else NO_TOKEN_ID for x in batch]
    ids = [x[3] for x in batch]

    prompts_f = [build_prompt(s1, s2) for s1, s2 in zip(sent1, sent2)]
    prompts_b = [build_prompt(s2, s1) for s1, s2 in zip(sent1, sent2)]

    enc_f = tokenizer(prompts_f, return_tensors='pt', padding=True, truncation=True)
    enc_b = tokenizer(prompts_b, return_tensors='pt', padding=True, truncation=True)

    logits_f = model(enc_f['input_ids'].to(device), enc_f['attention_mask'].to(device))
    logits_b = model(enc_b['input_ids'].to(device), enc_b['attention_mask'].to(device))

    # Forward/backward 두 방향 prompt의 logits를 평균하여 문장 순서에 덜 민감한 예측을 만든다.
    no_logits = (logits_f[:, NO_TOKEN_ID] + logits_b[:, NO_TOKEN_ID]) / 2
    yes_logits = (logits_f[:, YES_TOKEN_ID] + logits_b[:, YES_TOKEN_ID]) / 2

    preds = torch.where(
      yes_logits > no_logits,
      torch.full_like(yes_logits, YES_TOKEN_ID, dtype=torch.long),
      torch.full_like(no_logits, NO_TOKEN_ID, dtype=torch.long)
    )

    y_true.extend(labels)
    y_pred.extend(preds.cpu().tolist())
    sent_ids.extend(ids)

  acc = sum(int(p == y) for p, y in zip(y_pred, y_true)) / len(y_true)
  return acc, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase_bidirectional(raw_data, model, device, batch_size=8):
  """Gold label이 없는 test set에 대해 bidirectional inference로 prediction을 생성한다."""
  model.eval()
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token

  y_pred, sent_ids = [], []

  for start in tqdm(range(0, len(raw_data), batch_size), desc='bidir-test', disable=TQDM_DISABLE):
    batch = raw_data[start:start + batch_size]

    sent1 = [x[0] for x in batch]
    sent2 = [x[1] for x in batch]
    ids = [x[2] for x in batch]

    prompts_f = [build_prompt(s1, s2) for s1, s2 in zip(sent1, sent2)]
    prompts_b = [build_prompt(s2, s1) for s1, s2 in zip(sent1, sent2)]

    enc_f = tokenizer(prompts_f, return_tensors='pt', padding=True, truncation=True)
    enc_b = tokenizer(prompts_b, return_tensors='pt', padding=True, truncation=True)

    logits_f = model(enc_f['input_ids'].to(device), enc_f['attention_mask'].to(device))
    logits_b = model(enc_b['input_ids'].to(device), enc_b['attention_mask'].to(device))

    # Forward/backward 두 방향 prompt의 logits를 평균하여 문장 순서에 덜 민감한 예측을 만든다.
    no_logits = (logits_f[:, NO_TOKEN_ID] + logits_b[:, NO_TOKEN_ID]) / 2
    yes_logits = (logits_f[:, YES_TOKEN_ID] + logits_b[:, YES_TOKEN_ID]) / 2

    preds = torch.where(
      yes_logits > no_logits,
      torch.full_like(yes_logits, YES_TOKEN_ID, dtype=torch.long),
      torch.full_like(no_logits, NO_TOKEN_ID, dtype=torch.long)
    )

    y_pred.extend(preds.cpu().tolist())
    sent_ids.extend(ids)

  return y_pred, sent_ids


@torch.no_grad()
def test(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model for bidirectional inference from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  dev_para_acc, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase_bidirectional(
    para_dev_data,
    model,
    device,
    batch_size=args.batch_size
  )
  print(f"bidirectional dev paraphrase acc :: {dev_para_acc :.3f}")

  test_para_y_pred, test_para_sent_ids = model_test_paraphrase_bidirectional(
    para_test_data,
    model,
    device,
    batch_size=args.batch_size
  )

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--hard_neg_ratio", type=float, default=0.10)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--weight_decay", type=float, default=0.01)
  args = parser.parse_args()
  return args


def add_arguments(args):
  """모델 크기에 따라 결정되는 인수들을 추가."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-wd{args.weight_decay}-lora-qkv-r32-swap-hardneg-paraphrase.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  test(args)