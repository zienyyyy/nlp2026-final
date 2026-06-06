'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    시퀀스의 각 토큰에 대한 logit을 생성한다.
    언어 모델링 학습을 위해 모든 토큰 위치의 logit을 반환한다.
    """
    gpt_output = self.gpt(input_ids, attention_mask)
    hidden_states = gpt_output['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature

      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
      top_p_mask[..., 0] = True
      filtered_probs = sorted_probs * top_p_mask
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


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
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련. Early stopping 적용."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  optimizer = AdamW(model.parameters(), lr=args.lr)

  # Early stopping 설정
  best_loss = float('inf')
  patience_counter = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")

    # Early stopping 체크
    if train_loss < best_loss - args.min_delta:
      best_loss = train_loss
      patience_counter = 0
      save_model(model, optimizer, args, f'best_{args.filepath}')
      print(f"  → Best model 저장 (loss: {best_loss:.3f})")
    else:
      patience_counter += 1
      print(f"  → Early stopping patience: {patience_counter}/{args.patience}")
      if patience_counter >= args.patience:
        print(f"Early stopping at epoch {epoch}!")
        break

    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  """최적 모델로 제출용 소넷 생성."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # best 모델 우선, 없으면 마지막 epoch 모델 사용
  try:
    saved = torch.load(f'best_{args.filepath}', weights_only=False)
    print("best model 로드")
  except FileNotFoundError:
    saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
    print("마지막 epoch 모델 로드")

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


@torch.no_grad()
def run_temperature_experiment(args):
  """
  temperature와 top_p 조합별 소넷 생성 결과를 비교하는 실험.
  각 조합으로 held-out 소넷 첫 번째를 생성하고 출력한다.
  """
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  try:
    saved = torch.load(f'best_{args.filepath}', weights_only=False)
  except FileNotFoundError:
    saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
  # 첫 번째 소넷 하나만 사용
  sample = list(held_out_sonnet_dataset)[0]

  # 실험할 temperature / top_p 조합
  experiments = [
    (0.7, 0.9),
    (1.0, 0.9),
    (1.2, 0.9),
    (0.7, 0.7),
    (1.0, 0.7),
  ]

  print("\n" + "="*60)
  print("Temperature / Top-p 실험 결과")
  print("="*60)

  for temp, tp in experiments:
    print(f"\n--- temperature={temp}, top_p={tp} ---")
    encoding = model.tokenizer(sample[1], return_tensors='pt', padding=False, truncation=True).to(device)
    _, output = model.generate(encoding['input_ids'], temperature=temp, top_p=tp)
    print(f"{sample[1]}{output}\n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--top_p", type=float, default=0.9)

  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--model_size", type=str,
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # Early stopping parameters
  parser.add_argument("--patience", type=int, default=3,
                      help="Early stopping patience: loss가 개선되지 않는 epoch 수 허용치")
  parser.add_argument("--min_delta", type=float, default=0.001,
                      help="Early stopping 최소 개선 기준값")

  # 실험 모드
  parser.add_argument("--run_experiment", action='store_true',
                      help="temperature/top_p 실험 실행")

  args = parser.parse_args()
  return args


def add_arguments(args):
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
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)
  train(args)

  if args.run_experiment:
    run_temperature_experiment(args)

  generate_submission_sonnets(args)
