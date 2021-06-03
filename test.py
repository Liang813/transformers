import torch
from src.transformers import TransfoXLConfig, TransfoXLLMHeadModel

config = TransfoXLConfig()
lm = TransfoXLLMHeadModel(config)
test_tensor = torch.LongTensor([[0]])
print(lm(input_ids=test_tensor, labels=test_tensor)[0])
