import torch
from src.transformers import TransfoXLConfig, TransfoXLLMHeadModel
import traceback
try:
  config = TransfoXLConfig()
  lm = TransfoXLLMHeadModel(config)
  test_tensor = torch.LongTensor([[0]])
  print(lm(input_ids=test_tensor, labels=test_tensor)[0])
  assert(lm(input_ids=test_tensor, labels=test_tensor)[0].shape == [0,1])
except Exception as e:
  traceback.print_exc(file=open('/script/transformers3711-buggy.txt','w+'))
