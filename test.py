from pytorch_transformers import AutoTokenizer
import traceback
try:
  tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
  tokenizer.add_special_tokens({"additional_special_tokens": ("@a@", "@b@")})
  tokenizer.all_special_tokens
except Exception as e:
  traceback.print_exc(file=open('/script/transformers1152-buggy.txt','w+'))
  
