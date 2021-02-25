from pytorch_transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.add_special_tokens({"additional_special_tokens": ("@a@", "@b@")})
tokenizer.all_special_tokens
