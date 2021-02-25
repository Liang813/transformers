import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
outputs = model(input_ids)
prediction_scores = outputs[0]
