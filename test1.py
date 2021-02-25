import tensorflow as tf
from transformers import BertTokenizer, TFDistilBertForQuestionAnswering

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
start_positions = tf.constant([1])
end_positions = tf.constant([3])
outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
start_scores, end_scores = outputs[:2]
