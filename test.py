import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, TFTrainer, TFTrainingArguments

input_ids = [[1, 2, 3], [1, 2, 3]]
labels = [1, 2, 3]

dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids}, labels))
model = TFT5ForConditionalGeneration.from_pretrained("t5-base")

training_args = TFTrainingArguments(
    output_dir='./results',  # output directory
    logging_steps=100,
    max_steps=2,
    save_steps=2000,
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=8,  # batch size for evaluation
    warmup_steps=0,  # number of warmup steps for learning rate scheduler
    weight_decay=0.0,  # strength of weight decay
    learning_rate=5e-5,
    gradient_accumulation_steps=2
)

with training_args.strategy.scope():
    model = TFT5ForConditionalGeneration.from_pretrained("t5-base")

trainer = TFTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=dataset,  # training dataset
)

trainer.train()
