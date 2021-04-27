import tensorflow as tf
from src.transformers import RobertaConfig, TFRobertaForMaskedLM, create_optimizer
config = RobertaConfig()  
optimizer,lr = create_optimizer(1e-4,1000000,10000,0.1,1e-6,0.01)
training_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = TFRobertaForMaskedLM(config)
model.compile(optimizer=optimizer, loss=training_loss)
input = tf.random.uniform(shape=[1,25], maxval=100, dtype=tf.int32)
hist = model.fit(input, input, epochs=1, steps_per_epoch=1,verbose=0)
print("success")
