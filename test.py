from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.models import Model
from transformers import TFBertModel
try:
  input_layer = Input(shape = (512,), dtype='int64')
  bert = TFBertModel.from_pretrained('bert-base-chinese')(input_layer)
  bert = bert[0]
  dropout = Dropout(0.1)(bert)
  flat = Flatten()(dropout)
  classifier = Dense(units=5)(flat)
  model = Model(inputs=input_layer, outputs=classifier)
  model.summary()
except Exception as e:
  print(str(e))
  
