# https://github.com/huggingface/transformers/issues/917#issuecomment-525297746

import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel
import numpy as np
from scipy.special import softmax

PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> """

text = "The dog is very cute."

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased', output_attentions=True)

tokenize_input = tokenizer.tokenize(PADDING_TEXT + text)
tokenize_text = tokenizer.tokenize(text)

sum_lp = 0.0
for max_word_id in range((len(tokenize_input)-len(tokenize_text)), (len(tokenize_input))):

    sent = tokenize_input[:]

    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sent)])

    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, max_word_id:] = 1.0 

    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
    target_mapping[0, 0, max_word_id] = 1.0

    with torch.no_grad():
        next_token_logits, attentions = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)

    word_id = tokenizer.convert_tokens_to_ids([tokenize_input[max_word_id]])[0]
    predicted_prob = softmax(np.array(next_token_logits[0][-1]))
    lp = np.log(predicted_prob[word_id])

    sum_lp += lp

print("sentence logprob =", sum_lp)
