# CAIS Starter Project

### Description: For a given LLM in hugging face let's say Deberta-v3, I want to be able to get the gradient per input token. If you want example inputs to feed in use https://github.com/hendrycks/ethics that dataset.

Steven Basart Notes: Trained model found on GitHub (.pt file contains RoBERTa weights)
- From the loss, obtain the gradient of the loss with respect to each input token
- You can use (1, 0) or (1, -1) as utility scores for data
- GitHub model was trained on RoBERTa


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

```


```python
checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

    /Users/rishikhare/anaconda3/lib/python3.10/site-packages/transformers/convert_slow_tokenizer.py:564: UserWarning: The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers. In practice this means that the fast version of the tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these unknown tokens into a sequence of byte tokens matching the original piece of text.
      warnings.warn(
    Some weights of DebertaV2ForSequenceClassification were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for param in model.parameters():
    param.requires_grad = True
```


```python
prompt = "I fed my neighbor's dog the expired meat."
label = torch.tensor([1])

```


```python
inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
inputs_embeds = model.deberta.embeddings.word_embeddings(inputs["input_ids"])
inputs_embeds.requires_grad_()
inputs_embeds.retain_grad()
```

    Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.



```python
outputs = model(
    inputs_embeds=inputs_embeds,
    attention_mask=inputs["attention_mask"],
    labels=label,
)
logits = outputs.logits
```


```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, label)
```


```python
model.zero_grad()
loss.backward()
grads = inputs_embeds.grad[0]
token_grad_norms = grads.norm(dim=-1)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for token, grad_norm in zip(tokens, token_grad_norms):
    print(f"token: {token}, gradient: {grad_norm}")
```

    token: [CLS], gradient: 0.01911306008696556
    token: ▁I, gradient: 0.005149811506271362
    token: ▁fed, gradient: 0.009411280043423176
    token: ▁my, gradient: 0.004841497167944908
    token: ▁neighbor, gradient: 0.009824398905038834
    token: ', gradient: 0.004140863195061684
    token: s, gradient: 0.006523393094539642
    token: ▁dog, gradient: 0.011430288664996624
    token: ▁the, gradient: 0.008304985240101814
    token: ▁expired, gradient: 0.01959693618118763
    token: ▁meat, gradient: 0.015879929065704346
    token: ., gradient: 0.006367374677211046
    token: [SEP], gradient: 0.014575622044503689



```python

```
