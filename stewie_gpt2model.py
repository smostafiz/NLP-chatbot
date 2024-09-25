import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_gpt2_model():
    """The function loads a pre-trained GPT-2 language model and its corresponding tokenizer."""
    model_name = "gpt2"
    t_model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return t_model, tokenizer


def generate_gpt2_response(input_text, t_model, tokenizer):
    """The function takes an input text, a pre-trained GPT-2 language model, and its corresponding tokenizer,
    and it generates a response based on the input text."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    output = t_model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50,
                              do_sample=True, attention_mask=attention_mask)
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return bot_response
