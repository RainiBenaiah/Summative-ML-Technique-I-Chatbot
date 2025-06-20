from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Attempting to load model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Success! Model and tokenizer loaded.")