from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_n_ft_01 = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_T5small_01")
tokenizer_n_ft_01 = AutoTokenizer.from_pretrained("fine_tuned_T5small_01")
# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from peft import PeftModel

# # Path to your fine-tuned model
# model_dir = "fine_tuned_T5small_01"

# # Load the tokenizer from fine-tuned model directory
# tokenizer = AutoTokenizer.from_pretrained(model_dir)

# # Load the fine-tuned model (including LoRA adapters) directly
# model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="cpu")

# # Move model to CPU (ensuring it runs on the correct device)
# model.to("cpu")

# print("Fine-tuned model loaded successfully on CPU.")
