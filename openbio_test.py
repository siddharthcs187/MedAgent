# import transformers
# import torch
# print(transformers.__version__)

# model_id = "aaditya/Llama3-OpenBioLLM-8B"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cpu",
# )

# messages = [
#     {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."},
#     {"role": "user", "content": "How can i split a 3mg or 4mg waefin pill so i can get a 2.5mg pill?"},
# ]

# prompt = pipeline.tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
# )

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipeline(
#     prompt,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.1,
#     top_p=0.9,
# )
# print(outputs[0]["generated_text"][len(prompt):])

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("aaditya/Llama3-OpenBioLLM-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("aaditya/Llama3-OpenBioLLM-8B", trust_remote_code=True)
