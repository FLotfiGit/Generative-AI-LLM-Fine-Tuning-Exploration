pip install transformers torch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load Pretrained Model & Tokenizer
model_name = "facebook/opt-125m"  # You can change to GPT-3.5 if using OpenAI API
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Define Different Prompting Techniques
question = "What is the capital of France and why is it important?"

# 1. Instruction Prompting (Direct Command)
instruction_prompt = f"Answer the following question concisely: {question}"

# 2. Few-Shot Prompting (Providing Examples)
few_shot_prompt = f"""
Q: What is the capital of Germany? 
A: The capital of Germany is Berlin. It is important because it serves as the political and cultural hub of the country.

Q: What is the capital of France and why is it important? 
A: """

# 3. Chain-of-Thought (CoT) Prompting (Step-by-Step Reasoning)
cot_prompt = f"""
Think step by step:
1. Identify the country in the question.
2. Determine its capital.
3. Explain its significance.

{question}
"""

# Run Inference & Measure Performance
def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100)
    
    response_time = time.time() - start_time
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return response_text, response_time

# Generate responses using different prompting methods
instruction_response, instruction_time = generate_response(instruction_prompt)
few_shot_response, few_shot_time = generate_response(few_shot_prompt)
cot_response, cot_time = generate_response(cot_prompt)

# Analyzing Prompt Effectiveness
def analyze_response(response):
    words = response.split()
    return len(words), response

# Get response lengths
instruction_len, instruction_text = analyze_response(instruction_response)
few_shot_len, few_shot_text = analyze_response(few_shot_response)
cot_len, cot_text = analyze_response(cot_response)

# Display Results
print("**Instruction Prompt Response:**")
print(instruction_text)
print(f"Word Count: {instruction_len} | Response Time: {instruction_time:.2f} sec")

print("**Few-Shot Prompt Response:**")
print(few_shot_text)
print(f"Word Count: {few_shot_len} | Response Time: {few_shot_time:.2f} sec")

print(" **Chain-of-Thought Prompt Response:**")
print(cot_text)
print(f"Word Count: {cot_len} |  Response Time: {cot_time:.2f} sec")

# Performance Summary
print("**Prompt Engineering Performance Analysis:**")
print(f"Instruction Prompt: {instruction_len} words, {instruction_time:.2f} sec")
print(f"Few-Shot Prompt: {few_shot_len} words, {few_shot_time:.2f} sec")
print(f"Chain-of-Thought Prompt: {cot_len} words, {cot_time:.2f} sec")

# Key Observations:
if cot_len > few_shot_len and few_
