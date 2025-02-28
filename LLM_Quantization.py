pip install transformers optimum accelerate torch
pip install auto-gptq

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import time

# Load the OPT-125M model & tokenizer
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Sample text for evaluation
sample_text = "Artificial intelligence is transforming the world by"
input_ids = tokenizer(sample_text, return_tensors="pt").input_ids

# Baseline Performance (Before Quantization)
start_time = time.time()
with torch.no_grad():
    output = model.generate(input_ids, max_length=50)
baseline_time = time.time() - start_time
baseline_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("**Baseline Model Output:**")
print(baseline_output)
print(f"Inference Time (Baseline): {baseline_time:.3f} sec")

# GPTQ Quantization Configuration
quant_config = BaseQuantizeConfig(
    bits=4,  # Quantize to 4-bit
    group_size=128,  # Grouped quantization
    desc_act=True,  # Use activation quantization
)

# Quantize the model
quantized_model = AutoGPTQForCausalLM.from_pretrained(
    model_name, quantize_config=quant_config, torch_dtype=torch.float16
)
quantized_model.quantize()

# Post-Quantization Performance
start_time = time.time()
with torch.no_grad():
    output_quantized = quantized_model.generate(input_ids, max_length=50)
quantized_time = time.time() - start_time
quantized_output = tokenizer.decode(output_quantized[0], skip_special_tokens=True)

print("**Quantized Model Output:**")
print(quantized_output)
print(f"Inference Time (Quantized): {quantized_time:.3f} sec")

# Analyzing Performance Degradation
def compare_outputs(original, quantized):
    orig_words = original.split()
    quant_words = quantized.split()
    common = sum(1 for w1, w2 in zip(orig_words, quant_words) if w1 == w2)
    similarity = common / max(len(orig_words), len(quant_words))
    return similarity * 100

similarity_score = compare_outputs(baseline_output, quantized_output)

print("**Performance Analysis:**")
print(f"Text Similarity (Baseline vs Quantized): {similarity_score:.2f}%")
print(f"Inference Speed Improvement: {baseline_time / quantized_time:.2f}x faster")

