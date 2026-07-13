import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i} — {props.name}, total={total/1e9:.1f}GB, free={free/1e9:.1f}GB")

print("\nLoading Llama-3.1-8B-Instruct on cuda:0 (instance 1) ...")
from transformers import AutoModelForCausalLM
model0 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
free0, _ = torch.cuda.mem_get_info(0)
print(f"  cuda:0 free after first load: {free0/1e9:.1f}GB")

print("\nLoading Llama-3.1-8B-Instruct on cuda:1 (instance 2) ...")
model1 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:1",
)
free1, _ = torch.cuda.mem_get_info(1)
print(f"  cuda:1 free after second load: {free1/1e9:.1f}GB")

print("\nChecking GPTFuzz judge in HF cache ...")
try:
    from transformers import RobertaForSequenceClassification
    judge = RobertaForSequenceClassification.from_pretrained("hubert233/GPTFuzz").to("cuda:1")
    print("  GPTFuzz: OK")
    del judge
except Exception as e:
    print(f"  GPTFuzz: FAILED — {e}")

print("\nSanity check passed.")
