"""
Re-run inference for gcg_hook_target suffixes WITHOUT the Arditi hook.
Loads optimized suffixes from results.json, greedy decodes, scores with Gemma.
"""

import json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
IN_PATH    = "results/smoke10_exp21/results.json"
OUT_PATH   = "results/smoke10_exp21/results_no_hook.json"
MAX_NEW    = 200

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    with open(IN_PATH) as f:
        data = json.load(f)

    results = []
    for i, item in enumerate(data["results"]):
        pid    = item["prompt_id"]
        prompt = item["prompt"]
        suffix = item["suffix"]

        text = prompt + " " + suffix
        messages = [{"role": "user", "content": text}]
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok([chat], return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            out_ids = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        output = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)

        print(f"[{i+1}/10] ID={pid}: {output[:80]}", flush=True)
        results.append({"prompt_id": pid, "prompt": prompt,
                        "suffix": suffix, "output": output})

    with open(OUT_PATH, "w") as f:
        json.dump({"config": {"hook": False, "source": "gcg_hook_target"},
                   "results": results}, f, indent=2)
    print(f"\nDone -> {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
