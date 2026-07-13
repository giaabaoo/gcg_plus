import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaModel:
    """
    Llama-3.1-8B-Instruct wrapper for GCG attack experiments.

    Config fields (model.*):
        hf_path        : HuggingFace model path or local cache path
        dtype          : 'bfloat16' | 'float16' | 'float32'
        attn_impl      : attention implementation (e.g. 'eager')
        max_new_tokens : generation length
        temperature    : sampling temperature
        do_sample      : bool
    """

    def __init__(self, config):
        self.config = config

        dtype_map = {'bfloat16': torch.bfloat16,
                     'float16':  torch.float16,
                     'float32':  torch.float32}
        dtype = dtype_map.get(getattr(config.model, 'dtype', 'bfloat16'), torch.bfloat16)

        print(f'Loading {config.model.hf_path} ...')
        kwargs = dict(torch_dtype=dtype, device_map='auto')
        attn = getattr(config.model, 'attn_impl', None)
        if attn:
            kwargs['attn_implementation'] = attn

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(config.model.hf_path, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        # Fail fast if device_map='auto' silently fell back to CPU (e.g. allocated
        # GPU is full). 24h GCG on CPU is wasted compute.
        if self.device.type != 'cuda':
            raise RuntimeError(
                f'Model loaded on {self.device}, expected cuda. '
                f'Likely cause: allocated GPU is OOM. '
                f'Run "nvidia-smi" to check; resubmit the job.'
            )

    def format_prompt(self, prompt: str) -> str:
        msgs = [{'role': 'user', 'content': prompt}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt_text: str) -> str:
        """Plain generation — no hooks, no steering."""
        ids = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        input_len = ids['input_ids'].shape[1]

        max_new = getattr(self.config.model, 'max_new_tokens', 400)
        temp    = getattr(self.config.model, 'temperature', 0.7)
        sample  = getattr(self.config.model, 'do_sample', True)

        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=sample,
                temperature=temp,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
