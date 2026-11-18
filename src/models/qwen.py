"""GPU-only Hugging Face adapter. Dry-run and mock tests avoid these imports."""
class QwenGenerator:
    def __init__(self, model_id, revision):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available(): raise RuntimeError("Qwen inference requires an allocated CUDA GPU; use --dry-run or --mock locally")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.revision = self.model.config._commit_hash or revision
    def generate(self, messages, settings):
        torch = self.torch
        torch.manual_seed(settings["seed"])
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=settings["max_new_tokens"], do_sample=settings["do_sample"], pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0][inputs.input_ids.shape[1]:]
        self.last_generated_tokens = int(generated.shape[0])
        self.last_hit_max_new_tokens = self.last_generated_tokens >= settings["max_new_tokens"]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
