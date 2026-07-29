"""Local generation backend for Apple Silicon (MPS), with CPU/CUDA fallback.

Generation is deliberately local: the organism is the object of study, so its
responses should not depend on a hosted provider's own safety stack, silent
model updates, or rate limits. A 1.5B model runs comfortably on an M-series Mac.

Sampling is stochastic on purpose. IDT lives in a *distribution* over responses,
so a single greedy decode per prompt would collapse exactly the signal being
measured. Each sample gets its own seed, recorded, so any response can be
regenerated.
"""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 400


def resolve_device() -> str:
    """Pick the fastest available device. Imported here so the analysis stack
    (which has no torch) can import the rest of the package."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalChatModel:
    """A loaded chat model that generates one response per call.

    Loading is expensive, so the caller holds one instance for a whole run.
    """

    def __init__(self, config: GenerationConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(self, system_prompt: str, user_message: str, seed: int) -> str:
        """Generate one response. Raises on failure; the caller records the
        failure rather than letting it silently drop out of the corpus."""
        import torch

        torch.manual_seed(seed)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
