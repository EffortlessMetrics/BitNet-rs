"""
Cross-system model runners with deterministic execution.
Provides unified interface for BitNet.rs, llama.cpp, and HuggingFace.
"""
import json
import subprocess
import tempfile
import os
import shlex
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


class RunResult:
    """Container for model output with metadata."""
    def __init__(self, text: str, meta: Dict[str, Any], raw_json: Dict[str, Any]):
        self.text = text
        self.meta = meta
        self.raw = raw_json


def _run(cmd: str, timeout: int = 120, env: Optional[Dict] = None) -> Tuple[str, str, int]:
    """Execute command with timeout and return stdout, stderr, returncode."""
    p = subprocess.run(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        timeout=timeout,
        env=env or os.environ
    )
    return (
        p.stdout.decode("utf-8", errors="replace"), 
        p.stderr.decode("utf-8", errors="replace"), 
        p.returncode
    )


def normalize_text(s: str) -> str:
    """
    Normalize text for robust comparison.
    - Unicode NFKC normalization
    - Collapse whitespace
    - Keep content robust but not too strict
    """
    import unicodedata
    import re
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s


class BitNetRunner:
    """Runner for BitNet.rs CLI with deterministic settings."""
    
    def __init__(self, bin_path: str, model: str, tokenizer: Optional[str] = None, threads: int = 1):
        self.bin = bin_path
        self.model = model
        self.tokenizer = tokenizer
        self.threads = threads
    
    def run(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        seed: int = 42, 
        greedy: bool = True,
        dump_logits_steps: int = 0, 
        topk: int = 10, 
        timeout: int = 180
    ) -> RunResult:
        """Run BitNet with deterministic settings and return normalized output."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        
        # Force deterministic execution
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(self.threads),
            "MKL_NUM_THREADS": str(self.threads),
            "BLAS_NUM_THREADS": str(self.threads),
            "RAYON_NUM_THREADS": str(self.threads),
            "BITNET_DETERMINISTIC": "1",
            "BITNET_SEED": str(seed),
        })
        
        # Build command with greedy decoding
        args = [
            shlex.quote(self.bin), "run",
            "--model", shlex.quote(self.model),
            "--max-new-tokens", str(max_new_tokens),
            "--seed", str(seed),
            "--json-out", shlex.quote(tmp.name),
            "--temperature", "0" if greedy else "1",
            "--top-p", "1" if greedy else "0.95",
            "--top-k", "0" if greedy else "40",
        ]
        
        if self.tokenizer:
            args += ["--tokenizer", shlex.quote(self.tokenizer)]
        
        if dump_logits_steps > 0:
            args += ["--dump-logits", str(dump_logits_steps), "--topk", str(topk)]
        
        args += ["--prompt", shlex.quote(prompt)]
        
        cmd = " ".join(args)
        
        try:
            out, err, code = _run(cmd, timeout=timeout, env=env)
            
            # Read JSON output
            with open(tmp.name, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            text = normalize_text(raw.get("text", ""))
            
            meta = {
                "counts": raw.get("counts", {}),
                "timing_ms": raw.get("timing_ms", {}),
                "throughput_tps": raw.get("throughput_tps", {}),
                "tokenizer": raw.get("tokenizer", {}),
                "seed": seed,
                "exit_code": code,
            }
            
            return RunResult(text, meta, raw)
            
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class LlamaCppRunner:
    """Runner for llama.cpp with matching settings."""
    
    def __init__(self, bin_path: str, model: str, threads: int = 1):
        self.bin = bin_path
        self.model = model
        self.threads = threads
    
    def run(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        seed: int = 42, 
        greedy: bool = True,
        dump_logits_steps: int = 0, 
        topk: int = 10, 
        timeout: int = 180
    ) -> RunResult:
        """Run llama.cpp with greedy decoding."""
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.threads)
        
        # Build llama.cpp command (adjust flags to your version)
        cmd = (
            f"{shlex.quote(self.bin)} "
            f"-m {shlex.quote(self.model)} "
            f"-n {max_new_tokens} "
            f"--seed {seed} "
            f"--temp 0 --top-p 1 --top-k 0 "  # Force greedy
            f"-t {self.threads} "
            f"--no-penalize-nl "
            f"--silent-prompt "
            f"-p {shlex.quote(prompt)}"
        )
        
        if greedy:
            cmd += " --repeat-penalty 1.0"  # No repetition penalty
        
        out, err, code = _run(cmd, timeout=timeout, env=env)
        
        # Extract generated text (after prompt)
        text = out
        if prompt in text:
            text = text[text.index(prompt) + len(prompt):]
        text = normalize_text(text)
        
        return RunResult(
            text, 
            {"seed": seed, "exit_code": code}, 
            {"raw_stdout": out, "stderr": err}
        )


class HFRuntimeRunner:
    """Runner for HuggingFace transformers (optional, for reference)."""
    
    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float32
            ).to(self.device)
            self._model.eval()
    
    def run(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        seed: int = 42, 
        greedy: bool = True,
        dump_logits_steps: int = 0, 
        topk: int = 10, 
        timeout: int = 180
    ) -> RunResult:
        """Run HF model with greedy decoding."""
        import torch
        from transformers import set_seed
        
        self._load()
        set_seed(seed)
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with greedy decoding
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                do_sample=False,  # Force greedy
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt to get just generated text
        if full_text.startswith(prompt):
            text = full_text[len(prompt):]
        else:
            text = full_text
        
        text = normalize_text(text)
        
        return RunResult(
            text,
            {"seed": seed, "model_id": self.model_id},
            {"full_text": full_text}
        )