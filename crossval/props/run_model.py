"""
Cross-system model runners with deterministic execution.
Provides unified interface for bitnet-rs, llama.cpp, and HuggingFace.
"""
import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, cast

# Module-level type annotations for linting
meta: Dict[str, Any] = {}
rows: List[Dict[str, Any]] = []

if TYPE_CHECKING:
    import torch as _TorchModule  # type: ignore
    from transformers import AutoModelForCausalLM as _AutoModel, AutoTokenizer as _AutoTok  # type: ignore

# Runtime placeholders – deliberately typed as Optional[Any]
torch: Optional[Any] = None
AutoModelForCausalLM: Optional[Any] = None
AutoTokenizer: Optional[Any] = None


class RunResult:
    """Container for model output with metadata."""
    def __init__(self, text: str, meta: Dict[str, Any], raw_json: Dict[str, Any]):
        self.text = text
        self.meta = meta
        self.raw = raw_json


def _run(env: Optional[Dict[str, str]] = None) -> None:
    """Entry point that bails gracefully if heavy deps aren't available."""
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        return
    # Main logic would go here if deps were available
    return

# pyright: reportUnusedFunction=false
def _run_cmd(cmd: str, timeout: int = 120, env: Optional[Dict[str, str]] = None) -> Tuple[str, str, int]:
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
    """Runner for bitnet-rs CLI with deterministic settings."""

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

        # Deterministic environment (single-thread unless caller overrides)
        env = os.environ.copy()
        env.setdefault("BITNET_DETERMINISTIC", "1")
        env.setdefault("OMP_NUM_THREADS", str(self.threads))
        env.setdefault("MKL_NUM_THREADS", str(self.threads))
        env.setdefault("BLAS_NUM_THREADS", str(self.threads))
        env.setdefault("RAYON_NUM_THREADS", str(self.threads))

        # Build command as argv list (no shell quoting issues)
        args = [
            self.bin, "run",
            "--model", self.model,
            "--max-new-tokens", str(max_new_tokens),
            "--seed", str(int(seed) & 0xFFFFFFFF),  # Ensure valid u32
            "--json-out", tmp.name,
        ]

        if greedy:
            args += ["--greedy", "--deterministic", "--threads", "1"]
        else:
            args += ["--temperature", "1", "--top-p", "0.95", "--top-k", "40"]

        if self.tokenizer:
            args += ["--tokenizer", self.tokenizer]

        if dump_logits_steps > 0:
            args += ["--dump-logits", str(dump_logits_steps), "--topk", str(topk)]

        args += ["--prompt", prompt]

        try:
            # Run without shell to avoid quoting issues
            p = subprocess.run(
                args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False
            )
        except subprocess.TimeoutExpired as e:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(f"BitNet timed out after {timeout}s on prompt: {prompt[:100]}...") from e

        stdout = p.stdout.decode("utf-8", errors="replace")
        stderr = p.stderr.decode("utf-8", errors="replace")

        # Check return code
        if p.returncode != 0:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(
                f"BitNet exited with code {p.returncode}\n"
                f"STDERR:\n{stderr}\n"
                f"STDOUT:\n{stdout}\n"
                f"Prompt: {prompt[:100]}..."
            )

        # Parse JSON output
        try:
            with open(tmp.name, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to parse BitNet JSON output\n"
                f"STDERR:\n{stderr}\n"
                f"STDOUT:\n{stdout}\n"
                f"Prompt: {prompt[:100]}..."
            ) from e
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        text = normalize_text(raw.get("text", ""))

        meta: Dict[str, Any] = {
            "counts": raw.get("counts", {}),
            "timing_ms": raw.get("timing_ms", {}),
            "throughput_tps": raw.get("throughput_tps", {}),
            "tokenizer": raw.get("tokenizer", {}),
            "logits_dump": raw.get("logits_dump", []),  # For logit-parity
            "seed": seed,
        }

        return RunResult(text, meta, raw)

    def run_teacher_force(
        self,
        token_ids: Sequence[int],
        steps: int,
        topk: int,
        timeout: float = 120
    ) -> List[Dict[str, Any]]:
        """Run BitNet with teacher-forcing on a specific token path."""
        import tempfile
        import json
        import subprocess
        import os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()

        env = os.environ.copy()
        env.setdefault("BITNET_DETERMINISTIC", "1")
        env.setdefault("RAYON_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("BLAS_NUM_THREADS", "1")

        # Build argv - teacher-force doesn't need text-file content
        args = [
            self.bin, "eval",
            "--model", self.model,
            "--teacher-force-ids", ",".join(map(str, token_ids)),
            "--dump-logit-steps", str(steps),
            "--logits-topk", str(topk),
            "--json-out", tmp.name
        ]

        if self.tokenizer:
            args += ["--tokenizer", self.tokenizer]

        p = subprocess.run(
            args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False
        )

        if p.returncode != 0:
            os.unlink(tmp.name)
            raise RuntimeError(f"bitnet eval failed: {p.stderr.decode('utf-8','replace')}")

        with open(tmp.name, "r", encoding="utf-8") as f:
            data = json.load(f)
        os.unlink(tmp.name)

        return data.get("logits_dump", [])


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

        # Build llama.cpp command as argv list
        args = [
            self.bin,
            "-m", self.model,
            "-n", str(max_new_tokens),
            "--seed", str(int(seed) & 0xFFFFFFFF),
            "--temp", "0",
            "--top-p", "1",
            "--top-k", "0",  # Force greedy
            "-t", str(self.threads),
            "--no-penalize-nl",
            "--repeat-penalty", "1.0",  # No repetition penalty
            "-p", prompt,
        ]

        try:
            p = subprocess.run(
                args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llama.cpp timed out after {timeout}s") from e

        out = p.stdout.decode("utf-8", errors="replace")
        err = p.stderr.decode("utf-8", errors="replace")

        if p.returncode != 0:
            raise RuntimeError(
                f"llama.cpp exited with code {p.returncode}\n"
                f"STDERR:\n{err}\n"
                f"Prompt: {prompt[:100]}..."
            )

        # Extract generated text (after prompt)
        text = out
        if prompt in text:
            text = text[text.index(prompt) + len(prompt):]
        text = normalize_text(text)

        return RunResult(
            text,
            {"seed": seed},
            {"raw_stdout": out, "stderr": err}
        )


class HFRuntimeRunner:
    """Runner for HuggingFace transformers (optional, for reference)."""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Lazy load model and tokenizer."""
        if self._model is None:
            global torch, AutoModelForCausalLM, AutoTokenizer
            if torch is None:
                try:
                    import torch as _torch  # type: ignore
                    torch = cast(Any, _torch)
                except Exception:
                    return  # keep None; callers will early-return
            if AutoModelForCausalLM is None or AutoTokenizer is None:
                try:
                    from transformers import AutoModelForCausalLM as _AutoModel, AutoTokenizer as _AutoTok  # type: ignore
                    AutoModelForCausalLM = cast(Any, _AutoModel)
                    AutoTokenizer = cast(Any, _AutoTok)
                except Exception:
                    return

            # If imports failed, bail early
            if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
                return

            # Set deterministic PyTorch settings
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True, warn_only=True)

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32
            ).to(self.device)
            self._model.eval()

    def tokenizer(self) -> Any:
        """Get tokenizer for external use."""
        self._load()
        return self._tokenizer

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
        global torch
        if torch is None:
            import torch as _torch  # type: ignore
            torch = _torch
        from transformers import set_seed  # type: ignore

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

    def run_teacher_force(self, token_ids: Sequence[int], steps: int, topk: int) -> List[Dict[str, Any]]:
        """Run HF model with teacher-forcing on a specific token path."""
        global torch
        if torch is None:
            import torch as _torch  # type: ignore
            torch = _torch
        self._load()

        device = next(self._model.parameters()).device
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = self._model(input_ids=ids)
            L = out.logits.squeeze(0)  # [T, V]

        upto = min(steps, L.size(0)-1)
        dump: List[Dict[str, Any]] = []

        for t in range(upto):
            v = L[t]  # logits predicting token_ids[t+1]
            k = min(topk, v.size(-1))
            vals, idxs = torch.topk(v, k)
            topk_pairs = [(int(i), float(val)) for i, val in zip(idxs.tolist(), vals.tolist())]
            dump.append({
                "step": t,
                "topk": topk_pairs,
                "chosen_id": int(token_ids[t+1]) if t+1 < len(token_ids) else None
            })

        return dump


# Mark _run as used for linters
RUN = _run
