# Apple M4 Dense SLM Model Breadth Reference Sanity

`M4-MODEL-002` evaluates the exact candidates selected by `M4-MODEL-001` under
the local reference runner before any Rust M4 support work begins. This is still
artifact/reference evidence only. It does not register a model, change the M4
default, prove Rust M4 support, or widen any Metal or BitNet claim.

Machine-readable evidence is recorded in
`ci/quality/apple-m4-slm-model-breadth-reference-sanity.toml`.

The follow-up Gemma candidate cycle records machine-readable evidence in
`ci/quality/apple-m4-slm-model-breadth-gemma-reference-sanity.toml`.

## Runner

```text
/Users/steven/.cache/bitnet_cpp/build/bin/llama-cli
build 3962 (1f86f058)
Apple clang version 21.0.0 (clang-2100.0.123.102)
arm64-apple-darwin25.4.0
```

All reference generation attempts used CPU-only model layers with:

```text
-ngl 0 --no-warmup --temp 0 --top-k 1 --top-p 1 --min-p 0 --seed 42 --no-display-prompt
```

## Results

| Candidate | Result | Reason |
|---|---|---|
| `qwen3-0.6b-q8_0` | rejected for this round | The available reference runner reads GGUF metadata but fails before generation with `unknown model architecture: 'qwen3'`. |
| `smollm2-360m-instruct-q8_0` | reference-good | The reference runner loads the artifact, confirms tokenizer/chat-template metadata, and produces coherent short outputs for the bounded prompt suite. |
| `gemma-3-270m-it-q8_0` | rejected for this round | The available reference runner reads GGUF metadata but fails before generation with `unknown model architecture: 'gemma3'`. |

## Qwen3 Rejection

Artifact:

```text
repo = Qwen/Qwen3-0.6B-GGUF
revision = 23749fefcc72300e3a2ad315e1317431b06b590a
file = Qwen3-0.6B-Q8_0.gguf
size = 639446688 bytes
sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
license = apache-2.0
```

Observed metadata:

```text
general.architecture = qwen3
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre = qwen2
tokenizer.chat_template = present
general.file_type = 7
```

Reference runner failure:

```text
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'qwen3'
```

Decision:

```text
Do not promote Qwen3 to Rust M4 quality. Reconsider only after the reference
runner selected for this lane supports the qwen3 architecture.
```

## SmolLM2 Acceptance

Artifact:

```text
repo = HuggingFaceTB/SmolLM2-360M-Instruct-GGUF
revision = 593b5a2e04c8f3e4ee880263f93e0bd2901ad47f
file = smollm2-360m-instruct-q8_0.gguf
size = 386404992 bytes
sha256 = 48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201
license = apache-2.0
```

Observed metadata:

```text
general.architecture = llama
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre = smollm
tokenizer.chat_template = present
bos_token_id = 1
eos_token_id = 2
padding_token_id = 2
general.file_type = 7
```

Prompt template:

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
<prompt><|im_end|>
<|im_start|>assistant
```

Reference outputs:

| Prompt | Output |
|---|---|
| `What is 2+2? Answer briefly.` | `2+2 equals 4. [end of text]` |
| `Name the capital of France.` | `The capital of France is Paris. [end of text]` |
| `Write one short sentence about Rust.` | `Rust is a programming language that is known for its safety features, memory safety, and performance, making it a popular` |

The Rust sentence is truncated by the 24-token reference budget, but it is valid
UTF-8, non-empty, non-degenerate, and semantically plausible.

Decision:

```text
Promote smollm2-360m-instruct-q8_0 to M4-MODEL-003 for Rust M4
apple-m4-cpu-neon quality gating.
```

## Gemma 3 270M IT Rejection

Artifact:

```text
repo = ggml-org/gemma-3-270m-it-GGUF
revision = e7647be17ae1108f2f605ed061ca0608b171afff
file = gemma-3-270m-it-Q8_0.gguf
size = 291545600 bytes
sha256 = 0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3
license = gemma
```

Observed metadata:

```text
general.architecture = gemma3
general.name = Gemma 3 270m It
general.file_type = 7
tokenizer.ggml.model = llama
tokenizer.ggml.pre = default
tokenizer.chat_template = present
bos_token_id = 2
eos_token_id = 1
padding_token_id = 0
tokenizer.ggml.add_bos_token = true
tokenizer.ggml.add_eos_token = false
```

Reference runner failure:

```text
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'gemma3'
```

Cleanup:

```text
The 291545600-byte scratch GGUF was downloaded under target/ for this probe and
removed after metadata and failure evidence were recorded.
```

Decision:

```text
Do not promote Gemma 3 270M IT to Rust M4 quality. Reconsider only after the
reference runner selected for this lane supports the gemma3 architecture.
```

## Claim Boundary

This reference pass may claim only that `smollm2-360m-instruct-q8_0` is
reference-good for the bounded prompt suite and that `qwen3-0.6b-q8_0` is
blocked by the current reference runner.

The Gemma follow-up pass may claim only that `gemma-3-270m-it-q8_0` is blocked
by the current reference runner. It does not prove the Gemma Rust M4 path,
cache registration, default-model support, BitNet behavior, full
`apple-m4-metal` inference, Neural Engine execution, MPSGraph model inference,
QK256 support, or broad Apple Silicon performance.
