#!/usr/bin/env bash
set -euo pipefail

# Set up side-by-side model storage layout for dual format support

MODELS_DIR="${MODELS_DIR:-models}"
CACHE_DIR="${HOME}/.cache/bitnet-rs/models"

echo "==> Setting up dual-format model storage layout"

# Create directory structure
create_model_dirs() {
    local model_id="$1"
    local model_dir="${MODELS_DIR}/${model_id}"

    echo "→ Creating directories for ${model_id}"
    mkdir -p "${model_dir}/safetensors"
    mkdir -p "${model_dir}/gguf"

    # Create symlinks to cache if available
    if [ -d "${CACHE_DIR}/${model_id}" ]; then
        echo "  Linking from cache: ${CACHE_DIR}/${model_id}"

        # Link SafeTensors files
        for file in "${CACHE_DIR}/${model_id}"/*.safetensors; do
            if [ -f "$file" ]; then
                ln -sf "$file" "${model_dir}/safetensors/" 2>/dev/null || true
            fi
        done

        # Link tokenizer and config
        for file in tokenizer.json config.json tokenizer_config.json; do
            if [ -f "${CACHE_DIR}/${model_id}/${file}" ]; then
                ln -sf "${CACHE_DIR}/${model_id}/${file}" "${model_dir}/safetensors/" 2>/dev/null || true
            fi
        done
    fi
}

# Initialize model registry
init_registry() {
    local registry="${MODELS_DIR}/index.json"

    if [ ! -f "$registry" ]; then
        echo "→ Initializing model registry"
        cat > "$registry" <<EOF
{
  "version": "1.0",
  "models": {},
  "default_format": "safetensors"
}
EOF
    fi
}

# Register a model in the index
register_model() {
    local model_id="$1"
    local registry="${MODELS_DIR}/index.json"

    echo "→ Registering ${model_id} in index"

    # Use Python for JSON manipulation (more reliable than jq)
    python3 -c "
import json
import sys
from pathlib import Path

registry_path = Path('${registry}')
model_id = '${model_id}'
models_dir = Path('${MODELS_DIR}')

# Load existing registry
with open(registry_path) as f:
    data = json.load(f)

# Check what formats are available
model_dir = models_dir / model_id
formats = []
if (model_dir / 'safetensors').exists() and any((model_dir / 'safetensors').glob('*.safetensors')):
    formats.append('safetensors')
if (model_dir / 'gguf').exists() and any((model_dir / 'gguf').glob('*.gguf')):
    formats.append('gguf')

# Update registry
data['models'][model_id] = {
    'path': str(model_dir),
    'formats': formats,
    'verified': False
}

# Write back
with open(registry_path, 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Convert SafeTensors to GGUF if needed
convert_if_needed() {
    local model_id="$1"
    local model_dir="${MODELS_DIR}/${model_id}"

    # Check if SafeTensors exists but GGUF doesn't
    if [ -f "${model_dir}/safetensors/model.safetensors" ] && \
       [ ! -f "${model_dir}/gguf/model.gguf" ]; then

        echo "→ Converting ${model_id} to GGUF format"

        # Use our converter script
        if [ -f "scripts/convert_safetensors_to_gguf.py" ]; then
            python3 scripts/convert_safetensors_to_gguf.py \
                "${model_dir}/safetensors" \
                "${model_dir}/gguf/model.gguf" \
                || echo "  Warning: Conversion failed"
        else
            echo "  Warning: Converter script not found"
        fi
    fi
}

# Main setup
main() {
    echo "Models directory: ${MODELS_DIR}"
    echo "Cache directory: ${CACHE_DIR}"
    echo

    # Create base directory
    mkdir -p "${MODELS_DIR}"

    # Initialize registry
    init_registry

    # Set up known models
    if [ -d "${CACHE_DIR}/bitnet_b1_58-3B" ]; then
        create_model_dirs "bitnet_b1_58-3B"
        convert_if_needed "bitnet_b1_58-3B"
        register_model "bitnet_b1_58-3B"
    fi

    # Report status
    echo
    echo "==> Model storage layout complete"

    if [ -f "${MODELS_DIR}/index.json" ]; then
        echo
        echo "Registered models:"
        python3 -c "
import json
with open('${MODELS_DIR}/index.json') as f:
    data = json.load(f)
    for model_id, info in data.get('models', {}).items():
        formats = ', '.join(info.get('formats', []))
        print(f'  • {model_id}: {formats or \"no formats available\"}')"
    fi

    # Export paths for other scripts
    echo
    echo "Export these variables for validation:"
    echo "  export MODELS_DIR=\"${MODELS_DIR}\""

    # Find first available model
    if [ -f "${MODELS_DIR}/bitnet_b1_58-3B/safetensors/model.safetensors" ]; then
        echo "  export MODEL_PATH=\"${MODELS_DIR}/bitnet_b1_58-3B/safetensors/model.safetensors\""
        echo "  export TOKENIZER=\"${MODELS_DIR}/bitnet_b1_58-3B/safetensors/tokenizer.json\""
    fi

    if [ -f "${MODELS_DIR}/bitnet_b1_58-3B/gguf/model.gguf" ]; then
        echo "  export GGUF_MODEL=\"${MODELS_DIR}/bitnet_b1_58-3B/gguf/model.gguf\""
    fi
}

# Run main
main "$@"
