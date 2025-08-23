#!/usr/bin/env python3
"""
Convert SafeTensors model to GGUF format for BitNet.rs

This converter handles the BitNet specific quantization and metadata.
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

try:
    import safetensors.torch
    import torch
except ImportError:
    print("Error: Please install required packages:")
    print("  pip install safetensors torch")
    sys.exit(1)


# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# BitNet specific metadata keys
BITNET_KEYS = {
    "architecture": "bitnet.architecture",
    "context_length": "bitnet.context_length",
    "hidden_size": "bitnet.hidden_size",
    "num_layers": "bitnet.num_layers",
    "num_heads": "bitnet.num_heads",
    "vocab_size": "bitnet.vocab_size",
    "quantization": "bitnet.quantization_type",
    "weight_scale": "bitnet.weight_scale",
}


class GGUFWriter:
    """GGUF file writer for BitNet models"""
    
    def __init__(self, path: Path, arch: str = "bitnet"):
        self.path = path
        self.arch = arch
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Tuple[str, np.ndarray]] = []
        
    def add_metadata(self, key: str, value: Any, value_type: Optional[int] = None):
        """Add metadata entry"""
        if value_type is None:
            # Auto-detect type
            if isinstance(value, bool):
                value_type = GGUF_TYPE_BOOL
            elif isinstance(value, int):
                if value < 0:
                    value_type = GGUF_TYPE_INT32
                else:
                    value_type = GGUF_TYPE_UINT32
            elif isinstance(value, float):
                value_type = GGUF_TYPE_FLOAT32
            elif isinstance(value, str):
                value_type = GGUF_TYPE_STRING
            else:
                raise ValueError(f"Cannot auto-detect type for {type(value)}")
                
        self.metadata[key] = (value, value_type)
        
    def add_tensor(self, name: str, tensor: np.ndarray):
        """Add tensor to be written"""
        self.tensors.append((name, tensor))
        
    def write(self):
        """Write GGUF file"""
        with open(self.path, "wb") as f:
            # Write header
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))  # tensor count
            f.write(struct.pack("<Q", len(self.metadata)))  # metadata count
            
            # Write metadata
            for key, (value, vtype) in self.metadata.items():
                # Write key
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(key_bytes)))
                f.write(key_bytes)
                
                # Write value type
                f.write(struct.pack("<I", vtype))
                
                # Write value
                if vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", value))
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack("<i", value))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", value))
                elif vtype == GGUF_TYPE_STRING:
                    val_bytes = value.encode("utf-8")
                    f.write(struct.pack("<Q", len(val_bytes)))
                    f.write(val_bytes)
                elif vtype == GGUF_TYPE_BOOL:
                    f.write(struct.pack("<?", value))
                else:
                    raise ValueError(f"Unsupported value type: {vtype}")
                    
            # Align to 32 bytes for tensor data
            pos = f.tell()
            align_pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
            f.write(b"\x00" * align_pad)
            
            # Write tensor info
            for name, tensor in self.tensors:
                # Name
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<Q", len(name_bytes)))
                f.write(name_bytes)
                
                # Number of dimensions
                f.write(struct.pack("<I", len(tensor.shape)))
                
                # Shape
                for dim in tensor.shape:
                    f.write(struct.pack("<Q", dim))
                    
                # Data type (F32 = 0 for now, can extend for quantized)
                f.write(struct.pack("<I", 0))
                
                # Offset (will be calculated later)
                f.write(struct.pack("<Q", 0))
                
            # Write tensor data
            tensor_data_start = f.tell()
            for name, tensor in self.tensors:
                # Ensure F32
                if tensor.dtype != np.float32:
                    tensor = tensor.astype(np.float32)
                    
                # Align tensor data
                pos = f.tell()
                align_pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
                f.write(b"\x00" * align_pad)
                
                # Write tensor
                tensor.tofile(f)
                
            print(f"Wrote {len(self.tensors)} tensors to {self.path}")


def load_config(model_dir: Path) -> Dict[str, Any]:
    """Load model configuration"""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Warning: No config.json found in {model_dir}")
        return {}
        
    with open(config_path) as f:
        return json.load(f)


def convert_safetensors_to_gguf(
    input_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
    tokenizer_path: Optional[Path] = None,
) -> None:
    """Convert SafeTensors model to GGUF format"""
    
    print(f"Loading SafeTensors from {input_path}")
    tensors = safetensors.torch.load_file(str(input_path))
    
    # Create GGUF writer
    writer = GGUFWriter(output_path)
    
    # Add metadata from config
    if config:
        # Architecture info
        if "model_type" in config:
            writer.add_metadata("general.architecture", config["model_type"])
        if "hidden_size" in config:
            writer.add_metadata(BITNET_KEYS["hidden_size"], config["hidden_size"])
        if "num_hidden_layers" in config:
            writer.add_metadata(BITNET_KEYS["num_layers"], config["num_hidden_layers"])
        if "num_attention_heads" in config:
            writer.add_metadata(BITNET_KEYS["num_heads"], config["num_attention_heads"])
        if "vocab_size" in config:
            writer.add_metadata(BITNET_KEYS["vocab_size"], config["vocab_size"])
        if "max_position_embeddings" in config:
            writer.add_metadata(BITNET_KEYS["context_length"], config["max_position_embeddings"])
            
    # Add BitNet specific metadata
    writer.add_metadata(BITNET_KEYS["quantization"], "i2s")  # Default to i2s
    writer.add_metadata(BITNET_KEYS["weight_scale"], 1.0)
    writer.add_metadata("general.file_type", 1)  # F32
    
    # Convert and add tensors
    for name, tensor in tensors.items():
        # Convert tensor to numpy
        if hasattr(tensor, "numpy"):
            np_tensor = tensor.numpy()
        else:
            np_tensor = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)
            
        # Rename tensors to GGUF conventions if needed
        gguf_name = name
        # Example mappings (extend as needed):
        # "model.layers.0.self_attn.q_proj.weight" -> "blk.0.attn_q.weight"
        
        writer.add_tensor(gguf_name, np_tensor)
        
    # Write the GGUF file
    writer.write()
    print(f"Successfully converted to {output_path}")
    
    # Write metadata JSON for validation
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        meta = {
            "source": str(input_path),
            "format": "gguf",
            "version": GGUF_VERSION,
            "tensors": len(writer.tensors),
            "metadata_entries": len(writer.metadata),
        }
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SafeTensors to GGUF")
    parser.add_argument("input", type=Path, help="Input SafeTensors file or directory")
    parser.add_argument("output", type=Path, help="Output GGUF file")
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer file to embed")
    parser.add_argument("--config", type=Path, help="Model config.json file")
    args = parser.parse_args()
    
    # Load config
    config = None
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
    elif args.input.is_dir():
        config = load_config(args.input)
        # Find SafeTensors file
        st_files = list(args.input.glob("*.safetensors"))
        if not st_files:
            print(f"Error: No .safetensors files found in {args.input}")
            sys.exit(1)
        args.input = st_files[0]
        print(f"Using SafeTensors file: {args.input}")
        
    # Convert
    convert_safetensors_to_gguf(
        args.input,
        args.output,
        config=config,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()