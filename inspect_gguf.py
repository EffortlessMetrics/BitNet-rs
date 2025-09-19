import gguf
import sys

reader = gguf.GGUFReader(sys.argv[1])
print("Available metadata keys:")
for field in reader.fields.values():
    if "head" in field.name or "attn" in field.name:
        print(f"  {field.name}: {field.parts}")

