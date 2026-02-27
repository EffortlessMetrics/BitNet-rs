# BitNet C# FFI bindings

This folder contains a .NET P/Invoke binding for `crates/bitnet-ffi/include/bitnet.h`.

## What's improved

- Uses `LibraryImport` source-generated interop (faster and AOT-friendly).
- Uses UTF-8 marshaling helpers that always pass null-terminated strings.
- Maps all core model/inference/performance APIs from `bitnet.h`.
- Includes a small safe wrapper (`BitNetModelHandle`) that owns model lifetime.
- Surfaces native failures as `BitNetException` with `bitnet_get_last_error()` messages.

## Usage notes

- `BitNetNative.NativeLibraryName` defaults to `bitnet`.
  - Linux: `libbitnet.so`
  - macOS: `libbitnet.dylib`
  - Windows: `bitnet.dll`
- Ensure the native library is discoverable via `PATH`, `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`,
  or by loading it explicitly with `NativeLibrary.Load`.

## Minimal example

```csharp
using BitNet.Ffi;

int init = BitNetNative.Init();
if (init != (int)BitNetErrorCode.Success)
{
    throw new Exception(BitNetNative.GetLastError());
}

using var model = BitNetModelHandle.Load("models/model.gguf");
string text = model.Infer("Hello from C#");
Console.WriteLine(text);

_ = BitNetNative.Cleanup();
```
