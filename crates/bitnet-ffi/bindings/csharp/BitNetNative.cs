using System;
using System.Runtime.InteropServices;
using System.Text;

namespace BitNet.Ffi;

public enum BitNetErrorCode
{
    Success = 0,
    InvalidArgument = -1,
    ModelNotFound = -2,
    ModelLoadFailed = -3,
    InferenceFailed = -4,
    OutOfMemory = -5,
    ThreadSafety = -6,
    InvalidModelId = -7,
    ContextLengthExceeded = -8,
    UnsupportedOperation = -9,
    Internal = -10,
}

public enum BitNetBackendPreference : uint
{
    Auto = 0,
    Cpu = 1,
    Gpu = 2,
}

[StructLayout(LayoutKind.Sequential)]
public struct BitNetConfig
{
    public IntPtr ModelPath;
    public uint ModelFormat;
    public uint VocabSize;
    public uint HiddenSize;
    public uint NumLayers;
    public uint NumHeads;
    public uint IntermediateSize;
    public uint MaxPositionEmbeddings;
    public uint QuantizationType;
    public uint BlockSize;
    public float Precision;
    public uint NumThreads;
    public uint UseGpu;
    public uint BatchSize;
    public ulong MemoryLimit;
}

[StructLayout(LayoutKind.Sequential)]
public struct BitNetInferenceConfig
{
    public uint MaxLength;
    public uint MaxNewTokens;
    public float Temperature;
    public uint TopK;
    public float TopP;
    public float RepetitionPenalty;
    public float FrequencyPenalty;
    public float PresencePenalty;
    public ulong Seed;
    public uint DoSample;
    public uint BackendPreference;
    public uint EnableStreaming;
    public uint StreamBufferSize;
}

[StructLayout(LayoutKind.Sequential)]
public struct BitNetModelInfo
{
    public IntPtr Name;
    public IntPtr Version;
    public IntPtr Architecture;
    public uint VocabSize;
    public uint ContextLength;
    public uint HiddenSize;
    public uint NumLayers;
    public uint NumHeads;
    public uint IntermediateSize;
    public uint QuantizationType;
    public ulong FileSize;
    public ulong MemoryUsage;
    public uint IsGpuLoaded;
}

[StructLayout(LayoutKind.Sequential)]
public struct BitNetPerformanceMetrics
{
    public float TokensPerSecond;
    public float LatencyMs;
    public float MemoryUsageMb;
    public float GpuUtilization;
    public float TotalInferenceTimeMs;
    public float TimeToFirstTokenMs;
    public uint TokensGenerated;
    public uint PromptTokens;
}

public static partial class BitNetNative
{
    public const uint AbiVersion = 1;
    public const string NativeLibraryName = "bitnet";

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_abi_version")]
    public static partial uint GetAbiVersion();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_version")]
    private static partial IntPtr GetVersionPtr();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_init")]
    public static partial int Init();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_cleanup")]
    public static partial int Cleanup();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_model_load")]
    private static partial int ModelLoadNative(byte[] pathUtf8);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_model_load_with_config")]
    private static partial int ModelLoadWithConfigNative(byte[] pathUtf8, in BitNetConfig config);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_model_free")]
    public static partial int ModelFree(int modelId);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_model_is_loaded")]
    public static partial int ModelIsLoaded(int modelId);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_model_get_info")]
    public static partial int ModelGetInfo(int modelId, out BitNetModelInfo info);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_inference")]
    private static partial int InferenceNative(int modelId, byte[] promptUtf8, byte[] output, nuint maxLen);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_inference_with_config")]
    private static partial int InferenceWithConfigNative(
        int modelId,
        byte[] promptUtf8,
        in BitNetInferenceConfig config,
        byte[] output,
        nuint maxLen);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_get_last_error")]
    private static partial IntPtr GetLastErrorPtr();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_clear_last_error")]
    public static partial void ClearLastError();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_set_num_threads")]
    public static partial int SetNumThreads(uint numThreads);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_get_num_threads")]
    public static partial uint GetNumThreads();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_set_gpu_enabled")]
    public static partial int SetGpuEnabled(int enable);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_is_gpu_available")]
    public static partial int IsGpuAvailable();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_get_performance_metrics")]
    public static partial int GetPerformanceMetrics(int modelId, out BitNetPerformanceMetrics metrics);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_reset_performance_metrics")]
    public static partial int ResetPerformanceMetrics(int modelId);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_set_memory_limit")]
    public static partial int SetMemoryLimit(ulong limitBytes);

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_get_memory_usage")]
    public static partial ulong GetMemoryUsage();

    [LibraryImport(NativeLibraryName, EntryPoint = "bitnet_garbage_collect")]
    public static partial int GarbageCollect();

    public static string GetVersion() => Marshal.PtrToStringUTF8(GetVersionPtr()) ?? string.Empty;

    public static string GetLastError() => Marshal.PtrToStringUTF8(GetLastErrorPtr()) ?? string.Empty;

    public static int ModelLoad(string path) => ModelLoadNative(ToNullTerminatedUtf8(path));

    public static int ModelLoadWithConfig(string path, in BitNetConfig config) =>
        ModelLoadWithConfigNative(ToNullTerminatedUtf8(path), in config);

    public static int Inference(int modelId, string prompt, byte[] output)
    {
        var promptUtf8 = ToNullTerminatedUtf8(prompt);
        return InferenceNative(modelId, promptUtf8, output, (nuint)output.Length);
    }

    public static int InferenceWithConfig(
        int modelId,
        string prompt,
        in BitNetInferenceConfig config,
        byte[] output)
    {
        var promptUtf8 = ToNullTerminatedUtf8(prompt);
        return InferenceWithConfigNative(modelId, promptUtf8, in config, output, (nuint)output.Length);
    }

    public static string InferenceAsString(int modelId, string prompt, int maxOutputBytes = 8192)
    {
        var output = new byte[maxOutputBytes];
        int written = InferenceNative(modelId, ToNullTerminatedUtf8(prompt), output, (nuint)output.Length);
        if (written < 0)
        {
            throw new BitNetException((BitNetErrorCode)written, GetLastError());
        }

        return Encoding.UTF8.GetString(output, 0, written);
    }

    private static byte[] ToNullTerminatedUtf8(string value)
    {
        var utf8 = Encoding.UTF8.GetBytes(value);
        var buffer = new byte[utf8.Length + 1];
        Buffer.BlockCopy(utf8, 0, buffer, 0, utf8.Length);
        buffer[^1] = 0;
        return buffer;
    }
}

public sealed class BitNetException : Exception
{
    public BitNetErrorCode ErrorCode { get; }

    public BitNetException(BitNetErrorCode errorCode, string? message)
        : base(message ?? $"BitNet error: {(int)errorCode}")
    {
        ErrorCode = errorCode;
    }
}

public sealed class BitNetModelHandle : IDisposable
{
    public int ModelId { get; private set; }

    private BitNetModelHandle(int modelId)
    {
        ModelId = modelId;
    }

    public static BitNetModelHandle Load(string modelPath)
    {
        int modelId = BitNetNative.ModelLoad(modelPath);
        if (modelId < 0)
        {
            throw new BitNetException((BitNetErrorCode)modelId, BitNetNative.GetLastError());
        }

        return new BitNetModelHandle(modelId);
    }

    public string Infer(string prompt, int outputBufferBytes = 8192)
    {
        if (ModelId < 0)
        {
            throw new ObjectDisposedException(nameof(BitNetModelHandle));
        }

        return BitNetNative.InferenceAsString(ModelId, prompt, outputBufferBytes);
    }

    public void Dispose()
    {
        if (ModelId >= 0)
        {
            _ = BitNetNative.ModelFree(ModelId);
            ModelId = -1;
        }
    }
}
