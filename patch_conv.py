with open("crates/bitnet-kernels/src/convolution.rs", "r") as f:
    content = f.read()

content = content.replace(
"""                }
            }
        }
    }

    Ok(output)
}""",
"""                }
            }
        }
        _ => return Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::ExecutionFailed { reason: "Not implemented".to_string() })),
    }

    Ok(output)
}""")

with open("crates/bitnet-kernels/src/convolution.rs", "w") as f:
    f.write(content)
