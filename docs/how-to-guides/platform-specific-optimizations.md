# How to Apply Platform-Specific Optimizations

This guide explains how to apply platform-specific optimizations for BitNet.rs.

## Linux Optimizations

```bash
# Huge pages
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap
sudo swapoff -a

# Set process priority
nice -n -10 ./bitnet-server
```

## macOS Optimizations

```bash
# Increase file descriptor limits
ulimit -n 65536

# Set thread priority
sudo renice -10 -p $$
```

## Windows Optimizations

```powershell
# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Set process priority
Start-Process -FilePath "bitnet-server.exe" -WindowStyle Hidden -Priority High
```
