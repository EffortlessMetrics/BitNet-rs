Found 541 mutants to test
ok       Unmutated baseline in 45.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:675:48: replace < with == in DeviceAwareQuantizer::validate_gpu_cpu_parity in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:160:43: replace / with * in AccuracyReport::update_errors in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:105:18: replace < with > in LookupTable::dequantize in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:222:29: replace / with % in TL2Quantizer::get_or_create_lookup_table in 1.9s build + 1.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:305:61: replace * with / in CPUQuantizer::dequantize_i2s in 6.0s build + 0.5s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:257:44: replace << with >> in CPUQuantizer::quantize_i2s in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:371:9: replace TL1Quantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![]) in 3.2s build + 0.5s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:184:54: replace / with * in AccuracyReport::calculate_std in 1.8s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:358:9: replace TL2Quantizer::quantize_scalar -> Result<Vec<i8>> with Ok(vec![1]) in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:369:50: replace * with + in CPUQuantizer::dequantize_tl1 in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:183:9: replace I2SQuantizer::dequantize_scalar -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:675:48: replace < with > in DeviceAwareQuantizer::validate_gpu_cpu_parity in 1.9s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl1.rs:98:29: replace / with * in LookupTable::quantize in 1.9s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:111:9: replace VectorizedLookupTable::reverse_len -> usize with 1 in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:193:9: replace TL2Quantizer::with_config -> Self with Default::default() in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:373:9: replace I2SQuantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:80:22: replace * with + in dequantize_value in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:451:9: replace TL2Quantizer::quantize_avx2 -> Result<Vec<i8>> with Ok(vec![1]) in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:244:80: replace == with != in TL2Quantizer::from_ini_file in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:373:12: delete ! in I2SQuantizer::dequantize_neon in 2.2s build + 0.8s test
MISSED   crates/bitnet-quantization/src/utils.rs:99:5: replace calculate_snr -> Result<f32> with Ok(-1.0) in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:147:27: replace != with == in AccuracyReport::update_errors in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:337:9: replace TL2Quantizer::quantize_cuda -> Result<QuantizedTensor> with Ok(Default::default()) in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:165:9: replace I2SQuantizer::quantize_scalar -> Result<Vec<i8>> with Ok(vec![-1]) in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:166:74: replace / with % in AccuracyReport::update_errors in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:231:35: replace * with / in CPUQuantizer::quantize_i2s in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:222:49: replace - with + in TL2Quantizer::get_or_create_lookup_table in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:242:43: replace > with == in CPUQuantizer::quantize_i2s in 5.7s build + 1.0s test
MISSED   crates/bitnet-quantization/src/tl1.rs:397:9: replace TL1Quantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![0]) in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:211:44: replace / with * in TL2Quantizer::get_lookup_table in 1.9s build + 0.8s test
MISSED   crates/bitnet-quantization/src/i2s.rs:43:38: replace * with + in I2SLayout::with_block_size in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:669:34: replace - with + in DeviceAwareQuantizer::validate_gpu_cpu_parity in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:99:73: replace / with % in calculate_snr in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:331:56: replace * with / in TL1Quantizer::dequantize_scalar in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:622:55: delete ! in DeviceAwareQuantizer::quantize_with_validation in 1.8s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:466:9: replace TL2Quantizer::dequantize_avx512 -> Result<Vec<f32>> with Ok(vec![1.0]) in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:159:27: replace > with == in AccuracyReport::update_errors in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:155:5: replace calculate_optimal_block_size -> usize with 1 in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:394:12: delete ! in TL2Quantizer::quantize_avx2 in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/lib.rs:206:5: replace validate_round_trip -> Result<bool> with Ok(false) in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:376:9: replace TL2Quantizer::dequantize_scalar -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:100:9: replace VectorizedLookupTable::dequantize -> f32 with -1.0 in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:544:59: replace - with / in log_sum_exp in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:93:54: replace + with * in VectorizedLookupTable::quantize in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:14:13: replace / with * in calculate_scale in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:72:23: replace << with >> in quantize_value in 1.9s build + 0.5s test
MISSED   crates/bitnet-quantization/src/tl2.rs:358:9: replace TL2Quantizer::quantize_scalar -> Result<Vec<i8>> with Ok(vec![]) in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:264:38: replace * with / in TL1Quantizer::quantize_cuda in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:353:9: replace I2SQuantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![-1]) in 1.8s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:227:9: replace I2SQuantizer::quantize_simd -> Result<Vec<i8>> with Ok(vec![-1]) in 2.2s build + 0.8s test
MISSED   crates/bitnet-quantization/src/i2s.rs:353:12: delete ! in I2SQuantizer::quantize_neon in 3.3s build + 1.1s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:180:25: replace < with <= in AccuracyReport::calculate_std in 2.3s build + 0.9s test
MISSED   crates/bitnet-quantization/src/utils.rs:71:22: replace << with >> in quantize_value in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:144:5: replace validate_shapes -> Result<()> with Ok(()) in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:331:56: replace * with + in TL1Quantizer::dequantize_scalar in 1.5s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:296:58: replace * with + in CPUQuantizer::dequantize_i2s in 2.7s build + 0.5s test
MISSED   crates/bitnet-quantization/src/tl2.rs:212:48: delete - in TL2Quantizer::get_lookup_table in 1.8s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:451:9: replace TL2Quantizer::quantize_avx2 -> Result<Vec<i8>> with Ok(vec![-1]) in 2.7s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:420:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![]) in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:80:5: replace dequantize_value -> f32 with 0.0 in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:487:9: replace <impl QuantizerTrait for I2SQuantizer>::is_available -> bool with false in 2.7s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:456:9: replace TL2Quantizer::dequantize_avx2 -> Result<Vec<f32>> with Ok(vec![0.0]) in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:461:9: replace TL2Quantizer::quantize_avx512 -> Result<Vec<i8>> with Ok(vec![]) in 1.9s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:72:32: replace - with + in quantize_value in 1.6s build + 0.5s test
MISSED   crates/bitnet-quantization/src/tl1.rs:259:9: replace TL1Quantizer::quantize_cuda -> Result<QuantizedTensor> with Ok(Default::default()) in 1.5s build + 0.5s test
MISSED   crates/bitnet-quantization/src/i2s.rs:149:38: replace * with / in I2SQuantizer::quantize_cuda in 1.6s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:320:33: replace match guard self.config.use_avx2 with true in TL2Quantizer::dequantize in 2.5s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:349:9: replace CPUQuantizer::dequantize_tl1 -> Result<Vec<f32>> with Ok(vec![1.0]) in 3.0s build + 0.8s test
MISSED   crates/bitnet-quantization/src/utils.rs:43:18: replace |= with ^= in pack_2bit_values in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:368:52: replace - with / in CPUQuantizer::dequantize_tl1 in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:106:29: replace / with % in calculate_snr in 2.3s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:185:57: replace - with / in AccuracyReport::calculate_std in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:246:21: delete - in CPUQuantizer::quantize_i2s in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/i2s.rs:183:9: replace I2SQuantizer::dequantize_scalar -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/lib.rs:106:23: replace == with != in <impl Quantize for QuantizedTensor>::quantize in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:334:52: delete - in CPUQuantizer::quantize_tl1 in 2.3s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:92:56: replace - with + in calculate_mse in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:72:12: delete ! in I2SQuantizer::quantize in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:675:48: replace < with <= in DeviceAwareQuantizer::validate_gpu_cpu_parity in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:346:9: replace TL1Quantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![-1]) in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:105:18: replace < with == in LookupTable::dequantize in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:333:43: replace > with == in CPUQuantizer::quantize_tl1 in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:232:9: replace I2SQuantizer::dequantize_simd -> Result<Vec<f32>> with Ok(vec![0.0]) in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:543:5: replace log_sum_exp -> f64 with 0.0 in 1.9s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:296:58: replace * with / in CPUQuantizer::dequantize_i2s in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:71:37: replace - with + in quantize_value in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl1.rs:371:9: replace TL1Quantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:101:18: replace < with == in VectorizedLookupTable::dequantize in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:518:30: replace || with && in ReferenceCalculator::calculate_perplexity in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:681:67: replace / with * in DeviceAwareQuantizer::validate_gpu_cpu_parity in 2.0s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:529:57: replace < with <= in ReferenceCalculator::calculate_perplexity in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:368:45: replace / with % in CPUQuantizer::dequantize_tl1 in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:168:12: delete ! in AccuracyReport::update_errors in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl1.rs:407:9: replace TL1Quantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![]) in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:522:39: replace / with % in ReferenceCalculator::calculate_perplexity in 2.6s build + 0.6s test
MISSED   crates/bitnet-quantization/src/lib.rs:206:5: replace validate_round_trip -> Result<bool> with Ok(true) in 3.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/i2s.rs:165:9: replace I2SQuantizer::quantize_scalar -> Result<Vec<i8>> with Ok(vec![0]) in 5.1s build + 1.1s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:665:59: replace == with != in DeviceAwareQuantizer::validate_gpu_cpu_parity in 3.5s build + 1.1s test
MISSED   crates/bitnet-quantization/src/i2s.rs:149:38: replace * with + in I2SQuantizer::quantize_cuda in 2.5s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:242:57: replace / with * in CPUQuantizer::quantize_i2s in 2.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:93:9: replace VectorizedLookupTable::quantize -> i8 with 1 in 3.0s build + 0.7s test
MISSED   crates/bitnet-quantization/src/i2s.rs:43:9: replace I2SLayout::with_block_size -> Self with Default::default() in 2.5s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:545:13: replace + with - in log_sum_exp in 2.5s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:368:52: replace - with + in CPUQuantizer::dequantize_tl1 in 2.7s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:156:35: replace - with + in AccuracyReport::update_errors in 3.9s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:186:29: replace - with + in AccuracyReport::calculate_std in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:543:5: replace log_sum_exp -> f64 with -1.0 in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:104:9: replace LookupTable::dequantize -> f32 with 1.0 in 2.5s build + 1.0s test
MISSED   crates/bitnet-quantization/src/tl2.rs:100:9: replace VectorizedLookupTable::dequantize -> f32 with 0.0 in 3.2s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl1.rs:98:9: replace LookupTable::quantize -> i8 with 1 in 3.1s build + 0.9s test
MISSED   crates/bitnet-quantization/src/lib.rs:191:21: replace == with != in convert_quantization in 2.9s build + 1.3s test
MISSED   crates/bitnet-quantization/src/tl2.rs:342:38: replace * with + in TL2Quantizer::quantize_cuda in 5.6s build + 1.1s test
MISSED   crates/bitnet-quantization/src/tl2.rs:93:33: replace / with % in VectorizedLookupTable::quantize in 4.5s build + 1.0s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:529:36: replace && with || in ReferenceCalculator::calculate_perplexity in 3.3s build + 0.9s test
MISSED   crates/bitnet-quantization/src/i2s.rs:373:9: replace I2SQuantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![1.0]) in 3.1s build + 1.0s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:526:27: replace * with + in ReferenceCalculator::calculate_perplexity in 3.5s build + 1.0s test
MISSED   crates/bitnet-quantization/src/i2s.rs:165:9: replace I2SQuantizer::quantize_scalar -> Result<Vec<i8>> with Ok(vec![]) in 3.8s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:411:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![0.0]) in 3.9s build + 1.0s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:420:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![-1.0]) in 4.1s build + 1.2s test
MISSED   crates/bitnet-quantization/src/i2s.rs:227:9: replace I2SQuantizer::quantize_simd -> Result<Vec<i8>> with Ok(vec![1]) in 3.6s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:531:45: replace - with + in ReferenceCalculator::calculate_perplexity in 3.2s build + 0.8s test
MISSED   crates/bitnet-quantization/src/utils.rs:80:22: replace * with / in dequantize_value in 3.2s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:456:9: replace TL2Quantizer::dequantize_avx2 -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.7s build + 0.9s test
MISSED   crates/bitnet-quantization/src/tl2.rs:211:29: replace * with + in TL2Quantizer::get_lookup_table in 2.8s build + 0.9s test
MISSED   crates/bitnet-quantization/src/utils.rs:71:5: replace quantize_value -> i8 with 1 in 3.4s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl1.rs:327:38: replace - with + in TL1Quantizer::dequantize_scalar in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:456:9: replace TL2Quantizer::dequantize_avx2 -> Result<Vec<f32>> with Ok(vec![]) in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:71:31: replace - with + in quantize_value in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:72:32: replace - with / in quantize_value in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:43:38: replace * with / in I2SLayout::with_block_size in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:308:42: replace >= with < in CPUQuantizer::dequantize_i2s in 3.7s build + 1.0s test
MISSED   crates/bitnet-quantization/src/utils.rs:80:5: replace dequantize_value -> f32 with 1.0 in 3.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:85:16: replace != with == in calculate_mse in 2.4s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:211:44: replace / with % in TL2Quantizer::get_lookup_table in 2.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:544:59: replace - with + in log_sum_exp in 2.5s build + 0.7s test
MISSED   crates/bitnet-quantization/src/i2s.rs:144:9: replace I2SQuantizer::quantize_cuda -> Result<QuantizedTensor> with Ok(Default::default()) in 4.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:186:13: replace / with * in AccuracyReport::calculate_std in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:681:67: replace / with % in DeviceAwareQuantizer::validate_gpu_cpu_parity in 2.8s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:333:57: replace / with * in CPUQuantizer::quantize_tl1 in 3.0s build + 1.0s test
MISSED   crates/bitnet-quantization/src/i2s.rs:373:9: replace I2SQuantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![]) in 2.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:368:45: replace / with * in CPUQuantizer::dequantize_tl1 in 8.7s build + 3.4s test
MISSED   crates/bitnet-quantization/src/utils.rs:24:26: replace + with * in calculate_grouped_scales in 4.4s build + 2.2s test
MISSED   crates/bitnet-quantization/src/tl2.rs:451:9: replace TL2Quantizer::quantize_avx2 -> Result<Vec<i8>> with Ok(vec![0]) in 2.2s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:257:50: replace * with / in CPUQuantizer::quantize_i2s in 6.0s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:92:56: replace - with / in calculate_mse in 2.5s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:407:9: replace TL1Quantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![-1.0]) in 9.8s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:411:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.8s build + 7.0s test
MISSED   crates/bitnet-quantization/src/tl2.rs:281:35: replace match guard self.config.use_avx512 with true in TL2Quantizer::quantize in 3.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:334:70: replace * with + in CPUQuantizer::quantize_tl1 in 2.9s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:411:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.4s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:93:33: replace / with * in VectorizedLookupTable::quantize in 2.9s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl1.rs:104:9: replace LookupTable::dequantize -> f32 with 0.0 in 2.8s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:296:64: replace & with | in CPUQuantizer::dequantize_i2s in 6.4s build + 2.8s test
MISSED   crates/bitnet-quantization/src/utils.rs:99:5: replace calculate_snr -> Result<f32> with Ok(0.0) in 8.7s build + 1.1s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:243:47: replace > with >= in CPUQuantizer::quantize_i2s in 5.8s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:282:33: replace match guard self.config.use_avx2 with false in TL2Quantizer::quantize in 2.3s build + 0.9s test
MISSED   crates/bitnet-quantization/src/i2s.rs:353:9: replace I2SQuantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![1]) in 3.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/i2s.rs:183:9: replace I2SQuantizer::dequantize_scalar -> Result<Vec<f32>> with Ok(vec![]) in 6.8s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:573:9: replace DeviceAwareQuantizer::auto_detect -> bitnet_common::Result<Self> with Ok(Default::default()) in 3.0s build + 6.5s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:305:61: replace * with + in CPUQuantizer::dequantize_i2s in 2.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:411:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![]) in 4.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:536:33: replace / with * in ReferenceCalculator::calculate_perplexity in 2.6s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:159:27: replace > with < in AccuracyReport::update_errors in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:211:29: replace * with / in TL2Quantizer::get_lookup_table in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:102:20: replace == with != in calculate_snr in 4.2s build + 1.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:530:49: replace + with * in ReferenceCalculator::calculate_perplexity in 2.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:273:9: replace CPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.8s build + 2.1s test
MISSED   crates/bitnet-quantization/src/tl2.rs:211:48: replace - with + in TL2Quantizer::get_lookup_table in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:222:49: replace - with / in TL2Quantizer::get_or_create_lookup_table in 2.5s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:160:43: replace / with % in AccuracyReport::update_errors in 2.6s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:243:47: replace > with == in CPUQuantizer::quantize_i2s in 2.5s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:228:9: replace TL2Quantizer::from_ini_file -> Result<Self> with Ok(Default::default()) in 2.6s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:364:30: replace + with * in CPUQuantizer::dequantize_tl1 in 2.5s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:71:5: replace quantize_value -> i8 with 0 in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:287:40: replace * with / in CPUQuantizer::dequantize_i2s in 3.0s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:536:33: replace / with % in ReferenceCalculator::calculate_perplexity in 2.7s build + 1.0s test
MISSED   crates/bitnet-quantization/src/utils.rs:92:83: replace / with * in calculate_mse in 3.3s build + 1.1s test
MISSED   crates/bitnet-quantization/src/tl1.rs:134:9: replace TL1Quantizer::from_ini_file -> Result<Self> with Ok(Default::default()) in 4.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:101:18: replace < with > in VectorizedLookupTable::dequantize in 2.8s build + 1.1s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:531:45: replace - with / in ReferenceCalculator::calculate_perplexity in 3.3s build + 1.2s test
MISSED   crates/bitnet-quantization/src/tl2.rs:222:29: replace / with * in TL2Quantizer::get_or_create_lookup_table in 4.0s build + 1.2s test
MISSED   crates/bitnet-quantization/src/tl1.rs:371:12: delete ! in TL1Quantizer::dequantize_neon in 3.3s build + 1.0s test
MISSED   crates/bitnet-quantization/src/tl2.rs:456:9: replace TL2Quantizer::dequantize_avx2 -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:461:9: replace TL2Quantizer::quantize_avx512 -> Result<Vec<i8>> with Ok(vec![-1]) in 2.3s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:296:46: replace >> with << in CPUQuantizer::dequantize_i2s in 2.5s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:349:9: replace CPUQuantizer::dequantize_tl1 -> Result<Vec<f32>> with Ok(vec![0.0]) in 2.3s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:543:5: replace log_sum_exp -> f64 with 1.0 in 3.8s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:257:28: replace |= with &= in CPUQuantizer::quantize_i2s in 4.8s build + 0.7s test
MISSED   crates/bitnet-quantization/src/i2s.rs:353:9: replace I2SQuantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![]) in 2.2s build + 0.5s test
MISSED   crates/bitnet-quantization/src/tl2.rs:466:9: replace TL2Quantizer::dequantize_avx512 -> Result<Vec<f32>> with Ok(vec![]) in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:529:20: replace <= with > in ReferenceCalculator::calculate_perplexity in 2.8s build + 0.5s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:245:40: delete - in CPUQuantizer::quantize_i2s in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:333:57: replace / with % in CPUQuantizer::quantize_tl1 in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:101:18: replace < with <= in VectorizedLookupTable::dequantize in 2.3s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:85:5: replace calculate_mse -> Result<f32> with Ok(-1.0) in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl1.rs:346:9: replace TL1Quantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![]) in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:333:43: replace > with >= in CPUQuantizer::quantize_tl1 in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:363:35: replace * with / in CPUQuantizer::dequantize_tl1 in 2.8s build + 1.0s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:242:43: replace > with < in CPUQuantizer::quantize_i2s in 3.7s build + 1.1s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:180:9: replace AccuracyReport::calculate_std -> f64 with -1.0 in 3.7s build + 1.0s test
MISSED   crates/bitnet-quantization/src/tl1.rs:98:29: replace / with % in LookupTable::quantize in 3.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl1.rs:264:38: replace * with + in TL1Quantizer::quantize_cuda in 3.5s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:180:25: replace < with == in AccuracyReport::calculate_std in 3.1s build + 3.1s test
MISSED   crates/bitnet-quantization/src/tl2.rs:466:9: replace TL2Quantizer::dequantize_avx512 -> Result<Vec<f32>> with Ok(vec![-1.0]) in 3.0s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:669:34: replace - with / in DeviceAwareQuantizer::validate_gpu_cpu_parity in 5.8s build + 1.5s test
MISSED   crates/bitnet-quantization/src/tl2.rs:221:28: replace << with >> in TL2Quantizer::get_or_create_lookup_table in 3.3s build + 0.9s test
MISSED   crates/bitnet-quantization/src/tl2.rs:376:9: replace TL2Quantizer::dequantize_scalar -> Result<Vec<f32>> with Ok(vec![-1.0]) in 3.4s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:369:50: replace * with / in CPUQuantizer::dequantize_tl1 in 3.9s build + 0.9s test
MISSED   crates/bitnet-quantization/src/utils.rs:106:29: replace / with * in calculate_snr in 4.7s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl1.rs:371:9: replace TL1Quantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![0.0]) in 3.8s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:242:83: replace == with != in TL2Quantizer::from_ini_file in 3.2s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:180:9: replace AccuracyReport::calculate_std -> f64 with 0.0 in 3.5s build + 0.9s test
MISSED   crates/bitnet-quantization/src/tl1.rs:346:12: delete ! in TL1Quantizer::quantize_neon in 2.9s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:420:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.7s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:211:48: replace - with / in TL2Quantizer::get_lookup_table in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:282:33: replace match guard self.config.use_avx2 with true in TL2Quantizer::quantize in 2.8s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:394:9: replace GPUQuantizer::quantize_i2s -> Result<QuantizedTensor> with Ok(Default::default()) in 2.6s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:298:25: delete match arm 0 in CPUQuantizer::dequantize_i2s in 3.0s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:532:27: replace -= with += in ReferenceCalculator::calculate_perplexity in 2.8s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:184:54: replace / with % in AccuracyReport::calculate_std in 2.5s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:317:35: replace match guard self.config.use_avx512 with true in TL2Quantizer::dequantize in 2.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:99:5: replace calculate_snr -> Result<f32> with Ok(1.0) in 3.6s build + 1.5s test
MISSED   crates/bitnet-quantization/src/tl2.rs:417:12: delete ! in TL2Quantizer::dequantize_avx2 in 10.1s build + 2.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:281:35: replace match guard self.config.use_avx512 with false in TL2Quantizer::quantize in 14.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:349:9: replace CPUQuantizer::dequantize_tl1 -> Result<Vec<f32>> with Ok(vec![]) in 3.0s build + 1.0s test
MISSED   crates/bitnet-quantization/src/utils.rs:74:28: replace / with * in quantize_value in 4.6s build + 0.9s test
MISSED   crates/bitnet-quantization/src/tl1.rs:346:9: replace TL1Quantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![1]) in 3.0s build + 0.9s test
MISSED   crates/bitnet-quantization/src/tl2.rs:100:9: replace VectorizedLookupTable::dequantize -> f32 with 1.0 in 7.9s build + 5.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:376:9: replace TL2Quantizer::dequantize_scalar -> Result<Vec<f32>> with Ok(vec![0.0]) in 3.3s build + 1.3s test
MISSED   crates/bitnet-quantization/src/tl2.rs:320:33: replace match guard self.config.use_avx2 with false in TL2Quantizer::dequantize in 4.2s build + 1.0s test
MISSED   crates/bitnet-quantization/src/tl2.rs:358:9: replace TL2Quantizer::quantize_scalar -> Result<Vec<i8>> with Ok(vec![-1]) in 5.1s build + 1.4s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:334:63: replace + with - in CPUQuantizer::quantize_tl1 in 2.1s build + 1.0s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:169:73: replace / with * in AccuracyReport::update_errors in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:257:50: replace * with + in CPUQuantizer::quantize_i2s in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:644:9: replace DeviceAwareQuantizer::validate_gpu_cpu_parity -> Result<ParityReport> with Ok(Default::default()) in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:93:46: replace * with / in VectorizedLookupTable::quantize in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:166:74: replace / with * in AccuracyReport::update_errors in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:529:57: replace < with == in ReferenceCalculator::calculate_perplexity in 2.3s build + 0.7s test
MISSED   crates/bitnet-quantization/src/lib.rs:182:9: replace QuantizerTrait::is_available -> bool with false in 2.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:545:13: replace + with * in log_sum_exp in 2.5s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:518:9: replace ReferenceCalculator::calculate_perplexity -> f64 with 1.0 in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:257:36: replace & with | in CPUQuantizer::quantize_i2s in 2.5s build + 1.0s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:349:9: replace CPUQuantizer::dequantize_tl1 -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.5s build + 0.7s test
MISSED   crates/bitnet-quantization/src/i2s.rs:46:41: replace + with * in I2SLayout::with_block_size in 2.4s build + 0.9s test
MISSED   crates/bitnet-quantization/src/tl1.rs:162:12: delete ! in TL1Quantizer::quantize in 3.2s build + 0.9s test
MISSED   crates/bitnet-quantization/src/i2s.rs:258:12: delete ! in I2SQuantizer::dequantize_avx2 in 3.4s build + 1.3s test
MISSED   crates/bitnet-quantization/src/i2s.rs:373:9: replace I2SQuantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![0.0]) in 3.6s build + 1.0s test
MISSED   crates/bitnet-quantization/src/tl1.rs:148:87: replace == with != in TL1Quantizer::from_ini_file in 3.8s build + 1.5s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:172:43: replace <= with > in AccuracyReport::update_errors in 3.4s build + 1.0s test
MISSED   crates/bitnet-quantization/src/i2s.rs:238:12: delete ! in I2SQuantizer::quantize_avx2 in 2.8s build + 1.0s test
MISSED   crates/bitnet-quantization/src/utils.rs:114:47: replace != with == in extract_f32_data in 5.1s build + 2.1s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:257:36: replace & with ^ in CPUQuantizer::quantize_i2s in 3.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:85:5: replace calculate_mse -> Result<f32> with Ok(0.0) in 4.6s build + 2.4s test
MISSED   crates/bitnet-quantization/src/tl2.rs:358:9: replace TL2Quantizer::quantize_scalar -> Result<Vec<i8>> with Ok(vec![0]) in 11.2s build + 1.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:334:63: replace + with * in CPUQuantizer::quantize_tl1 in 4.0s build + 1.1s test
MISSED   crates/bitnet-quantization/src/tl1.rs:397:9: replace TL1Quantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![-1]) in 8.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:106:13: replace * with / in calculate_snr in 2.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:529:57: replace < with > in ReferenceCalculator::calculate_perplexity in 2.2s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:260:12: delete ! in TL2Quantizer::quantize in 2.5s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:597:13: delete match arm QuantizationType::TL2 in DeviceAwareQuantizer::quantize_with_validation in 2.3s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:222:44: replace / with % in TL2Quantizer::get_or_create_lookup_table in 3.1s build + 2.5s test
MISSED   crates/bitnet-quantization/src/tl1.rs:537:9: replace <impl QuantizerTrait for TL1Quantizer>::is_available -> bool with false in 2.7s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:186:29: replace - with / in AccuracyReport::calculate_std in 2.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:186:13: replace / with % in AccuracyReport::calculate_std in 3.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:333:43: replace > with < in CPUQuantizer::quantize_tl1 in 3.4s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:71:5: replace quantize_value -> i8 with -1 in 2.6s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:420:9: replace GPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![0.0]) in 2.5s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:287:40: replace * with + in CPUQuantizer::dequantize_i2s in 3.1s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl2.rs:222:44: replace / with * in TL2Quantizer::get_or_create_lookup_table in 3.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:149:50: replace && with || in CpuFeatures::best_kernel in 3.4s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:147:9: replace AccuracyReport::update_errors with () in 6.3s build + 0.9s test
MISSED   crates/bitnet-quantization/src/utils.rs:74:28: replace / with % in quantize_value in 3.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:290:42: replace >= with < in CPUQuantizer::dequantize_i2s in 4.8s build + 3.3s test
MISSED   crates/bitnet-quantization/src/tl1.rs:104:9: replace LookupTable::dequantize -> f32 with -1.0 in 5.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:376:9: replace TL2Quantizer::dequantize_scalar -> Result<Vec<f32>> with Ok(vec![]) in 6.3s build + 8.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:245:38: replace < with > in CPUQuantizer::quantize_i2s in 3.0s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:296:64: replace & with ^ in CPUQuantizer::dequantize_i2s in 7.2s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:300:30: delete - in CPUQuantizer::dequantize_i2s in 2.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/utils.rs:85:5: replace calculate_mse -> Result<f32> with Ok(1.0) in 2.7s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl1.rs:105:18: replace < with <= in LookupTable::dequantize in 2.3s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl1.rs:407:9: replace TL1Quantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![0.0]) in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:38:9: replace <impl std::fmt::Display for QuantizationType>::fmt -> std::fmt::Result with Ok(Default::default()) in 2.3s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:334:70: replace * with / in CPUQuantizer::quantize_tl1 in 2.2s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:250:89: replace == with != in TL2Quantizer::from_ini_file in 2.6s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:526:27: replace * with / in ReferenceCalculator::calculate_perplexity in 9.7s build + 0.5s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:273:9: replace CPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.6s build + 0.5s test
MISSED   crates/bitnet-quantization/src/utils.rs:99:73: replace / with * in calculate_snr in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/tl2.rs:149:29: replace && with || in CpuFeatures::best_kernel in 3.0s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:294:53: replace + with * in CPUQuantizer::dequantize_i2s in 2.0s build + 6.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:180:9: replace AccuracyReport::calculate_std -> f64 with 1.0 in 2.4s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:156:35: replace - with / in AccuracyReport::update_errors in 2.7s build + 0.8s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:185:57: replace - with + in AccuracyReport::calculate_std in 4.1s build + 1.0s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:242:43: replace > with >= in CPUQuantizer::quantize_i2s in 2.8s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:299:25: delete match arm 1 in CPUQuantizer::dequantize_i2s in 2.4s build + 0.8s test
MISSED   crates/bitnet-quantization/src/tl2.rs:466:9: replace TL2Quantizer::dequantize_avx512 -> Result<Vec<f32>> with Ok(vec![0.0]) in 2.2s build + 0.6s test
MISSED   crates/bitnet-quantization/src/utils.rs:80:5: replace dequantize_value -> f32 with -1.0 in 2.7s build + 0.9s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:522:39: replace / with * in ReferenceCalculator::calculate_perplexity in 3.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:180:25: replace < with > in AccuracyReport::calculate_std in 2.7s build + 0.8s test
MISSED   crates/bitnet-quantization/src/utils.rs:23:23: replace * with / in calculate_grouped_scales in 2.1s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:232:9: replace I2SQuantizer::dequantize_simd -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.6s build + 0.7s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:245:38: replace < with <= in CPUQuantizer::quantize_i2s in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:46:41: replace + with - in I2SLayout::with_block_size in 2.0s build + 0.5s test
MISSED   crates/bitnet-quantization/src/tl2.rs:461:9: replace TL2Quantizer::quantize_avx512 -> Result<Vec<i8>> with Ok(vec![0]) in 1.7s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:232:9: replace I2SQuantizer::dequantize_simd -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:242:57: replace / with % in CPUQuantizer::quantize_i2s in 1.7s build + 0.7s test
MISSED   crates/bitnet-quantization/src/tl1.rs:407:9: replace TL1Quantizer::dequantize_neon -> Result<Vec<f32>> with Ok(vec![1.0]) in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/i2s.rs:353:9: replace I2SQuantizer::quantize_neon -> Result<Vec<i8>> with Ok(vec![0]) in 1.9s build + 0.6s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:245:38: replace < with == in CPUQuantizer::quantize_i2s in 1.9s build + 0.5s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:273:9: replace CPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![0.0]) in 1.8s build + 0.5s test
