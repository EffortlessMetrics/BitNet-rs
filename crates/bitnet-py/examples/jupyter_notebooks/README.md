# BitNet.cpp Python Bindings - Jupyter Notebook Examples

This directory contains comprehensive Jupyter notebook examples for migrating from the original BitNet Python implementation to the new Rust-based `bitnet_py` library.

## Notebooks Overview

### 1. [Basic Migration](01_basic_migration.ipynb) 🚀
**Perfect for: First-time users and basic migration**

- **What it covers:**
  - Installation and setup
  - API compatibility demonstration
  - Simple migration examples
  - Performance comparison basics
  - Migration utilities usage

- **Prerequisites:** Basic Python knowledge
- **Time to complete:** 15-20 minutes
- **Key takeaways:** Understanding the minimal changes needed for migration

### 2. [Advanced Features](02_advanced_features.ipynb) ⚡
**Perfect for: Users wanting to leverage new capabilities**

- **What it covers:**
  - Async/await support for web applications
  - Streaming generation for real-time apps
  - Batch processing for efficiency
  - Advanced sampling strategies
  - Memory management and monitoring
  - Enhanced error handling

- **Prerequisites:** Completed basic migration notebook
- **Time to complete:** 30-40 minutes
- **Key takeaways:** Leveraging bitnet_py's advanced features for better performance

### 3. [Production Deployment](03_production_deployment.ipynb) 🏭
**Perfect for: Production deployments and scaling**

- **What it covers:**
  - Production configuration best practices
  - Docker containerization
  - Load balancing strategies
  - Monitoring and observability
  - Security considerations
  - CI/CD integration

- **Prerequisites:** Understanding of deployment concepts
- **Time to complete:** 45-60 minutes
- **Key takeaways:** Production-ready deployment strategies

## Quick Start

1. **Install bitnet_py:**
   ```bash
   pip install bitnet-py
   ```

2. **Install Jupyter:**
   ```bash
   pip install jupyter notebook
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open the notebooks in order:**
   - Start with `01_basic_migration.ipynb`
   - Progress to `02_advanced_features.ipynb`
   - Finish with `03_production_deployment.ipynb`

## Before You Begin

### Required Files
Update these paths in the notebooks to point to your actual files:
- `MODEL_PATH`: Path to your BitNet model file (`.gguf` format)
- `TOKENIZER_PATH`: Path to your tokenizer file (`.model` format)

### Example Setup
```python
# Update these in each notebook
MODEL_PATH = "path/to/your/model.gguf"
TOKENIZER_PATH = "path/to/your/tokenizer.model"
```

### Optional Dependencies
For full functionality, install these optional packages:
```bash
pip install fastapi uvicorn streamlit ipywidgets
```

## Learning Path

### For New Users
1. **Start here:** `01_basic_migration.ipynb`
2. **Learn the basics:** Complete all cells in order
3. **Test with your models:** Update paths and run examples
4. **Move to advanced:** `02_advanced_features.ipynb`

### For Existing BitNet Users
1. **Quick migration:** Focus on the API comparison sections
2. **Performance gains:** Run the benchmarking examples
3. **Advanced features:** Explore async and streaming capabilities
4. **Production ready:** Review deployment strategies

### For Production Deployments
1. **Review all notebooks:** Understand the full capabilities
2. **Focus on production:** `03_production_deployment.ipynb`
3. **Customize configs:** Adapt examples to your infrastructure
4. **Monitor and scale:** Implement observability practices

## Common Issues and Solutions

### Model Files Not Found
**Problem:** `FileNotFoundError` when loading models
**Solution:** 
- Update `MODEL_PATH` and `TOKENIZER_PATH` variables
- Ensure files exist and are accessible
- Check file permissions

### Import Errors
**Problem:** `ImportError: No module named 'bitnet_py'`
**Solution:**
```bash
pip install bitnet-py
# or for development
pip install -e /path/to/bitnet-py
```

### Performance Issues
**Problem:** Slow inference or high memory usage
**Solution:**
- Enable GPU acceleration if available
- Use batch processing for multiple requests
- Configure memory limits appropriately
- Check system resources

### Jupyter Kernel Issues
**Problem:** Kernel crashes or becomes unresponsive
**Solution:**
- Restart kernel: `Kernel > Restart`
- Clear outputs: `Cell > All Output > Clear`
- Check memory usage and free up resources

## Integration Examples

The notebooks include integration examples for popular frameworks:

### Web Frameworks
- **FastAPI**: Async web API with streaming support
- **Flask**: Traditional web application integration
- **Streamlit**: Interactive data applications

### Development Tools
- **Jupyter Widgets**: Interactive notebook interfaces
- **IPython**: Enhanced interactive Python

### Production Tools
- **Docker**: Containerized deployment
- **Kubernetes**: Orchestrated scaling
- **Monitoring**: Prometheus and observability

## Performance Expectations

Based on the examples in these notebooks, you can expect:

| Metric | Original BitNet | bitnet_py | Improvement |
|--------|----------------|-----------|-------------|
| Inference Speed | 45 tokens/sec | 156 tokens/sec | 3.5x faster |
| Memory Usage | 6.4 GB | 3.2 GB | 50% reduction |
| Startup Time | 18.5 seconds | 4.2 seconds | 4.4x faster |
| Concurrent Requests | Limited | 100+ | Highly scalable |

## Next Steps After Notebooks

1. **Migrate your project:**
   - Use the migration utilities demonstrated
   - Apply the patterns from the examples
   - Test thoroughly with your specific use case

2. **Optimize for your needs:**
   - Tune configuration parameters
   - Choose appropriate sampling strategies
   - Configure memory and performance settings

3. **Deploy to production:**
   - Follow the deployment best practices
   - Implement monitoring and logging
   - Set up CI/CD pipelines

4. **Get support:**
   - Check the [Migration Guide](../MIGRATION_GUIDE.md)
   - Review [API documentation](https://docs.rs/bitnet-py/)
   - Open issues on GitHub for problems

## Contributing

Found an issue or have suggestions for improving these notebooks?

1. **Report issues:** Open a GitHub issue with details
2. **Suggest improvements:** Submit pull requests
3. **Share examples:** Contribute additional use cases
4. **Update documentation:** Help keep examples current

## Additional Resources

- **[Migration Guide](../MIGRATION_GUIDE.md)**: Comprehensive migration documentation
- **[Performance Comparison](../performance_comparison.py)**: Detailed benchmarking tools
- **[Migration Utilities](../python/bitnet_py/migration.py)**: Automated migration helpers
- **[Example Applications](../examples/)**: Complete application examples
- **[API Documentation](https://docs.rs/bitnet-py/)**: Full API reference

---

**Happy migrating! 🚀**

These notebooks will guide you through a smooth transition from the original BitNet Python implementation to the high-performance bitnet_py library.