# How to Deploy BitNet.rs

This guide provides instructions for deploying the `BitNet.rs` applications in various environments, from local execution to containerized and cloud-based setups.

## Deployment Options

`BitNet.rs` offers several ways to deploy and run its inference engine:

1.  **Command-Line Interface (CLI):** For direct, one-off inference tasks.
2.  **Standalone HTTP Server:** For providing a persistent, network-accessible inference service.
3.  **Docker Containers:** For portable, isolated deployments.
4.  **Kubernetes:** For scalable, orchestrated deployments in a cluster environment.
5.  **Cloud Platforms:** With specific guides for AWS, GCP, and Azure.

---

## 1. Using the Command-Line Interface (CLI)

The `bitnet-cli` application is the simplest way to run inference directly from your terminal. It is suitable for scripting, testing, and single inference requests.

### Prerequisites

- You have either [built the project from source](./building.md) or downloaded a pre-built binary.
- You have a compatible model file (e.g., in GGUF format).

### Example Usage

To run inference on a prompt, use the `infer` subcommand:

```bash
# Ensure the bitnet-cli binary is in your PATH or run it from the target directory
./target/release/bitnet-cli infer \
  --model /path/to/your/model.gguf \
  --prompt "Explain quantum computing in simple terms."
```

The CLI also includes tools for benchmarking, model conversion, and more. Use `--help` to see all available commands:

```bash
./target/release/bitnet-cli --help
```

---

## 2. Running the Standalone HTTP Server

The `bitnet-server` application exposes the inference engine as an HTTP service with an OpenAI-compatible API. This is ideal for integrating `BitNet.rs` with other applications over a network.

### Example Usage

1.  **Start the server:**
    Point the server to a model file and specify a port.

    ```bash
    ./target/release/bitnet-server \
      --port 8080 \
      --model /path/to/your/model.gguf
    ```

2.  **Test the server:**
    You can send requests to the `/v1/completions` endpoint using `curl` or any other HTTP client.

    ```bash
    curl -X POST http://localhost:8080/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Hello, world!", "max_tokens": 50}'
    ```

---

## 3. Using Docker

Docker provides a convenient way to run `BitNet.rs` in a containerized environment, ensuring consistency and isolation. Pre-built images are available from the GitHub Container Registry (`ghcr.io`).

### Running the Pre-built Image

This command pulls the latest image and starts an interactive session:

```bash
docker run --rm -it ghcr.io/microsoft/bitnet:latest
```

To run the server with a local model mounted into the container:

```bash
docker run --rm -it -p 8080:8080 \
  -v /path/to/your/models:/models \
  ghcr.io/microsoft/bitnet:latest \
  bitnet-server --port 8080 --model /models/model.gguf
```

### Building Custom Images

The repository includes `Dockerfile` definitions in the `docker/` directory for building custom images (e.g., `Dockerfile.cpu`, `Dockerfile.gpu`). You can also use the `docker-compose.yml` files for more complex multi-service setups.

---

## 4. Deploying on Kubernetes

For scalable and resilient deployments, `BitNet.rs` can be deployed on a Kubernetes cluster. The repository includes manifests and a Helm chart to simplify this process.

### Kubernetes Manifests

The `k8s/` directory contains standard Kubernetes manifest files for various components:
- `deployment-cpu.yaml` / `deployment-gpu.yaml`: For deploying the application on CPU or GPU nodes.
- `service.yaml`: To expose the deployment as a network service.
- `hpa.yaml`: For configuring Horizontal Pod Autoscaling.
- And more for configuration, storage, and networking.

You can apply these manifests using `kubectl apply -f k8s/`.

### Helm Chart

For a more manageable deployment, a Helm chart is available in the `helm/bitnet/` directory. Helm helps you manage the lifecycle of Kubernetes applications.

To install the chart:

```bash
helm install my-bitnet-release helm/bitnet/ --values helm/bitnet/values.yaml
```

You can customize the deployment by modifying the `values.yaml` file.

---

## 5. Cloud Platform Deployments

The `deployment/` directory contains specific instructions and templates for deploying `BitNet.rs` on major cloud platforms:

- **`deployment/aws/`**: Contains guides and resources for deploying on Amazon Web Services.
- **`deployment/gcp/`**: Contains guides and resources for deploying on Google Cloud Platform.
- **`deployment/azure/`**: Contains guides and resources for deploying on Microsoft Azure.

Please refer to the `README.md` file within each directory for platform-specific instructions.
