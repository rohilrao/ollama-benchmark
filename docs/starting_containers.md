# Ollama Container Deployment Guide

This document outlines the usage, configuration, and key operational insights for the `start_ollama_container.sh` deployment script.

For now, we rely on a script to start containers. This could eventually move to a Compose file.

## Overview
* The `start_ollama_container.sh` script automates the deployment of an Ollama container (currently: `docker.io/ollama/ollama:0.18.3`) using Podman. 
* The container is configured to request all available NVIDIA GPUs using the `--device nvidia.com/gpu=all` flag. Ollama will dynamically use all available gpu memory to handle concurrent requests.
* Container names are automatically generated based on the current configuration parameters, using the exact format: `ollama_np<NUM_PARALLEL>_mlm<MAX_LOADED_MODELS>_api<API_PORT>`.
* The script scans the host directory `/ollama_models` for subdirectories containing a `Modelfile` to automatically build models within the container. (See: [Ollama model files and storage](#ollama-model-files-and-storage))
* Model data is persistently stored on the host at `/ollama_storage`, which is then mounted to `/root/.ollama` inside the container. (See: [Ollama Built Models](#ollama-built-models))

## Command-Line Options
 
Execute the script using the format: `./start_ollama_container.sh [OPTIONS]`.
 
| Option | Description | Default Value |
| :--- | :--- | :--- |
| `--num-parallel <n>` | Sets the `OLLAMA_NUM_PARALLEL` variable to handle concurrent requests. | 4 |
| `--max-loaded-models <n>` | Sets the `OLLAMA_MAX_LOADED_MODELS` variable. | 3 |
| `--keep-alive <duration>` | Sets the `OLLAMA_KEEP_ALIVE` variable — how long a loaded model stays in VRAM after its last request before being unloaded. Accepts a plain number of seconds (e.g. `300`), a duration string (`30s`, `5m`, `1h`), or `-1` to keep models loaded indefinitely. | `5m` |
| `--max-queue <n>` | Sets the `OLLAMA_MAX_QUEUE` variable — the maximum number of requests that can queue before Ollama returns a `503` instead of accepting more. | 512 |
| `--context-length <n>` | Sets the `OLLAMA_CONTEXT_LENGTH` variable — the default context window used for models that don't explicitly set `num_ctx` in their `Modelfile`. | 32768 |
| `--api-port <port>` | Defines the host port mapped to the container's standard 11434 API port. | 11434 |
| `--force-recreate` | A flag (no value). When passed, removes and recreates every model that already exists, overriding it with the current `Modelfile`/GGUF. When omitted, existing models are left as-is. | No value |
| `-h`, `--help` | Shows the help message and exits the script. | N/A |
 
> **Note:** Every parameter above has a built-in default defined near the top of the script, under the `Defaults` section. You can change behavior in one of two ways:
> 1. **Edit the defaults directly in the script** — useful for a permanent change that should apply every time the script is run without extra flags.
> 2. **Override per-run via the corresponding CLI flag** — useful for one-off or environment-specific deployments without touching the script itself.
>
> CLI flags always take precedence over the script's built-in defaults for that invocation.
>
> **Note on `--force-recreate`:** unlike the other options, this is a bare flag, not a `--flag <value>` option. Its presence on the command line turns it on. To leave it disabled, simply omit it.


## Server defaults vs. per-request overrides
 
The parameters set by this script act as **defaults** that in some cases (mentioned below) can be overridden by individual client API calls on a per-request basis.
 
* `OLLAMA_KEEP_ALIVE` can be overridden per-request via client code (`keep_alive=0`, `keep_alive=-1`, `keep_alive="10m"`, etc.); The `--keep-alive` value set by this script applies only to requests that don't specify their own `keep_alive`. There is no server-side ceiling, so any caller can override it per-request, including `keep_alive=-1` to pin a model in VRAM indefinitely. Client code SHOULD ABSOLUTELY avoid `keep_alive=-1` unless there's a deliberate reason a model needs to stay resident.
* `OLLAMA_CONTEXT_LENGTH` can also be overridden per-request (`options={"num_ctx": <n>}`), but only up to the model's actual maximum context length — a model's architecture and quantization define a hard ceiling, so requesting a larger `num_ctx` than the model supports won't work regardless of the server default.
* `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, and `OLLAMA_MAX_QUEUE` are fixed at container start, and CANNOT be overriden per-request.
 
Example of a per-request override in Python:
 
````python
import ollama
 
response = ollama.chat(
    model="qwen3:32b-q8",
    messages=[{"role": "user", "content": "Explain KV cache in 2 sentences."}],
    keep_alive=0,                # overrides OLLAMA_KEEP_ALIVE for this call only
    options={"num_ctx": 8192},   # overrides OLLAMA_CONTEXT_LENGTH for this call only
)
print(response["message"]["content"])
````
 
---

## Ollama model files and storage

* All Ollama models are stored under ```/models/ollama/ollama_models```

* /ollama-models folder structure
```text
/ollama_models
│
├── /Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M
│   ├── Modelfile
│   ├── model.gguf
│   └── model_tag.txt
│
└── /Qwen3-32B-Q8_0
    ├── Modelfile
    ├── model.gguf
    └── model_tag.txt
```

* Every folder for a model contains:
   - model.gguf (model weights)
   - Modelfile (model instructions - including system prompt, context length etc. - that can be overwritten at run time)
   - model_tag.txt (this is the name you use to actually call a model at runtime: for example: mistral-small3.2:24b)  

* All the three file types are required mandatorily by our script for creating a model.

NOTE: Please use the same naming convention whenever you add a new model.

## Ollama Built Models

Once you have downloaded the correct GGUF file and managed to obtain accurate `Modelfile` instructions, you must ask Ollama to create the model so it can be used for inference. 

For example, Ollama uses the following command to create a model: 
````ollama create <model-tag> -f <model_file_path>````

>model-tag is the name we use to actually refer to the model at inference time. It is also the name used by ollama when you run a command like ```ollama list``` (inside the ollama container).
>The model-tag can be found in the corresponding `model_tag.txt` file

The resulting files that Ollama creates are stored under: ```/models/ollama/ollama_storage```

* **Ollama's storage architecture** is heavily inspired by Docker containers. Instead of storing a model as one giant, monolithic file, it breaks it down into reusable layers. It creates the following types of data:
  * **Manifests:** A lightweight JSON blueprint that acts as the table of contents, linking a specific model name to its required data.
  * **Blobs (Binary Large Objects):** The actual immutable data files (like the heavy model weights, system prompts, and templates). Because they are split into layers, multiple models can share the same base weights without duplicating disk space.

**Note:** Model creation can take upto a couple of minutes per model. If you are deploying many models, this process can take much longer. Therefore, the script persistently stores all the created blobs and manifests on the servers `ollama_storage` folder, mounting it as a volume to the container. This allows the container to access the built models directly upon restart without having to rebuild them from scratch every time.

**⚠️Important:** Any changes to a `Modelfile`, `GGUF` file or `model_tag.txt` file require rebuilding the model blobs. In this scenario, you must stop the existing Ollama container and re-run the deployment script using the `--force-recreate` flag. This will remove the outdated existing model and recreate it from scratch with your new changes.

**Workflow to apply model changes:**

1. Stop the running container:
```podman stop <container_name>```
(Note: You can find your exact container name by running podman ps)

2. Re-run the deployment script with the force flag:
bash ./start_ollama_container.sh --force-recreate

## Important Notes

* **Mandatory `model_tag.txt`:** The script will strictly skip model creation for any directory that contains a `Modelfile`,`gguf` file but lacks a `model_tag.txt` file.
* **Port Conflicts Prevent Execution:** The script utilizes the `ss` command during pre-flight checks to verify if the requested API port is available; if it is already in use, it will raise an error and exit before attempting deployment.
* **Updating Existing Models:** By default, the script skips model creation if a model already exists in the container. You must explicitly pass the `--force-recreate` flag if you have updated a `Modelfile` or a GGUF file and need those changes applied.
* **Existing Containers Replaced:** If a container with the same name already exists, the script will automatically replace it using the `--replace` flag.

---

## Container-side path mapping

The host directory `/ollama_models` is bind-mounted **read-only** into the container at `/root/models` (`-v "${SRC_DIR}:/root/models:ro"`). This means every `Modelfile` referenced during model creation is actually read from inside the container at:

```
/root/models/<model_folder_name>/Modelfile
```

For example, the `Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M` folder on the host becomes:

```
/root/models/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M/Modelfile
```

This is the exact path the script passes to `ollama create -f <path>` when building each model. If you ever need to debug model creation manually by shelling into the container (`podman exec -it <container_name> bash`), use this `/root/models/...` path, not the host path `/ollama_models/...` — the host path doesn't exist inside the container.

The host's persistent storage directory `/ollama_storage` is mounted the same way, but **read-write** and at a different container path: `/root/.ollama` (`-v "${OLLAMA_STORAGE_BIND}:/root/.ollama"`). This is where Ollama keeps the manifests and blobs for every model it builds. So when debugging inside the container, the two relevant mappings are:

| Host path | Container path | Mode |
| :--- | :--- | :--- |
| `/ollama_models` | `/root/models` | read-only |
| `/ollama_storage` | `/root/.ollama` | read-write |

Knowing both mappings is useful if you ever need to manually inspect built models (e.g. `ls /root/.ollama/models/manifests` inside the container) or confirm that a model's source files are visible where the script expects them.
