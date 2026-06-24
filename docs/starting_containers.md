# Ollama Container Deployment Guide

This document outlines the usage, configuration, and key operational insights for the `start_ollama_container.sh` deployment script.

For now, we rely on a script to do start containers. We should migrate this to a compose file that starts along with the alisa backend.

## Overview

The script automates the deployment of an Ollama container (`docker.io/ollama/ollama:0.18.3`) using Podman. 

It handles container lifecycle management, resource allocation, port validation, and automated model creation from local source files.

---

## GPU Resource handling
* The container is configured to request all available NVIDIA GPUs using the `--device nvidia.com/gpu=all` flag.
* In case of concurrent requests, models will be loaded across all available GPUs

* **Podman Over Docker:** The script exclusively relies on `podman` commands for container execution and management; it will fail if Podman is not installed.
* **Hardware Acceleration:** The container is configured to request all available NVIDIA GPUs using the `--device nvidia.com/gpu=all` flag.
* **Dynamic Container Naming:** Container names are automatically generated based on the current configuration parameters, using the exact format: `ollama_np<NUM_PARALLEL>_mlm<MAX_LOADED_MODELS>_api<API_PORT>`.
* **Automated Model Provisioning:** The script scans the host directory `/ollama_models` for subdirectories containing a `Modelfile` to automatically build models within the container.
* **Persistent Storage:** Model data is persistently stored on the host at `/ollama_storage`, which is then mounted to `/root/.ollama` inside the container.

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

* the model_tag.txt is required by our script to build the models

NOTE: Please use the same naming convention whenever you add a new model.

## Ollama Built Models

Once you have downloaded the correct GGUF file and managed to obtain accurate `Modelfile` instructions, you must ask Ollama to create the model so it can be used for inference. 

For example, Ollama uses the following command to create a model: 
````ollama create <model-tag> -f <model_file_path>````

>model-tag is the name we use to actually refer to the model at inference time. It is also the name used by ollama when you run ollama list.
>the model-tag can be found in the corresponding model_tag.txt file

The resulting files that Ollama creates using the GGUF and Modelfile you provide are stored under: ```/models/ollama/ollama_storage```

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

* **Mandatory `model_tag.txt`:** The script will strictly skip model creation for any directory that contains a `Modelfile` but lacks a `model_tag.txt` file.
* **Port Conflicts Prevent Execution:** The script utilizes the `ss` command during pre-flight checks to verify if the requested API or Web ports are available; if they are in use, it will raise an error and exit before attempting deployment.
* **Updating Existing Models:** By default, the script skips model creation if a model already exists in the container. You must explicitly pass the `--force-recreate` flag if you have updated a `Modelfile` or a GGUF file and need those changes applied.
* **Existing Containers Replaced:** If a container with the same name already exists, the script will automatically replace it using the `--replace` flag.

---

## Command-Line Options

Execute the script using the format: `bash ./start_ollama_container.sh [OPTIONS]`.

| Option | Description | Default Value |
| :--- | :--- | :--- |
| `--num-parallel <n>` | Sets the `OLLAMA_NUM_PARALLEL` variable to handle concurrent requests. | 4 |
| `--max-loaded-models <n>` | Sets the `OLLAMA_MAX_LOADED_MODELS` variable. | 3 |
| `--api-port <port>` | Defines the host port mapped to the container's standard 11434 API port. | 11450 |
| `--web-port <port>` | Defines the host port mapped to the container's 8080 port for a web UI. | 3004 |
| `--enable-web-port <bool>` | Determines whether the web port is published. | false |
| `--force-recreate` | Removes and recreates models, overriding any existing ones. | false |
| `-h`, `--help` | Shows the help message and exits the script. | N/A |
