#!/bin/bash
set -e

# ==========================================
# Configuration Variables
# ==========================================

SRC_DIR="ollama-models"

# Host ports
HOST_API_PORT="11440"
HOST_WEB_PORT="3004"
ENABLE_WEB_PORT=true

# Ollama Runtime Settings
OLLAMA_NUM_PARALLEL="8"
OLLAMA_MAX_LOADED_MODELS="3"

# Deployment Settings
API_WAIT_TIME=5

# Container image
OLLAMA_CONTAINER_IMAGE="docker.io/ollama/ollama:0.18.3"

# Persistent Ollama storage on host/server
OLLAMA_STORAGE_BIND="/ollama_storage"

# Name of your container
CONTAINER_NAME="ollama_np${OLLAMA_NUM_PARALLEL}_mlm${OLLAMA_MAX_LOADED_MODELS}"

# ==========================================
# Helper Functions
# ==========================================

port_in_use() {
    local port="$1"
    ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"
}

container_exists() {
    podman container exists "$CONTAINER_NAME"
}

container_running() {
    podman ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"
}

# ==========================================
# 0. Pre-flight Checks
# ==========================================

echo ""
echo "##### Starting container ######"
echo "Running pre-flight checks..."

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: SRC_DIR does not exist: $SRC_DIR"
    echo "Fix SRC_DIR before running the script."
    exit 1
fi

if [ ! -d "$OLLAMA_STORAGE_BIND" ]; then
    echo "Ollama storage directory does not exist. Creating: $OLLAMA_STORAGE_BIND"
    mkdir -p "$OLLAMA_STORAGE_BIND"
fi

if ! command -v podman >/dev/null 2>&1; then
    echo "ERROR: podman is not installed or not available in PATH."
    exit 1
fi

if ! command -v ss >/dev/null 2>&1; then
    echo "WARNING: 'ss' command not found. Cannot check whether ports are already in use."
else
    if port_in_use "$HOST_API_PORT"; then
        echo "ERROR: HOST_API_PORT ${HOST_API_PORT} is already in use."
        echo "Choose a different HOST_API_PORT or stop the process using it."
        exit 1
    fi

    if [ "$ENABLE_WEB_PORT" = true ] && port_in_use "$HOST_WEB_PORT"; then
        echo "ERROR: HOST_WEB_PORT ${HOST_WEB_PORT} is already in use."
        echo "Choose a different HOST_WEB_PORT or stop the process using it."
        exit 1
    fi
fi

if container_running; then
    echo "WARNING: A running container named '${CONTAINER_NAME}' already exists."
    echo "Because --replace is used, it will be replaced."
elif container_exists; then
    echo "WARNING: A stopped container named '${CONTAINER_NAME}' already exists."
    echo "Because --replace is used, it will be replaced."
fi

if ! find "$SRC_DIR" -mindepth 2 -maxdepth 2 -name "Modelfile" | grep -q .; then
    echo "WARNING: No Modelfile found inside subfolders of: $SRC_DIR"
    echo "Expected structure:"
    echo "$SRC_DIR/model-folder/Modelfile"
fi

echo "Pre-flight checks complete."

# ==========================================
# 1. Start the Container
# ==========================================
echo ""
echo "Starting Ollama container: ${CONTAINER_NAME}"
echo ""
echo "Mounting source models from: ${SRC_DIR}"
echo ""
echo "Using persistent Ollama storage folder: ${OLLAMA_STORAGE_BIND}"
echo ""
echo "Ollama API: http://localhost:${HOST_API_PORT}"
echo ""
echo "##### OLLAMA ENV VARIABLES: #####"
echo "OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
echo "OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"
echo "#################################"
echo ""

PORT_ARGS=(-p "${HOST_API_PORT}:11434")

if [ "$ENABLE_WEB_PORT" = true ]; then
    PORT_ARGS+=(-p "${HOST_WEB_PORT}:8080")
fi

podman run -d \
  "${PORT_ARGS[@]}" \
  -e OLLAMA_HOST=0.0.0.0 \
  -e OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
  -e OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS}" \
  -v "${SRC_DIR}:/root/models:ro" \
  -v "${OLLAMA_STORAGE_BIND}:/root/.ollama" \
  --device nvidia.com/gpu=all \
  --name "${CONTAINER_NAME}" \
  --replace \
  "${OLLAMA_CONTAINER_IMAGE}"

# ==========================================
# 2. Wait for Service Initialization
# ==========================================

echo "Waiting ${API_WAIT_TIME} seconds for Ollama to initialize..."
sleep "${API_WAIT_TIME}"


# ==========================================
# 3. Build the Models
# ==========================================

echo "Scanning for Modelfiles in ${SRC_DIR}..."

for model_dir in "${SRC_DIR}"/*/; do
    if [ -f "${model_dir}Modelfile" ]; then
        folder_name=$(basename "${model_dir}")

        case "$folder_name" in
            "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M")
                tag_name="mistral-small3.2:24b"
                ;;
            "Mistral-Small-3.2-24B-32K-Instruct-2506-Q4_K_M")
                tag_name="mistral-small3.2:24b-32k"
                ;;
            "mistral-7b-instruct-v0.3-q4_k_m")
                tag_name="mistral-v0.3"
                ;;
            "Qwen3-32B-Q8_0")
                tag_name="qwen3:32b-q8"
                ;;
            "Qwen3VL-32B-Instruct-Q4_K_M")
                tag_name="qwen3-vl-instruct"
                ;;
            *)
                tag_name=$(echo "${folder_name}" | tr '[:upper:]' '[:lower:]')
                ;;
        esac

        echo "Creating model: ${tag_name} from ${folder_name}..."

        podman exec "${CONTAINER_NAME}" \
          ollama create "${tag_name}" \
          -f "/root/models/${folder_name}/Modelfile"
    fi
done


echo "Deployment complete."
echo "Check models with:"
echo "podman exec ${CONTAINER_NAME} ollama list" 



