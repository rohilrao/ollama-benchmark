#!/bin/bash
set -e # stop on first error

# ==========================================
# Usage
# ==========================================
#
# bash ./start_ollama_container.sh [OPTIONS]
#
# Inspired from: https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests
# Options:
#   --num-parallel <n>          OLLAMA_NUM_PARALLEL (default: 4)
#   --max-loaded-models <n>     OLLAMA_MAX_LOADED_MODELS (default: 3)
#   --keep-alive <duration>     OLLAMA_KEEP_ALIVE (default: 5m) - how long a loaded model stays in VRAM after its last request. Accepts a duration like "5m", "30s", "1h", a plain number of seconds, or "-1" to keep models loaded indefinitely.
#   --max-queue <n>             OLLAMA_MAX_QUEUE (default: 512) - max number of requests that can queue before Ollama returns a 503
#   --context-length <n>        OLLAMA_CONTEXT_LENGTH (default: 32768) - default context window for models that don't explicitly set num_ctx in their Modelfile
#   --api-port <port>           Host port for the Ollama API (default: 11434) - raises an error if api port is already in use
#   --web-port <port>           Web ports are disable by default but could be enabled for a web UI like OllamaWebUI (default: 3000) - raises an error if web port is already in use
#   --enable-web-port <bool>    Whether to publish the web port: true|false (default: false)
#   --force-recreate            Remove and recreate any model that already exists (Required in case you updated the Modelfile or changed the GGUF)
#   -h, --help                  Show this help message and exit
#
# NOTE: Every parameter below has a built-in default defined in the "Defaults" section.
#       You can either edit those defaults directly in the script for a permanent change,
#       or override them per-run using the corresponding CLI flag without touching the script.
#
# ==========================================
# Defaults
# ==========================================

OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_KEEP_ALIVE="5m"
OLLAMA_MAX_QUEUE=512
OLLAMA_CONTEXT_LENGTH=32768
HOST_API_PORT=11434
HOST_WEB_PORT=3000
ENABLE_WEB_PORT=false
FORCE_RECREATE_MODELS=false

# ==========================================
# Configuration Variables
# ==========================================

BASE_DIR="/models/ollama/"
SRC_DIR="${BASE_DIR}/ollama_models"
API_WAIT_TIME=5 #script waits 5 seconds after starting the container to give Ollama container time to initialize 
OLLAMA_READY_TIMEOUT=60
OLLAMA_CONTAINER_IMAGE="docker.io/ollama/ollama:0.18.3"
OLLAMA_STORAGE_BIND="${BASE_DIR}/ollama_storage" # Persistent location to store ollama models (created from Modelfiles and GGUF files) -> will be mounted to /root/.ollama in the container

# ==========================================
# Helper Functions (defined early; needed by arg parsing)
# ==========================================

print_usage() {
    cat <<'EOF'
Usage: bash ./start_ollama_container.sh [OPTIONS]

Options:
  --num-parallel <n>          OLLAMA_NUM_PARALLEL (default: 4)
  --max-loaded-models <n>     OLLAMA_MAX_LOADED_MODELS (default: 3)
  --keep-alive <duration>     OLLAMA_KEEP_ALIVE (default: 5m) - how long a loaded model stays in VRAM after its last request. Accepts a duration like "5m", "30s", "1h", a plain number of seconds, or "-1" to keep models loaded indefinitely.
  --max-queue <n>              OLLAMA_MAX_QUEUE (default: 512) - max number of requests that can queue before Ollama returns a 503
  --context-length <n>         OLLAMA_CONTEXT_LENGTH (default: 4096) - default context window for models that don't explicitly set num_ctx in their Modelfile
  --api-port <port>           Host port for the Ollama API (default: 11434 or whatever is specifed in the script) - raises an error if api port is already in use
  --web-port <port>           Host port for the web UI (default: 3004) - raises an error if web port is already in use
  --enable-web-port <bool>    Whether to publish the web port: true|false (default: false)
  --force-recreate            Remove and recreate any model that already exists (Required in case you updated the Modelfile or changed the GGUF)
  -h, --help                  Show this help message and exit

NOTE: Every parameter above has a built-in default defined near the top of the script.
      You can either edit those defaults directly in the script for a permanent change,
      or override them per-run using the corresponding CLI flag without touching the script.
EOF
}

is_positive_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_bool() {
    [ "$1" = "true" ] || [ "$1" = "false" ]
}

is_valid_keep_alive() {
    # Accepts: -1 (keep loaded indefinitely), a plain number of seconds (e.g. 300),
    # or a duration string like 30s, 5m, 1h
    [[ "$1" =~ ^-1$ ]] || [[ "$1" =~ ^[0-9]+(s|m|h)?$ ]]
}

require_value() {
    local flag="$1"
    local value="$2"

    if [ -z "$value" ] || [[ "$value" == --* ]]; then
        echo "ERROR: ${flag} requires a value." >&2
        print_usage
        exit 1
    fi
}

# ==========================================
# Argument Parsing
# ==========================================

while [ $# -gt 0 ]; do
    case "$1" in
        --num-parallel)
            require_value "$1" "$2"
            OLLAMA_NUM_PARALLEL="$2"
            shift 2
            ;;
        --max-loaded-models)
            require_value "$1" "$2"
            OLLAMA_MAX_LOADED_MODELS="$2"
            shift 2
            ;;
        --keep-alive)
            require_value "$1" "$2"
            OLLAMA_KEEP_ALIVE="$2"
            shift 2
            ;;
        --max-queue)
            require_value "$1" "$2"
            OLLAMA_MAX_QUEUE="$2"
            shift 2
            ;;
        --context-length)
            require_value "$1" "$2"
            OLLAMA_CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --api-port)
            require_value "$1" "$2"
            HOST_API_PORT="$2"
            shift 2
            ;;
        --web-port)
            require_value "$1" "$2"
            HOST_WEB_PORT="$2"
            shift 2
            ;;
        --enable-web-port)
            require_value "$1" "$2"
            ENABLE_WEB_PORT="$2"
            shift 2
            ;;
        --force-recreate)
            FORCE_RECREATE_MODELS=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

# ==========================================
# Argument Validation
# ==========================================

if ! is_positive_int "$OLLAMA_NUM_PARALLEL"; then
    echo "ERROR: --num-parallel must be a positive integer" >&2
    exit 1
fi
if ! is_positive_int "$OLLAMA_MAX_LOADED_MODELS"; then
    echo "ERROR: --max-loaded-models must be a positive integer" >&2
    exit 1
fi
if ! is_valid_keep_alive "$OLLAMA_KEEP_ALIVE"; then
    echo "ERROR: --keep-alive must be -1, a plain number of seconds, or a duration like 30s/5m/1h" >&2
    exit 1
fi
if ! is_positive_int "$OLLAMA_MAX_QUEUE"; then
    echo "ERROR: --max-queue must be a positive integer" >&2
    exit 1
fi
if ! is_positive_int "$OLLAMA_CONTEXT_LENGTH"; then
    echo "ERROR: --context-length must be a positive integer" >&2
    exit 1
fi
if ! is_positive_int "$HOST_API_PORT"; then
    echo "ERROR: --api-port must be a positive integer" >&2
    exit 1
fi
if ! is_positive_int "$HOST_WEB_PORT"; then
    echo "ERROR: --web-port must be a positive integer" >&2
    exit 1
fi
if ! is_bool "$ENABLE_WEB_PORT"; then
    echo "ERROR: --enable-web-port must be 'true' or 'false'" >&2
    exit 1
fi

CONTAINER_NAME="ollama_np${OLLAMA_NUM_PARALLEL}_mlm${OLLAMA_MAX_LOADED_MODELS}_api${HOST_API_PORT}"

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

normalize_model_name() {
    local model_name="$1"
    if [[ "$model_name" != *":"* ]]; then
        echo "${model_name}:latest"
    else
        echo "$model_name"
    fi
}

model_exists() {
    local lookup_name="$1"
    podman exec "${CONTAINER_NAME}" ollama list \
        | awk 'NR>1 {print $1}' \
        | grep -qx "${lookup_name}"
}

wait_for_ollama() {
    echo "Waiting for Ollama to become ready..."
    local elapsed=0
    until podman exec "${CONTAINER_NAME}" ollama list >/dev/null 2>&1; do
        if [ "$elapsed" -ge "$OLLAMA_READY_TIMEOUT" ]; then
            echo "ERROR: Ollama did not become ready within ${OLLAMA_READY_TIMEOUT} seconds."
            exit 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "Ollama is ready."
}

create_or_reuse_model() {
    local tag_name="$1"
    local folder_name="$2"
    local modelfile_path="/root/models/${folder_name}/Modelfile"
    local lookup_name
    lookup_name=$(normalize_model_name "$tag_name")

    if model_exists "$lookup_name"; then
        if [ "$FORCE_RECREATE_MODELS" = true ]; then
            echo "Force recreate enabled. Removing existing model: ${lookup_name}"
            podman exec "${CONTAINER_NAME}" ollama rm "${lookup_name}" || true
            echo "Recreating model: ${tag_name} from ${folder_name}..."
            podman exec "${CONTAINER_NAME}" ollama create "${tag_name}" -f "${modelfile_path}"
        else
            echo "Model already exists: ${lookup_name}. Skipping create."
        fi
    else
        echo "Model missing. Creating model: ${tag_name} from ${folder_name}..."
        podman exec "${CONTAINER_NAME}" ollama create "${tag_name}" -f "${modelfile_path}"
    fi
}

# ==========================================
# 0. Pre-flight Checks
# ==========================================

echo ""
echo "##### Starting container ######"
echo "Running pre-flight checks..."

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: SRC_DIR does not exist: $SRC_DIR"
    exit 1
fi

if [ ! -d "$OLLAMA_STORAGE_BIND" ]; then
    echo "Creating storage directory: $OLLAMA_STORAGE_BIND"
    mkdir -p "$OLLAMA_STORAGE_BIND"
fi

if ! command -v podman >/dev/null 2>&1; then
    echo "ERROR: podman is not installed."
    exit 1
fi

if ! command -v ss >/dev/null 2>&1; then
    echo "WARNING: 'ss' command not found. Cannot verify ports."
else
    if port_in_use "$HOST_API_PORT"; then
        echo "ERROR: HOST_API_PORT ${HOST_API_PORT} is already in use."
        exit 1
    fi
    if [ "$ENABLE_WEB_PORT" = true ] && port_in_use "$HOST_WEB_PORT"; then
        echo "ERROR: HOST_WEB_PORT ${HOST_WEB_PORT} is already in use."
        exit 1
    fi
fi

if container_running || container_exists; then
    echo "NOTE: Container '${CONTAINER_NAME}' exists and will be replaced."
fi

# ==========================================
# 1. Start the Container
# ==========================================

echo ""
echo "Starting Ollama container: ${CONTAINER_NAME}"

PORT_ARGS=(-p "${HOST_API_PORT}:11434")
if [ "$ENABLE_WEB_PORT" = true ]; then
    PORT_ARGS+=(-p "${HOST_WEB_PORT}:8080")
fi

podman run -d \
    "${PORT_ARGS[@]}" \
    -e OLLAMA_HOST=0.0.0.0 \
    -e OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
    -e OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS}" \
    -e OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" \
    -e OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE}" \
    -e OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH}" \
    -v "${SRC_DIR}:/root/models:ro" \
    -v "${OLLAMA_STORAGE_BIND}:/root/.ollama" \
    --device nvidia.com/gpu=all \
    --name "${CONTAINER_NAME}" \
    --replace \
    "${OLLAMA_CONTAINER_IMAGE}"

# ==========================================
# 2. Wait for Service Initialization
# ==========================================

sleep "${API_WAIT_TIME}"
wait_for_ollama

# ==========================================
# 3. Create Missing Models / Reuse Existing Models
# ==========================================

echo ""
echo "Scanning for Modelfiles in ${SRC_DIR}..."

for model_dir in "${SRC_DIR}"/*/; do
    if [ -f "${model_dir}Modelfile" ]; then
        folder_name=$(basename "${model_dir}")
        
        # Look for the metadata file to determine the tag name
        if [ -f "${model_dir}model_tag.txt" ]; then
            # Read the file and strip any rogue spaces or carriage returns
            tag_name=$(cat "${model_dir}model_tag.txt" | tr -d '\r\n ')
            
            # Proceed with creation since the tag exists
            create_or_reuse_model "$tag_name" "$folder_name"
        else
            # Strictly skip creation if no model_tag.txt exists
            echo "WARNING: No model_tag.txt found in ${folder_name}. Skipping model creation."
        fi
    fi
done

# ==========================================
# 4. Done
# ==========================================

echo ""
echo "Deployment complete. API is available at http://localhost:${HOST_API_PORT}"
