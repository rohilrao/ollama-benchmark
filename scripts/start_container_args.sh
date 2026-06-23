#!/bin/bash
set -e

# ==========================================
# Usage
# ==========================================
#
# bash ./deploy_ollama.sh [OPTIONS]
#
# Options:
#   --num-parallel <n>          OLLAMA_NUM_PARALLEL (default: 8)
#   --max-loaded-models <n>     OLLAMA_MAX_LOADED_MODELS (default: 3)
#   --api-port <port>           Host port for the Ollama API (default: 11440)
#   --web-port <port>           Host port for the web UI (default: 3004)
#   --enable-web-port <bool>    Whether to publish the web port: true|false (default: true)
#   --force-recreate            Remove and recreate any model that already exists
#   -h, --help                  Show this help message and exit
#
# Examples:
#   bash ./deploy_ollama.sh --num-parallel 8 --max-loaded-models 3 --api-port 11440 --web-port 3004
#   bash ./deploy_ollama.sh --api-port 11441 --web-port 3005 --force-recreate
#

# ==========================================
# Defaults
# ==========================================

OLLAMA_NUM_PARALLEL=16
OLLAMA_MAX_LOADED_MODELS=3
HOST_API_PORT=11450
HOST_WEB_PORT=3004
ENABLE_WEB_PORT=false
FORCE_RECREATE_MODELS=false

# ==========================================
# Configuration Variables
# ==========================================

SRC_DIR="/ollama-models"

# Deployment Settings
API_WAIT_TIME=5
OLLAMA_READY_TIMEOUT=60

# Container image
OLLAMA_CONTAINER_IMAGE="docker.io/ollama/ollama:0.18.3"

# Persistent Ollama storage on host/server
OLLAMA_STORAGE_BIND="/ollama_storage"

# ==========================================
# Helper Functions (defined early; needed by arg parsing)
# ==========================================

print_usage() {
    cat <<'EOF'
Usage: bash ./deploy_ollama.sh [OPTIONS]

Options:
  --num-parallel <n>          OLLAMA_NUM_PARALLEL (default: 8)
  --max-loaded-models <n>     OLLAMA_MAX_LOADED_MODELS (default: 3)
  --api-port <port>           Host port for the Ollama API (default: 11440)
  --web-port <port>           Host port for the web UI (default: 3004)
  --enable-web-port <bool>    Whether to publish the web port: true|false (default: true)
  --force-recreate            Remove and recreate any model that already exists
  -h, --help                  Show this help message and exit

Examples:
  bash ./deploy_ollama.sh --num-parallel 8 --max-loaded-models 3 --api-port 11440 --web-port 3004
  bash ./deploy_ollama.sh --api-port 11441 --web-port 3005 --force-recreate
EOF
}

is_positive_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_bool() {
    [ "$1" = "true" ] || [ "$1" = "false" ]
}

require_value() {
    local flag="$1"
    local value="$2"
    if [ -z "$value" ]; then
        echo "ERROR: ${flag} requires a value." >&2
        print_usage
        exit 1
    fi
}

# ==========================================
# Argument Parsing (flags only — no positional args)
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
    echo "ERROR: --num-parallel must be a positive integer, got: ${OLLAMA_NUM_PARALLEL}" >&2
    exit 1
fi

if ! is_positive_int "$OLLAMA_MAX_LOADED_MODELS"; then
    echo "ERROR: --max-loaded-models must be a positive integer, got: ${OLLAMA_MAX_LOADED_MODELS}" >&2
    exit 1
fi

if ! is_positive_int "$HOST_API_PORT"; then
    echo "ERROR: --api-port must be a positive integer, got: ${HOST_API_PORT}" >&2
    exit 1
fi

if ! is_positive_int "$HOST_WEB_PORT"; then
    echo "ERROR: --web-port must be a positive integer, got: ${HOST_WEB_PORT}" >&2
    exit 1
fi

if ! is_bool "$ENABLE_WEB_PORT"; then
    echo "ERROR: --enable-web-port must be 'true' or 'false', got: ${ENABLE_WEB_PORT}" >&2
    exit 1
fi

# Name of your container (includes ports so multiple instances with the
# same parallel/loaded-model settings but different ports don't collide)
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

# Takes the already-normalized lookup name to avoid recomputing it
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

            podman exec "${CONTAINER_NAME}" \
                ollama create "${tag_name}" \
                -f "${modelfile_path}"
        else
            echo "Model already exists: ${lookup_name}. Skipping create."
        fi
    else
        echo "Model missing. Creating model: ${tag_name} from ${folder_name}..."

        podman exec "${CONTAINER_NAME}" \
            ollama create "${tag_name}" \
            -f "${modelfile_path}"
    fi
}

# ==========================================
# 0. Pre-flight Checks
# ==========================================

echo ""
echo "##### Starting container ######"
echo "Running pre-flight checks..."
echo ""
echo "Resolved configuration:"
echo "OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
echo "OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"
echo "HOST_API_PORT=${HOST_API_PORT}"
echo "HOST_WEB_PORT=${HOST_WEB_PORT}"
echo "ENABLE_WEB_PORT=${ENABLE_WEB_PORT}"
echo "CONTAINER_NAME=${CONTAINER_NAME}"
echo ""

if [ "$FORCE_RECREATE_MODELS" = true ]; then
    echo "FORCE_RECREATE_MODELS=true"
    echo "Existing model tags will be removed and recreated from Modelfiles."
else
    echo "FORCE_RECREATE_MODELS=false"
    echo "Existing model tags will be reused."
fi

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
    echo "NOTE: A running container named '${CONTAINER_NAME}' already exists and will be replaced (--replace)."
elif container_exists; then
    echo "NOTE: A stopped container named '${CONTAINER_NAME}' already exists and will be replaced (--replace)."
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
echo "Using persistent Ollama storage folder: ${OLLAMA_STORAGE_BIND}"
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

        create_or_reuse_model "$tag_name" "$folder_name"
    fi
done

# ==========================================
# 4. Done
# ==========================================

echo ""
echo "Deployment complete."
echo ""
echo "Check models with:"
echo "podman exec ${CONTAINER_NAME} ollama list"
echo ""
echo "Check running loaded models with:"
echo "podman exec ${CONTAINER_NAME} ollama ps"
echo ""
echo "API endpoint:"
echo "http://localhost:${HOST_API_PORT}"
