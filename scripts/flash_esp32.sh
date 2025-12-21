#!/usr/bin/env bash
set -euo pipefail

# flash_esp32.sh
# Usage: ./scripts/flash_esp32.sh [--port /dev/tty.XYZ] [--board ai-thinker|xiao-s3|esp32dev] [--monitor]

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/.pio_upload"
SRC_DIR="$BUILD_DIR/src"
INCLUDE_DIR="$BUILD_DIR/include"

PORT=""
BOARD="ai-thinker"
MONITOR=false

print_usage(){
  cat <<EOF
Usage: $0 [--port /dev/cu.usbserial-XXXX] [--board ai-thinker|xiao-s3|esp32dev] [--monitor]

This script will:
  - Create a temporary PlatformIO project in .pio_upload/
  - Copy compile/compile.ino -> src/main.cpp and camera_pins.h -> include/
  - Install platformio into the active Python venv if missing
  - Build and upload using PlatformIO

Options:
  --port    Serial port to upload to (e.g. /dev/cu.SLAB_USBtoUART). If omitted,
            the script will try to auto-detect the most recently added port.
  --board   Target board (default: ai-thinker). Supported: ai-thinker, xiao-s3, esp32dev
  --monitor Start serial monitor after upload (baud 115200)
  --help    Show this message
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --board) BOARD="$2"; shift 2;;
    --monitor) MONITOR=true; shift 1;;
    --help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

# ensure venv python is used if present
if [[ -f "$REPO_ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/venv/bin/activate"
fi

# ensure PlatformIO installed
if ! command -v pio >/dev/null 2>&1; then
  echo "PlatformIO CLI not found. Installing into current Python environment..."
  python -m pip install --upgrade pip setuptools wheel >/dev/null
  python -m pip install -U platformio
fi

# create project structure
rm -rf "$BUILD_DIR"
mkdir -p "$SRC_DIR" "$INCLUDE_DIR"

# copy sketch -> src/main.cpp
SKETCH_SRC="$REPO_ROOT/compile/compile.ino"
CAM_PINS="$REPO_ROOT/compile/camera_pins.h"
SECRETS_EXAMPLE="$REPO_ROOT/compile/secrets.example.h"
SECRETS_LOCAL="$REPO_ROOT/compile/secrets.h"

if [[ ! -f "$SKETCH_SRC" ]]; then
  echo "ERROR: compile.ino not found at $SKETCH_SRC"
  exit 1
fi

# convert .ino to main.cpp by adding extern C include guard if needed
# A simple copy is fine for most sketches; PlatformIO will compile .cpp too.
cp "$SKETCH_SRC" "$SRC_DIR/main.cpp"
cp "$CAM_PINS" "$INCLUDE_DIR/camera_pins.h"

# Provide secrets.h (WiFi/server config). Prefer local secrets.h, otherwise fall back to secrets.example.h
if [[ -f "$SECRETS_EXAMPLE" ]]; then
  cp "$SECRETS_EXAMPLE" "$INCLUDE_DIR/secrets.example.h"
fi

if [[ -f "$SECRETS_LOCAL" ]]; then
  cp "$SECRETS_LOCAL" "$INCLUDE_DIR/secrets.h"
elif [[ -f "$SECRETS_EXAMPLE" ]]; then
  cp "$SECRETS_EXAMPLE" "$INCLUDE_DIR/secrets.h"
  echo "WARNING: compile/secrets.h not found; using secrets.example.h. Create compile/secrets.h with your real WiFi/server settings."
else
  cat > "$INCLUDE_DIR/secrets.h" <<'EOF'
#pragma once

// Copy this file to compile/secrets.h and fill in real values.
#define WIFI_SSID_STR "YOUR_WIFI_SSID"
#define WIFI_PASS_STR "YOUR_WIFI_PASSWORD"

// Your FastAPI server LAN IP / port (ESP32 must reach it)
#define SERVER_HOST_STR "192.168.1.100"
#define SERVER_PORT_NUM 8081

// Optional UDP target (only used when ENABLE_UDP=1)
#define UDP_HOST_STR "127.0.0.1"
#define UDP_PORT_NUM 12345
EOF
  echo "WARNING: compile/secrets.example.h not found; generated a placeholder include/secrets.h."
fi
# Copy optional stub or helper headers from compile/ if present
if [[ -f "$REPO_ROOT/compile/ESP_I2S.h" ]]; then
  cp "$REPO_ROOT/compile/ESP_I2S.h" "$INCLUDE_DIR/ESP_I2S.h"
fi

# write a platformio.ini
cat > "$BUILD_DIR/platformio.ini" <<'INI'
[env:ai-thinker]
platform = espressif32
board = esp32cam
framework = arduino
monitor_speed = 115200
lib_deps =
  gilmaimon/ArduinoWebsockets

[env:xiao-s3]
platform = espressif32
; Use a PSRAM-capable S3 DevKitC-1 variant so psramFound() works and camera buffers can live in PSRAM
board = rymcu-esp32-s3-devkitc-1
framework = arduino
monitor_speed = 115200
lib_deps =
  gilmaimon/ArduinoWebsockets

[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
  gilmaimon/ArduinoWebsockets
INI
 

# detect port if not provided
if [[ -z "$PORT" ]]; then
  echo "Auto-detecting serial ports..."
  # list candidate ports (macOS ships Bash 3.2: no `mapfile`)
  CANDIDATES=()
  while IFS= read -r p; do
    [[ -n "$p" ]] && CANDIDATES+=("$p")
  done < <(ls -1 /dev/cu.* /dev/tty.* 2>/dev/null || true)
  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "No serial ports found under /dev. Please plug in your USB-serial adapter and re-run with --port"
    exit 1
  fi
  echo "Detected ports:" 
  for p in "${CANDIDATES[@]}"; do echo "  $p"; done

  # try to pick the most recently modified device node as a heuristic
  PORT=$(ls -t /dev/cu.* /dev/tty.* 2>/dev/null | head -n 1 || true)
  echo "Auto-selected port: $PORT"
  read -r -p "Use this port? [Y/n] " yn
  yn=${yn:-Y}
  if [[ ! $yn =~ ^[Yy] ]]; then
    echo "Please re-run with --port /dev/xxxx"
    exit 1
  fi
fi

# select env
case "$BOARD" in
  ai-thinker) ENV=ai-thinker;;
  xiao-s3) ENV=xiao-s3;;
  esp32dev) ENV=esp32dev;;
  *) echo "Unknown board: $BOARD"; exit 1;;
esac

echo "Building for env=$ENV, upload port=$PORT"
cd "$BUILD_DIR"

# build
pio run -e "$ENV"

# upload (use --upload-port for PlatformIO)
# On ESP32-S3 native USB, the device node can briefly disappear during reset.
wait_for_port() {
  local port_path="$1"
  local timeout_s="${2:-10}"
  local waited=0
  while [[ ! -e "$port_path" ]]; do
    if (( waited >= timeout_s )); then
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 0
}

if ! wait_for_port "$PORT" 10; then
  echo "ERROR: Serial port not found: $PORT"
  echo "Hint: Unplug/replug the board, or re-run with --port /dev/cu.usbmodemXXXX"
  exit 1
fi

UPLOAD_OK=false
for attempt in 1 2 3; do
  echo "Uploading (attempt $attempt/3)..."
  if pio run -e "$ENV" -t upload --upload-port "$PORT"; then
    UPLOAD_OK=true
    break
  fi
  echo "Upload failed. Waiting for port to re-appear and retrying..."
  wait_for_port "$PORT" 10 || true
  sleep 1
done

if [[ "$UPLOAD_OK" != "true" ]]; then
  echo "ERROR: Upload failed after 3 attempts."
  echo "Hint: Try the other device node (e.g. /dev/tty.usbmodemXXXX), or press BOOT then RESET and re-run."
  exit 1
fi

if $MONITOR; then
  echo "Starting serial monitor on $PORT (115200). Press Ctrl-C to exit."
  pio device monitor -p "$PORT" -b 115200
fi

echo "Done. If upload succeeded, device should reboot and start the firmware."
