#!/bin/sh
# This script installs AnythingLLMDesktop on Linux.
# On systems with AppArmor enabled, the AppImage needs to also create a userspace
# apparmor profile so that the AppImage can be run without SUID requirements due to root chromium ownership.
#
# Todo: Detect the current location of the AppImage so that we can update the application
# in-place without the user needed to manually move the application to the new location.
# This is also useful so that the apparmor location is always correct if the user edits it.
set -eu

status() { echo "$*" >&2; }
error() { echo "ERROR $*"; exit 1; }
warning() { echo "WARNING: $*"; }

# Detect AppArmor major version so we generate compatible profile syntax.
# AppArmor 4.x (Ubuntu 24.04+) supports abi declarations and userns rules;
# older versions (Ubuntu 20.04/22.04) need classic syntax only.
get_apparmor_major_version() {
    if command -v apparmor_parser >/dev/null 2>&1; then
        apparmor_parser --version 2>/dev/null | grep -oE '[0-9]+' | head -1
    else
        echo "0"
    fi
}

# Create an AppArmor profile for AnythingLLMDesktop for systems with AppArmor enabled
# https://askubuntu.com/questions/1512287/obsidian-appimage-the-suid-sandbox-helper-binary-was-found-but-is-not-configu/1528215#1528215
create_apparmor_profile() {
    status "Checking for sudo privileges..."
    sudo -v
    if [ $? -ne 0 ]; then
        error "Failed to get sudo privileges! Aborting..."
        exit 1
    fi

    AA_MAJOR=$(get_apparmor_major_version)
    status "Creating AppArmor profile for AnythingLLM (AppArmor version ${AA_MAJOR}.x detected)..."

    if [ "${AA_MAJOR}" -ge 4 ] 2>/dev/null; then
        APP_ARMOR_CONTENT=$(cat <<'EOF'
# AnythingLLMDesktop AppArmor profile (4.x syntax)
abi <abi/4.0>,
include <tunables/global>

profile anythingllmdesktop /**/AnythingLLMDesktop.AppImage flags=(unconfined) {
  userns,
}
EOF
)
    else
        APP_ARMOR_CONTENT=$(cat <<'EOF'
# AnythingLLMDesktop AppArmor profile (classic syntax for AppArmor 2.x/3.x)
#include <tunables/global>

profile anythingllmdesktop /**/AnythingLLMDesktop.AppImage flags=(unconfined) {
}
EOF
)
    fi

    echo "$APP_ARMOR_CONTENT" | sudo tee /etc/apparmor.d/anythingllmdesktop > /dev/null
    status "Reloading AppArmor service..."
    if sudo apparmor_parser -r /etc/apparmor.d/anythingllmdesktop 2>/dev/null; then
        status "AppArmor profile created - you can now run AnythingLLMDesktop without SUID requirements."
    elif sudo systemctl reload apparmor.service 2>/dev/null; then
        status "AppArmor profile created - you can now run AnythingLLMDesktop without SUID requirements."
    else
        warning "AppArmor reload failed. The profile was written to /etc/apparmor.d/anythingllmdesktop"
        warning "but could not be loaded. You may need to reboot or run:"
        warning "  sudo apparmor_parser -r /etc/apparmor.d/anythingllmdesktop"
        warning "If issues persist, you can disable the profile for this app only:"
        warning "  sudo ln -sf /etc/apparmor.d/anythingllmdesktop /etc/apparmor.d/disable/"
        warning "  sudo systemctl reload apparmor"
    fi
}

check_to_create_apparmor_profile() {
    # Check if the system has AppArmor enabled
    if [ -f /sys/kernel/security/apparmor/profiles ]; then
        if ! [ -f /etc/apparmor.d/anythingllmdesktop ]; then
            status "AppArmor is enabled on this system."
            status "\e[31m[Warning]\e[0m You will get an error about SUID permission issues when running the AppImage without creating an AppArmor profile."
            status "This requires sudo privileges. If you are unsure, you can create an AppArmor profile manually."
            read -p "Do you want to auto-create an AppArmor profile for AnythingLLM now? (y/n): " create_apparmor
            case "$create_apparmor" in [yY]) create_apparmor="y" ;; esac
            if [ "$create_apparmor" = "y" ]; then
                create_apparmor_profile
            else
                status "AppArmor is enabled on this system."
                status "AppArmor profile creation skipped. You may not be able to run AnythingLLMDesktop without it."
            fi
        else
            status "AppArmor profile already exists."
            read -p "Do you want to overwrite it with the latest version? (y/n): " overwrite_apparmor
            case "$overwrite_apparmor" in [yY]) overwrite_apparmor="y" ;; esac
            if [ "$overwrite_apparmor" = "y" ]; then
                create_apparmor_profile
            fi
        fi
    else
        status "AppArmor could not be automatically detected or does not exist. If you get an SUID error on startup, you may need to create an AppArmor profile for AnythingLLMDesktop.AppImage manually."
    fi  
}

check_or_create_desktop_profile() {
    if ! [ -f $HOME/.local/share/applications/anythingllmdesktop.desktop ]; then
        status "Desktop profile not found. Creating..."

        # Default Exec command
        EXEC_CMD="$INSTALL_DIR/AnythingLLMDesktop.AppImage"

        # Check for Wayland + KDE specifically
        # We check XDG_SESSION_TYPE for "wayland"
        # We check XDG_CURRENT_DESKTOP for "KDE" (handles "KDE", "KDE-Plasma", etc)
        # We use ':-' to safely handle unbound variables in 'set -u' mode
        if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
            case "${XDG_CURRENT_DESKTOP:-}" in
                *"KDE"*)
                    status "Detected KDE Plasma on Wayland. Adding specific flags for IME support."
                    EXEC_CMD="$INSTALL_DIR/AnythingLLMDesktop.AppImage --enable-features=UseOzonePlatform --ozone-platform=wayland --enable-wayland-ime"
                    ;;
            esac
        fi

        DESKTOP_CONTENT=$(cat <<EOF
[Desktop Entry]
StartupWMClass=anythingllm-desktop
Type=Application
Name=AnythingLLM Desktop
Exec=$EXEC_CMD
Icon=$HOME/.config/anythingllm-desktop/storage/icon.png
Categories=Utility;
EOF
)
        mkdir -p $HOME/.local/share/applications
        echo "$DESKTOP_CONTENT" | tee $HOME/.local/share/applications/anythingllmdesktop.desktop > /dev/null
        status "Desktop profile created!"
    fi
}

arch=$(uname -m)
[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'
if [ "$(id -u)" -eq 0 ]; then
    status "This script should not be run as root. Please run it as a regular user."
    exit 1
fi

# Allow custom installation directory via ANYTHING_LLM_INSTALL_DIR environment variable
# Defaults to $HOME if not set
INSTALL_DIR="${ANYTHING_LLM_INSTALL_DIR:-$HOME}"

status "#########################################################"
status " Welcome to the AnythingLLM Desktop Installer"
status " by Mintplex Labs Inc (team@mintplexlabs.com)"
status " Architecture: $arch"
status " Install Directory: $INSTALL_DIR"
status "#########################################################"

if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then
    APPIMAGE_URL="https://cdn.anythingllm.com/latest/AnythingLLMDesktop-Arm64.AppImage"
else
    APPIMAGE_URL="https://cdn.anythingllm.com/latest/AnythingLLMDesktop.AppImage"
fi
APPIMAGE_FILE="AnythingLLMDesktop.AppImage"

mkdir -p "$INSTALL_DIR"
SHOULD_DOWNLOAD="true"
if [ -f "$INSTALL_DIR/$APPIMAGE_FILE" ]; then
    status "Existing installation found at $INSTALL_DIR/$APPIMAGE_FILE"
    read -p "Do you want to re-download and overwrite it? (y/n): " overwrite
    case "$overwrite" in [yY]) ;; *) SHOULD_DOWNLOAD="false" ;; esac
fi

if [ "$SHOULD_DOWNLOAD" = "true" ]; then
    status "Downloading AnythingLLM Desktop..."
    curl --fail --show-error --location --progress-bar -o "$INSTALL_DIR/$APPIMAGE_FILE" "$APPIMAGE_URL"
    chmod +x "$INSTALL_DIR/$APPIMAGE_FILE"
fi

# ====================== Ollama Engine Download ======================
OLLAMA_VERSION="0.20.7"
OLLAMA_ENGINE_DIR="$HOME/.config/anythingllm-desktop/storage/engines/ollama"

install_ollama_engine() {
    if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then
        OLLAMA_ARCHIVE="linux-arm64.tar.zst"
    else
        OLLAMA_ARCHIVE="linux-amd64.tar.zst"
    fi
    OLLAMA_URL="https://cdn.anythingllm.com/support/ollama/${OLLAMA_VERSION}/${OLLAMA_ARCHIVE}"

    if ! command -v zstd >/dev/null 2>&1; then
        warning "zstd is required to extract the Ollama engine."
        warning "Install with: apt install zstd (Debian/Ubuntu) or yum install zstd (RHEL/CentOS)"
        warning "Skipping Ollama engine download. The built-in LLM engine will not be available."
        return 1
    fi

    # Wipe existing engine directory for a clean install
    if [ -d "$OLLAMA_ENGINE_DIR" ]; then
        status "Removing existing Ollama engine for clean install..."
        rm -rf "$OLLAMA_ENGINE_DIR"
    fi

    mkdir -p "$OLLAMA_ENGINE_DIR"
    status "Downloading Ollama engine v${OLLAMA_VERSION}..."
    if ! curl --fail --show-error --location --progress-bar -o "$OLLAMA_ENGINE_DIR/${OLLAMA_ARCHIVE}" "$OLLAMA_URL"; then
        warning "Failed to download Ollama engine. The built-in LLM engine will not be available."
        rm -rf "$OLLAMA_ENGINE_DIR"
        return 1
    fi

    status "Extracting Ollama engine..."
    tar --zstd -xf "$OLLAMA_ENGINE_DIR/${OLLAMA_ARCHIVE}" -C "$OLLAMA_ENGINE_DIR"
    rm -f "$OLLAMA_ENGINE_DIR/${OLLAMA_ARCHIVE}"

    # Rename the binary to our expected name
    if [ -f "$OLLAMA_ENGINE_DIR/bin/ollama" ]; then
        mv "$OLLAMA_ENGINE_DIR/bin/ollama" "$OLLAMA_ENGINE_DIR/bin/llm"
    fi

    # Mark binaries as executable
    chmod +x "$OLLAMA_ENGINE_DIR/bin/llm"

    # If nvidia-smi is not available, drop CUDA folders to save disk space
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        status "No NVIDIA GPU detected (nvidia-smi not found). Removing CUDA libraries to save space..."
        rm -rf "$OLLAMA_ENGINE_DIR/lib/ollama/cuda_v12" 2>/dev/null
        rm -rf "$OLLAMA_ENGINE_DIR/lib/ollama/cuda_v13" 2>/dev/null
    else
        status "NVIDIA GPU detected. Keeping CUDA libraries."
    fi

    # Write version marker
    echo -n "$OLLAMA_VERSION" > "$OLLAMA_ENGINE_DIR/.ollama-version"
    status "Ollama engine v${OLLAMA_VERSION} installed successfully."
}

SHOULD_INSTALL_OLLAMA="true"
# New installs run local models on lemond, so Ollama is only installed/updated for users who already have
# Ollama models downloaded. Every pulled (or partially pulled) model leaves a sha256-* file in blobs/ -
# the models/ollama folder itself is not enough since the app creates it on every launch.
# When skipped, an existing Ollama engine folder is left exactly as it is.
OLLAMA_MODELS_DIR="$HOME/.config/anythingllm-desktop/storage/models/ollama"
has_ollama_models() {
    # POSIX sh - an unmatched glob stays literal, so check the match actually exists
    for blob in "$OLLAMA_MODELS_DIR"/blobs/sha256-*; do
        [ -e "$blob" ] && return 0
    done
    return 1
}

if ! has_ollama_models; then
    status "No downloaded Ollama models found. Skipping Ollama engine."
    SHOULD_INSTALL_OLLAMA="false"
elif [ -f "$OLLAMA_ENGINE_DIR/.ollama-version" ]; then
    INSTALLED_OLLAMA_VERSION=$(cat "$OLLAMA_ENGINE_DIR/.ollama-version" 2>/dev/null || echo "")
    if [ "$INSTALLED_OLLAMA_VERSION" = "$OLLAMA_VERSION" ] && [ -f "$OLLAMA_ENGINE_DIR/bin/llm" ]; then
        status "Ollama engine v${OLLAMA_VERSION} already installed. Skipping download."
        SHOULD_INSTALL_OLLAMA="false"
    else
        status "Ollama engine version mismatch (installed: ${INSTALLED_OLLAMA_VERSION}, expected: ${OLLAMA_VERSION}). Updating..."
    fi
fi

if [ "$SHOULD_INSTALL_OLLAMA" = "true" ]; then
    install_ollama_engine
fi

# swap_engine_dir <new dir> <current dir> - replaces the current engine folder with a fully installed new one. The new
# version is always built next to the current one first, so a failed or offline update keeps the working engine.
swap_engine_dir() {
    rm -rf "$2.old"
    if [ -e "$2" ] && ! mv "$2" "$2.old"; then
        return 1
    fi
    if ! mv "$1" "$2"; then
        [ -e "$2.old" ] && mv "$2.old" "$2"
        return 1
    fi
    rm -rf "$2.old"
}

# ====================== lemond Engine Download ======================
# lemond is the Lemonade server that manages the llama.cpp engines. The llama.cpp engines are separate downloads.
LEMOND_VERSION="2026.40.0"
LEMOND_ENGINE_DIR="$HOME/.config/anythingllm-desktop/storage/engines/lemond"

install_lemond_engine() {
    if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then
        LEMOND_PLATFORM="linux-arm64"
    else
        LEMOND_PLATFORM="linux-x64"
    fi
    LEMOND_ARCHIVE="lemond-${LEMOND_VERSION}-${LEMOND_PLATFORM}.tar.gz"
    LEMOND_URL="https://cdn.anythingllm.com/support/lemonade/${LEMOND_VERSION}/${LEMOND_ARCHIVE}"

    staging="${LEMOND_ENGINE_DIR}.new"
    rm -rf "$staging"
    mkdir -p "$staging"

    status "Downloading lemond engine v${LEMOND_VERSION}..."
    # The archive has a single top-level folder - strip it so lemond lands directly in the engine dir
    if ! curl --fail --show-error --location --progress-bar -o "$staging/${LEMOND_ARCHIVE}" "$LEMOND_URL" \
        || ! tar -xzf "$staging/${LEMOND_ARCHIVE}" -C "$staging" --strip-components=1 \
        || [ ! -f "$staging/lemond" ]; then
        rm -rf "$staging"
        lemond_install_failed "Failed to download lemond engine v${LEMOND_VERSION}."
        return 1
    fi
    rm -f "$staging/${LEMOND_ARCHIVE}"
    chmod +x "$staging/lemond"
    echo -n "$LEMOND_VERSION" > "$staging/.lemond-version"

    if ! swap_engine_dir "$staging" "$LEMOND_ENGINE_DIR"; then
        rm -rf "$staging"
        lemond_install_failed "Could not replace the installed lemond engine."
        return 1
    fi
    status "lemond engine v${LEMOND_VERSION} installed successfully."
}

lemond_install_failed() {
    if [ -f "$LEMOND_ENGINE_DIR/lemond" ]; then
        warning "$1 Keeping the installed lemond engine."
    else
        warning "$1 AnythingLLM will download it the next time a local model is used."
    fi
}

SHOULD_INSTALL_LEMOND="true"
if [ -f "$LEMOND_ENGINE_DIR/.lemond-version" ]; then
    INSTALLED_LEMOND_VERSION=$(cat "$LEMOND_ENGINE_DIR/.lemond-version" 2>/dev/null || echo "")
    if [ "$INSTALLED_LEMOND_VERSION" = "$LEMOND_VERSION" ] && [ -f "$LEMOND_ENGINE_DIR/lemond" ]; then
        status "lemond engine v${LEMOND_VERSION} already installed. Skipping download."
        SHOULD_INSTALL_LEMOND="false"
    else
        status "lemond engine version mismatch (installed: ${INSTALLED_LEMOND_VERSION}, expected: ${LEMOND_VERSION}). Updating..."
    fi
fi

if [ "$SHOULD_INSTALL_LEMOND" = "true" ]; then
    install_lemond_engine
fi

# ====================== llama.cpp Engine Download ======================
# The llama.cpp builds lemond runs, pruned to llama-server + what it loads and hosted next to the lemond version.
# Each backend gets its own folder so a GPU is never visible through two backends at once:
#   cpu/    base package
#   vulkan/ base package + libggml-vulkan.so add-on (identical base, so it is downloaded once and copied)
#   cuda/   full CUDA package (upstream builds CUDA separately) - NVIDIA only, see select_llamacpp_cuda
LLAMACPP_VERSION="b11425"
LLAMACPP_BASE_URL="https://cdn.anythingllm.com/support/lemonade/${LEMOND_VERSION}/llamacpp-${LLAMACPP_VERSION}"
LLAMACPP_ENGINE_DIR="$HOME/.config/anythingllm-desktop/storage/engines/llamacpp"

# glibc_at_least 2.38 - true when the system glibc is at least the given version
glibc_at_least() {
    have=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
    [ -n "$have" ] && [ "$(printf '%s\n%s\n' "$1" "$have" | sort -V | head -n 1)" = "$1" ]
}

# Prints the CUDA package to install (cuda-13.4 / cuda-12.8) or nothing, in which case NVIDIA users run on Vulkan.
select_llamacpp_cuda() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    gpu_info=$(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
    [ -n "$gpu_info" ] || return 0
    driver_major=$(echo "${gpu_info%%,*}" | tr -d ' ' | cut -d. -f1)
    compute_cap=$(echo "${gpu_info#*,}" | tr -d ' ')

    # lemond only runs its CUDA backend on Turing (7.5) or newer
    if [ "$(printf '7.5\n%s\n' "$compute_cap" | sort -V | head -n 1)" != "7.5" ]; then
        status "NVIDIA GPU compute capability ${compute_cap} is below 7.5 - using Vulkan instead of CUDA." >&2
        return 0
    fi
    # The CUDA builds are made on Ubuntu 24.04 and need glibc 2.38+ (Ubuntu 24.04+, Debian 13+, Fedora 39+)
    if ! glibc_at_least 2.38; then
        status "CUDA requires glibc 2.38+ (Ubuntu 24.04 or newer) - using Vulkan instead of CUDA." >&2
        return 0
    fi
    if [ "$driver_major" -ge 580 ] 2>/dev/null; then
        echo "cuda-13.4"
    elif [ "$LLAMACPP_PLATFORM" = "linux-x64" ] && [ "$driver_major" -ge 570 ] 2>/dev/null; then
        echo "cuda-12.8"
    else
        status "NVIDIA driver ${driver_major} is too old for CUDA (needs 580+, or 570+ on x86_64) - using Vulkan instead of CUDA." >&2
    fi
}

# fetch_llamacpp <archive> <dest dir> - downloads and unpacks one (flat) package into the dest dir
fetch_llamacpp() {
    mkdir -p "$2"
    curl --fail --show-error --location --progress-bar -o "$2/$1" "${LLAMACPP_BASE_URL}/$1" || return 1
    tar -xzf "$2/$1" -C "$2" || return 1
    rm -f "$2/$1"
}

install_llamacpp_engine() {
    staging="${LLAMACPP_ENGINE_DIR}.new"
    rm -rf "$staging"
    complete="true"

    status "Downloading llama.cpp ${LLAMACPP_VERSION} engine..."
    if ! fetch_llamacpp "llamacpp-${LLAMACPP_VERSION}-${LLAMACPP_PLATFORM}.tar.gz" "$staging/cpu"; then
        rm -rf "$staging"
        if [ -f "$LLAMACPP_ENGINE_DIR/cpu/llama-server" ]; then
            warning "Failed to download the llama.cpp engine. Keeping the installed llama.cpp engine."
        else
            warning "Failed to download the llama.cpp engine. AnythingLLM will download it the next time a local model is used."
        fi
        return 1
    fi

    status "Downloading llama.cpp Vulkan support..."
    cp -R "$staging/cpu" "$staging/vulkan"
    if ! fetch_llamacpp "llamacpp-${LLAMACPP_VERSION}-${LLAMACPP_PLATFORM}-vulkan.tar.gz" "$staging/vulkan"; then
        warning "Failed to download llama.cpp Vulkan support. Local models will run on the CPU."
        rm -rf "$staging/vulkan"
        complete="false"
    fi

    if [ -n "$LLAMACPP_CUDA" ]; then
        status "NVIDIA GPU detected - downloading llama.cpp ${LLAMACPP_CUDA} support (this is a large download)..."
        if ! fetch_llamacpp "llamacpp-${LLAMACPP_VERSION}-${LLAMACPP_PLATFORM}-${LLAMACPP_CUDA}.tar.gz" "$staging/cuda"; then
            warning "Failed to download llama.cpp CUDA support. Local models will run on Vulkan."
            rm -rf "$staging/cuda"
            complete="false"
        fi
    fi

    # Only mark complete when every backend we wanted landed, so a failed add-on is retried on the next install
    [ "$complete" = "true" ] && echo -n "$LLAMACPP_MARKER" > "$staging/.llamacpp-version"

    if ! swap_engine_dir "$staging" "$LLAMACPP_ENGINE_DIR"; then
        rm -rf "$staging"
        warning "Could not replace the installed llama.cpp engine. Keeping it."
        return 1
    fi
    status "llama.cpp engine ${LLAMACPP_VERSION} installed successfully."
}

if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then
    LLAMACPP_PLATFORM="linux-arm64"
    LLAMACPP_MIN_GLIBC="2.38"
else
    LLAMACPP_PLATFORM="linux-x64"
    LLAMACPP_MIN_GLIBC="2.34"
fi

if ! glibc_at_least "$LLAMACPP_MIN_GLIBC"; then
    warning "llama.cpp on ${LLAMACPP_PLATFORM} requires glibc ${LLAMACPP_MIN_GLIBC}+ (found $(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown)). Local models will not be available."
else
    LLAMACPP_CUDA=$(select_llamacpp_cuda)
    # The marker records the CUDA choice too, so a driver/OS upgrade picks up CUDA on the next installer run
    LLAMACPP_MARKER="${LLAMACPP_VERSION}:${LLAMACPP_CUDA:-none}"
    if [ "$(cat "$LLAMACPP_ENGINE_DIR/.llamacpp-version" 2>/dev/null)" = "$LLAMACPP_MARKER" ] && [ -f "$LLAMACPP_ENGINE_DIR/cpu/llama-server" ]; then
        status "llama.cpp engine ${LLAMACPP_VERSION} (${LLAMACPP_CUDA:-no CUDA}) already installed. Skipping download."
    else
        install_llamacpp_engine
    fi
fi

# ====================== FastFlowLM (NPU) Engine Download ======================
# FLM is lemond's AMD XDNA2 NPU engine (ROCm/FastFlowLM portable build, x86_64 only). It is only installed when an
# XDNA2 NPU is present, so no other machine ever gets engines/flm. It also needs the amdxdna kernel driver (Linux 6.14+).
FLM_VERSION="1.0.7"
FLM_ENGINE_DIR="$HOME/.config/anythingllm-desktop/storage/engines/flm"

# True when an AMD XDNA2 NPU is on the PCI bus - the same IDs lemond checks (AMD 1022:17f0, NPU class, XDNA2 revisions)
has_xdna2_npu() {
    for dev in /sys/bus/pci/devices/*; do
        [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x1022" ] || continue
        [ "$(cat "$dev/device" 2>/dev/null)" = "0x17f0" ] || continue
        [ "$(cat "$dev/class" 2>/dev/null)" = "0x118000" ] || continue
        case "$(cat "$dev/revision" 2>/dev/null)" in 0x10|0x11|0x20) return 0 ;; esac
    done
    return 1
}

install_flm_engine() {
    FLM_ARCHIVE="flm-v${FLM_VERSION}-linux-x64.tar.gz"
    staging="${FLM_ENGINE_DIR}.new"
    rm -rf "$staging"
    mkdir -p "$staging"

    status "AMD NPU detected - downloading FastFlowLM v${FLM_VERSION} engine..."
    if ! curl --fail --show-error --location --progress-bar -o "$staging/${FLM_ARCHIVE}" "https://cdn.anythingllm.com/support/lemonade/${LEMOND_VERSION}/flm-v${FLM_VERSION}/${FLM_ARCHIVE}" \
        || ! tar -xzf "$staging/${FLM_ARCHIVE}" -C "$staging" \
        || [ ! -f "$staging/flm-real" ]; then
        rm -rf "$staging"
        warning "Failed to download FastFlowLM. NPU models will not be available."
        return 1
    fi
    rm -f "$staging/${FLM_ARCHIVE}"
    chmod +x "$staging/flm" "$staging/flm-real"
    echo -n "$FLM_VERSION" > "$staging/.flm-version"

    if ! swap_engine_dir "$staging" "$FLM_ENGINE_DIR"; then
        rm -rf "$staging"
        warning "Could not replace the installed FastFlowLM engine. Keeping it."
        return 1
    fi
    status "FastFlowLM engine v${FLM_VERSION} installed successfully."
}

if [ "$LLAMACPP_PLATFORM" = "linux-x64" ] && has_xdna2_npu; then
    if ! glibc_at_least 2.39; then
        warning "FastFlowLM (AMD NPU) requires glibc 2.39+ (Ubuntu 24.04 or newer). NPU models will not be available."
    elif [ "$(cat "$FLM_ENGINE_DIR/.flm-version" 2>/dev/null)" = "$FLM_VERSION" ] && [ -f "$FLM_ENGINE_DIR/flm-real" ]; then
        status "FastFlowLM engine v${FLM_VERSION} already installed. Skipping download."
    else
        install_flm_engine
    fi
fi

status "AnythingLLM Desktop is ready to run!"
status "$INSTALL_DIR/$APPIMAGE_FILE to start AnythingLLMDesktop"
status "\e[36mHeads up!\e[0m You can rerun this installer anytime to get the latest version of AnythingLLM without effecting your existing data."
status "Documentation: https://docs.anythingllm.com"
status "Issues: https://github.com/Mintplex-Labs/anything-llm"
status "\e[36mThanks for using AnythingLLM!\e[0m\n\n"

status "Next, we will create a desktop profile and AppArmor profile for AnythingLLMDesktop."
status "This is required for the AppImage to be able to run without SUID requirements."
status "You can manually create these profiles if you prefer."

check_or_create_desktop_profile
check_to_create_apparmor_profile

read -p "Do you want to start AnythingLLMDesktop now? (y/n): " start
case "$start" in [yY]) start="y" ;; esac
if [ "$start" = "y" ]; then
    "$INSTALL_DIR/$APPIMAGE_FILE"
fi