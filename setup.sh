#!/usr/bin/env bash
# ClinicSearch — One-shot environment setup for macOS Apple Silicon (M1/M2/M3/M4).
#
# This installs everything needed:
#   - Homebrew (if missing)
#   - pyenv + Python 3.12.8
#   - Ollama (via Homebrew cask if missing)
#   - Project Python dependencies into a virtualenv
#   - Pulls both Ollama models (Gemma 4 E2B + EmbeddingGemma)
#
# Usage:
#   bash setup.sh
#
# Run from the project root (where requirements.txt lives).

set -euo pipefail

GREEN="\033[92m"
RED="\033[91m"
YELLOW="\033[93m"
BLUE="\033[94m"
RESET="\033[0m"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║  ClinicSearch Setup — macOS Apple Silicon                  ║${RESET}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${RESET}"

# 0. Sanity checks
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}✗ This script is for macOS only.${RESET}"
    exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "${YELLOW}⚠ You appear to be on Intel Mac. This was designed for Apple Silicon.${RESET}"
    echo "  Continuing, but performance will be much slower."
fi

# Disk space check — need at least 25GB
AVAILABLE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [[ "$AVAILABLE_GB" -lt 25 ]]; then
    echo -e "${RED}✗ Need at least 25GB free disk space (you have ${AVAILABLE_GB}GB).${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} Disk space: ${AVAILABLE_GB}GB available"

# 1. Xcode CLI tools
if ! xcode-select -p &>/dev/null; then
    echo -e "${BLUE}Installing Xcode Command Line Tools...${RESET}"
    xcode-select --install
    echo "  Approve the GUI prompt, then re-run this script."
    exit 0
fi
echo -e "${GREEN}✓${RESET} Xcode CLI tools installed"

# 2. Homebrew
if ! command -v brew &>/dev/null; then
    echo -e "${BLUE}Installing Homebrew...${RESET}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add to PATH for current session
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
echo -e "${GREEN}✓${RESET} Homebrew: $(brew --version | head -1)"

# 3. pyenv
if ! command -v pyenv &>/dev/null; then
    echo -e "${BLUE}Installing pyenv...${RESET}"
    brew install pyenv pyenv-virtualenv
fi
echo -e "${GREEN}✓${RESET} pyenv: $(pyenv --version)"

# 4. System libs
echo -e "${BLUE}Installing system libraries (portaudio, ffmpeg)...${RESET}"
brew list portaudio &>/dev/null || brew install portaudio
brew list ffmpeg &>/dev/null || brew install ffmpeg
echo -e "${GREEN}✓${RESET} System libraries installed"

# 5. Ollama
if ! command -v ollama &>/dev/null; then
    echo -e "${BLUE}Installing Ollama...${RESET}"
    brew install ollama
fi
echo -e "${GREEN}✓${RESET} Ollama installed"

# Start Ollama if not running
if ! curl -fsS http://localhost:11434/api/tags &>/dev/null; then
    echo -e "${BLUE}Starting Ollama service...${RESET}"
    brew services start ollama
    sleep 3
fi

# 6. Python 3.12.8
PYTHON_VERSION="3.12.8"
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo -e "${BLUE}Installing Python ${PYTHON_VERSION} via pyenv (5-10 minutes)...${RESET}"
    pyenv install "${PYTHON_VERSION}"
fi
echo -e "${GREEN}✓${RESET} Python ${PYTHON_VERSION} available"

# Pin Python version for this project
pyenv local "${PYTHON_VERSION}"

# 7. Virtual environment
VENV_DIR=".venv"
PYTHON_BIN="$(pyenv prefix ${PYTHON_VERSION})/bin/python"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo -e "${BLUE}Creating virtual environment at ${VENV_DIR}...${RESET}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo -e "${GREEN}✓${RESET} Virtualenv active: $(which python)"

# 8. Project deps
echo -e "${BLUE}Installing project dependencies (this takes 3-5 minutes)...${RESET}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}✓${RESET} Project dependencies installed"

# 9. Pull Ollama models
echo -e "${BLUE}Pulling Gemma 4 E2B (~1.6 GB, this takes 5-10 minutes)...${RESET}"
ollama pull gemma4:e2b

echo -e "${BLUE}Pulling EmbeddingGemma (~340 MB)...${RESET}"
ollama pull embeddinggemma:300m-qat-q8_0

echo -e "${GREEN}✓${RESET} Models ready:"
ollama list

# 10. Create .env from template
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo -e "${GREEN}✓${RESET} Created .env from template"
fi

# 11. Make scripts executable
chmod +x apk_wrapper/build_apk.sh 2>/dev/null || true

# 12. Final verification
echo
echo -e "${BLUE}Running environment verification...${RESET}"
echo
python scripts/verify_environment.py

echo
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║  ✓ SETUP COMPLETE                                          ║${RESET}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${RESET}"
echo
echo -e "${BLUE}Next steps:${RESET}"
echo "  1. source .venv/bin/activate   # if not already active"
echo "  2. python scripts/download_data.py"
echo "  3. # Download WHO EML and MSF PDFs manually (see download_data.py output)"
echo "  4. python -m src.ingest"
echo "  5. python -m src.embed"
echo "  6. streamlit run app.py --server.address=0.0.0.0 --server.port=8501"
echo
