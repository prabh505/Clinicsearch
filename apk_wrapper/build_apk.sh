#!/usr/bin/env bash
# ClinicSearch — APK build script.
#
# Wraps the running Streamlit app into an installable Android APK
# using Jipok/website-to-apk, which:
#   - Downloads its own Android SDK (no Android Studio needed)
#   - Uses a single config file
#   - Outputs a signed, sideloadable APK
#
# Prerequisites:
#   - Streamlit must be running on http://<your-laptop-ip>:8501
#   - Your phone and laptop must be on the same WiFi
#   - You must know your laptop's LAN IP (run: ipconfig getifaddr en0)
#
# Usage:
#   1. Edit apk_wrapper/webapk.conf and replace YOUR_LAPTOP_IP with your real IP
#   2. Run: ./apk_wrapper/build_apk.sh
#   3. The final APK appears at apk_wrapper/clinicsearch.apk
#   4. Transfer to your Android phone via AirDrop, Google Drive, or USB
#   5. Enable "Install unknown apps" for the file source, then tap to install

set -euo pipefail

GREEN="\033[92m"
RED="\033[91m"
YELLOW="\033[93m"
BLUE="\033[94m"
RESET="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/.build"
TOOL_DIR="$BUILD_DIR/website-to-apk"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║  ClinicSearch APK Builder                                  ║${RESET}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${RESET}"

# 1. Check webapk.conf has been edited
CONFIG_FILE="$SCRIPT_DIR/webapk.conf"
if grep -q "YOUR_LAPTOP_IP" "$CONFIG_FILE" 2>/dev/null; then
    echo -e "${RED}✗ Edit $CONFIG_FILE first.${RESET}"
    echo "  Replace YOUR_LAPTOP_IP with your laptop's WiFi IP address."
    echo "  Find it with: ipconfig getifaddr en0"
    exit 1
fi

# 2. Check icon exists
ICON_FILE="$SCRIPT_DIR/icon.png"
if [[ ! -f "$ICON_FILE" ]]; then
    echo -e "${YELLOW}⚠ Icon not found at $ICON_FILE${RESET}"
    echo "  Creating a placeholder icon (512x512 blue with stethoscope emoji)..."
    # Create a simple solid-color PNG as fallback using Python
    python3 - <<EOF
import struct, zlib
# 512x512 single-color PNG (sky blue) — minimal valid PNG
W, H = 512, 512
def png_chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
sig = b"\x89PNG\r\n\x1a\n"
ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
# Generate a sky-blue (#0EA5E9) image
row = b"\x00" + (b"\x0E\xA5\xE9" * W)
raw = row * H
idat = zlib.compress(raw, 9)
data = sig + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b"")
open("$ICON_FILE", "wb").write(data)
print("Placeholder icon created.")
EOF
fi

# 3. Clone website-to-apk if not already present
mkdir -p "$BUILD_DIR"
if [[ ! -d "$TOOL_DIR" ]]; then
    echo -e "${BLUE}Cloning Jipok/website-to-apk...${RESET}"
    git clone --depth 1 https://github.com/Jipok/website-to-apk "$TOOL_DIR"
else
    echo -e "${GREEN}✓${RESET} website-to-apk already cloned"
fi

# 4. Fetch Java 17 (the tool ships a helper for this)
cd "$TOOL_DIR"
if [[ ! -d "jdk" ]] && [[ ! -d "JDK" ]]; then
    echo -e "${BLUE}Fetching Java 17...${RESET}"
    ./make.sh get_java
fi

# 5. Generate keystore if not present
if [[ ! -f "keystore.jks" ]]; then
    echo -e "${BLUE}Generating signing keystore (one-time)...${RESET}"
    ./make.sh keygen
fi

# 6. Copy our config and icon into the tool directory
cp "$CONFIG_FILE" "$TOOL_DIR/webapk.conf"
cp "$ICON_FILE" "$TOOL_DIR/icon.png"

# 7. Build the APK
echo -e "${BLUE}Building APK (this may take 3-5 minutes the first time)...${RESET}"
./make.sh build

# 8. Locate output APK
OUTPUT_APK=""
for candidate in app/build/outputs/apk/release/*.apk app/build/outputs/apk/debug/*.apk *.apk; do
    if [[ -f "$candidate" ]]; then
        OUTPUT_APK="$candidate"
        break
    fi
done

if [[ -z "$OUTPUT_APK" ]]; then
    echo -e "${RED}✗ Build completed but APK not found.${RESET}"
    echo "  Check $TOOL_DIR for the output."
    exit 1
fi

FINAL_APK="$SCRIPT_DIR/clinicsearch.apk"
cp "$OUTPUT_APK" "$FINAL_APK"
SIZE_KB=$(( $(stat -f%z "$FINAL_APK" 2>/dev/null || stat -c%s "$FINAL_APK") / 1024 ))

echo
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║  ✓ APK BUILD SUCCESSFUL                                    ║${RESET}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${RESET}"
echo
echo -e "  Output: ${BLUE}$FINAL_APK${RESET}"
echo -e "  Size:   ${SIZE_KB} KB"
echo
echo -e "${BLUE}Next steps:${RESET}"
echo "  1. Make sure Streamlit is running on your laptop"
echo "  2. Transfer the APK to your Android phone (AirDrop / Drive / USB)"
echo "  3. On the phone, enable 'Install unknown apps' for that file source"
echo "  4. Tap the APK and install"
echo "  5. Open the app — it should connect to your laptop over WiFi"
echo
