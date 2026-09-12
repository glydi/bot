#!/bin/bash
# Build "GLYDI.app" -- a double-clickable launcher for the bot window.
#
# Beyond convenience, the bundle solves a real problem: macOS grants camera
# and microphone access per application. Run from a terminal the bot inherits
# the terminal's permissions, and on this machine the camera returned black
# frames because Terminal had never been granted access. As its own bundle
# with its own usage descriptions, macOS prompts for "GLYDI" directly.
#
# The release binary is copied into the bundle, so the app keeps working if
# the checkout is rebuilt; models, the database and the ttsd helper are
# still read from the repo (paths are resolved through GLYDI_ROOT).
#
# Usage: rust/make_app.sh [destination-dir]   (default: ~/Desktop)

set -euo pipefail

RUST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$RUST/.." && pwd)"
DEST="${1:-$HOME/Desktop}"
APP="$DEST/GLYDI.app"
BIN="$RUST/target/release/glydi"

[ -x "$BIN" ] || { echo "no release binary; run: cd rust && cargo build --release -p glydi --features vision,kokoro" >&2; exit 1; }
[ -x "$RUST/ttsd/ttsd" ] || make -C "$RUST/ttsd" >/dev/null

echo "repo: $REPO"
echo "app:  $APP"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$REPO/build/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
cp "$BIN" "$APP/Contents/MacOS/glydi-bin"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>GLYDI</string>
    <key>CFBundleDisplayName</key><string>GLYDI</string>
    <key>CFBundleIdentifier</key><string>ai.glydi.bot</string>
    <key>CFBundleExecutable</key><string>GLYDI</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.2.0</string>
    <key>CFBundleVersion</key><string>2</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key>
    <string>GLYDI uses the camera to recognise the people it is talking to, so it can remember their names.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>GLYDI listens through the microphone so you can talk to it.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/GLYDI" <<LAUNCHER
#!/bin/bash
# Runs the bot in the foreground so the .app stays alive in the Dock and can
# be quit from there. Errors go to a dialog, since there is no terminal.
# The binary is "glydi-bin": on a case-insensitive filesystem "glydi" and
# this "GLYDI" launcher would be the same file.
set -uo pipefail

export GLYDI_ROOT="$REPO"
export ORT_DYLIB_PATH="\${ORT_DYLIB_PATH:-/opt/homebrew/lib/libonnxruntime.dylib}"
export PATH="/opt/homebrew/bin:\$PATH"
HERE="\$(cd "\$(dirname "\$0")" && pwd)"
LOG="\$GLYDI_ROOT/data/launch.log"
mkdir -p "\$GLYDI_ROOT/data"

fail() {
    osascript -e "display alert \"GLYDI\" message \"\$1\" as critical" >/dev/null 2>&1 || true
    exit 1
}

cd "\$GLYDI_ROOT" || fail "Cannot find the bot at \$GLYDI_ROOT."

# A .env in the repo is honoured, as it was for the Python build.
if [ -f "\$GLYDI_ROOT/.env" ]; then
    set -a; . "\$GLYDI_ROOT/.env"; set +a
fi

if pgrep -x glydi-bin >/dev/null 2>&1 || pgrep -f "GLYDI.app/Contents/MacOS/glydi-bin run" >/dev/null 2>&1; then
    osascript -e 'display notification "Already running." with title "GLYDI"' >/dev/null 2>&1 || true
    exit 0
fi

# The default brain is local and needs a running model server.
if ! curl -s -m 2 "\${GLYDI_LOCAL_LLM_URL:-http://localhost:11434/v1}/models" >/dev/null; then
    if command -v ollama >/dev/null 2>&1; then
        nohup ollama serve >>"\$LOG" 2>&1 &
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            curl -s -m 1 "\${GLYDI_LOCAL_LLM_URL:-http://localhost:11434/v1}/models" >/dev/null && break
            sleep 1
        done
    else
        fail "Ollama is not installed. Run: brew install ollama && ollama pull qwen2.5:3b"
    fi
fi

exec "\$HERE/glydi-bin" run >>"\$LOG" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/GLYDI" "$APP/Contents/MacOS/glydi-bin"

# Ad-hoc signature: enough for TCC to key the camera/mic grants to this
# bundle and remember them across launches.
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || echo "note: codesign unavailable; permissions may be asked again each launch"

echo "built $APP"
