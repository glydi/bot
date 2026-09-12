#!/bin/bash
# Build "Glydi.app" -- the animated Glydi face, with what the bot sees and hears.
#
# This exists as its own bundle for one reason: macOS attributes camera access
# to the *application*, and a binary launched from a terminal inherits that
# terminal's grant. On this machine that produced the worst possible failure
# mode -- AVCaptureDevice reporting `authorized` while every frame came back
# black, from both our own capture code and ffmpeg. A signed bundle with its
# own NSCameraUsageDescription asks for permission in its own name and gets
# real frames.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/Desktop}"
APP="$DEST/Glydi.app"

"$REPO/.venv/bin/python" "$REPO/tools/make_icon.py" "$REPO/build" >/dev/null

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$REPO/build/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Glydi</string>
    <key>CFBundleDisplayName</key><string>Glydi</string>
    <key>CFBundleIdentifier</key><string>ai.glydi.app</string>
    <key>CFBundleExecutable</key><string>Glydi</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key>
    <string>Glydi uses the camera to recognise the people it is talking to, so it can remember their names.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Glydi listens through the microphone so you can talk to it.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/Glydi" <<LAUNCHER
#!/bin/bash
set -uo pipefail
REPO="$REPO"
LOG="\$REPO/data/glydi.log"
mkdir -p "\$REPO/data"
cd "\$REPO" || exit 1
# The bundle declares NSCameraUsageDescription, so macOS prompts in the app's
# own name; OpenCV does not need to run its own authorization dance.
export OPENCV_AVFOUNDATION_SKIP_AUTH=1
# The local model server. The bot's prompt outgrows Ollama's default 4096
# context after a few minutes of talking, and past that every reply
# re-processes the whole prompt (0.8s replies became 20s). Run it with a
# bigger window, keep the model resident, and let the background fact
# extraction share the server instead of queueing in front of the reply.
# Four slots, not two: the chat, the fact extraction and the summary each
# have their own prompt prefix, and with fewer slots than prefixes the chat's
# cached prompt is evicted every turn and rebuilt from scratch (seen: 12s
# to first word).
if ! curl -sf -m 2 http://localhost:11434/v1/models >/dev/null 2>&1; then
    brew services stop ollama >/dev/null 2>&1 || true
    (OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=10m OLLAMA_NUM_PARALLEL=4 \
     OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
     nohup ollama serve >/dev/null 2>&1 &)
    for i in \$(seq 1 20); do curl -sf -m 2 http://localhost:11434/v1/models >/dev/null 2>&1 && break; sleep 1; done
fi

exec "\$REPO/.venv/bin/python" -m glydi_bot.webapp >>"\$LOG" 2>&1
LAUNCHER

chmod +x "$APP/Contents/MacOS/Glydi"

# The ad-hoc signature gives the bundle a stable identity for macOS to hang the
# camera and microphone grants on. Without it the permission can be forgotten
# between launches, or attributed to the parent process instead.
codesign --force --deep --sign - "$APP" 2>/dev/null \
  || echo "note: could not ad-hoc sign (permissions may re-prompt)"

touch "$APP"
echo "built: $APP"
