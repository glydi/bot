#!/bin/bash
# Build "Glydi Bot.app" -- a double-clickable launcher for the native bot window.
#
# Beyond convenience, the bundle solves a real problem: macOS grants camera and
# microphone access per-application. Run from a terminal the bot inherits the
# terminal's permissions (and on this machine the camera returned black frames
# because Terminal had never been granted access). As its own signed bundle with
# its own usage descriptions, macOS prompts for "Glydi Bot" directly.
#
# Usage: tools/make_app.sh [destination-dir]   (default: ~/Desktop)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/Desktop}"
APP="$DEST/Glydi Bot.app"

echo "repo: $REPO"
echo "app:  $APP"

"$REPO/.venv/bin/python" "$REPO/tools/make_icon.py" "$REPO/build" >/dev/null

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$REPO/build/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Glydi Bot</string>
    <key>CFBundleDisplayName</key><string>Glydi Bot</string>
    <key>CFBundleIdentifier</key><string>ai.glydi.bot</string>
    <key>CFBundleExecutable</key><string>GlydiBot</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key>
    <string>Glydi Bot uses the camera to recognise the people it is talking to, so it can remember their names.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Glydi Bot listens through the microphone so you can talk to it.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/GlydiBot" <<LAUNCHER
#!/bin/bash
# Runs the bot in the foreground so the .app stays alive in the Dock and can be
# quit from there.
set -uo pipefail

REPO="$REPO"
LOG="\$REPO/data/launch.log"
mkdir -p "\$REPO/data"

note() { osascript -e "display notification \"\$1\" with title \"Glydi Bot\"" >/dev/null 2>&1 || true; }
fail() {
    osascript -e "display alert \"Glydi Bot\" message \"\$1\" as critical" >/dev/null 2>&1 || true
    exit 1
}

cd "\$REPO" || fail "Cannot find the bot at \$REPO."
[ -x "\$REPO/.venv/bin/python" ] || fail "The Python environment is missing. Run: python3.12 -m venv .venv && .venv/bin/pip install -e '.[identity]'"

# Anchored to the interpreter so a shell or editor that merely mentions the
# module name is not mistaken for a running bot.
if pgrep -f "[Pp]ython[0-9.]* -m glydi_bot\\.native\$" >/dev/null 2>&1; then
    note "Already running."
    exit 0
fi

# The default brain is local and needs no key -- just a running model server.
# A hosted provider needs its key present.
PROVIDER="\$(grep -E '^GLYDI_LLM=' "\$REPO/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \"')"
case "\${PROVIDER:-local}" in
    claude) NEEDED=ANTHROPIC_API_KEY ;;
    gemini) NEEDED=GOOGLE_API_KEY ;;
    openai) NEEDED=OPENAI_API_KEY ;;
    *)      NEEDED= ;;
esac
if [ -n "\$NEEDED" ]; then
    KEY="\$(grep -E "^\${NEEDED}=" "\$REPO/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \"')"
    if [ -z "\$KEY" ]; then
        osascript -e "display alert \"Glydi Bot needs \$NEEDED\" message \"Add it to the .env file, then launch again.\" as critical" >/dev/null 2>&1 || true
        open -e "\$REPO/.env" 2>/dev/null || true
        exit 1
    fi
else
    LLM_URL="\$(grep -E '^GLYDI_LOCAL_LLM_URL=' "\$REPO/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \"')"
    if ! curl -sf -m 2 "\${LLM_URL:-http://localhost:11434/v1}/models" >/dev/null 2>&1; then
        # Ollama installed but not running is the common case; start it.
        if command -v ollama >/dev/null 2>&1; then
            # Keep the model resident between conversations, and let the
            # background fact extraction run beside the next turn instead of
            # in front of it.
            (OLLAMA_KEEP_ALIVE=10m OLLAMA_NUM_PARALLEL=2 ollama serve >/dev/null 2>&1 &)
            sleep 2
        fi
        if ! curl -sf -m 2 "\${LLM_URL:-http://localhost:11434/v1}/models" >/dev/null 2>&1; then
            osascript -e "display alert \"Glydi Bot needs a local model server\" message \"Install Ollama (brew install ollama), then: ollama pull qwen2.5:3b\" as critical" >/dev/null 2>&1 || true
            exit 1
        fi
    fi
fi

note "Starting…"
exec "\$REPO/.venv/bin/python" -m glydi_bot.native >>"\$LOG" 2>&1
LAUNCHER

chmod +x "$APP/Contents/MacOS/GlydiBot"

# An ad-hoc signature gives the bundle a stable identity, which is what macOS
# ties camera and microphone permissions to. Without it, permission can be
# forgotten between launches.
codesign --force --deep --sign - "$APP" 2>/dev/null || echo "note: could not ad-hoc sign (permissions may re-prompt)"

touch "$APP"  # nudge Finder's icon cache
echo "built: $APP"
