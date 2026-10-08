#!/usr/bin/env bash
# Install glydi.service so the bot starts at boot on the Jetson.
#
#     deploy/jetson/install-service.sh            # headless (default)
#     deploy/jetson/install-service.sh --kiosk    # with the face window on the GNOME/X11 session
#     deploy/jetson/install-service.sh --cage     # with the face window under cage (no desktop)
#
# Run as the login user that owns the checkout (the one install.sh ran
# as); it sudo's for the copy into /etc/systemd/system. Fills __REPO__,
# __USER__ and __UID__ in the chosen template, installs it as
# glydi.service, reloads systemd and starts it. Re-run after editing a
# template or moving the checkout; it replaces the installed copy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
UNIT_NAME="glydi.service"
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"

template="$HERE/glydi.service"
case "${1:-}" in
    "") ;;
    --kiosk) template="$HERE/glydi-kiosk.service" ;;
    --cage)
        template="$HERE/glydi-cage.service"
        command -v cage >/dev/null 2>&1 || { echo "no cage binary; sudo apt-get install -y cage (kiosk.md, \"Chest screen under cage\")" >&2; exit 1; }
        ;;
    -h | --help)
        sed -n '2,12p' "$0"
        exit 0
        ;;
    *)
        echo "unknown option: $1 (try --kiosk or --cage)" >&2
        exit 2
        ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    echo "run this as the login user, not root: the unit runs as you" >&2
    exit 1
fi
user="$(id -un)"
uid="$(id -u)"
bin="$REPO/rust/target/release/glydi"

[ -x "$bin" ] || { echo "no release binary at $bin; run deploy/jetson/install.sh first" >&2; exit 1; }
[ -f "$REPO/.env" ] || { echo "no $REPO/.env; cp deploy/jetson/env.jetson .env and edit it" >&2; exit 1; }

echo "template: $template"
echo "repo:     $REPO"
echo "user:     $user ($uid)"

# Substitute into a temp file, then install with fixed ownership and mode
# (a unit must not be group/world writable or systemd refuses it).
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
sed -e "s|__REPO__|$REPO|g" -e "s|__USER__|$user|g" -e "s|__UID__|$uid|g" "$template" >"$tmp"
sudo install -o root -g root -m 0644 "$tmp" "$UNIT_PATH"

# Ollama first (its own installer enabled it; make sure), then ours.
sudo systemctl enable ollama >/dev/null 2>&1 || true
sudo systemctl daemon-reload
sudo systemctl enable --now "$UNIT_NAME"

echo
echo "installed $UNIT_PATH"
systemctl --no-pager status "$UNIT_NAME" || true
echo
echo "follow the log:   journalctl -u glydi -f"
echo "stop / start:     sudo systemctl stop glydi   /  sudo systemctl start glydi"
echo "disable at boot:  sudo systemctl disable --now glydi"
