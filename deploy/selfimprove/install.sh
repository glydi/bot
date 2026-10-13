#!/usr/bin/env bash
# Install the nightly self-improvement timer (03:00, Persistent=true).
#
#     deploy/selfimprove/install.sh
#
# Run as the login user that owns the checkout and runs glydi.service; it
# sudo's for the copy into /etc/systemd/system. Re-run after moving the repo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
[ "$(id -u)" -ne 0 ] || { echo "run as the login user, not root" >&2; exit 1; }
[ -f "$REPO/.env" ] || { echo "no $REPO/.env (deploy/jetson/env.jetson)" >&2; exit 1; }
user="$(id -un)"
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
sed -e "s|__REPO__|$REPO|g" -e "s|__USER__|$user|g" "$HERE/glydi-selfimprove.service" >"$tmp"
sudo install -o root -g root -m 0644 "$tmp" /etc/systemd/system/glydi-selfimprove.service
sudo install -o root -g root -m 0644 "$HERE/glydi-selfimprove.timer" /etc/systemd/system/glydi-selfimprove.timer
sudo systemctl daemon-reload
sudo systemctl enable --now glydi-selfimprove.timer
systemctl list-timers glydi-selfimprove.timer --no-pager
echo "run once now: sudo systemctl start glydi-selfimprove; journalctl -u glydi-selfimprove -f"
