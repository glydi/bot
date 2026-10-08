# GLYDI on a plain Ubuntu PC

A stand-in for the Jetson: an x86_64 box running Ubuntu Server 24.04 LTS,
headless. The Jetson's own files in `deploy/jetson/` are reused; only the
architecture-specific bits differ, and `setup.sh` handles those.

## Install the OS

1. Write `ubuntu-24.04.x-live-server-amd64.iso` to a USB stick (Rufus on
   Windows: select the stick, select the ISO, Start, defaults). Check the
   ISO against `SHA256SUMS` from releases.ubuntu.com first.
2. Boot the PC from the stick (F12 / F2 / Del boot menu, depending on the
   board). In the installer: English, keyboard, "Ubuntu Server" (not the
   minimised one), use the whole disk, create the login user, **tick
   "Install OpenSSH server"**, no snaps. Reboot, pull the stick.
3. Log in, note the IP (`ip -4 a`), and from then on work over `ssh`.

## Install the bot

```sh
sudo apt-get install -y git
git clone git@github.com:glydi/bot.git ~/bot     # or https://github.com/glydi/bot.git
cd ~/bot
deploy/ubuntu/setup.sh                            # 20-40 minutes
deploy/jetson/install-service.sh                  # start at boot, headless
journalctl -u glydi -f
```

`setup.sh` installs the build packages, Rust, onnxruntime, Ollama with the
same drop-in and the same `glydi-1.5b` model as the Jetson, fetches the
speech and vision models, writes `.env` from `deploy/jetson/env.jetson`
with the x86_64 paths, builds the release binary and runs `glydi check`.

Copy the fine-tune's GGUF to `models/glydi-1.5b.q4_K_M.gguf` before the
first run (or re-run `setup.sh` after copying it) so `glydi-1.5b` is the
trained model rather than the untuned base. `scp` it from the training PC.

With an NVIDIA card: install CUDA 12 and cuDNN 9 from NVIDIA's apt repo,
then `GLYDI_GPU=1 deploy/ubuntu/setup.sh` takes the CUDA build of
onnxruntime and turns the GPU switches on in `.env`. Without them the
GPU providers fail to register and everything runs on the CPU, logged.

## Talk to it without a microphone

```sh
set -a; . .env; set +a
rust/target/release/glydi run --headless --no-camera --no-mic --text
```
