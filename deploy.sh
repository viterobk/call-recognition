#!/bin/sh
set -eu

cd "$(dirname "$0")"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Нужен драйвер NVIDIA с поддержкой CUDA 12.8. Команда nvidia-smi не найдена." >&2
  exit 1
fi

$SUDO apt-get update
$SUDO apt-get install -y ffmpeg git python3 python3-venv python3-pip curl ca-certificates

python3 -m venv .venv

old=$(.venv/bin/python -m pip freeze | awk -F'[=[]' '/^(nvidia-|cuda-|triton)/ {print $1}')
if [ -n "$old" ]; then
  # shellcheck disable=SC2086
  .venv/bin/python -m pip uninstall -y $old
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | $SUDO sh
fi

if command -v systemctl >/dev/null 2>&1; then
  $SUDO systemctl enable --now ollama
fi

ollama pull gemma4:e2b

echo "Готово. Запуск: .venv/bin/python main.py"
