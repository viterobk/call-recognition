# Распознавание звонков

Сервер с драйвером NVIDIA CUDA 13.3. Скрипт ставит ffmpeg, Python-окружение с PyTorch `2.14.1+cu132`, Ollama и модель `gemma4:e2b`. Драйвер NVIDIA скрипт не устанавливает: `nvidia-smi` уже должен видеть видеокарту. Колеса PyTorch под сам CUDA 13.3 нет, а сборка 13.4 требует драйвер новее. `cu132` на драйвере 13.3 работает.

Положите mp3 в `sample_sounds`. Приложение обрабатывает все файлы сразу и пишет результат в `results`.

## Скрипт

```bash
git clone git@github.com:viterobk/call-recognition.git
cd call-recognition
chmod +x deploy.sh
./deploy.sh
.venv/bin/python main.py
```

## Те же шаги вручную

```bash
git clone git@github.com:viterobk/call-recognition.git
cd call-recognition

sudo apt-get update
sudo apt-get install -y ffmpeg git python3 python3-venv python3-pip curl ca-certificates

python3 -m venv .venv

old=$(.venv/bin/python -m pip freeze | awk -F'[=[]' '/^(nvidia-|cuda-|triton)/ {print $1}')
if [ -n "$old" ]; then
  .venv/bin/python -m pip uninstall -y $old
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull gemma4:e2b

.venv/bin/python main.py
```

Удаление пакетов `nvidia-*`, `cuda-*` и `triton` нужно, если в этом окружении раньше стоял PyTorch под CUDA 12.8: пакеты `nvidia-*-cu12` конфликтуют с CUDA 13. На чистом окружении список пустой, и шаг ничего не удаляет.

Суммаризация идёт в 5 потоков. Приложение перезапускает службу `ollama` и выставляет ей 5 параллельных запросов и контекст 8192. Запускайте `main.py` от root или через sudo, иначе служба не перенастроится и останется один поток.
