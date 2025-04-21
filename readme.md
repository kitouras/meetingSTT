## Установка

Требования: python >=3.10, Windows 11, Powershell, git, CUDA >=11.8

Установка: 
1) Установить [ffmpeg](https://www.ffmpeg.org/download.html) и добавить в PATH.
2) Запустить install.bat в Powershell и дождаться установки

## Запуск

1) Общие шаги:
    - Запустить сервер с LLM на 1234 порту (например, в [LM Studio](https://lmstudio.ai/)), модель можно задать в llm_api_model в `settings.json`, по умолчанию `yandexgpt-5-lite-8b-instruct`
    - Привести аудио в нужный формат wav, если оно в другом формате. Например для mp3 через утилиту [ffmpeg](https://ffmpeg.org/) (для других форматов аналогично):
        `ffmpeg -i /path/to/audio_name.mp3 -ar 16000 -ac 1 -acodec pcm_s16le meeting_audio.wav`
    - Задать значение hugging_face_token в `settings.json` на свой [Hugging Face токен](https://huggingface.co/settings/tokens)
2) Опция 1: локальный скрипт (`main.py`):
    - Запустить скрипт с нужным аудио через `.\.venv\Scripts\python.exe main.py {your_audio_name}.wav`
    - Вывод результатов диаризации, транскриции и суммаризации будет в консоли
    - В `last_transcription.txt` и `last_summary.txt`сохранятся последние транскрипция и саммари соответственно
2) Опция 2: локальный сервер (`api.py`):
    - Запустить скрипт через `.\.venv\Scripts\python.exe api.py`, дождаться загрузки моделей и интерфейса
    - В интерфейсе выбрать wav-файл встречи и нажать кнопку "Суммаризировать"
    - Дождаться окончания обработки и получить саммари
    - В `last_transcription.txt` и `last_summary.txt`сохранятся последние транскрипция и саммари соответственно
