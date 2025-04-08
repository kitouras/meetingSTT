## Установка

Требования: python >=3.10, Windows 11, Powershell, CUDA >=11.8

Установка: запустить install.bat в Powershell, дождаться установки

## Запуск

1) Общие шаги:
    - Запустить сервер с LLM на 1234 порту (например, в [LM Studio](https://lmstudio.ai/)), модель можно задать в llm_api_model в `settings.json`, по умолчанию `yandexgpt-5-lite-8b-instruct`
    - Привести аудио в нужный формат wav, если оно в другом формате. Например для mp3 через утилиту [ffmpeg](https://ffmpeg.org/) (для других форматов аналогично):
        `ffmpeg -i /path/to/audio_name.mp3 -ar 16000 -ac 1 -acodec pcm_s16le meeting_audio.wav`
    - Задать значение hugging_face_token в `settings.json` на свой [Hugging Face токен](https://huggingface.co/settings/tokens)
2) Опция 1: локальный скрипт (`main.py`):
    - Запустить скрипт с нужным аудио через `python main.py {your_audio_name}.wav`
    - Вывод результатов диаризации, транскриции и суммаризации будет в консоли
    - В `last_transcription.txt` и `last_summary.txt`сохранятся последние транскрипция и саммари соответственно
2) Опция 2: локальный сервер (`api.py`):
    - Запустить скрипт через `python api.py`, дождаться загрузки vosk модели (1-2 мин.)
    - Делать запросы на эндпоинт `http://127.0.0.1:5000/summarize` c прикрепленным к запросу файлом `audio_file`, например с помощью curl:
        `curl -X POST -F "audio_file=@{your_audio_name}.wav" http://127.0.0.1:5000/summarize`
    - В ответе на запрос будет содержаться json с саммари встречи
    - В консоли с сервером будет содержаться результаты диаризации, транскрипции и суммаризации
    - В `last_transcription.txt` и `last_summary.txt`сохранятся последние транскрипция и саммари соответственно
