# Развертывание приложения

Требования: CUDA >= 11.8

## Предварительные шаги

1) Установить [Docker Desktop](https://www.docker.com/products/docker-desktop/) и запустить его
2) Задать значение `hugging_face_token` в `settings.json` на свой [Hugging Face токен](https://huggingface.co/settings/tokens) для диаризации [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1)
3) Скачать модель [`yandexgpt-5-lite-8b-instruct`](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF/blob/main/YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf) и поместить ее в папку `models`
4) Привести обрабатываемое аудио в нужный формат wav. Например для mp3 через утилиту [ffmpeg](https://ffmpeg.org/) (для других форматов аналогично):
    `ffmpeg -i /path/to/audio_name.mp3 -ar 16000 -ac 1 -acodec pcm_s16le meeting_audio.wav`

## Запуск

1) Открыть терминал в папке приложения и выполнить команду: `docker-compose up --build` (после сборки флаг `--build` можно опустить)
2) Дождаться сообщений о запуске сервера `llamacpp` и веб-сервера `app` (первый запуск будет долгим)

## Использование

1) Открыть браузер и перейти по адресу `http://127.0.0.1:5001`.
2) В интерфейсе выбрать подготовленный wav-файл встречи и нажать кнопку "Суммаризировать".
3) Дождаться окончания обработки и получить саммари
4) Транскрипция и саммари также сохраняются в файлы `last_transcription.txt` и `last_summary.txt` внутри контейнера `app`

## Остановка:

1) В терминале, где запущен `docker-compose`, нажмите `Ctrl+C`.
2) Чтобы удалить контейнеры и сеть (но не volume с моделью), выполните: `docker-compose down`
