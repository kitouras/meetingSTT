# meetingSTT

## Описание

Приложение предназначено для обработки аудиозаписей встреч. Оно выполняет следующие функции:
- Диаризация аудиозаписей в формате WAV с использованием Pyannote.
- Транскрибация аудиозаписей с использованием GigaAM.
- Суммаризация полученной транскрипции с помощью большой языковой модели (по умолчанию Gemma3 12B).
- Предоставление пользователю результатов транскрибации и суммаризации.
- Возможность скачивания транскрипции и суммаризации в формате PDF.

Требования: Python >= 3.10, CUDA >= 11.8, git, [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## Предварительные шаги

1) Склонировать репозиторий через `git clone https://github.com/kitouras/meetingSTT.git` либо скачать архив через Code => Download ZIP и распаковать его.
2) Задать значение `hugging_face_token` в `settings.json` на свой [Hugging Face токен](https://huggingface.co/settings/tokens) для диаризации [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1) (нужно подтвердить лицензию)
2) Подтвердить [лицензию Gemma](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf), скачать модель [`gemma-3-12b-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf/blob/main/gemma-3-12b-it-q4_0.gguf) и поместить ее в папку `models`
3) Привести обрабатываемое аудио в нужный формат wav (если еще не приведено). Например для mp3 через утилиту [ffmpeg](https://ffmpeg.org/) (для других форматов аналогично):

```
ffmpeg -i /path/to/audio_name.mp3 -ar 16000 -ac 1 -acodec pcm_s16le meeting_audio.wav
```

## Запуск

1) Убедится, что Docker Desktop запущен (можно поставить в автозагрузку в докере через "Settings" - "General" - "Start Docker Desktop when you sign in to your computer")
2) Запустить `start.bat`, дождаться загрузки интерфейса.

## Остановка:

1) Закрыть вкладку с приложением, либо нажать `Ctrl+C` в том терминале, который запустил `start.bat`.

## Использование

1) В интерфейсе выбрать подготовленный wav-файл встречи и нажать кнопку "Суммаризировать".
2) Дождаться окончания обработки и получить саммари и транскрипцию
