## Embedding Web Service (FastAPI)

Русский | English

### Описание / Description

- Простой веб‑сервис на FastAPI с POST `/embed`, который принимает текст и возвращает эмбеддинг.
- Использует локальную библиотеку эмбеддингов `fastembed` с моделью `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (подходит для русского и английского).

### Требования / Requirements

- Python 3.9+
- Интернет для первого скачивания модели (кэшируется локально)

### Установка / Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt  # Windows PowerShell
# или / or
source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

### Запуск / Run

```bash
./.venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload  # Windows
# или / or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload  # активированный venv
```

### Проверка / Healthcheck

```bash
curl http://127.0.0.1:8000/health
```

### Использование / Usage

POST `/embed`

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Пример текста / Example text"}'
```

Пример ответа / Example response:

```json
{
  "embedding": [0.0123, -0.0456, ...],
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "dim": 384
}
```

### Примечания / Notes

- Первая генерация может занять время из‑за загрузки модели. Модель кэшируется.
- Эндпоинт `/docs` содержит авто‑документацию Swagger UI.
