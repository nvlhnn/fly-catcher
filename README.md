# Fly Counter API

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

## Endpoint

`POST /count`

Form fields:
- `file`: image upload (multipart/form-data)
- `return_marked` (optional): `true` to receive a PNG with highlighted detections

Successful response:

- When `return_marked` is omitted or `false`:

```json
{
  "count": 3
}
```

- When `return_marked` is `true`:

Response is an `image/png` stream (filename `marked.png`).

## Prepare the code

Ensure your ASGI app is `app` in `app/main.py` (it is).

Add an ASGI entrypoint file at the repo root named `asgi.py` with the following contents:

```python
import os, sys
sys.path.append(os.path.dirname(__file__))
from app.main import app as application
```
