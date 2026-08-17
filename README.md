# Piłkarska AI v5 — Render + Telegram

## Render
Build Command:
`pip install -r requirements.txt`

Start Command:
`python football_ai_v4.py`

## Environment Variables
Ustaw:
- `API_FOOTBALL_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Pozostałe mają wartości domyślne z `render.yaml`.

## Telegram
Po uruchomieniu bot obsługuje:
- `/start`
- `/typy`
- `/wyniki`
- `/status`
- `/help`

Bot wykonuje też automatyczny skan zgodnie z `SCAN_MINUTES`.
