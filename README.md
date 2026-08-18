# Piłkarska AI — wersja testowa dla API-Football Free

Ta wersja jest przygotowana specjalnie do sprawdzenia, czy bot może działać bez płatnego planu API-Football.

## Najważniejsze
- historia modelu używa sezonu **2024**, który Twój plan Free zgłasza jako dostępny;
- **bieżące i nadchodzące mecze nie dostają parametru `season`** — są pobierane przez `from/to` albo `date`;
- automatyczny skaner jest domyślnie **WYŁĄCZONY**, żeby nie zużywać 100 zapytań/dzień planu Free;
- `/typy` uruchamia analizę tylko wtedy, gdy użytkownik wyśle komendę;
- `/wyniki` pobiera mecze dnia;
- cache ogranicza liczbę powtórnych zapytań;
- logi pokazują pozostały dzienny limit API, jeśli API zwraca ten nagłówek;
- Telegram na Renderze działa przez webhook.

## Dlaczego AUTO_SCAN jest wyłączony?
Plan Free ma 100 zapytań dziennie. Przy 6 ligach automatyczne skanowanie co godzinę mogłoby łatwo przekroczyć ten limit. Dlatego w tej wersji testujemy najpierw `/typy` i `/wyniki` na żądanie.

## Render
Build Command:
`pip install -r requirements.txt`

Start Command:
`python football_ai_v4.py`

## Environment Variables
W Render zachowaj swoje obecne zmienne:
- `API_FOOTBALL_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Dodatkowo wersja testowa może używać:
- `AUTO_SCAN=0` (zalecane)
- `API_MIN_INTERVAL=7`
- `API_MAX_RETRIES=3`

**Nie dodawaj pliku `.env` do repozytorium.**

## Test po deployu
1. `/start`
2. `/status`
3. `/wyniki`
4. `/typy`

Jeśli `/typy` zwróci błąd dotyczący sezonu, wklej dokładny komunikat z Telegrama oraz fragment Render Logs z momentu wykonania komendy.
