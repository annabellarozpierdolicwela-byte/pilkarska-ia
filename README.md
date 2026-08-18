# Piłka AI / Master Of AI — wersja Render

## Najważniejsza poprawka
W tej wersji każde wywołanie API-Football `/fixtures` przekazuje parametr `season`.

Bot nie ma już twardo wpisanego `season=2024` dla aktualnych meczów. Dla każdej ustawionej ligi pobiera aktualny sezon przez `/leagues?id=...&current=true` i zapisuje go w cache na 24 godziny. Jeśli na początku sezonu aktualna liga ma zbyt mało zakończonych spotkań do modelu, bot może dodatkowo pobrać poprzedni sezon jako historię.

Nadchodzące mecze są pobierane przez `/fixtures` z `league + season + from/to`.
Mecze dnia są pobierane przez `/fixtures` z `league + season + date`.

## Komendy Telegram
- `/start`
- `/typy`
- `/wyniki`
- `/status`
- `/help`

## Render
Build Command:
`pip install -r requirements.txt`

Start Command:
`python football_ai_v4.py`

Nie zmieniaj tych poleceń.

## Environment Variables
W Render → Service → Environment powinny być ustawione:

- `API_FOOTBALL_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TIMEZONE=Europe/Warsaw`
- `SCAN_MINUTES=60`
- `MAX_MATCHES=2`
- `MIN_SCORE_PROB=0.08`
- `MIN_COUPON_IMPROVEMENT=0.02`
- `API_MIN_INTERVAL=7.0`
- `API_MAX_RETRIES=3`
- `LEAGUE_IDS=39,140,135,78,106,2`

Nie publikuj pliku `.env` i nie wpisuj prawdziwych kluczy do repozytorium.

## Telegram na Render
Bot używa webhooka. Po uruchomieniu Render automatycznie ustawia webhook na `/telegram/webhook`.
Nie uruchamiaj jednocześnie tego samego tokena Telegrama przez lokalny polling.

## Limity API
Kod ma odstęp między żądaniami, retry dla błędów 429 oraz cache sezonów, historii i nadchodzących meczów. Pierwsze wykonanie `/typy` po wygaśnięciu cache może wykonać więcej zapytań; kolejne wykonania korzystają z cache.

## Co sprawdzić po deployu
1. Otwórz Render i poczekaj na `Deploy successful`.
2. Wejdź w Telegram.
3. Wyślij `/status`.
4. Wyślij `/wyniki`.
5. Wyślij `/typy`.
6. Jeśli pojawi się błąd, sprawdź Render → Logs. Kod loguje błąd API zamiast ukrywać go.
