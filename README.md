# Piłkarska AI — SAFE SOURCES / TEST

Ta wersja nie jest uzależniona od limitu API-Football.

## Źródła
1. TheSportsDB — bieżące/nadchodzące mecze i wyniki dnia. Darmowy klucz 123.
2. OpenFootball football.json — publiczna historia 2025/26 dla modelu, bez klucza.
3. API-Football — opcjonalny fallback, tylko jeśli ustawisz `API_FOOTBALL_KEY`.
4. Lokalny cache — jeśli źródło chwilowo nie odpowiada, bot nie kończy procesu błędem.

## Najważniejsza zmiana
`/typy` NIE MUSI zwrócić typu. Jeśli dane nie pokazują wyraźnej przewagi, bot odpowiada:

"🟡 BRAK MOCNEGO SYGNAŁU NA TERAZ"

Nie generuje wtedy kuponu na siłę.

## Progi
- `MIN_SIGNAL_PROB=0.57`
- `MIN_SIGNAL_EDGE=0.10`
- `MIN_HISTORY=50`

Można je później zmienić, ale na testach lepiej zostawić domyślne.

## Render
Build:
`pip install -r requirements.txt`

Start:
`python football_ai_v5.py`

Wymagane:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Opcjonalne:
- `API_FOOTBALL_KEY`

`THESPORTSDB_API_KEY` może pozostać jako `123`.

## Komendy
- `/start`
- `/typy`
- `/wyniki`
- `/status`
- `/help`

Ważne: żaden system predykcyjny nie może zagwarantować poprawności typów. Ta wersja ma przede wszystkim nie wymuszać typowania, gdy sygnał jest słaby.
