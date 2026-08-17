# Piłkarska AI v4

Automatyczny skaner prognoz dokładnego wyniku + Telegram.

## Co robi
- pobiera historię i nadchodzące mecze z API-Football,
- liczy formę, gole dom/wyjazd i Elo,
- tworzy rozkład prawdopodobieństwa wyników 0:0–6:6,
- wybiera maksymalnie 2 mecze,
- nadaje sygnał: BARDZO WYSOKI / WYSOKI / NISKI,
- zapamiętuje poprzedni wysłany kupon,
- wysyła nowy kupon tylko, jeśli jego ocena jest lepsza od poprzedniej o ustawiony próg,
- zapisuje historię prognoz.

## Ważne
Siła sygnału jest rankingiem modelu, a nie gwarancją wygranej. Dokładne wyniki
są bardzo trudnym celem. Przed użyciem do realnych zakładów trzeba zrobić
chronologiczny backtest na niewidzianych danych.

## Uruchomienie
1. Skopiuj `.env.example` do `.env`.
2. Wpisz API_FOOTBALL_KEY, TELEGRAM_BOT_TOKEN i TELEGRAM_CHAT_ID.
3. `pip install -r requirements.txt`
4. `python football_ai_v4.py`

Nie publikuj pliku `.env`.
