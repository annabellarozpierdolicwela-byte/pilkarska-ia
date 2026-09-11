import os
import time
import json
import math
import logging
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / '.env')

API_KEY = os.getenv('API_FOOTBALL_KEY', '').strip()
TSDB_KEY = os.getenv('THESPORTSDB_API_KEY', '123').strip() or '123'
BASE_API = 'https://v3.football.api-sports.io'
TSDB_BASE = 'https://www.thesportsdb.com/api/v1/json'

TEAMS = [
    'Boca Juniors',
    'Central Cordoba',
    'Gimnasia Y Tiro de Salta',
    'Tristan Suarez',
]
PAIRS = [
    ('Boca Juniors', 'Central Cordoba'),
    ('Gimnasia Y Tiro de Salta', 'Tristan Suarez'),
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('history_probe')


def norm_team(name: str) -> str:
    x = str(name or '').lower().strip()
    for token in (' fc', ' afc', ' cf', ' sc', ' fk'):
        x = x.replace(token, '')
    x = ''.join(ch for ch in unicodedata.normalize('NFKD', x) if not unicodedata.combining(ch))
    aliases = {
        'gimnasia y tiro de salta': 'gimnasia y tiro',
        'gimnasia y tiro': 'gimnasia y tiro',
        'tristan suarez': 'tristan suarez',
        'central cordoba santiago del estero': 'central cordoba',
        'central cordoba de santiago del estero': 'central cordoba',
    }
    return aliases.get(x, x)


def api_get(path, params):
    if not API_KEY:
        raise RuntimeError('Brak API_FOOTBALL_KEY')
    r = requests.get(BASE_API + path, headers={'x-apisports-key': API_KEY}, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get('errors'):
        raise RuntimeError(str(payload['errors']))
    return payload.get('response', [])


def api_team_search(team):
    data = api_get('/teams', {'search': team})
    wanted = norm_team(team)
    exact = [x for x in data if norm_team((x.get('team') or {}).get('name')) == wanted]
    chosen = exact[0] if exact else (data[0] if data else None)
    return (chosen or {}).get('team', {}).get('id')


def api_history(team):
    tid = api_team_search(team)
    if not tid:
        raise RuntimeError(f'Nie znaleziono ID API-Football dla {team}')
    data = api_get('/fixtures', {'team': int(tid), 'last': 100, 'status': 'FT-AET-PEN'})
    rows = []
    wanted = norm_team(team)
    for f in data:
        teams = f.get('teams') or {}
        home = str((teams.get('home') or {}).get('name') or '').strip()
        away = str((teams.get('away') or {}).get('name') or '').strip()
        hg = pd.to_numeric((f.get('goals') or {}).get('home'), errors='coerce')
        ag = pd.to_numeric((f.get('goals') or {}).get('away'), errors='coerce')
        dt = pd.to_datetime((f.get('fixture') or {}).get('date'), utc=True, errors='coerce')
        if not home or not away or pd.isna(hg) or pd.isna(ag) or pd.isna(dt):
            continue
        rows.append({'date': dt, 'home': home, 'away': away, 'hg': float(hg), 'ag': float(ag),
                     'home_key': norm_team(home), 'away_key': norm_team(away), 'source': 'api-football'})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(['date','home_key','away_key']).sort_values('date')


def api_h2h(a, b):
    ai = api_team_search(a)
    bi = api_team_search(b)
    if not ai or not bi:
        return pd.DataFrame()
    data = api_get('/fixtures/headtohead', {'h2h': f'{int(ai)}-{int(bi)}', 'last': 20})
    rows = []
    for f in data:
        t = f.get('teams') or {}
        home = str((t.get('home') or {}).get('name') or '').strip()
        away = str((t.get('away') or {}).get('name') or '').strip()
        g = f.get('goals') or {}
        hg = pd.to_numeric(g.get('home'), errors='coerce')
        ag = pd.to_numeric(g.get('away'), errors='coerce')
        dt = pd.to_datetime((f.get('fixture') or {}).get('date'), utc=True, errors='coerce')
        if home and away and not pd.isna(hg) and not pd.isna(ag) and not pd.isna(dt):
            rows.append({'date': dt, 'home': home, 'away': away, 'hg': float(hg), 'ag': float(ag), 'source': 'api-football-h2h'})
    return pd.DataFrame(rows).drop_duplicates(['date','home','away']).sort_values('date') if rows else pd.DataFrame()


def tsdb_get(path, params=None):
    r = requests.get(f'{TSDB_BASE}/{TSDB_KEY}/{path}', params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def tsdb_history(team, limit=20):
    data = tsdb_get('searchteams.php', {'t': team})
    teams = data.get('teams') or []
    wanted = norm_team(team)
    chosen = next((x for x in teams if norm_team(x.get('strTeam')) == wanted), teams[0] if teams else None)
    if not chosen or not chosen.get('idTeam'):
        return pd.DataFrame()
    data = tsdb_get('eventslast.php', {'id': chosen['idTeam']})
    rows = []
    for e in (data.get('results') or data.get('events') or [])[:max(1, limit)]:
        home = str(e.get('strHomeTeam') or '').strip()
        away = str(e.get('strAwayTeam') or '').strip()
        hg = pd.to_numeric(e.get('intHomeScore'), errors='coerce')
        ag = pd.to_numeric(e.get('intAwayScore'), errors='coerce')
        dt = pd.to_datetime(e.get('strTimestamp') or f"{e.get('dateEvent','')} {e.get('strTime','')}" , utc=True, errors='coerce')
        if home and away and not pd.isna(hg) and not pd.isna(ag) and not pd.isna(dt):
            rows.append({'date': dt, 'home': home, 'away': away, 'hg': float(hg), 'ag': float(ag),
                         'home_key': norm_team(home), 'away_key': norm_team(away), 'source': 'thesportsdb'})
    return pd.DataFrame(rows).sort_values('date') if rows else pd.DataFrame()


def counts(df, team):
    if df.empty:
        return 0, 0
    key = norm_team(team)
    return int((df['home_key'] == key).sum()), int((df['away_key'] == key).sum())


def main():
    print('MASTER OF AI — HISTORY PROBE')
    print('Cel: 20 HOME + 20 AWAY + H2H dla 4 wskazanych drużyn.')
    print('Uwaga: Flashscore nie jest tu udawane jako API; tester mierzy realny zwrot z dostępnych API.')
    print()

    api_frames = {}
    for team in TEAMS:
        try:
            df = api_history(team)
            api_frames[team] = df
            h, a = counts(df, team)
            print(f'API-Football | {team:28s} HOME={h:2d} AWAY={a:2d} TOTAL={len(df):3d}')
        except Exception as exc:
            print(f'API-Football | {team:28s} ERROR: {exc}')
        time.sleep(0.2)

    print('\nAPI-Football H2H')
    for a, b in PAIRS:
        try:
            df = api_h2h(a, b)
            print(f'H2H | {a} — {b}: {len(df):2d} spotkań')
        except Exception as exc:
            print(f'H2H | {a} — {b}: ERROR: {exc}')

    print('\nTheSportsDB (kontrola)')
    for team in TEAMS:
        try:
            df = tsdb_history(team, 20)
            h, a = counts(df, team)
            print(f'TheSportsDB | {team:28s} HOME={h:2d} AWAY={a:2d} TOTAL={len(df):3d}')
        except Exception as exc:
            print(f'TheSportsDB | {team:28s} ERROR: {exc}')

    print('\nINTERPRETACJA')
    print('1) Jeżeli API-Football daje >=20 HOME dla gospodarza i >=20 AWAY dla gościa, te dane mogą zasilić MIN20.')
    print('2) Jeżeli daje mniej, tester pokazuje dokładnie gdzie brakuje historii.')
    print('3) H2H jest liczone osobno i nie jest sztucznie uzupełniane.')
    print('4) Flashscore wymaga osobnego, zgodnego z ich zasadami sposobu integracji; ten tester nie podszywa się pod takie API.')

if __name__ == '__main__':
    main()
