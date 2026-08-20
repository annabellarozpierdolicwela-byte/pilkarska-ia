import os
import time
import json
import math
import logging
import re
import threading
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# ============================================================
# MASTER OF AI v5
# Multi-source football predictor:
#   1) Football-Data.co.uk -> historical/current CSV data
#   2) API-Football -> small fallback for fixtures only
#   3) Local CSV cache -> no repeated downloads
#
# IMPORTANT:
# - Never put API keys directly in this file.
# - Put them in Render Environment Variables / .env locally.
# - This is a statistical predictor, NOT a guarantee of results.
# ============================================================

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
TSDB_KEY = os.getenv("THESPORTSDB_API_KEY", "123").strip() or "123"
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TZ = os.getenv("TIMEZONE", "Europe/Warsaw")
SCAN = int(os.getenv("SCAN_MINUTES", "120"))
AUTO_SCAN = os.getenv("AUTO_SCAN", "0").lower() in ("1", "true", "yes", "on")

# Safety: never allow the bot to intentionally use more than this many
# API-Football requests in one UTC day.
API_DAILY_BUDGET = int(os.getenv("API_DAILY_BUDGET", "35"))
API_MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "7.0"))

MAXM = int(os.getenv("MAX_MATCHES", "3"))
MINP = float(os.getenv("MIN_SCORE_PROB", "0.05"))

# STRICT EXACT-SCORE MODE
# - The active coupon is always exactly 2 matches.
# - Both legs are exact-score predictions.
# - The same 2 fixtures are kept while they remain upcoming.
# - A later signal is accepted only when BOTH legs remain at least as strong
#   as the current pair and at least one leg becomes meaningfully stronger.
# - Weak exact-score candidates are rejected and never announced.
EXACT_SCORE_MIN_PROB = float(os.getenv("EXACT_SCORE_MIN_PROB", "0.10"))
EXACT_SCORE_MIN_AGREEMENT = float(os.getenv("EXACT_SCORE_MIN_AGREEMENT", "0.75"))
EXACT_SCORE_MIN_EDGE = float(os.getenv("EXACT_SCORE_MIN_EDGE", "0.02"))
EXACT_SCORE_HISTORY_MIN = int(os.getenv("EXACT_SCORE_HISTORY_MIN", "6"))

# Exact-score signal strength tiers.
STRONG_SIGNAL_P = 0.75
VERY_STRONG_SIGNAL_P = 0.85
INCREDIBLE_SIGNAL_P = 0.92

# User requested a fixed hourly check.
SIGNAL_CHECK_MINUTES = 60

# Current season is determined automatically:
# football season 2026/27 => 2026
CURRENT_SEASON = int(os.getenv("CURRENT_SEASON", str(datetime.now().year)))

# Football-Data league codes.
# Champions League is intentionally NOT included here because
# Football-Data's standard league CSVs do not cover every competition.
FD_LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "B1": "Belgian Pro League",
    "G1": "Greece",
    "T1": "Turkey",
    "SC0": "Scottish Premiership",
    "PL": "Poland",
}

# API-Football IDs corresponding to the old project.
# They are used ONLY as a fallback for fixtures.
API_LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    88: "Eredivisie",
    94: "Primeira Liga",
    106: "Ekstraklasa",
    2: "UEFA Champions League",
}

BASE_API = "https://v3.football.api-sports.io"
FD_BASE = "https://www.football-data.co.uk/mmz4281"

log = logging.getLogger("master_of_ai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_api_lock = threading.Lock()
_api_last_call = 0.0
_api_used_file = DATA / "api_usage.json"

HISTORY_CACHE = DATA / "master_history.csv"
FIXTURE_CACHE = DATA / "upcoming.csv"
PREDICTIONS_LOG = DATA / "predictions.csv"
LAST_COUPON = DATA / "last_coupon.json"
LAST_SIGNAL = DATA / "last_signal.json"
EXACT_PAIR_STATE = DATA / "exact_pair_state.json"

HISTORY_CACHE_HOURS = int(os.getenv("HISTORY_CACHE_HOURS", "12"))
FIXTURE_CACHE_MINUTES = int(os.getenv("FIXTURE_CACHE_MINUTES", "15"))
TSDB_DAYS_AHEAD = int(os.getenv("TSDB_DAYS_AHEAD", "7"))
# Do not reject a real fixture merely because a team has little/no history.
# The predictor already contains sensible league-average priors for this case.
MIN_TEAM_HISTORY = int(os.getenv("MIN_TEAM_HISTORY", "0"))
TSDB_BASE = "https://www.thesportsdb.com/api/v1/json"

# ============================================================
# BASIC HELPERS
# ============================================================

def now():
    return datetime.now(ZoneInfo(TZ))


def season_code(year):
    """2025 -> '2526', 2026 -> '2627'."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def norm_team(name):
    """Normalizes team names so different sources can be matched."""
    if not isinstance(name, str):
        return ""
    x = (
        name.lower()
        .replace(" fc", "")
        .replace(" afc", "")
        .replace(" cf", "")
        .replace(" sc", "")
        .replace(" fk", "")
        .replace("  ", " ")
        .strip()
    )
    aliases = {
        "man united": "manchester united",
        "man utd": "manchester united",
        "man city": "manchester city",
        "tottenham": "tottenham hotspur",
        "spurs": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "psg": "paris saint-germain",
        "paris sg": "paris saint-germain",
        "inter": "internazionale",
        "internazionale": "internazionale",
        "ath madrid": "atletico madrid",
        "athletic bilbao": "athletic club",
        "ath bilbao": "athletic club",
        "bayern munich": "bayern munich",
        "bayern münchen": "bayern munich",
    }
    return aliases.get(x, x)


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


# ============================================================
# API-FOOTBALL: HARD BUDGET
# ============================================================

def _utc_day():
    return datetime.utcnow().strftime("%Y-%m-%d")


def _load_api_usage():
    if not _api_used_file.exists():
        return {"date": _utc_day(), "used": 0}
    try:
        data = json.loads(_api_used_file.read_text(encoding="utf-8"))
        if data.get("date") != _utc_day():
            return {"date": _utc_day(), "used": 0}
        return data
    except Exception:
        return {"date": _utc_day(), "used": 0}


def _save_api_usage(data):
    _api_used_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def api_budget_available():
    return _load_api_usage()["used"] < API_DAILY_BUDGET


def api(path, params):
    """
    API-Football fallback with a strict local daily budget.
    If the budget is exhausted, the application DOES NOT call the API.
    """
    global _api_last_call

    if not API_KEY:
        raise RuntimeError("Brak API_FOOTBALL_KEY — API-Football jest wyłączone.")

    usage = _load_api_usage()
    if usage["used"] >= API_DAILY_BUDGET:
        raise RuntimeError(
            f"Lokalny limit bezpieczeństwa API-Football osiągnięty "
            f"({usage['used']}/{API_DAILY_BUDGET})."
        )

    with _api_lock:
        wait = API_MIN_INTERVAL - (time.monotonic() - _api_last_call)
        if wait > 0:
            time.sleep(wait)

        _api_last_call = time.monotonic()

        r = requests.get(
            BASE_API + path,
            headers={"x-apisports-key": API_KEY},
            params=params,
            timeout=30,
        )

        # Count every actual API request.
        usage["used"] += 1
        _save_api_usage(usage)

    if r.status_code == 429:
        raise RuntimeError("API-Football zwróciło 429 — limit/rate limit.")

    r.raise_for_status()
    payload = r.json()

    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))

    remaining = r.headers.get("x-ratelimit-requests-remaining")
    if remaining is not None:
        log.info("API-Football remaining: %s", remaining)

    return payload.get("response", [])


# ============================================================
# FOOTBALL-DATA.CO.UK
# ============================================================

def football_data_url(season_year, league_code):
    return f"{FD_BASE}/{season_code(season_year)}/{league_code}.csv"


def download_fd(season_year, league_code):
    """
    Downloads one CSV and caches it locally.
    Football-Data publishes CSV files for many leagues/seasons,
    including current and historical seasons.
    """
    cache = DATA / f"fd_{season_year}_{league_code}.csv"

    # Keep an existing file unless explicitly refreshed.
    if cache.exists() and time.time() - cache.stat().st_mtime < 24 * 3600:
        try:
            return pd.read_csv(cache, encoding="latin1")
        except Exception:
            pass

    url = football_data_url(season_year, league_code)

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) < 100:
            return pd.DataFrame()

        cache.write_bytes(r.content)

        return pd.read_csv(
            cache,
            encoding="latin1",
            on_bad_lines="skip",
        )
    except Exception as exc:
        log.warning("Football-Data %s %s: %s", season_year, league_code, exc)
        return pd.DataFrame()


def fd_to_internal(df, league_code, season_year):
    """
    Converts Football-Data columns to the common format used by the model.
    """
    if df.empty:
        return pd.DataFrame()

    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce",
    )
    out["home"] = df["HomeTeam"].astype(str).str.strip()
    out["away"] = df["AwayTeam"].astype(str).str.strip()
    out["hg"] = pd.to_numeric(df["FTHG"], errors="coerce")
    out["ag"] = pd.to_numeric(df["FTAG"], errors="coerce")
    out["league_code"] = league_code
    out["league"] = FD_LEAGUES.get(league_code, league_code)
    out["season"] = season_year

    # Optional market probabilities.
    if "B365H" in df.columns:
        out["odd_h"] = pd.to_numeric(df["B365H"], errors="coerce")
    else:
        out["odd_h"] = float("nan")

    if "B365D" in df.columns:
        out["odd_d"] = pd.to_numeric(df["B365D"], errors="coerce")
    else:
        out["odd_d"] = float("nan")

    if "B365A" in df.columns:
        out["odd_a"] = pd.to_numeric(df["B365A"], errors="coerce")
    else:
        out["odd_a"] = float("nan")

    # For historical model training, only completed matches are useful.
    out = out.dropna(subset=["date", "hg", "ag"])
    return out


def load_history():
    """
    Builds a local historical database from several free CSV sources.
    We use multiple seasons so the model is not trained only on 2024.
    """
    seasons_back = int(os.getenv("HISTORY_SEASONS", "5"))
    frames = []

    for year in range(CURRENT_SEASON - seasons_back, CURRENT_SEASON + 1):
        for code in FD_LEAGUES:
            raw = download_fd(year, code)
            converted = fd_to_internal(raw, code, year)
            if not converted.empty:
                frames.append(converted)

    if not frames:
        if HISTORY_CACHE.exists():
            try:
                return pd.read_csv(HISTORY_CACHE)
            except Exception:
                pass
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["home_key"] = df["home"].map(norm_team)
    df["away_key"] = df["away"].map(norm_team)

    df = (
        df.drop_duplicates(
            subset=["date", "home_key", "away_key"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    try:
        df.to_csv(HISTORY_CACHE, index=False)
    except Exception:
        pass

    return df


# ============================================================
# UPCOMING FIXTURES
# ============================================================

def parse_fd_upcoming():
    """
    Football-Data CSVs also contain future fixtures in many current-season
    files. This is our first choice because it costs zero API-Football calls.
    """
    frames = []

    # Current season first; previous season is useful around season boundaries.
    for code in FD_LEAGUES:
        raw = download_fd(CURRENT_SEASON, code)
        if raw.empty:
            continue

        if not {"Date", "HomeTeam", "AwayTeam"}.issubset(raw.columns):
            continue

        x = pd.DataFrame()
        x["date"] = pd.to_datetime(
            raw["Date"],
            dayfirst=True,
            errors="coerce",
        )
        x["home"] = raw["HomeTeam"].astype(str).str.strip()
        x["away"] = raw["AwayTeam"].astype(str).str.strip()

        if "FTHG" in raw.columns:
            x["hg"] = pd.to_numeric(raw["FTHG"], errors="coerce")
        else:
            x["hg"] = float("nan")

        if "FTAG" in raw.columns:
            x["ag"] = pd.to_numeric(raw["FTAG"], errors="coerce")
        else:
            x["ag"] = float("nan")

        x["league"] = FD_LEAGUES.get(code, code)
        x["source"] = "football-data"
        x["status"] = x.apply(
            lambda r: "FT"
            if pd.notna(r.hg) and pd.notna(r.ag)
            else "NS",
            axis=1,
        )

        frames.append(x)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date"])

    # Normalize all fixture dates to UTC before comparing them.
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = dates
    df = df.dropna(subset=["date"])

    # Keep upcoming games only.
    start = pd.Timestamp.now(tz="UTC")
    df = df[df["date"] >= start]

    return df.sort_values("date").drop_duplicates(
        subset=["date", "home", "away"],
    )


def api_fallback_fixtures():
    """
    API-Football is used only when Football-Data did not provide enough
    upcoming fixtures. We deliberately use today's date + next 2 days.
    Maximum calls are controlled by API_DAILY_BUDGET.
    """
    if not API_KEY or not api_budget_available():
        return pd.DataFrame()

    frames = []
    start = now().date()
    end = start + timedelta(days=2)

    for lid, league_name in API_LEAGUES.items():
        if not api_budget_available():
            break

        try:
            data = api(
                "/fixtures",
                {
                    "league": lid,
                    "season": CURRENT_SEASON,
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                    "timezone": TZ,
                },
            )

            for f in data:
                teams = f.get("teams", {})
                goals = f.get("goals", {})
                fixture = f.get("fixture", {})
                status = f.get("fixture", {}).get("status", {}).get("short", "")

                frames.append({
                    "id": fixture.get("id"),
                    "date": fixture.get("date"),
                    "home": teams.get("home", {}).get("name"),
                    "away": teams.get("away", {}).get("name"),
                    "hg": goals.get("home"),
                    "ag": goals.get("away"),
                    "league": league_name,
                    "status": status,
                    "source": "api-football",
                })

        except Exception as exc:
            log.warning("API fallback league %s: %s", lid, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["date"])


# ============================================================
# THESPORTSDB — LIVE UPCOMING FIXTURES
# ============================================================

def tsdb_get(path, params=None):
    url = f"{TSDB_BASE}/{TSDB_KEY}/{path}"
    r = requests.get(url, params=params or {}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and payload.get("message") and not payload.get("events"):
        log.warning("TheSportsDB %s: %s", path, payload.get("message"))
    return payload


def _tsdb_event_datetime(event):
    # Prefer the API timestamp when available. Fall back to local date + time.
    stamp = event.get("strTimestamp")
    if stamp:
        dt = pd.to_datetime(stamp, utc=True, errors="coerce")
        if pd.notna(dt):
            return dt

    date_event = str(event.get("dateEvent") or "").strip()
    str_time = str(event.get("strTime") or "").strip()
    if not date_event:
        return pd.NaT

    raw = f"{date_event} {str_time}".strip()
    dt = pd.to_datetime(raw, errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    try:
        return dt.tz_localize(TZ).tz_convert("UTC")
    except Exception:
        return pd.Timestamp(dt, tz="UTC")


def parse_tsdb_upcoming():
    """
    Gets real upcoming football matches from TheSportsDB.
    The free API supports day-based schedules; seven days gives the bot
    a dependable live fixture pool even when Football-Data has no current
    season file yet.
    """
    frames = []
    today = now().date()
    days = max(1, min(TSDB_DAYS_AHEAD, 10))

    for offset in range(days):
        day = today + timedelta(days=offset)
        try:
            payload = tsdb_get("eventsday.php", {
                "d": day.isoformat(),
                "s": "Soccer",
            })
        except Exception as exc:
            log.warning("TheSportsDB %s: %s", day, exc)
            continue

        events = payload.get("events") or []
        for e in events:
            home = str(e.get("strHomeTeam") or "").strip()
            away = str(e.get("strAwayTeam") or "").strip()
            if not home or not away:
                continue

            dt = _tsdb_event_datetime(e)
            if pd.isna(dt):
                continue

            status = str(e.get("strStatus") or "").upper().strip()
            hg = safe_float(e.get("intHomeScore"))
            ag = safe_float(e.get("intAwayScore"))
            if status in {"FT", "AET", "PEN", "CANC", "CANCELLED", "POSTPONED"}:
                if status in {"FT", "AET", "PEN"}:
                    match_status = status
                else:
                    match_status = status
            else:
                match_status = "NS"

            frames.append({
                "id": e.get("idEvent"),
                "date": dt,
                "home": home,
                "away": away,
                "hg": hg,
                "ag": ag,
                "league": str(e.get("strLeague") or ""),
                "source": "thesportsdb",
                "status": match_status,
            })

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= pd.Timestamp.now(tz="UTC")]
    df["home_key"] = df["home"].map(norm_team)
    df["away_key"] = df["away"].map(norm_team)
    return df.sort_values("date").drop_duplicates(
        subset=["date", "home_key", "away_key"], keep="first"
    )


def upcoming():
    """
    Zero API calls in the normal case.
    Cache is used so repeated /typy commands do not download again.
    """
    if FIXTURE_CACHE.exists():
        age = time.time() - FIXTURE_CACHE.stat().st_mtime
        if age < FIXTURE_CACHE_MINUTES * 60:
            try:
                cached = pd.read_csv(FIXTURE_CACHE)
                cached["date"] = pd.to_datetime(
                    cached["date"], utc=True, errors="coerce"
                )
                cached = cached.dropna(subset=["date"]).sort_values("date")
                cached = cached[cached["date"] >= pd.Timestamp.now(tz="UTC")]
                # Old cache files from previous versions did not contain the
                # live TheSportsDB source. Rebuild those instead of trusting
                # potentially incomplete fixture data.
                has_live_source = "source" in cached.columns and cached["source"].astype(str).str.contains("thesportsdb", case=False, na=False).any()
                if not cached.empty and has_live_source:
                    return cached
            except Exception:
                pass

    sources = []

    # 1) TheSportsDB: live fixtures.
    tsdb_df = parse_tsdb_upcoming()
    if not tsdb_df.empty:
        sources.append(tsdb_df)

    # 2) Football-Data: useful when its current CSV already contains fixtures.
    fd_df = parse_fd_upcoming()
    if not fd_df.empty:
        sources.append(fd_df)

    # 3) API-Football: last-resort fallback.
    if sum(len(x) for x in sources) < 3:
        api_df = api_fallback_fixtures()
        if not api_df.empty:
            sources.append(api_df)

    if not sources:
        return pd.DataFrame()

    df = pd.concat(sources, ignore_index=True)

    if df.empty:
        return df

    df["home_key"] = df["home"].map(norm_team)
    df["away_key"] = df["away"].map(norm_team)
    # Prefer live TheSportsDB over static/duplicated copies of the same game.
    source_rank = {"thesportsdb": 0, "api-football": 1, "football-data": 2}
    df["_source_rank"] = df.get("source", "football-data").map(source_rank).fillna(9)
    df = df.sort_values(["date", "_source_rank"])
    df = df.drop_duplicates(
        subset=["date", "home_key", "away_key"],
        keep="first",
    ).drop(columns=["_source_rank"], errors="ignore")

    try:
        df.to_csv(FIXTURE_CACHE, index=False)
    except Exception:
        pass

    return df


# ============================================================
# STATISTICAL ENGINE
# ============================================================

def poisson(k, lam):
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def recent_team_matches(df, team, n=12, before=None):
    x = df[
        (df.home_key == norm_team(team)) |
        (df.away_key == norm_team(team))
    ].copy()

    if before is not None:
        b = pd.to_datetime(before, utc=True, errors="coerce")
        dates = pd.to_datetime(x["date"], utc=True, errors="coerce")
        x = x[dates < b]

    return x.sort_values("date").tail(n)


def team_strength(df, team, before=None):
    """
    Returns attack, defence, form and home/away-specific averages.
    """
    x = recent_team_matches(df, team, n=15, before=before)

    if x.empty:
        return {
            "gf": 1.20,
            "ga": 1.20,
            "form": 0.50,
            "home_gf": 1.20,
            "home_ga": 1.20,
            "away_gf": 1.20,
            "away_ga": 1.20,
            "matches": 0,
        }

    rows = []
    for _, r in x.iterrows():
        if norm_team(r.home) == norm_team(team):
            gf, ga = r.hg, r.ag
            venue = "home"
        else:
            gf, ga = r.ag, r.hg
            venue = "away"

        rows.append((gf, ga, venue))

    z = pd.DataFrame(rows, columns=["gf", "ga", "venue"])

    # Recency weighting: recent matches matter more.
    weights = [0.65 ** i for i in range(len(z) - 1, -1, -1)]
    z["w"] = weights

    def wmean(col):
        return float((z[col] * z["w"]).sum() / z["w"].sum())

    gf = wmean("gf")
    ga = wmean("ga")

    points = []
    for _, r in z.iterrows():
        points.append(
            3 if r.gf > r.ga else 1 if r.gf == r.ga else 0
        )
    z["points"] = points
    form = float((z["points"] * z["w"]).sum() / (3 * z["w"].sum()))

    home_z = z[z.venue == "home"]
    away_z = z[z.venue == "away"]

    home_gf = (
        float((home_z.gf * home_z.w).sum() / home_z.w.sum())
        if not home_z.empty else gf
    )
    home_ga = (
        float((home_z.ga * home_z.w).sum() / home_z.w.sum())
        if not home_z.empty else ga
    )
    away_gf = (
        float((away_z.gf * away_z.w).sum() / away_z.w.sum())
        if not away_z.empty else gf
    )
    away_ga = (
        float((away_z.ga * away_z.w).sum() / away_z.w.sum())
        if not away_z.empty else ga
    )

    return {
        "gf": gf,
        "ga": ga,
        "form": form,
        "home_gf": home_gf,
        "home_ga": home_ga,
        "away_gf": away_gf,
        "away_ga": away_ga,
        "matches": len(z),
    }


def elo_ratings(df, before=None):
    """
    Elo calculated chronologically. If before is supplied, future matches
    are excluded to avoid data leakage.
    """
    x = df.sort_values("date").copy()
    if before is not None:
        b = pd.to_datetime(before, utc=True, errors="coerce")
        dates = pd.to_datetime(x["date"], utc=True, errors="coerce")
        x = x[dates < b]

    elo = {}

    for _, r in x.iterrows():
        h = norm_team(r.home)
        a = norm_team(r.away)

        rh = elo.get(h, 1500.0)
        ra = elo.get(a, 1500.0)

        expected_h = 1 / (1 + 10 ** ((ra - rh - 55) / 400))
        actual_h = (
            1.0 if r.hg > r.ag
            else 0.5 if r.hg == r.ag
            else 0.0
        )

        margin = max(
            1.0,
            math.log1p(abs(float(r.hg) - float(r.ag))) * 1.6,
        )

        k = 22 * margin

        elo[h] = rh + k * (actual_h - expected_h)
        elo[a] = ra + k * ((1 - actual_h) - (1 - expected_h))

    return elo


def league_goal_average(df, league, before=None):
    x = df[df.league == league].copy()

    if before is not None:
        b = pd.to_datetime(before, utc=True, errors="coerce")
        dates = pd.to_datetime(x["date"], utc=True, errors="coerce")
        x = x[dates < b]

    x = x.dropna(subset=["hg", "ag"]).tail(800)

    if x.empty:
        return 1.35, 1.10

    return float(x.hg.mean()), float(x.ag.mean())


def h2h_adjustment(df, home, away, before=None):
    """
    Small H2H adjustment only. H2H is deliberately low-weight because
    old meetings are much less informative than current team strength.
    """
    hk = norm_team(home)
    ak = norm_team(away)

    x = df[
        (
            (df.home_key == hk) & (df.away_key == ak)
        ) |
        (
            (df.home_key == ak) & (df.away_key == hk)
        )
    ].copy()

    if before is not None:
        b = pd.to_datetime(before, utc=True, errors="coerce")
        dates = pd.to_datetime(x["date"], utc=True, errors="coerce")
        x = x[dates < b]

    x = x.sort_values("date").tail(6)

    if len(x) < 2:
        return 1.0, 1.0

    home_goals = []
    away_goals = []

    for _, r in x.iterrows():
        if norm_team(r.home) == hk:
            home_goals.append(r.hg)
            away_goals.append(r.ag)
        else:
            home_goals.append(r.ag)
            away_goals.append(r.hg)

    h = sum(home_goals) / len(home_goals)
    a = sum(away_goals) / len(away_goals)

    # Very conservative influence.
    return (
        max(0.90, min(1.10, 1 + (h - 1.35) * 0.035)),
        max(0.90, min(1.10, 1 + (a - 1.05) * 0.035)),
    )


def market_probabilities(row):
    """
    Uses Football-Data odds only as a weak extra signal.
    We do NOT allow odds to dominate the statistical model.
    """
    oh = safe_float(row.get("odd_h", float("nan")))
    od = safe_float(row.get("odd_d", float("nan")))
    oa = safe_float(row.get("odd_a", float("nan")))

    if not all(math.isfinite(x) and x > 1 for x in (oh, od, oa)):
        return None

    inv = [1 / oh, 1 / od, 1 / oa]
    s = sum(inv)

    return {
        "h": inv[0] / s,
        "d": inv[1] / s,
        "a": inv[2] / s,
    }


def score_matrix(lh, la, max_goals=7):
    matrix = []

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson(h, lh) * poisson(a, la)
            matrix.append((p, h, a))

    return sorted(matrix, reverse=True)


def _empirical_goal_distribution(df, team, venue, before=None, n=20, max_goals=7):
    """Weighted empirical goal distribution for one team at a specific venue."""
    x = recent_team_matches(df, team, n=n, before=before).copy()
    wanted = norm_team(team)

    if venue == "home":
        x = x[x["home_key"] == wanted]
        goals = pd.to_numeric(x["hg"], errors="coerce").dropna().tolist()
    else:
        x = x[x["away_key"] == wanted]
        goals = pd.to_numeric(x["ag"], errors="coerce").dropna().tolist()

    if not goals:
        return [1.0 / (max_goals + 1)] * (max_goals + 1), 0

    # Recent matches receive more weight; Laplace smoothing prevents zero cells.
    weights = [0.72 ** i for i in range(len(goals) - 1, -1, -1)]
    buckets = [0.75] * (max_goals + 1)
    for g, w in zip(goals, weights):
        idx = int(max(0, min(max_goals, round(float(g)))))
        buckets[idx] += w

    total = sum(buckets)
    return [b / total for b in buckets], len(goals)


def _score_agreement(blended, poisson_dist, empirical_dist):
    """How strongly the independent score models agree on the best score."""
    p_score = (poisson_dist[0][1], poisson_dist[0][2])
    e_matrix = sorted(
        [(p, h, a) for h, p_h in enumerate(empirical_dist["home"])
         for a, p_a in enumerate(empirical_dist["away"])
         for p in [p_h * p_a]],
        reverse=True,
    )
    e_score = (e_matrix[0][1], e_matrix[0][2])

    if p_score == e_score:
        return 1.0
    if abs(p_score[0] - e_score[0]) + abs(p_score[1] - e_score[1]) == 1:
        return 0.75
    if abs(p_score[0] - e_score[0]) + abs(p_score[1] - e_score[1]) == 2:
        return 0.50
    return 0.25


def _exact_score_matrix(lh, la, home_emp, away_emp, max_goals=7):
    """Blend Poisson and empirical venue-specific scoring distributions."""
    poisson_rows = []
    blended = []

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            pp = poisson(h, lh) * poisson(a, la)
            ep = home_emp[h] * away_emp[a]
            # 65% structural Poisson model + 35% empirical form model.
            p = 0.65 * pp + 0.35 * ep
            poisson_rows.append((pp, h, a))
            blended.append((p, h, a))

    return sorted(blended, reverse=True), sorted(poisson_rows, reverse=True)


def predict(df, home, away, league=None, match_date=None):
    """
    Strict exact-score model.

    Inputs:
      - recent recency-weighted goals
      - home/away-specific performance
      - Elo strength
      - recent form
      - H2H
      - league scoring environment
      - empirical venue-specific goal distributions
      - Poisson score distribution

    The exact-score signal is announced only when the models agree enough
    and the exact-score probability clears a conservative threshold.
    """
    hs = team_strength(df, home, before=match_date)
    aws = team_strength(df, away, before=match_date)
    elo = elo_ratings(df, before=match_date)

    eh = elo.get(norm_team(home), 1500.0)
    ea = elo.get(norm_team(away), 1500.0)

    if league:
        lg_h, lg_a = league_goal_average(df, league, before=match_date)
    else:
        lg_h, lg_a = 1.35, 1.10

    # Base attacking/defensive blend.
    home_attack = 0.55 * hs["home_gf"] + 0.45 * hs["gf"]
    home_defence = 0.55 * hs["home_ga"] + 0.45 * hs["ga"]
    away_attack = 0.55 * aws["away_gf"] + 0.45 * aws["gf"]
    away_defence = 0.55 * aws["away_ga"] + 0.45 * aws["ga"]

    lh = 0.62 * home_attack + 0.38 * away_defence
    la = 0.62 * away_attack + 0.38 * home_defence

    # League environment stabilizes sparse team samples.
    lh = 0.78 * lh + 0.22 * lg_h
    la = 0.78 * la + 0.22 * lg_a

    # Elo + form.
    elo_diff = (eh + 55 - ea) / 400
    form_diff = hs["form"] - aws["form"]
    lh *= math.exp(0.16 * elo_diff + 0.07 * form_diff)
    la *= math.exp(-0.12 * elo_diff - 0.05 * form_diff)

    # H2H: intentionally small.
    h2h_h, h2h_a = h2h_adjustment(df, home, away, before=match_date)
    lh *= h2h_h
    la *= h2h_a

    lh = max(0.20, min(4.20, lh))
    la = max(0.15, min(3.80, la))

    home_emp, home_n = _empirical_goal_distribution(
        df, home, "home", before=match_date, n=20
    )
    away_emp, away_n = _empirical_goal_distribution(
        df, away, "away", before=match_date, n=20
    )

    blended_dist, poisson_dist = _exact_score_matrix(
        lh, la, home_emp, away_emp, max_goals=7
    )

    # Independent empirical score model for agreement checking.
    empirical_model = {"home": home_emp, "away": away_emp}
    agreement = _score_agreement(
        blended_dist, poisson_dist, empirical_model
    )

    raw_p, best_h, best_a = blended_dist[0]
    second_p = blended_dist[1][0] if len(blended_dist) > 1 else 0.0
    margin = max(0.0, raw_p - second_p)

    # Exact-score confidence is not a claim of true hit probability.
    history_quality = min(
        1.0,
        (home_n + away_n) / max(1, 2 * EXACT_SCORE_HISTORY_MIN),
    )
    confidence = (
        0.55 * min(raw_p / 0.18, 1.0)
        + 0.25 * agreement
        + 0.10 * min(margin / 0.04, 1.0)
        + 0.10 * history_quality
    )
    confidence = max(0.0, min(1.0, confidence))

    top_scores = []
    for p, h, a in blended_dist[:8]:
        top_scores.append({
            "score": f"{h}:{a}",
            "prob": float(p),
        })

    # 1X2 is kept internally for diagnostics/backtesting only.
    p_home = sum(p for p, h, a in blended_dist if h > a)
    p_draw = sum(p for p, h, a in blended_dist if h == a)
    p_away = sum(p for p, h, a in blended_dist if h < a)

    return {
        "home": home,
        "away": away,
        "score": f"{best_h}:{best_a}",
        "prob": float(raw_p),
        "raw_prob": float(raw_p),
        "score_confidence": float(confidence),
        "score_agreement": float(agreement),
        "score_margin": float(margin),
        "lh": lh,
        "la": la,
        "elo_home": eh,
        "elo_away": ea,
        "elo_diff": eh - ea,
        "form_home": hs["form"],
        "form_away": aws["form"],
        "matches_home": hs["matches"],
        "matches_away": aws["matches"],
        "venue_history_home": home_n,
        "venue_history_away": away_n,
        "league_goal_home": lg_h,
        "league_goal_away": lg_a,
        "p1x2": {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        },
        "recommended_result": (
            "1" if p_home >= p_draw and p_home >= p_away
            else "X" if p_draw >= p_home and p_draw >= p_away
            else "2"
        ),
        "top_scores": top_scores,
    }


# ============================================================
# BETTING MARKETS / MATCH OPTIONS
# ============================================================

def poisson_cdf(k, lam):
    if k < 0:
        return 0.0
    return sum(poisson(i, lam) for i in range(k + 1))


def over_probability(lam, line):
    """Probability of Over line for standard .5 goal totals."""
    if line % 1 != 0.5:
        return None
    return 1.0 - poisson_cdf(int(line), lam)


def under_probability(lam, line):
    if line % 1 != 0.5:
        return None
    return poisson_cdf(int(line), lam)


def result_probability_from_lambdas(lh, la):
    dist = score_matrix(lh, la, 8)
    home = sum(p for p, h, a in dist if h > a)
    draw = sum(p for p, h, a in dist if h == a)
    away = sum(p for p, h, a in dist if h < a)
    return {"1": home, "X": draw, "2": away}


def market_options_from_prediction(pred):
    """
    Builds bettable options from the same Poisson goal model.
    These are MODEL probabilities, not bookmaker odds.
    """
    lh = float(pred["lh"])
    la = float(pred["la"])
    total = lh + la

    # Approximate first/second-half scoring intensity. The model deliberately
    # keeps this conservative because the historical CSV has no minute-level
    # goal timing for every source.
    first_h = total * 0.44
    second_h = total * 0.56

    totals = []
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        over = over_probability(total, line)
        under = under_probability(total, line)
        totals.append({
            "line": line,
            "over": over,
            "under": under,
            "best": f"Over {line}" if over >= under else f"Under {line}",
            "best_prob": max(over, under),
        })

    first_half_totals = []
    second_half_totals = []
    for line in (0.5, 1.5, 2.5):
        fo = over_probability(first_h, line)
        fu = under_probability(first_h, line)
        so = over_probability(second_h, line)
        su = under_probability(second_h, line)
        first_half_totals.append({
            "line": line, "over": fo, "under": fu,
            "best": f"Over {line}" if fo >= fu else f"Under {line}",
            "best_prob": max(fo, fu),
        })
        second_half_totals.append({
            "line": line, "over": so, "under": su,
            "best": f"Over {line}" if so >= su else f"Under {line}",
            "best_prob": max(so, su),
        })

    # Handicap is represented as a simple goal-line market. For the common
    # -1.5/+1.5 lines, calculate the probability that the selected team wins
    # after applying the handicap.
    handicaps = []
    for line in (-1.5, -0.5, 0.5, 1.5):
        dist = score_matrix(lh, la, 8)
        home_cover = sum(p for p, h, a in dist if (h + line) > a)
        away_cover = sum(p for p, h, a in dist if (a - line) > h)
        handicaps.append({
            "line": line,
            "home": home_cover,
            "away": away_cover,
            "best": (
                f"{pred['home']} {line:+.1f}"
                if home_cover >= away_cover
                else f"{pred['away']} {-line:+.1f}"
            ),
            "best_prob": max(home_cover, away_cover),
        })

    # BTTS is useful as an additional market and follows directly from
    # independent Poisson scoring.
    btts_yes = (1 - math.exp(-lh)) * (1 - math.exp(-la))

    # Team-goal markets.
    home_over_0_5 = 1 - math.exp(-lh)
    away_over_0_5 = 1 - math.exp(-la)
    home_over_1_5 = 1 - poisson_cdf(1, lh)
    away_over_1_5 = 1 - poisson_cdf(1, la)

    return {
        "totals": totals,
        "first_half_totals": first_half_totals,
        "second_half_totals": second_half_totals,
        "handicap": handicaps,
        "btts": {"yes": btts_yes, "no": 1 - btts_yes},
        "team_goals": {
            "home_over_0_5": home_over_0_5,
            "home_over_1_5": home_over_1_5,
            "away_over_0_5": away_over_0_5,
            "away_over_1_5": away_over_1_5,
        },
        "expected_goals": {
            "home": lh,
            "away": la,
            "total": total,
            "first_half": first_h,
            "second_half": second_h,
        },
    }


def api_team_search(team_name):
    if not API_KEY or not api_budget_available():
        return None
    try:
        data = api("/teams", {"search": team_name})
        if not data:
            return None
        wanted = norm_team(team_name)
        exact = [
            x for x in data
            if norm_team(x.get("team", {}).get("name", "")) == wanted
        ]
        item = exact[0] if exact else data[0]
        return item.get("team", {}).get("id")
    except Exception as exc:
        log.warning("Nie udało się znaleźć zespołu %s: %s", team_name, exc)
        return None


def api_fixture_for_match(match):
    """Find API-Football fixture ID and team IDs for a selected match."""
    if not API_KEY or not api_budget_available():
        return None

    league_name = str(match.get("league", ""))
    league_id = next(
        (lid for lid, name in API_LEAGUES.items()
         if norm_team(name) == norm_team(league_name)),
        None,
    )
    if league_id is None:
        return None

    dt = pd.to_datetime(match.get("date"), utc=True, errors="coerce")
    if pd.isna(dt):
        return None

    try:
        data = api(
            "/fixtures",
            {
                "league": league_id,
                "season": CURRENT_SEASON,
                "date": dt.strftime("%Y-%m-%d"),
                "timezone": TZ,
            },
        )
        for f in data:
            teams = f.get("teams", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            if (
                norm_team(home) == norm_team(match["home"])
                and norm_team(away) == norm_team(match["away"])
            ):
                return {
                    "fixture_id": f.get("fixture", {}).get("id"),
                    "home_id": teams.get("home", {}).get("id"),
                    "away_id": teams.get("away", {}).get("id"),
                }
    except Exception as exc:
        log.warning("Nie znaleziono fixture API dla %s - %s: %s",
                    match["home"], match["away"], exc)
    return None


def api_player_scorers(team_id):
    """
    Return the best available goal scorers for a team.
    Requires API-Football player coverage. If unavailable, return [].
    """
    if not team_id or not API_KEY or not api_budget_available():
        return []
    try:
        data = api(
            "/players",
            {"team": int(team_id), "season": CURRENT_SEASON, "page": 1},
        )
        rows = []
        for item in data:
            player = item.get("player", {})
            stats = item.get("statistics") or []
            if not stats:
                continue
            goals = (stats[0].get("goals") or {}).get("total")
            apps = (stats[0].get("games") or {}).get("appearences")
            if goals is None:
                continue
            rows.append({
                "name": player.get("name", "Nieznany"),
                "goals": int(goals or 0),
                "appearances": int(apps or 0),
            })
        rows.sort(key=lambda x: (x["goals"], x["appearances"]), reverse=True)
        return rows[:3]
    except Exception as exc:
        log.warning("Nie udało się pobrać strzelców team=%s: %s", team_id, exc)
        return []


def enrich_scorers(chosen):
    """
    Adds scorer candidates only when API-Football can identify the fixture.
    We never invent a player name from team-level statistics.
    """
    for match in chosen:
        match["scorers"] = {
            "home": [],
            "away": [],
            "note": "Brak danych o składach/strzelcach z API-Football."
        }
        fixture = api_fixture_for_match(match)
        if not fixture:
            continue

        home = api_player_scorers(fixture.get("home_id"))
        away = api_player_scorers(fixture.get("away_id"))

        if home or away:
            match["scorers"] = {
                "home": home,
                "away": away,
                "note": "Kandydaci wg dostępnych danych sezonowych API-Football."
            }


def enrich_bookmaker_odds(chosen):
    """
    Optional bookmaker odds. The model remains usable when odds are absent.
    We store raw market labels instead of assuming fixed bookmaker bet IDs.
    """
    for match in chosen:
        match["bookmaker_odds"] = []
        fixture = api_fixture_for_match(match)
        if not fixture or not fixture.get("fixture_id") or not API_KEY:
            continue
        if not api_budget_available():
            break
        try:
            data = api(
                "/odds",
                {"fixture": fixture["fixture_id"], "page": 1},
            )
            compact = []
            for bookmaker in data:
                bname = bookmaker.get("bookmaker", {}).get("name", "")
                for bet in bookmaker.get("bets", []):
                    values = bet.get("values", [])
                    if values:
                        compact.append({
                            "bookmaker": bname,
                            "market": bet.get("name", ""),
                            "values": values[:12],
                        })
            match["bookmaker_odds"] = compact[:80]
        except Exception as exc:
            log.warning("Nie udało się pobrać kursów: %s", exc)


def _decimal_odd(value):
    try:
        x = float(value)
        return x if x > 1.0 else None
    except Exception:
        return None


def _norm_label(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def best_value_bets(match):
    """
    Compare model probabilities with bookmaker decimal odds when available.
    Edge = model probability * decimal odds - 1.
    Only positive-edge candidates are returned.
    """
    markets = match.get("markets", {})
    odds_rows = match.get("bookmaker_odds", [])
    candidates = []

    p1x2 = match.get("p1x2", {})
    result_map = {
        "home": float(p1x2.get("home", 0)),
        "draw": float(p1x2.get("draw", 0)),
        "away": float(p1x2.get("away", 0)),
    }

    for row in odds_rows:
        market = _norm_label(row.get("market"))
        bookmaker = row.get("bookmaker", "")
        for val in row.get("values", []):
            label = _norm_label(val.get("value"))
            odd = _decimal_odd(val.get("odd"))
            if odd is None:
                continue

            prob = None
            display = None

            # Match winner / 1X2.
            if "match winner" in market or market in ("1x2", "winner"):
                key = {
                    "home": "home", "draw": "draw", "away": "away",
                    "1": "home", "x": "draw", "2": "away",
                }.get(label)
                if key:
                    prob = result_map[key]
                    display = f"1X2 {label.upper()}"

            # Full-time totals.
            m = re.search(r"(over|under)\s*(0\.5|1\.5|2\.5|3\.5|4\.5)", label)
            if prob is None and ("over/under" in market or "total" in market) and m:
                side, line_s = m.group(1), m.group(2)
                line = float(line_s)
                for t in markets.get("totals", []):
                    if t["line"] == line:
                        prob = t[side]
                        display = f"{side.title()} {line:g} gole"
                        break

            # First-half totals.
            if prob is None and ("1st half" in market or "first half" in market or "half" in market) and m:
                side, line_s = m.group(1), m.group(2)
                line = float(line_s)
                for t in markets.get("first_half_totals", []):
                    if t["line"] == line:
                        prob = t[side]
                        display = f"1. połowa {side.title()} {line:g}"
                        break

            # BTTS.
            if prob is None and ("both teams" in market or "btts" in market):
                if label in ("yes", "no"):
                    prob = markets.get("btts", {}).get(label)
                    display = f"BTTS {label.upper()}"

            if prob is not None and prob > 0:
                edge = prob * odd - 1
                if edge > 0.02:
                    candidates.append({
                        "market": display,
                        "bookmaker": bookmaker,
                        "odd": odd,
                        "model_prob": prob,
                        "edge": edge,
                    })

    candidates.sort(key=lambda x: x["edge"], reverse=True)
    return candidates[:5]


def add_market_analysis(chosen):
    """Add model markets and optional player/market data."""
    for match in chosen:
        match["markets"] = market_options_from_prediction(match)
    # Optional API enrichment must NEVER break the core prediction.
    try:
        enrich_scorers(chosen)
    except Exception as exc:
        log.warning("Scorer enrichment skipped: %s", exc)

    try:
        enrich_bookmaker_odds(chosen)
    except Exception as exc:
        log.warning("Odds enrichment skipped: %s", exc)

    for match in chosen:
        try:
            match["value_bets"] = best_value_bets(match)
        except Exception as exc:
            log.warning("Value-bet analysis skipped: %s", exc)
            match["value_bets"] = []

    return chosen



# ============================================================
# PREDICTIONS + QUALITY CONTROL
# ============================================================

def main_signal(p1x2):
    """Select the strongest 1X2 outcome and its probability."""
    outcomes = {
        "1": float(p1x2.get("home", 0.0)),
        "X": float(p1x2.get("draw", 0.0)),
        "2": float(p1x2.get("away", 0.0)),
    }
    code = max(outcomes, key=outcomes.get)
    names = {
        "1": "Gospodarz wygra",
        "X": "Remis",
        "2": "Goście wygrają",
    }
    return code, names[code], outcomes[code]



def signal_level(confidence):
    """Three exact-score strength tiers requested by the user."""
    p = float(confidence or 0.0)
    if p >= INCREDIBLE_SIGNAL_P:
        return "🤯 NIESAMOWICIE MOCNY SYGNAŁ AI"
    if p >= VERY_STRONG_SIGNAL_P:
        return "🔥 BARDZO MOCNY SYGNAŁ AI"
    if p >= STRONG_SIGNAL_P:
        return "🟢 MOCNY SYGNAŁ AI"
    return "⚪ SŁABY — ODRZUCONY"


def label(p):
    return signal_level(p)


def coupon_score(items):
    if not items:
        return 0.0
    return min(float(x.get("score_confidence", 0.0)) for x in items)


def _pair_strength(items):
    """The pair is only as strong as its weaker exact-score leg."""
    if not items or len(items) < 2:
        return 0.0
    return min(float(x.get("score_confidence", 0.0)) for x in items)


def _fixture_key(match):
    return (
        norm_team(match.get("home", "")),
        norm_team(match.get("away", "")),
        str(match.get("date", "")),
    )


def _signal_fingerprint(match):
    return "|".join(
        [
            norm_team(match.get("home", "")),
            norm_team(match.get("away", "")),
            str(match.get("score", "")),
            str(match.get("date", "")),
        ]
    )


def _load_pair_state():
    if not EXACT_PAIR_STATE.exists():
        return None
    try:
        data = json.loads(EXACT_PAIR_STATE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        pair = data.get("matches")
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        return data
    except Exception:
        return None


def _save_pair_state(matches, reason="initial"):
    if not matches or len(matches) != 2:
        return None

    pair_strength = _pair_strength(matches)
    data = {
        "timestamp": now().isoformat(),
        "reason": reason,
        "signal_level": signal_level(pair_strength),
        "pair_strength": pair_strength,
        "matches": matches,
    }
    EXACT_PAIR_STATE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


def _state_matches_current_fixture(state_match, candidate):
    return _fixture_key(state_match) == _fixture_key(candidate)


def _is_strong_exact_score(match):
    """
    Very strict admission gate.
    Weak candidates are rejected before they can enter the active coupon.
    """
    confidence = float(match.get("score_confidence", 0.0))
    probability = float(match.get("prob", 0.0))
    agreement = float(match.get("score_agreement", 0.0))
    home_n = int(match.get("venue_history_home", 0))
    away_n = int(match.get("venue_history_away", 0))

    return (
        probability >= EXACT_SCORE_MIN_PROB
        and agreement >= EXACT_SCORE_MIN_AGREEMENT
        and confidence >= STRONG_SIGNAL_P
        and (home_n + away_n) >= EXACT_SCORE_HISTORY_MIN * 2
    )


def _candidate_matches(history, upcoming_df):
    """Build only strong exact-score candidates."""
    cand = []

    for _, r in upcoming_df.iterrows():
        if str(r.get("status", "")).upper() in {
            "FT", "AET", "PEN", "CANC", "CANCELLED", "POSTPONED"
        }:
            continue

        match_date = r["date"]
        home = str(r["home"])
        away = str(r["away"])
        league = str(r.get("league", ""))

        if not home or not away or home == "nan" or away == "nan":
            continue

        p = predict(
            history,
            home,
            away,
            league=league,
            match_date=match_date,
        )

        if not _is_strong_exact_score(p):
            continue

        p.update({
            "date": str(match_date),
            "league": league,
            "signal_code": "EXACT",
            "signal": "Dokładny wynik",
            "signal_prob": float(p["prob"]),
            "label": signal_level(p["score_confidence"]),
            "source": str(r.get("source", "football-data")),
        })
        cand.append(p)

    cand.sort(
        key=lambda x: (
            -float(x.get("score_confidence", 0.0)),
            -float(x.get("prob", 0.0)),
            -float(x.get("score_agreement", 0.0)),
            pd.to_datetime(x.get("date"), utc=True, errors="coerce"),
        )
    )
    return cand


def _choose_initial_pair(candidates):
    """Choose two different fixtures and optimize for the weakest leg."""
    if len(candidates) < 2:
        return None

    best_pair = None
    best_key = None

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if _fixture_key(a) == _fixture_key(b):
                continue

            pair = [a, b]
            key = (
                _pair_strength(pair),
                (float(a.get("score_confidence", 0.0))
                 + float(b.get("score_confidence", 0.0))) / 2.0,
                min(float(a.get("prob", 0.0)), float(b.get("prob", 0.0))),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pair = pair

    return best_pair


def _pair_is_expired(state):
    """Reset after both original fixtures are no longer upcoming."""
    if not state or not state.get("matches"):
        return True

    active_count = 0
    now_utc = pd.Timestamp.now(tz="UTC")

    for m in state["matches"]:
        dt = pd.to_datetime(m.get("date"), utc=True, errors="coerce")
        if pd.notna(dt) and dt > now_utc:
            active_count += 1

    return active_count < 2


def _improved_pair(current_pair, fresh_candidates):
    """
    Update the SAME two fixtures only.
    A new pair is accepted only when:
      - both legs are at least as strong as the current legs;
      - and at least one leg is meaningfully stronger.
    This prevents weaker signals from replacing a good one.
    """
    by_fixture = {_fixture_key(x): x for x in fresh_candidates}
    updated = []
    any_improvement = False

    for current in current_pair:
        fresh = by_fixture.get(_fixture_key(current))
        if fresh is None:
            updated.append(current)
            continue

        old_conf = float(current.get("score_confidence", 0.0))
        old_prob = float(current.get("prob", 0.0))
        new_conf = float(fresh.get("score_confidence", 0.0))
        new_prob = float(fresh.get("prob", 0.0))

        # New score must not be weaker on either confidence or raw probability.
        not_weaker = new_conf >= old_conf and new_prob >= old_prob
        meaningful = (
            new_conf >= old_conf + EXACT_SCORE_MIN_EDGE
            or new_prob >= old_prob + EXACT_SCORE_MIN_EDGE
            or str(fresh.get("score")) != str(current.get("score"))
            and new_conf > old_conf
            and new_prob > old_prob
        )

        if not_weaker:
            updated.append(fresh if meaningful else current)
        else:
            updated.append(current)

        if meaningful and not_weaker:
            any_improvement = True

    return updated, any_improvement


def _pair_summary_message(pair, prefix):
    strength = _pair_strength(pair)
    level = signal_level(strength)
    lines = [
        prefix,
        "",
        f"📡 SIŁA AKO: {strength:.1%}",
        level,
        "",
    ]
    for i, m in enumerate(pair, 1):
        lines += [
            f"{i}️⃣ {m['home']} — {m['away']}",
            f"🎯 Dokładny wynik: {m.get('score', 'brak')}",
            f"📈 Szansa modelu: {float(m.get('prob', 0.0)):.1%}",
            f"🧠 Siła sygnału: {float(m.get('score_confidence', 0.0)):.1%}",
            f"🤝 Zgodność modeli: {float(m.get('score_agreement', 0.0)):.0%}",
            "",
        ]
    lines.append("🎟️ AKO: 2 dokładne wyniki na te same 2 mecze.")
    return "\n".join(lines)


def make_predictions():
    history = load_history()

    if history.empty or len(history) < 200:
        raise RuntimeError(
            f"Za mało historii z darmowych źródeł: {len(history)} meczów."
        )

    upcoming_df = upcoming()
    if upcoming_df.empty:
        return [], None

    candidates = _candidate_matches(history, upcoming_df)
    state = _load_pair_state()

    # First run / expired coupon: choose exactly two strong fixtures.
    if state is None or _pair_is_expired(state):
        pair = _choose_initial_pair(candidates)
        return pair or [], coupon_score(pair) if pair else None

    current_pair = state.get("matches", [])
    fresh_pair, improved = _improved_pair(current_pair, candidates)

    # Return the SAME two fixtures. The caller decides whether to announce
    # an improvement or just a no-better-signal hourly status.
    return fresh_pair, (coupon_score(fresh_pair) if fresh_pair else None)


# ============================================================
# AUTOMATION / SIGNAL STATE
# ============================================================

def process_hourly_signal(chat_id=None, force_first=False):
    """
    Hourly controller:
      - establishes the first 2-match exact-score AKO when none exists;
      - otherwise monitors only those same 2 fixtures;
      - sends an alert only for a genuinely stronger pair;
      - every hour sends a status telling whether a better type appeared.
    """
    history = load_history()
    if history.empty or len(history) < 200:
        send(
            "⏰ KONTROLA GODZINNA\n\n"
            "Nie udało się znaleźć lepszego typu — za mało danych "
            "do bezpiecznej analizy.",
            chat_id,
        )
        return False

    upcoming_df = upcoming()
    candidates = _candidate_matches(history, upcoming_df) if not upcoming_df.empty else []
    state = _load_pair_state()

    if state is None or _pair_is_expired(state):
        pair = _choose_initial_pair(candidates)
        if pair and len(pair) == 2:
            _save_pair_state(pair, reason="first_coupons")
            save_last(pair, coupon_score(pair))
            save_predictions(pair, coupon_score(pair))
            send(
                prediction_message(pair, coupon_score(pair)),
                chat_id,
            )
            send(
                "⏰ KONTROLA GODZINNA\n\n"
                "✅ Utworzono bazowy kupon 2× dokładny wynik.\n"
                f"{signal_level(_pair_strength(pair))}",
                chat_id,
            )
            return True

        send(
            "⏰ KONTROLA GODZINNA\n\n"
            "❌ Nie udało się znaleźć wystarczająco mocnego zestawu "
            "2 dokładnych wyników. Słabe sygnały zostały odrzucone.",
            chat_id,
        )
        return False

    current = state["matches"]
    fresh, improved = _improved_pair(current, candidates)

    if improved:
        _save_pair_state(fresh, reason="stronger_signal")
        save_last(fresh, coupon_score(fresh))
        save_predictions(fresh, coupon_score(fresh))
        send(
            _pair_summary_message(
                fresh,
                "🚨 ZNALEZIONO MOCNIEJSZY SYGNAŁ!",
            ),
            chat_id,
        )
        send(
            f"⏰ KONTROLA GODZINNA\n\n"
            f"✅ Lepszy typ się znalazł.\n"
            f"{signal_level(_pair_strength(fresh))}",
            chat_id,
        )
        return True

    # No improvement: keep the current pair unchanged.
    send(
        "⏰ KONTROLA GODZINNA\n\n"
        "🔒 Nie znaleziono lepszego typu od obecnego.\n"
        "✅ Słabsze sygnały zostały odrzucone.\n"
        f"{signal_level(_pair_strength(current))}\n\n"
        "Obecny kupon pozostaje bez zmian: 2 dokładne wyniki na te same 2 mecze.",
        chat_id,
    )
    return False


# ============================================================
# TELEGRAM
# ============================================================

def tg(method, data=None):
    if not TG_TOKEN:
        raise RuntimeError("Brak TELEGRAM_BOT_TOKEN")

    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
        data=data or {},
        timeout=25,
    )
    r.raise_for_status()

    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(str(payload))

    return payload


def send(msg, chat_id=None):
    chat_id = chat_id or TG_CHAT

    if not TG_TOKEN or not chat_id:
        print(msg, flush=True)
        return

    tg("sendMessage", {
        "chat_id": chat_id,
        "text": msg,
    })



def prediction_message(chosen, score):
    if not chosen or len(chosen) != 2:
        return (
            "🟡 BRAK WYSTARCZAJĄCO MOCNEGO AKO\n\n"
            "AI odrzuciło słabe sygnały.\n"
            "Nie wysyłam losowych dokładnych wyników."
        )

    pair_strength = _pair_strength(chosen)
    lines = [
        "🤖 MASTER OF AI — ŚCISŁY TRYB DOKŁADNEGO WYNIKU",
        "",
        "🎟️ AKO (2): 2 MECZE × 2 DOKŁADNE WYNIKI",
        f"📡 SIŁA CAŁEGO AKO: {pair_strength:.1%}",
        signal_level(pair_strength),
        "",
    ]

    for i, x in enumerate(chosen, 1):
        conf = float(x.get("score_confidence", 0.0))
        prob = float(x.get("prob", 0.0))
        agreement = float(x.get("score_agreement", 0.0))

        lines += [
            f"⚽ MECZ {i}: {x['home']} — {x['away']}",
            f"🎯 DOKŁADNY WYNIK: {x.get('score', 'brak')}",
            f"📈 Szansa modelu: {prob:.1%}",
            f"🧠 Siła sygnału: {conf:.1%}",
            f"🏷️ {signal_level(conf)}",
            f"🤝 Zgodność modeli: {agreement:.0%}",
            f"📊 Forma: {float(x.get('form_home', 0)):.0%} vs {float(x.get('form_away', 0)):.0%}",
            f"📚 Historia: {x.get('matches_home', 0)} vs {x.get('matches_away', 0)} meczów",
            f"🏠/✈️ Próba venue: {x.get('venue_history_home', 0)} vs {x.get('venue_history_away', 0)}",
            f"♟️ Elo: {x.get('elo_home', 1500):.0f} vs {x.get('elo_away', 1500):.0f}",
            f"⚽ Oczekiwane gole: {x.get('lh', 0):.2f} vs {x.get('la', 0):.2f}",
            "",
            "🔎 NAJLEPSZE ALTERNATYWY DOKŁADNEGO WYNIKU:",
        ]

        for j, t in enumerate((x.get("top_scores") or [])[:3], 1):
            lines.append(f"{j}. {t['score']} — {float(t['prob']):.1%}")

        lines.append("")

    lines += [
        "✅ Słabe sygnały są automatycznie odrzucane.",
        "✅ Ten sam kupon obejmuje te same 2 mecze.",
        "✅ Co godzinę AI sprawdza, czy znalazł się lepszy typ.",
        "⚠️ Dokładny wynik nie jest gwarantowany, nawet przy bardzo mocnym sygnale.",
    ]
    return "\n".join(lines)



def compare_and_announce_signal(chosen, chat_id=None):
    """
    Backward-compatible wrapper. The strict controller now handles the
    complete 2-leg coupon and the hourly status.
    """
    if chosen and len(chosen) == 2:
        old = _load_pair_state()
        if old is None:
            _save_pair_state(chosen, reason="compatibility")
            send(prediction_message(chosen, coupon_score(chosen)), chat_id)


def save_predictions(chosen, score):
    rows = []

    for x in chosen:
        rows.append({
            "timestamp": now().isoformat(),
            "date": x["date"],
            "home": x["home"],
            "away": x["away"],
            "league": x["league"],
            "score_prediction": x["score"],
            "probability": x["prob"],
            "raw_probability": x["raw_prob"],
            "signal_code": x["signal_code"],
            "signal": x["signal"],
            "signal_probability": x["signal_prob"],
            "p_home": x["p1x2"]["home"],
            "p_draw": x["p1x2"]["draw"],
            "p_away": x["p1x2"]["away"],
            "source": x["source"],
            "expected_goals_home": x.get("markets", {}).get("expected_goals", {}).get("home"),
            "expected_goals_away": x.get("markets", {}).get("expected_goals", {}).get("away"),
            "btts_yes": x.get("markets", {}).get("btts", {}).get("yes"),
        })

    if not rows:
        return

    nd = pd.DataFrame(rows)

    if PREDICTIONS_LOG.exists():
        try:
            old = pd.read_csv(PREDICTIONS_LOG)
            nd = pd.concat([old, nd], ignore_index=True)
        except Exception:
            pass

    nd.to_csv(PREDICTIONS_LOG, index=False)


def load_last():
    if not LAST_COUPON.exists():
        return None

    try:
        return json.loads(
            LAST_COUPON.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def save_last(chosen, score):
    LAST_COUPON.write_text(
        json.dumps({
            "timestamp": now().isoformat(),
            "score": score,
            "items": chosen,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_last_signal():
    if not LAST_SIGNAL.exists():
        return None
    try:
        return json.loads(LAST_SIGNAL.read_text(encoding="utf-8"))
    except Exception:
        return None


def strongest_signal(chosen):
    if not chosen:
        return None

    # Strongest signal = highest main 1X2 probability.
    return max(
        chosen,
        key=lambda x: (
            float(x.get("signal_prob", 0.0)),
            -pd.to_datetime(
                x.get("date"),
                utc=True,
                errors="coerce"
            ).timestamp()
            if pd.notna(pd.to_datetime(x.get("date"), utc=True, errors="coerce"))
            else 0,
        ),
    )


def save_last_signal(chosen):
    if not chosen:
        return None

    best = max(
        chosen,
        key=lambda x: (
            float(x.get("score_confidence", 0.0)),
            float(x.get("prob", 0.0)),
        ),
    )

    old_seen = _load_seen_exact_signals()
    for item in chosen:
        old_seen.add(_signal_fingerprint(item))

    data = {
        "timestamp": now().isoformat(),
        "home": best["home"],
        "away": best["away"],
        "date": best["date"],
        "league": best["league"],
        "signal": "Dokładny wynik",
        "signal_code": "EXACT",
        "score": best["score"],
        "score_prob": best.get("prob", 0.0),
        "score_confidence": best.get("score_confidence", 0.0),
        "score_agreement": best.get("score_agreement", 0.0),
        "seen": sorted(old_seen),
    }
    LAST_SIGNAL.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


# ============================================================
# RESULT CHECKING
# ============================================================

def evaluate_predictions():
    """
    Basic backtest against matches already present in the local history.
    This evaluates the model without pretending it can know future data.

    It is intentionally conservative: it reports exact-score hits and
    1X2 hits, but does not claim profitability.
    """
    if not PREDICTIONS_LOG.exists():
        return "Brak zapisanych predykcji."

    pred = pd.read_csv(PREDICTIONS_LOG)
    hist = load_history()

    if pred.empty or hist.empty:
        return "Brak danych do rozliczenia."

    pred["date"] = pd.to_datetime(pred["date"], utc=True, errors="coerce")
    hist["date"] = pd.to_datetime(hist["date"], utc=True, errors="coerce")

    exact = 0
    result_1x2 = 0
    total = 0

    for _, p in pred.iterrows():
        x = hist[
            (hist.date >= p.date - pd.Timedelta(days=2)) &
            (hist.date <= p.date + pd.Timedelta(days=2)) &
            (hist.home_key == norm_team(p.home)) &
            (hist.away_key == norm_team(p.away))
        ]

        if x.empty:
            continue

        r = x.sort_values("date").iloc[-1]
        total += 1

        actual = f"{int(r.hg)}:{int(r.ag)}"
        if actual == str(p.score_prediction):
            exact += 1

        ph = float(p.p_home)
        pdra = float(p.p_draw)
        pa = float(p.p_away)

        pred_res = "H" if ph >= pdra and ph >= pa else (
            "D" if pdra >= ph and pdra >= pa else "A"
        )

        real_res = "H" if r.hg > r.ag else (
            "D" if r.hg == r.ag else "A"
        )

        if pred_res == real_res:
            result_1x2 += 1

    if total == 0:
        return "Nie udało się jeszcze rozliczyć żadnego zakończonego meczu."

    return (
        "📊 ROZLICZENIE MASTER OF AI\n\n"
        f"Rozliczone mecze: {total}\n"
        f"Dokładny wynik: {exact}/{total} = {exact/total:.1%}\n"
        f"1X2: {result_1x2}/{total} = {result_1x2/total:.1%}\n\n"
        "To statystyka modelu, nie gwarancja przyszłej skuteczności."
    )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

def status_message():
    usage = _load_api_usage()

    return (
        "🤖 MASTER OF AI działa.\n\n"
        f"📚 Historia: Football-Data + lokalny cache\n"
        f"🌍 Lig: {len(FD_LEAGUES)}\n"
        f"🎯 Tryb: TYLKO DOKŁADNE WYNIKI\n"
        f"🎟️ Pierwszy sygnał: 2 mecze | kolejne: 1\n"
        f"📊 Min. exact-score: {EXACT_SCORE_MIN_PROB:.0%}\n"
        f"🟢 Mocny exact-score od: {STRONG_SIGNAL_P:.0%}\n"
        f"🔥 Bardzo mocny exact-score od: {VERY_STRONG_SIGNAL_P:.0%}\n"
        f"🤯 Niesamowicie mocny exact-score od: {INCREDIBLE_SIGNAL_P:.0%}\n"
        f"⏰ Kontrola: co 60 minut\n"
        f"🛡️ API-Football dzienny budżet: "
        f"{usage['used']}/{API_DAILY_BUDGET}\n"
        f"⏱️ API odstęp: {API_MIN_INTERVAL:.1f}s\n"
        f"📅 Sezon: {CURRENT_SEASON}/{CURRENT_SEASON+1}\n"
        f"📡 Mecze live: TheSportsDB + Football-Data + API fallback\n"
        f"🌍 Strefa: {TZ}"
    )


def handle_update(u):
    msg = u.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = str((msg.get("chat") or {}).get("id", ""))

    if not text or not chat_id:
        return

    cmd = text.split()[0].split("@")[0].lower()

    try:
        if cmd == "/start":
            send(
                "🤖 Witaj w Master Of AI!\n\n"
                "/typy — analiza i dokładne wyniki\n"
                "/wyniki — dzisiejsze mecze\n"
                "/statystyki — skuteczność zapisanych typów\n"
                "/status — status systemu\n"
                "/kontrola — natychmiastowa kontrola lepszego sygnału\n"
                "/help — pomoc",
                chat_id,
            )

        elif cmd in ("/help", "/pomoc"):
            send(
                "📌 KOMENDY\n\n"
                "/typy — uruchamia analizę\n"
                "/wyniki — pokazuje dostępne wyniki\n"
                "/statystyki — rozlicza zapisane typy\n"
                "/status — status źródeł i API\n"
                "/kontrola — sprawdza od razu, czy jest lepszy typ",
                chat_id,
            )

        elif cmd == "/status":
            send(status_message(), chat_id)

        elif cmd == "/statystyki":
            send(evaluate_predictions(), chat_id)

        elif cmd == "/rynki":
            send(
                "📌 RYNKI ANALIZOWANE PRZEZ MASTER OF AI\\n\\n"
                "1️⃣ 1X2 — gospodarz/remis/goście\\n"
                "🛡️ Handicap — m.in. -1.5, -0.5, +0.5, +1.5\\n"
                "⚽ Gole — Over/Under 0.5–4.5\\n"
                "🤝 BTTS — obie drużyny strzelą\\n"
                "1️⃣ 1. połowa — Over/Under\\n"
                "2️⃣ 2. połowa — Over/Under\\n"
                "🎯 Gole drużyny — Over 0.5/1.5\\n"
                "👤 Strzelcy — kandydaci, jeśli API dostarczy dane\\n\\n"
                "Prawdopodobieństwa rynków pochodzą z modelu. "
                "Kurs bukmachera nie jest prawdopodobieństwem AI.",
                chat_id,
            )

        elif cmd == "/wyniki":
            h = load_history()
            if h.empty:
                send("Brak danych w lokalnej bazie.", chat_id)
            else:
                today = now().date()
                x = h[
                    pd.to_datetime(h.date, utc=True).dt.date == today
                ].sort_values("date")

                if x.empty:
                    send("Brak zapisanych wyników na dziś.", chat_id)
                else:
                    lines = [
                        f"📅 WYNIKI — {today.strftime('%d.%m.%Y')}",
                        "",
                    ]
                    for _, r in x.iterrows():
                        lines.append(
                            f"{r.home} - {r.away} "
                            f"⚽ {int(r.hg)}:{int(r.ag)}"
                        )
                    send("\n".join(lines[:101]), chat_id)

        elif cmd == "/typy":
            send(
                "⏳ Master Of AI wykonuje ścisłą analizę dokładnych wyników...",
                chat_id,
            )

            state = _load_pair_state()

            if state is None or _pair_is_expired(state):
                history = load_history()
                upcoming_df = upcoming()
                candidates = _candidate_matches(history, upcoming_df) if not upcoming_df.empty else []
                chosen = _choose_initial_pair(candidates) if len(candidates) >= 2 else None

                if not chosen:
                    send(
                        "❌ Nie znaleziono 2 wystarczająco mocnych dokładnych wyników. "
                        "Słabe sygnały zostały odrzucone.",
                        chat_id,
                    )
                    return

                save_state = _save_pair_state(chosen, reason="manual_typy")
                save_predictions(chosen, coupon_score(chosen))
                save_last(chosen, coupon_score(chosen))
                send(prediction_message(chosen, coupon_score(chosen)), chat_id)
            else:
                send(
                    prediction_message(
                        state.get("matches", []),
                        state.get("pair_strength", 0.0),
                    ),
                    chat_id,
                )

        elif cmd == "/kontrola":
            process_hourly_signal(chat_id=chat_id, force_first=True)

        else:
            send("Nie znam tej komendy. Wpisz /help.", chat_id)

    except Exception as exc:
        log.exception("Błąd komendy")
        send(
            f"❌ Błąd: {exc}",
            chat_id,
        )


# ============================================================
# STARTUP VALIDATION
# ============================================================

def validate_config():
    """Validate configuration without making network calls."""
    if not TZ:
        raise RuntimeError("TIMEZONE nie może być puste.")
    if MAXM < 1:
        raise RuntimeError("MAX_MATCHES musi być >= 1.")
    if MINP < 0 or MINP > 1:
        raise RuntimeError("MIN_SCORE_PROB musi być w zakresie 0..1.")
    if not (0 < STRONG_SIGNAL_P < VERY_STRONG_SIGNAL_P < INCREDIBLE_SIGNAL_P <= 1):
        raise RuntimeError(
            "Progi exact-score muszą spełniać: mocny < bardzo mocny < niesamowicie mocny."
        )
    if API_DAILY_BUDGET < 0:
        raise RuntimeError("API_DAILY_BUDGET nie może być ujemny.")
    if API_MIN_INTERVAL < 0:
        raise RuntimeError("API_MIN_INTERVAL nie może być ujemny.")
    if not 0 <= EXACT_SCORE_MIN_PROB <= 1:
        raise RuntimeError("EXACT_SCORE_MIN_PROB musi być w zakresie 0..1.")
    if not 0 <= EXACT_SCORE_MIN_AGREEMENT <= 1:
        raise RuntimeError("EXACT_SCORE_MIN_AGREEMENT musi być w zakresie 0..1.")
    if EXACT_SCORE_MIN_EDGE < 0:
        raise RuntimeError("EXACT_SCORE_MIN_EDGE nie może być ujemny.")
    if EXACT_SCORE_HISTORY_MIN < 0:
        raise RuntimeError("EXACT_SCORE_HISTORY_MIN nie może być ujemny.")


# ============================================================
# TELEGRAM / RENDER
# ============================================================

app = Flask(__name__)


@app.post("/telegram/webhook")
def telegram_webhook():
    update = request.get_json(silent=True) or {}
    threading.Thread(
        target=handle_update,
        args=(update,),
        daemon=True,
    ).start()
    return jsonify({"ok": True})


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "Master Of AI v6",
    })


@app.get("/health")
def health_check():
    return jsonify({
        "status": "healthy",
    })


def configure_telegram_webhook():
    public_url = os.getenv(
        "RENDER_EXTERNAL_URL",
        "",
    ).strip().rstrip("/")

    if not TG_TOKEN or not public_url:
        return False

    try:
        result = tg("setWebhook", {
            "url": public_url + "/telegram/webhook",
            "allowed_updates": json.dumps(["message"]),
            "drop_pending_updates": "false",
        })

        return bool(result.get("ok"))

    except Exception:
        log.exception("Nie udało się ustawić webhooka.")
        return False



def worker():
    log.info("Monitoring ścisłego AKO uruchomiony — kontrola co 60 minut.")

    while True:
        try:
            process_hourly_signal()
        except Exception as exc:
            log.exception("Błąd monitoringu godzinnego: %s", exc)

        time.sleep(60 * 60)


def make_exact_score_ako(matches):
    """Jeden kupon: dokładnie 2 mecze i dokładnie 1 wynik na każdy mecz."""
    candidates = []
    for match in matches:
        score = match.get("score")
        prob = match.get("score_prob", 0.0)

        if not score:
            for item in (match.get("top_scores") or []):
                if isinstance(item, dict):
                    score = item.get("score") or item.get("result")
                    prob = item.get("prob", item.get("probability", 0.0))
                    if score:
                        break

        if score:
            try:
                prob = float(prob or 0.0)
            except (TypeError, ValueError):
                prob = 0.0
            candidates.append((prob, match, score))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) < 2:
        return None

    selected = candidates[:2]
    return {
        "matches": [x[1] for x in selected],
        "scores": [x[2] for x in selected],
        "probabilities": [x[0] for x in selected],
        "combined_probability": selected[0][0] * selected[1][0],
    }


def main():
    validate_config()

    if not TG_TOKEN:
        log.warning("Brak TELEGRAM_BOT_TOKEN.")

    if not API_KEY:
        log.warning(
            "Brak API_FOOTBALL_KEY — działa tryb bez awaryjnego API."
        )

    configure_telegram_webhook()

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()

    port = int(os.getenv("PORT", "10000"))

    log.info(
        "Master Of AI v6 uruchomiony na porcie %s.",
        port,
    )

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
