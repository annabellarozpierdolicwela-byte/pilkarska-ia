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

# v25 stable fast-command state
_FAST_STATE = {
    "last_scan_signature": None,
    "last_scan_time": None,
    "last_check_time": None,
    "last_check_signature": None,
}
_FAST_CACHE_TTL = 25

def _fast_signature(value):
    try:
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        return str(value)[:200]


def _check_changes(current_payload):
    sig = _fast_signature(current_payload)
    previous = _FAST_STATE.get("last_check_signature")
    changed = previous is not None and previous != sig
    _FAST_STATE["last_check_signature"] = sig
    _FAST_STATE["last_check_time"] = now()
    return changed, sig


def _fast_reply(send, chat_id, message):
    send(message, chat_id)


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
AUTO_SCAN = os.getenv("AUTO_SCAN", "1").lower() in ("1", "true", "yes", "on")
# Automatic daily delivery window (Europe/Warsaw by default): 09:00 through 20:00.
AUTO_SEND_HOURS = tuple(
    int(x.strip()) for x in os.getenv(
        "AUTO_SEND_HOURS", "9,10,11,12,13,14,15,16,17,18,19,20"
    ).split(",") if x.strip()
)
MARKET_MIN_PROB = float(os.getenv("MARKET_MIN_PROB", "0.72"))
MARKET_STRONG_PROB = float(os.getenv("MARKET_STRONG_PROB", "0.78"))
MARKET_MAX_LEGS = int(os.getenv("MARKET_MAX_LEGS", "7"))
# TEST MODE: /typy shows the best markets found even when no market
# reaches the production AKO threshold. This is for validation/backtesting.
TEST_MODE = os.getenv("TEST_MODE", "1").lower() in ("1", "true", "yes", "on")
TEST_TOP_MARKETS = int(os.getenv("TEST_TOP_MARKETS", "10"))
EXACT_WATCH_INTERVAL_MINUTES = int(os.getenv("EXACT_WATCH_INTERVAL_MINUTES", "30"))
EXACT_WATCH_TOP_SCORES = int(os.getenv("EXACT_WATCH_TOP_SCORES", "3"))
EXACT_WATCH_INTERNAL_SCORES = int(os.getenv("EXACT_WATCH_INTERNAL_SCORES", "14"))
EXACT_SCORE_MAX_GOALS = int(os.getenv("EXACT_SCORE_MAX_GOALS", "9"))
EXACT_LONG_TAIL_MIN_TOTAL = int(os.getenv("EXACT_LONG_TAIL_MIN_TOTAL", "4"))
EXACT_LONG_TAIL_MIN_RELATIVE = float(os.getenv("EXACT_LONG_TAIL_MIN_RELATIVE", "0.28"))
EXACT_LONG_TAIL_MAX_BOOST = float(os.getenv("EXACT_LONG_TAIL_MAX_BOOST", "1.35"))
EXACT_WATCH_RANK_BLEND = os.getenv("EXACT_WATCH_RANK_BLEND", "1").lower() in ("1", "true", "yes", "on")
EXACT_WATCH_CONTEXT_STRENGTH = float(os.getenv("EXACT_WATCH_CONTEXT_STRENGTH", "0.12"))
EXACT_WATCH_DIXON_COLES_RHO = float(os.getenv("EXACT_WATCH_DIXON_COLES_RHO", "-0.08"))
EXACT_WATCH_CALIBRATION_FILE = DATA / "exact_watch_calibration.json"
EXACT_WATCH_TOP_COMBINATIONS = int(os.getenv("EXACT_WATCH_TOP_COMBINATIONS", "5"))
# EXACT DOUBLE INTELLIGENCE: stricter ensemble selection inspired by the
# observed "five independent double coupons" workflow. No bookmaker odds.
EXACT_ENGINE_VERSION = "25.0-REAL-FIXTURES"
EXACT_SCORE_ENSEMBLE_BLEND = float(os.getenv("EXACT_SCORE_ENSEMBLE_BLEND", "0.72"))
EXACT_SCORE_CALIBRATION_WEIGHT = float(os.getenv("EXACT_SCORE_CALIBRATION_WEIGHT", "0.10"))
EXACT_SCORE_MIN_FIXTURE_QUALITY = float(os.getenv("EXACT_SCORE_MIN_FIXTURE_QUALITY", "0.085"))
EXACT_PAIR_MIN_STRENGTH = float(os.getenv("EXACT_PAIR_MIN_STRENGTH", "0.018"))
# Adaptive exact-score gate: exact-score probabilities are naturally much lower
# than 1X2/goal-market probabilities. Do not require an unrealistically high
# product of two single-score probabilities.
EXACT_PAIR_SOFT_STRENGTH = float(os.getenv("EXACT_PAIR_SOFT_STRENGTH", "0.012"))
EXACT_PAIR_DIVERSITY_PENALTY = float(os.getenv("EXACT_PAIR_DIVERSITY_PENALTY", "0.10"))
EXACT_ENGINE_MAX_SCORE_GOALS = int(os.getenv("EXACT_ENGINE_MAX_SCORE_GOALS", "7"))
# v15: reverse-engineered presentation/selection style from observed public examples: BASELINE -> AI RANKING -> AI LIFT -> TOP3 -> TOP5 DOUBLE.
# Baseline is a pure Poisson score probability; AI ranking adds the existing
# empirical/Dixon-Coles/context/history ensemble. No bookmaker odds are used.
EXACT_V13_AI_LIFT_WEIGHT = float(os.getenv("EXACT_V13_AI_LIFT_WEIGHT", "0.16"))
EXACT_V13_STABILITY_WEIGHT = float(os.getenv("EXACT_V13_STABILITY_WEIGHT", "0.18"))
EXACT_V13_LIFT_CAP = float(os.getenv("EXACT_V13_LIFT_CAP", "1.60"))
EXACT_V13_TOP3_MASS_WEIGHT = float(os.getenv("EXACT_V13_TOP3_MASS_WEIGHT", "0.10"))
# v15: extra independent signals. They are used only when the historical
# dataset actually contains the corresponding columns.
EXACT_V15_XG_WEIGHT = float(os.getenv("EXACT_V15_XG_WEIGHT", "0.16"))
EXACT_V15_RECENCY_WEIGHT = float(os.getenv("EXACT_V15_RECENCY_WEIGHT", "0.12"))
EXACT_V15_MONTE_CARLO_SIMS = int(os.getenv("EXACT_V15_MONTE_CARLO_SIMS", "12000"))
EXACT_V15_BAYES_SHRINK = float(os.getenv("EXACT_V15_BAYES_SHRINK", "0.18"))
EXACT_V15_FORM_STABILITY_WEIGHT = float(os.getenv("EXACT_V15_FORM_STABILITY_WEIGHT", "0.10"))
# v25: advanced signals discussed as plausible extensions of the observed
# "baseline -> AI ranking" design. They never invent unavailable data.
EXACT_V17_LINEUP_WEIGHT = float(os.getenv("EXACT_V17_LINEUP_WEIGHT", "0.10"))
EXACT_V17_STYLE_WEIGHT = float(os.getenv("EXACT_V17_STYLE_WEIGHT", "0.08"))
EXACT_V17_STRENGTH_WEIGHT = float(os.getenv("EXACT_V17_STRENGTH_WEIGHT", "0.10"))
EXACT_V17_CALIBRATION_WEIGHT = float(os.getenv("EXACT_V17_CALIBRATION_WEIGHT", "0.12"))
EXACT_V17_LATE_SIGNAL_WEIGHT = float(os.getenv("EXACT_V17_LATE_SIGNAL_WEIGHT", "0.10"))
EXACT_WATCH_STATE = DATA / "exact_watch_state.json"

# Safety: never allow the bot to intentionally use more than this many
# API-Football requests in one UTC day.
API_DAILY_BUDGET = int(os.getenv("API_DAILY_BUDGET", "35"))
API_MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "7.0"))

# UEFA-first intelligence mode.
# 3 = Champions League, 2 = Europa League, 1 = Conference League.
UEFA_PRIORITY_ENABLED = os.getenv("UEFA_PRIORITY_ENABLED", "1").lower() in ("1", "true", "yes", "on")
UEFA_STRICT_FOCUS = os.getenv("UEFA_STRICT_FOCUS", "1").lower() in ("1", "true", "yes", "on")
UEFA_REPLACEMENT_TIER_LOCK = os.getenv("UEFA_REPLACEMENT_TIER_LOCK", "1").lower() in ("1", "true", "yes", "on")
HISTORY_SEASONS = int(os.getenv("HISTORY_SEASONS", "8"))
MAX_DAILY_SIGNAL_UPDATES = int(os.getenv("MAX_DAILY_SIGNAL_UPDATES", "5"))

MAXM = int(os.getenv("MAX_MATCHES", "2"))
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
EXACT_SCORE_MIN_MARGIN = float(os.getenv("EXACT_SCORE_MIN_MARGIN", "0.008"))
EXACT_PAIR_REPLACEMENT_EDGE = float(os.getenv("EXACT_PAIR_REPLACEMENT_EDGE", "0.02"))
DIAGNOSTIC_LIMIT = int(os.getenv("DIAGNOSTIC_LIMIT", "5"))

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
    "HR1": "Croatia",
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
    210: "Croatia HNL",
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    848: "UEFA Europa Conference League",
}

# Normalized competition tiers used by the selector.
UEFA_COMPETITION_TIERS = {
    "uefa champions league": 3,
    "champions league": 3,
    "uefa europa league": 2,
    "europa league": 2,
    "uefa europa conference league": 1,
    "uefa conference league": 1,
    "europa conference league": 1,
    "conference league": 1,
}

NON_UEFA_TIER = 0

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
REJECTION_LOG = DATA / "rejected_signals.csv"
DAILY_SIGNAL_STATE = DATA / "daily_signal_state.json"
AUTO_SLOT_STATE = DATA / "auto_slot_state.json"

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
    seasons_back = max(1, HISTORY_SEASONS)
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


def _manual_test_fixtures():
    """Fixed fixture pool for TEST_MODE: four user-selected matches for today."""
    today = now().date()
    rows = [
        {"home": "Valencia", "away": "Real Betis", "league": "LaLiga", "status": "NS", "date": f"{today}T19:00:00+00:00", "source": "manual-test"},
        {"home": "Sabah FK", "away": "Hapoel Be'er Sheva", "league": "UEFA Champions League Qualifying", "status": "NS", "date": f"{today}T16:45:00+00:00", "source": "manual-test"},
        {"home": "Bodo/Glimt", "away": "NEC Nijmegen", "league": "UEFA Champions League Qualifying", "status": "NS", "date": f"{today}T19:00:00+00:00", "source": "manual-test"},
        {"home": "LASK Linz", "away": "Celtic", "league": "UEFA Champions League Qualifying", "status": "NS", "date": f"{today}T19:00:00+00:00", "source": "manual-test"},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["home_key"] = df["home"].map(norm_team)
    df["away_key"] = df["away"].map(norm_team)
    return df


def _today_utc():
    """Return today's UTC timestamp for manually watched fixtures."""
    return pd.Timestamp.now(tz="UTC").normalize()


def _resolve_manual_fixture_dates(fixtures):
    """Resolve real kickoff times from the bot's existing fixture feed."""
    fixtures = fixtures.copy()
    if fixtures.empty:
        return fixtures

    feeds = []
    for name in ("fetch_fixtures", "get_fixtures", "load_fixtures", "fetch_upcoming_fixtures"):
        fn = globals().get(name)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, pd.DataFrame) and not data.empty:
                    feeds.append(data)
            except Exception:
                pass

    if not feeds:
        return fixtures

    feed = pd.concat(feeds, ignore_index=True)
    lower = {str(c).lower(): c for c in feed.columns}

    def col(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    hc = col("home", "home_team", "hometeam")
    ac = col("away", "away_team", "awayteam")
    dc = col("date", "datetime", "fixture_date", "kickoff", "match_date")
    if not hc or not ac or not dc:
        return fixtures

    now_utc = pd.Timestamp.now(tz="UTC")

    for i, row in fixtures.iterrows():
        home = str(row["home"]).strip().casefold()
        away = str(row["away"]).strip().casefold()
        m = feed[
            feed[hc].astype(str).str.strip().str.casefold().eq(home)
            & feed[ac].astype(str).str.strip().str.casefold().eq(away)
        ]
        if m.empty:
            continue

        dates = pd.to_datetime(m[dc], utc=True, errors="coerce").dropna()
        future = dates[dates > now_utc]
        if not future.empty:
            fixtures.at[i, "date"] = future.min()

    return fixtures


def _manual_exact_watch_fixtures():
    """Fixed fixtures supplied for the current exact-score test."""
    return pd.DataFrame([
        {"home": "FC Thun", "away": "Lech Poznań", "league": "UEFA", "date": None},
        {"home": "Iberia 1999", "away": "Jagiellonia Białystok", "league": "UEFA", "date": None},
        {"home": "Monaco", "away": "Górnik Zabrze", "league": "UEFA", "date": None},
        {"home": "Barcelona", "away": "Athletic Bilbao", "league": "UEFA", "date": None},
        {"home": "Partizan", "away": "Getafe", "league": "UEFA", "date": None},
        {"home": "Hviti Riddarinn", "away": "Fjolnir Reykjavik", "league": "UEFA", "date": None},
    ])

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


def competition_tier(league_name):
    """UEFA-first competition tier: 3 UCL, 2 UEL, 1 UECL, 0 elsewhere."""
    key = re.sub(r"\s+", " ", str(league_name or "").strip().lower())
    return UEFA_COMPETITION_TIERS.get(key, NON_UEFA_TIER)


def is_uefa_competition(league_name):
    return competition_tier(league_name) > 0


def competition_priority(league_name):
    """Large score bonus for UEFA competitions without making non-UEFA invisible."""
    tier = competition_tier(league_name)
    return {
        3: 1.00,
        2: 0.94,
        1: 0.88,
        0: 0.00,
    }.get(tier, 0.0)


def _safe_mode(values):
    vals = [int(v) for v in values if pd.notna(v)]
    if not vals:
        return None, 0
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=lambda k: (counts[k], -abs(k - sum(vals)/len(vals))))
    return best, counts[best]


def historical_patterns(df, home, away, before=None):
    """
    Detect repeatable historical patterns without treating them as destiny.
    Uses recent venue-specific scoring, score-mode recurrence, result streaks
    and low-weight pair-specific H2H score recurrence.
    """
    hs = recent_team_matches(df, home, n=30, before=before)
    aw = recent_team_matches(df, away, n=30, before=before)

    def team_view(x, team, venue):
        wanted = norm_team(team)
        if venue == "home":
            x = x[x["home_key"] == wanted].copy()
            gf = pd.to_numeric(x["hg"], errors="coerce")
            ga = pd.to_numeric(x["ag"], errors="coerce")
        else:
            x = x[x["away_key"] == wanted].copy()
            gf = pd.to_numeric(x["ag"], errors="coerce")
            ga = pd.to_numeric(x["hg"], errors="coerce")
        x = x.assign(_gf=gf, _ga=ga).dropna(subset=["_gf", "_ga"])
        scores = [f"{int(gf)}:{int(ga)}" for gf, ga in zip(x["_gf"], x["_ga"])]
        wins = sum(1 for gf, ga in zip(x["_gf"], x["_ga"]) if gf > ga)
        draws = sum(1 for gf, ga in zip(x["_gf"], x["_ga"]) if gf == ga)
        losses = sum(1 for gf, ga in zip(x["_gf"], x["_ga"]) if gf < ga)
        mode_gf, mode_gf_n = _safe_mode(x["_gf"].tolist())
        mode_ga, mode_ga_n = _safe_mode(x["_ga"].tolist())
        return {
            "scores": scores,
            "wins": wins, "draws": draws, "losses": losses,
            "mode_gf": mode_gf, "mode_ga": mode_ga,
            "mode_gf_count": mode_gf_n, "mode_ga_count": mode_ga_n,
            "sample": len(x),
        }

    h = team_view(hs, home, "home")
    a = team_view(aw, away, "away")

    pair = df[
        (((df.home_key == norm_team(home)) & (df.away_key == norm_team(away))) |
         ((df.home_key == norm_team(away)) & (df.away_key == norm_team(home))))
    ].copy()
    if before is not None:
        b = pd.to_datetime(before, utc=True, errors="coerce")
        pair = pair[pd.to_datetime(pair["date"], utc=True, errors="coerce") < b]
    pair = pair.sort_values("date").tail(12)

    repeated_scores = {}
    for _, r in pair.iterrows():
        if norm_team(r.home) == norm_team(home):
            score = f"{int(r.hg)}:{int(r.ag)}"
        else:
            score = f"{int(r.ag)}:{int(r.hg)}"
        repeated_scores[score] = repeated_scores.get(score, 0) + 1

    h_share = max(h["mode_gf_count"], h["mode_ga_count"]) / max(1, h["sample"])
    a_share = max(a["mode_gf_count"], a["mode_ga_count"]) / max(1, a["sample"])
    h_result_bias = (h["wins"] + 0.5 * h["draws"]) / max(1, h["sample"])
    a_result_bias = (a["wins"] + 0.5 * a["draws"]) / max(1, a["sample"])

    return {
        "home_mode_gf": h["mode_gf"] or 0.0,
        "home_mode_ga": h["mode_ga"] or 0.0,
        "away_mode_gf": a["mode_gf"] or 0.0,
        "away_mode_ga": a["mode_ga"] or 0.0,
        "home_pattern_share": h_share,
        "away_pattern_share": a_share,
        "home_result_bias": h_result_bias,
        "away_result_bias": a_result_bias,
        "h2h_repeated_score": max(repeated_scores.values()) if repeated_scores else 0,
        "h2h_sample": len(pair),
    }


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


def score_matrix(lh, la, max_goals=9):
    matrix = []

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson(h, lh) * poisson(a, la)
            matrix.append((p, h, a))

    return sorted(matrix, reverse=True)


def _empirical_goal_distribution(df, team, venue, before=None, n=20, max_goals=9):
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



def _normalize_score_distribution(rows):
    """Normalize score probabilities and protect against numerical drift."""
    if not rows:
        return []
    vals = [(max(0.0, float(p)), int(h), int(a)) for p, h, a in rows]
    total = sum(p for p, _, _ in vals)
    if total <= 0:
        return [(1.0 / len(vals), h, a) for _, h, a in vals]
    return sorted([(p / total, h, a) for p, h, a in vals], reverse=True)


def _tail_mass_adjustment(score_h, score_a, base_prob, all_rows):
    """
    Hidden v25 signal: compare the candidate with the total probability mass
    of its goal band. A 5:1 is rewarded only when the model itself puts
    meaningful mass into high-scoring states.
    """
    try:
        total = sum(p for p, _, _ in all_rows)
        if total <= 0:
            return 1.0
        high = sum(
            p for p, h, a in all_rows
            if (h + a >= 4)
        ) / total
        candidate = max(1e-12, base_prob)
        # Small bounded correction; never manufactures a high score.
        if score_h + score_a >= 4:
            return max(0.88, min(1.16, 1.0 + 0.55 * (high - 0.20)))
        return max(0.92, min(1.08, 1.0 - 0.20 * max(0.0, high - 0.35)))
    except Exception:
        return 1.0


def _exact_score_matrix(lh, la, home_emp, away_emp, max_goals=9):
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

    blended = _normalize_score_distribution(blended)
    poisson_rows = _normalize_score_distribution(poisson_rows)
    return blended, poisson_rows



def dixon_coles_tau(home_goals, away_goals, lam_home, lam_away, rho=-0.08):
    """Dixon-Coles low-score correction for correlated 0/0, 1/0, 0/1, 1/1 cells."""
    h, a = int(home_goals), int(away_goals)
    if h == 0 and a == 0:
        return max(0.50, 1 - (lam_home * lam_away * rho))
    if h == 1 and a == 0:
        return max(0.50, 1 + (lam_away * rho))
    if h == 0 and a == 1:
        return max(0.50, 1 + (lam_home * rho))
    if h == 1 and a == 1:
        return max(0.50, 1 - rho)
    return 1.0


def exact_score_context_multiplier(home, away, score_h, score_a, context=None):
    """Context multiplier for exact-score ranking; it never uses bookmaker odds."""
    if not context:
        return 1.0
    mult = 1.0
    first_h = context.get("first_leg_home_goals")
    first_a = context.get("first_leg_away_goals")
    if first_h is not None and first_a is not None:
        try:
            deficit = int(first_a) - int(first_h)
            # Current home side trails the tie: modestly increase scenarios with
            # >=2 home goals and modestly decrease passive 0/1-goal home outcomes.
            if deficit > 0:
                if score_h >= 2:
                    mult *= 1.0 + EXACT_WATCH_CONTEXT_STRENGTH
                elif score_h == 0:
                    mult *= 1.0 - EXACT_WATCH_CONTEXT_STRENGTH * 0.65
                elif score_h == 1:
                    mult *= 1.0 - EXACT_WATCH_CONTEXT_STRENGTH * 0.20
            elif deficit < 0:
                if score_a >= 2:
                    mult *= 1.0 + EXACT_WATCH_CONTEXT_STRENGTH * 0.65
                elif score_a == 0:
                    mult *= 1.0 - EXACT_WATCH_CONTEXT_STRENGTH * 0.20
        except Exception:
            pass
    return max(0.75, min(1.30, mult))


def _exact_score_calibration_factor(home, away, score):
    """Very small, conservative self-calibration from resolved historical watch samples."""
    try:
        cal = _exact_watch_calibration_load()
        samples = cal.get("samples", []) if isinstance(cal, dict) else []
        if not samples:
            return 1.0
        key_h, key_a = norm_team(home), norm_team(away)
        attempts = hits = 0
        for row in samples[-500:]:
            if norm_team(row.get("home", "")) != key_h or norm_team(row.get("away", "")) != key_a:
                continue
            preds = row.get("predicted_top3") or []
            if score in preds:
                attempts += 1
                hits += int(row.get("actual") == score)
        if attempts < 3:
            return 1.0
        # Shrink strongly toward 1.0; this prevents a tiny sample from overfitting.
        observed = (hits + 2.0) / (attempts + 4.0)
        return 0.90 + 0.20 * observed
    except Exception:
        return 1.0




def _safe_mean(values, default=0.0):
    try:
        vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        return sum(vals) / len(vals) if vals else default
    except Exception:
        return default


def _team_strength_snapshot(df, home, away):
    """Estimate relative strength only from available historical goals."""
    try:
        if df is None or df.empty or "home" not in df.columns or "away" not in df.columns:
            return None
        d = df.copy()
        hkey, akey = norm_team(home), norm_team(away)
        hr = d[d["home"].astype(str).map(norm_team) == hkey]
        ar = d[d["away"].astype(str).map(norm_team) == akey]
        if len(hr) < 3 or len(ar) < 3:
            return None
        hg = pd.to_numeric(hr["hg"], errors="coerce").dropna()
        ag = pd.to_numeric(ar["ag"], errors="coerce").dropna()
        if hg.empty or ag.empty:
            return None
        return {"home_attack": float(hg.mean()), "away_attack": float(ag.mean()),
                "n": int(min(len(hg), len(ag)))}
    except Exception:
        return None


def _lineup_signal(context):
    """Use lineup/absence information only if explicitly supplied by the caller."""
    if not isinstance(context, dict):
        return 1.0
    # Accepted numeric signal: >1 supports the candidate, <1 weakens it.
    for key in ("lineup_factor", "xi_factor", "availability_factor"):
        try:
            if key in context and context[key] is not None:
                return max(0.85, min(1.15, float(context[key])))
        except Exception:
            pass
    return 1.0


def _style_signal(context):
    """Optional tactical/style factor; no web or invented data."""
    if not isinstance(context, dict):
        return 1.0
    for key in ("style_factor", "tactical_factor", "matchup_factor"):
        try:
            if key in context and context[key] is not None:
                return max(0.88, min(1.12, float(context[key])))
        except Exception:
            pass
    return 1.0


def _late_signal(context):
    """Optional late-information factor, e.g. confirmed XI."""
    if not isinstance(context, dict):
        return 1.0
    for key in ("late_factor", "late_signal", "pre_match_factor"):
        try:
            if key in context and context[key] is not None:
                return max(0.90, min(1.10, float(context[key])))
        except Exception:
            pass
    return 1.0


def _find_col(df, names):
    """Return the first matching column name, case-insensitively."""
    if df is None or not hasattr(df, "columns"):
        return None
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def _optional_xg_snapshot(df, home, away, before=None):
    """
    Optional xG/xGA signal. It is deliberately ignored when the data source
    does not contain xG columns, so the bot never invents xG.
    """
    try:
        if df is None or df.empty:
            return None
        d = df.copy()
        if before is not None and "date" in d.columns:
            d = d[d["date"] < before]
        xg_h = _find_col(d, ["home_xg", "xg_home", "home_xg_for", "hxg"])
        xg_a = _find_col(d, ["away_xg", "xg_away", "away_xg_for", "axg"])
        if not xg_h or not xg_a:
            return None
        rows_h = d[d["home"].astype(str).map(norm_team) == norm_team(home)].tail(12)
        rows_a = d[d["away"].astype(str).map(norm_team) == norm_team(away)].tail(12)
        if rows_h.empty or rows_a.empty:
            return None
        vals_h = pd.to_numeric(rows_h[xg_h], errors="coerce").dropna()
        vals_a = pd.to_numeric(rows_a[xg_a], errors="coerce").dropna()
        if len(vals_h) < 3 or len(vals_a) < 3:
            return None
        return {
            "home_xg": float(vals_h.mean()),
            "away_xg": float(vals_a.mean()),
            "n": int(min(len(vals_h), len(vals_a))),
        }
    except Exception:
        return None


def _bayesian_shrink(value, prior, strength):
    """Shrink sparse estimates toward a stable prior."""
    try:
        s = max(0.0, min(0.80, float(strength)))
        return (1.0 - s) * float(value) + s * float(prior)
    except Exception:
        return value


def _score_mc_probability(lh, la, h, a, sims=12000, seed_text=""):
    """
    Deterministic Monte-Carlo check of the Poisson score cell.
    A deterministic seed makes repeated 30-minute scans reproducible.
    """
    import random
    try:
        sims = max(1000, min(int(sims), 50000))
        seed = abs(hash(seed_text)) % (2**32)
        rng = random.Random(seed)
        hits = 0
        # Knuth Poisson sampler; sufficient for the modest football-goal means here.
        def draw(lam):
            L = math.exp(-lam)
            k, prod = 0, 1.0
            while prod > L and k < 30:
                k += 1
                prod *= rng.random()
            return k - 1
        for _ in range(sims):
            if draw(lh) == h and draw(la) == a:
                hits += 1
        return hits / sims
    except Exception:
        return poisson(h, lh) * poisson(a, la)




def _historical_calibration_factor(history, score, default=1.0):
    """
    Conservative empirical calibration. It compares how often a score cell
    occurred versus its model-like mass when enough resolved rows exist.
    Never creates a factor outside [0.85, 1.15].
    """
    try:
        if history is None or history.empty or "hg" not in history.columns or "ag" not in history.columns:
            return default
        h, a = map(int, score.split(":"))
        actual = ((pd.to_numeric(history["hg"], errors="coerce") == h) &
                  (pd.to_numeric(history["ag"], errors="coerce") == a)).mean()
        n = len(history)
        if n < 50:
            return default
        # Shrink aggressively toward 1.0; this is calibration, not overfitting.
        return max(0.85, min(1.15, 1.0 + EXACT_V17_CALIBRATION_WEIGHT * (actual / max(0.01, 0.08) - 1.0)))
    except Exception:
        return default



def _score_robustness(pred, h, a, context=None):
    """Stress-test a score under small model perturbations."""
    try:
        lh, la = float(pred["lh"]), float(pred["la"])
        vals = []
        for fh, fa in ((0.92,0.92),(0.96,1.04),(1.00,1.00),(1.04,0.96),(1.08,1.08)):
            p = poisson(h, max(0.05, lh*fh)) * poisson(a, max(0.05, la*fa))
            vals.append(p)
        if not vals:
            return 0.0
        mean = sum(vals)/len(vals)
        var = sum((x-mean)**2 for x in vals)/len(vals)
        cv = math.sqrt(var)/max(mean, 1e-12)
        return max(0.70, min(1.10, 1.0 - 0.45*cv))
    except Exception:
        return 1.0


def exact_score_intelligence_rank(pred, context=None):
    """
    v25 Exact Score engine: BASELINE -> AI RANKING -> AI LIFT, modeled on the observed score table format.

    BASELINE:
        Pure Poisson probability for the exact score, normalized across the
        full score matrix. This is the transparent mathematical starting point.

    AI RANKING:
        Existing blended Poisson/empirical model + Dixon-Coles + context +
        historical calibration + model agreement/stability.

    AI LIFT:
        Relative change of the AI ranking versus the pure-Poisson baseline.
        A score does not become a pick merely because its lift is high; lift is
        a secondary signal. The primary ranking remains the model's absolute
        ensemble support.

    No bookmaker odds are read or used here.
    """
    rows = []
    base = pred.get("score_components", [])
    if not base:
        for item in pred.get("top_scores", []):
            score = item.get("score", "")
            p = float(item.get("prob", 0.0))
            rows.append({
                **item,
                "intelligence": max(p, 1e-12),
                "rank_probability": max(p, 1e-12),
                "baseline_probability": max(p, 1e-12),
                "ai_lift": 1.0,
                "stability": 0.0,
                "context_multiplier": 1.0,
            })
        total = sum(x["rank_probability"] for x in rows) or 1.0
        for x in rows:
            x["rank_probability"] /= total
        return sorted(rows, key=lambda x: x["rank_probability"], reverse=True)

    agreement = float(pred.get("score_agreement", 0.0))
    margin = max(0.0, float(pred.get("score_margin", 0.0)))
    history_quality = min(
        1.0,
        (float(pred.get("venue_history_home", 0)) +
         float(pred.get("venue_history_away", 0))) /
        max(1.0, 2.0 * EXACT_SCORE_HISTORY_MIN),
    )
    pattern_quality = float(pred.get("pattern_quality", 0.0))
    home, away = pred.get("home", ""), pred.get("away", "")

    # Full-matrix pure-Poisson baseline. This is deliberately independent of
    # empirical/context corrections and therefore gives us a clean comparator.
    poisson_total = sum(max(0.0, float(x.get("poisson", 0.0))) for x in base) or 1.0

    raw_rows = []
    for item in base:
        h, a_score = int(item["h"]), int(item["a"])
        score = f"{h}:{a_score}"
        blend = max(1e-9, float(item["blend"]))
        poisson_p = max(1e-9, float(item["poisson"]))
        empirical_p = max(1e-9, float(item["empirical"]))
        baseline_p = poisson_p / poisson_total

        dc = dixon_coles_tau(
            h, a_score, float(pred["lh"]), float(pred["la"]),
            EXACT_WATCH_DIXON_COLES_RHO
        )
        dc_p = max(1e-9, poisson_p * dc)
        ctx = exact_score_context_multiplier(home, away, h, a_score, context)
        calibration = _exact_score_calibration_factor(home, away, score)

        # Geometric ensemble: several independent views must support the score.
        geo = math.exp(
            0.34 * math.log(blend) +
            0.20 * math.log(poisson_p) +
            0.18 * math.log(empirical_p) +
            0.14 * math.log(dc_p)
        )

        # v25 Monte-Carlo cross-check. It does not replace the analytical model.
        mc = _score_mc_probability(
            float(pred["lh"]), float(pred["la"]), h, a,
            EXACT_V15_MONTE_CARLO_SIMS,
            seed_text=f"{home}|{away}|{h}:{a}"
        )
        mc = max(1e-9, mc)

        # Optional xG evidence: only active when the source really contains xG.
        xg_factor = 1.0
        xg = pred.get("xg_snapshot")
        if xg:
            expected_h = _bayesian_shrink(float(pred["lh"]), float(xg["home_xg"]), EXACT_V15_BAYES_SHRINK)
            expected_a = _bayesian_shrink(float(pred["la"]), float(xg["away_xg"]), EXACT_V15_BAYES_SHRINK)
            # Compare the candidate's implied goal area with the xG-informed area.
            # Small, capped adjustment prevents xG from dominating.
            d = abs(h - expected_h) + abs(a - expected_a)
            xg_factor = max(0.78, min(1.22, math.exp(-0.10 * d)))
        else:
            xg_factor = 1.0

        stability = (
            0.27 * min(geo / 0.12, 1.0) +
            0.20 * agreement +
            0.13 * min(margin / 0.04, 1.0) +
            0.12 * history_quality +
            0.10 * pattern_quality +
            0.18 * min(mc / max(poisson_p, 1e-9), 1.20) / 1.20
        )

        historical_cal = _historical_calibration_factor(
            context.get("history_df") if isinstance(context, dict) else None,
            f"{h}:{a}",
            1.0
        )
        lineup_factor = _lineup_signal(context)
        style_factor = _style_signal(context)
        late_factor = _late_signal(context)

        robustness = _score_robustness(pred, h, a, context)
        tail_adjustment = _tail_mass_adjustment(h, a, geo, blended_dist)

        # High-score scenarios receive a modest evidence-based boost only when
        # the recent goal environment + model distribution support them.
        long_tail_factor = 1.0
        if (h + a) >= EXACT_LONG_TAIL_MIN_TOTAL and volatility_signal > 0.35:
            long_tail_factor = 1.0 + (EXACT_LONG_TAIL_MAX_BOOST - 1.0) * volatility_signal

        strength_factor = 1.0
        strength = pred.get("strength_snapshot")
        if strength:
            # Relative strength is a bounded correction, not a replacement for Poisson.
            gap = abs(float(strength["home_attack"]) - float(strength["away_attack"]))
            strength_factor = max(0.94, min(1.06, 1.0 + 0.015 * min(gap, 4.0)))

        intelligence = (
            geo *
            (0.74 + 0.26 * stability) *
            ctx *
            (1.0 + EXACT_SCORE_CALIBRATION_WEIGHT * (calibration - 1.0)) *
            (1.0 + EXACT_V15_XG_WEIGHT * (xg_factor - 1.0)) *
            (1.0 + EXACT_V15_RECENCY_WEIGHT * min(stability, 1.0)) *
            (1.0 + EXACT_V17_LINEUP_WEIGHT * (lineup_factor - 1.0)) *
            (1.0 + EXACT_V17_STYLE_WEIGHT * (style_factor - 1.0)) *
            (1.0 + EXACT_V17_STRENGTH_WEIGHT * (strength_factor - 1.0)) *
            (1.0 + EXACT_V17_LATE_SIGNAL_WEIGHT * (late_factor - 1.0)) * historical_cal *
            long_tail_factor
        )

        raw_rows.append({
            "score": score,
            "prob": blend,
            "poisson_prob": poisson_p,
            "empirical_prob": empirical_p,
            "dixon_coles_prob": dc_p,
            "ensemble_probability": geo,
            "monte_carlo_probability": mc,
            "xg_factor": xg_factor,
            "long_tail_factor": long_tail_factor,
            "volatility_signal": volatility_signal,
            "stability": stability,
            "context_multiplier": ctx,
            "calibration_factor": calibration,
            "baseline_probability": baseline_p,
            "intelligence": max(1e-12, intelligence),
        })

    total = sum(x["intelligence"] for x in raw_rows) or 1.0
    for x in raw_rows:
        x["rank_probability"] = x["intelligence"] / total
        # AI lift is intentionally capped. A small baseline can otherwise make
        # a mediocre score look artificially spectacular.
        raw_lift = x["rank_probability"] / max(x["baseline_probability"], 1e-9)
        x["ai_lift"] = max(0.40, min(EXACT_V13_LIFT_CAP, raw_lift))

    raw_rows.sort(
        key=lambda x: (
            x["rank_probability"],
            x["ai_lift"],
            x["ensemble_probability"],
            x["stability"],
        ),
        reverse=True,
    )

    # v25: do not let the first three slots become permanently dominated by
    # 0:0/1:0/1:1/2:1. Keep the mathematically strongest score first, then allow
    # a genuinely supported high-scoring scenario into TOP 3. This is a
    # diversification rule, not a forced long-shot pick.
    if raw_rows:
        top = raw_rows[0]
        tail_candidates = [
            x for x in raw_rows[1:]
            if (int(x["score"].split(":")[0]) + int(x["score"].split(":")[1]) >= EXACT_LONG_TAIL_MIN_TOTAL)
            and x["rank_probability"] >= top["rank_probability"] * EXACT_LONG_TAIL_MIN_RELATIVE
            and x.get("volatility_signal", 0.0) >= 0.35
        ]
        if tail_candidates:
            tail = max(tail_candidates, key=lambda x: x["rank_probability"] * (1.0 + 0.20 * x.get("volatility_signal", 0.0)))
            remaining = [x for x in raw_rows if x is not tail and x is not top]
            raw_rows = [top, tail] + remaining
            raw_rows.sort(key=lambda x: (x["rank_probability"], x["ai_lift"]), reverse=True)
            # Reinsert the tail as the second scenario only if it is still
            # materially supported; otherwise preserve pure ranking.
            if tail in raw_rows and raw_rows[1] is not tail:
                raw_rows.remove(tail)
                raw_rows.insert(1, tail)

    return raw_rows

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

    # Historical-repeat features: a small regularizer, never a dominant factor.
    patterns = historical_patterns(df, home, away, before=match_date)
    repeat_h = (patterns["home_pattern_share"] - 0.20) * 0.06
    repeat_a = (patterns["away_pattern_share"] - 0.20) * 0.06
    lh *= math.exp(repeat_h)
    la *= math.exp(repeat_a)

    lh = max(0.20, min(4.20, lh))
    la = max(0.15, min(3.80, la))

    home_emp, home_n = _empirical_goal_distribution(
        df, home, "home", before=match_date, n=20
    )
    away_emp, away_n = _empirical_goal_distribution(
        df, away, "away", before=match_date, n=20
    )

    blended_dist, poisson_dist = _exact_score_matrix(
        lh, la, home_emp, away_emp, max_goals=EXACT_SCORE_MAX_GOALS
    )

    # v25: high-score / long-tail signal. It is evidence-driven, not a forced
    # "5:1 generator". We inspect recent combined-goal volatility and whether
    # the model itself assigns meaningful mass to 4+ goal scenarios.
    recent_all = pd.concat([
        recent_team_matches(df, home, n=15, before=match_date),
        recent_team_matches(df, away, n=15, before=match_date),
    ], ignore_index=True).drop_duplicates(subset=["date", "home_key", "away_key"])
    recent_totals = []
    if not recent_all.empty:
        for _, rr in recent_all.iterrows():
            try:
                recent_totals.append(float(rr.hg) + float(rr.ag))
            except Exception:
                pass
    avg_total = _safe_mean(recent_totals, default=float(lh + la))
    high_total_share = (sum(1 for x in recent_totals if x >= EXACT_LONG_TAIL_MIN_TOTAL) /
                        max(1, len(recent_totals)))
    model_high_mass = sum(float(p) for p, h, a in blended_dist
                          if h + a >= EXACT_LONG_TAIL_MIN_TOTAL)
    volatility_signal = max(0.0, min(1.0,
        0.45 * min(avg_total / 4.5, 1.0) +
        0.30 * min(high_total_share / 0.45, 1.0) +
        0.25 * min(model_high_mass / 0.50, 1.0)
    ))

    score_components = []
    for p, h, a in blended_dist:
        pp = poisson(h, lh) * poisson(a, la)
        ep = home_emp[h] * away_emp[a]
        score_components.append({
            "h": h, "a": a, "blend": float(p),
            "poisson": float(pp), "empirical": float(ep),
        })

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
    pattern_quality = min(
        1.0,
        0.35 * patterns["home_pattern_share"]
        + 0.35 * patterns["away_pattern_share"]
        + 0.30 * min(patterns["h2h_repeated_score"] / 3.0, 1.0)
    )
    confidence = (
        0.50 * min(raw_p / 0.18, 1.0)
        + 0.22 * agreement
        + 0.10 * min(margin / 0.04, 1.0)
        + 0.10 * history_quality
        + 0.08 * pattern_quality
    )
    confidence = max(0.0, min(1.0, confidence))

    top_scores = []
    for p, h, a in blended_dist[:10]:
        scenario = "standard"
        if h + a >= EXACT_LONG_TAIL_MIN_TOTAL:
            scenario = "high-goal scenario"
        top_scores.append({
            "score": f"{h}:{a}",
            "prob": float(p),
            "scenario": scenario,
        })

    # Optional xG snapshot. We do not fabricate it when the source lacks xG.
    xg_snapshot = _optional_xg_snapshot(df, home, away, before=match_date)
    strength_snapshot = _team_strength_snapshot(df, home, away)

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
        "pattern_quality": float(pattern_quality),
        "historical_patterns": patterns,
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
        "score_components": score_components,
        "xg_snapshot": xg_snapshot, "strength_snapshot": strength_snapshot,
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
    """Internal probability of the selected double exact-score pair."""
    if not items or len(items) < 2:
        return 0.0
    probs = []
    for x in items[:2]:
        scores = x.get("scores") or []
        if not scores:
            return 0.0
        probs.append(float(scores[0].get("rank_probability", scores[0].get("prob", 0.0))))
    return math.prod(probs)


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


def _exact_score_rejection_reasons(match):
    """Return precise reasons why an exact-score candidate was rejected."""
    reasons = []
    probability = float(match.get("prob", 0.0))
    agreement = float(match.get("score_agreement", 0.0))
    confidence = float(match.get("score_confidence", 0.0))
    margin = float(match.get("score_margin", 0.0))
    home_n = int(match.get("venue_history_home", 0))
    away_n = int(match.get("venue_history_away", 0))

    if probability < EXACT_SCORE_MIN_PROB:
        reasons.append(f"dokładny wynik ma tylko {probability:.1%} (minimum {EXACT_SCORE_MIN_PROB:.1%})")
    if agreement < EXACT_SCORE_MIN_AGREEMENT:
        reasons.append(f"modele są zbyt mało zgodne ({agreement:.0%}, minimum {EXACT_SCORE_MIN_AGREEMENT:.0%})")
    if confidence < STRONG_SIGNAL_P:
        reasons.append(f"siła sygnału {confidence:.1%} < {STRONG_SIGNAL_P:.0%}")
    if margin < EXACT_SCORE_MIN_MARGIN:
        reasons.append(f"zbyt mała przewaga nad 2. wynikiem ({margin:.1%})")
    if (home_n + away_n) < EXACT_SCORE_HISTORY_MIN * 2:
        reasons.append(
            f"za mało historii stadionowej ({home_n + away_n}, minimum {EXACT_SCORE_HISTORY_MIN * 2})"
        )
    return reasons


def _is_strong_exact_score(match):
    """Very strict admission gate; weak candidates never enter the active coupon."""
    return not _exact_score_rejection_reasons(match)


def _record_rejections(rejected):
    """Persist rejection diagnostics without treating rejected tips as signals."""
    if not rejected:
        return
    rows = []
    for item in rejected:
        rows.append({
            "timestamp": now().isoformat(),
            "date": item.get("date", ""),
            "home": item.get("home", ""),
            "away": item.get("away", ""),
            "league": item.get("league", ""),
            "score": item.get("score", ""),
            "probability": item.get("prob", 0.0),
            "confidence": item.get("score_confidence", 0.0),
            "agreement": item.get("score_agreement", 0.0),
            "margin": item.get("score_margin", 0.0),
            "history": int(item.get("venue_history_home", 0)) + int(item.get("venue_history_away", 0)),
            "reasons": " | ".join(item.get("rejection_reasons", [])),
        })
    try:
        df = pd.DataFrame(rows)
        if REJECTION_LOG.exists():
            old = pd.read_csv(REJECTION_LOG)
            df = pd.concat([old, df], ignore_index=True)
        # Keep the diagnostic file bounded so Render storage does not grow forever.
        df.tail(5000).to_csv(REJECTION_LOG, index=False)
    except Exception as exc:
        log.warning("Nie udało się zapisać diagnostyki odrzuceń: %s", exc)


def _candidate_matches(history, upcoming_df, with_diagnostics=False):
    """Build strong exact-score candidates and separately collect rejected ones."""
    cand = []
    rejected = []

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

        p = predict(history, home, away, league=league, match_date=match_date)
        p.update({
            "date": str(match_date),
            "league": league,
            "competition_tier": competition_tier(league),
            "competition_priority": competition_priority(league),
            "uefa_focus": bool(is_uefa_competition(league)),
            "signal_code": "EXACT",
            "signal": "Dokładny wynik",
            "signal_prob": float(p["prob"]),
            "label": signal_level(p["score_confidence"]),
            "source": str(r.get("source", "football-data")),
        })

        reasons = _exact_score_rejection_reasons(p)
        if reasons:
            p["rejection_reasons"] = reasons
            rejected.append(p)
            continue

        cand.append(p)

    cand.sort(
        key=lambda x: (
            -int(x.get("competition_tier", 0)) if UEFA_PRIORITY_ENABLED else 0,
            -float(x.get("score_confidence", 0.0)),
            -float(x.get("prob", 0.0)),
            -float(x.get("score_agreement", 0.0)),
            -float(x.get("score_margin", 0.0)),
            pd.to_datetime(x.get("date"), utc=True, errors="coerce"),
        )
    )
    rejected.sort(key=lambda x: float(x.get("score_confidence", 0.0)), reverse=True)
    _record_rejections(rejected)
    if with_diagnostics:
        return cand, rejected
    return cand


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


def _pair_rank(pair):
    """Rank a 2-leg AKO: UEFA tier first, then weakest-leg exact-score quality."""
    if not pair or len(pair) != 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    confs = [float(x.get("score_confidence", 0.0)) for x in pair]
    probs = [float(x.get("prob", 0.0)) for x in pair]
    agreements = [float(x.get("score_agreement", 0.0)) for x in pair]
    tiers = [int(x.get("competition_tier", competition_tier(x.get("league", "")))) for x in pair]
    uefa_count = sum(1 for t in tiers if t > 0)
    # For a pair, mixed UEFA/non-UEFA is below an all-UEFA pair.
    pair_tier_score = (min(tiers) / 3.0) if UEFA_PRIORITY_ENABLED else 0.0
    all_uefa_bonus = 1.0 if uefa_count == 2 else 0.0
    return (
        pair_tier_score,
        all_uefa_bonus,
        min(confs),
        min(probs),
        sum(agreements) / 2.0,
    )


def _best_pair(candidates):
    """Choose the strongest two-fixture pair; never use two legs from one fixture."""
    if len(candidates) < 2:
        return None
    best = None
    best_rank = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if _fixture_key(a) == _fixture_key(b):
                continue
            pair = [a, b]
            rank = _pair_rank(pair)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = pair
    return best


def _choose_initial_pair(candidates):
    return _best_pair(candidates)


def _pair_strictly_better(new_pair, old_pair):
    """Only replace the active pair with a genuinely stronger pair.
    When UEFA strict focus is active, a UEFA pair cannot be replaced by
    a domestic pair of lower competition tier.
    """
    if not new_pair or not old_pair or len(new_pair) != 2 or len(old_pair) != 2:
        return False

    old_uefa = all(competition_tier(x.get("league", "")) > 0 for x in old_pair)
    new_uefa = all(competition_tier(x.get("league", "")) > 0 for x in new_pair)
    if UEFA_STRICT_FOCUS and old_uefa and not new_uefa:
        return False

    old_min_tier = min(competition_tier(x.get("league", "")) for x in old_pair)
    new_min_tier = min(competition_tier(x.get("league", "")) for x in new_pair)
    if UEFA_REPLACEMENT_TIER_LOCK and new_min_tier < old_min_tier:
        return False

    new_rank = _pair_rank(new_pair)
    old_rank = _pair_rank(old_pair)

    # Competition priority is treated as a gate, not something quality can simply cancel out.
    if new_rank[0] < old_rank[0]:
        return False

    quality_new = new_rank[2:]
    quality_old = old_rank[2:]
    not_weaker = all(n >= o for n, o in zip(quality_new, quality_old))
    meaningful = (
        quality_new[0] >= quality_old[0] + EXACT_PAIR_REPLACEMENT_EDGE
        or quality_new[1] >= quality_old[1] + EXACT_PAIR_REPLACEMENT_EDGE
        or quality_new[2] >= quality_old[2] + EXACT_PAIR_REPLACEMENT_EDGE
    )
    return not_weaker and meaningful


def _improved_pair(current_pair, fresh_candidates):
    """Search the whole upcoming pool, but replace the active pair only with a genuinely stronger pair."""
    best = _best_pair(fresh_candidates)
    if best is None or not _pair_strictly_better(best, current_pair):
        return current_pair, False, None
    return best, True, best


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

def _utc_local_day_key():
    return now().strftime("%Y-%m-%d")


def _load_daily_signal_state():
    today = _utc_local_day_key()
    if not DAILY_SIGNAL_STATE.exists():
        return {"date": today, "updates": 0}
    try:
        data = json.loads(DAILY_SIGNAL_STATE.read_text(encoding="utf-8"))
        if data.get("date") != today:
            return {"date": today, "updates": 0}
        return {
            "date": today,
            "updates": int(data.get("updates", 0)),
        }
    except Exception:
        return {"date": today, "updates": 0}


def _daily_signal_allowed():
    return _load_daily_signal_state()["updates"] < MAX_DAILY_SIGNAL_UPDATES


def _count_daily_signal_update(reason, pair):
    data = _load_daily_signal_state()
    data["updates"] += 1
    data["last_reason"] = reason
    data["timestamp"] = now().isoformat()
    data["pair_fingerprint"] = [
        _signal_fingerprint(x) for x in (pair or [])
    ]
    DAILY_SIGNAL_STATE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


def _hourly_status_message(scanned, rejected, current=None, extra=""):
    lines = [
        "⏰ KONTROLA GODZINNA — BEZ WYJĄTKÓW",
        "",
        f"🔎 Przeanalizowano meczów: {scanned}",
        f"❌ Odrzucono słabych sygnałów: {rejected}",
        f"🧠 Priorytet: Liga Mistrzów → Liga Europy → Liga Konferencji",
    ]
    daily = _load_daily_signal_state()
    lines += [
        f"📨 Nowych aktualizacji sygnału dziś: {daily['updates']}/{MAX_DAILY_SIGNAL_UPDATES}",
    ]
    if current and len(current) == 2:
        lines += [
            "",
            f"🔒 Aktywne mecze: {current[0]['home']} — {current[0]['away']} | {current[1]['home']} — {current[1]['away']}",
            f"🎯 Wyniki: {current[0].get('score')} + {current[1].get('score')}",
            f"📡 Siła AKO: {_pair_strength(current):.1%}",
        ]
    if extra:
        lines += ["", extra]
    return "\n".join(lines)



# ============================================================
# DAILY COMBINED-MARKET ENGINE
# ============================================================

PRIORITY_DOMESTIC = {
    "serie a": 1.00,
    "la liga": 1.00,
    "bundesliga": 1.00,
    "premier league": 1.00,
    "ligue 1": 0.98,
    "ekstraklasa": 0.98,
    "primeira liga": 0.96,
    "portugal": 0.96,
    "turkey": 0.94,
    "turkish super lig": 0.94,
    "croatia": 0.92,
    "croatian hnl": 0.92,
}

def _league_priority_for_daily(league):
    key = re.sub(r"\s+", " ", str(league or "").strip().lower())
    tier = competition_tier(key)
    if tier == 3:
        return 1.12
    if tier == 2:
        return 1.09
    if tier == 1:
        return 1.06
    for name, value in PRIORITY_DOMESTIC.items():
        if name in key:
            return value
    return 0.80


def _market_candidate(match, label, probability, code):
    probability = float(probability or 0.0)
    if probability < MARKET_MIN_PROB:
        return None
    # Confidence is deliberately below 100% even when the model probability is high.
    # It is a model score, not a guarantee.
    base_conf = 0.72 * probability + 0.28 * float(match.get("score_confidence", 0.0))
    priority = _league_priority_for_daily(match.get("league", ""))
    quality = base_conf * priority
    return {
        "home": match["home"],
        "away": match["away"],
        "date": match.get("date", ""),
        "league": match.get("league", ""),
        "competition_tier": competition_tier(match.get("league", "")),
        "competition_priority": priority,
        "fixture_key": _fixture_key(match),
        "market_code": code,
        "market": label,
        "probability": probability,
        "confidence": min(0.99, base_conf),
        "quality": quality,
        "source": match.get("source", ""),
        "expected_goals": match.get("markets", {}).get("expected_goals", {}),
    }


def _daily_market_candidates(history, upcoming_df):
    """Analyze today's fixtures and return strong non-exact-score market signals."""
    out = []
    rejected = 0
    if upcoming_df is None or upcoming_df.empty:
        return out, rejected

    today = now().date()
    for _, r in upcoming_df.iterrows():
        status = str(r.get("status", "")).upper()
        if status in {"FT", "AET", "PEN", "CANC", "CANCELLED", "POSTPONED"}:
            continue

        dt = pd.to_datetime(r.get("date"), utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        if dt.tz_convert(TZ).date() != today:
            continue

        home, away = str(r.get("home", "")), str(r.get("away", ""))
        league = str(r.get("league", ""))
        if not home or not away or home == "nan" or away == "nan":
            continue

        try:
            pred = predict(history, home, away, league=league, match_date=dt)
            pred["markets"] = market_options_from_prediction(pred)
        except Exception as exc:
            log.warning("Daily market prediction failed %s-%s: %s", home, away, exc)
            rejected += 1
            continue

        p1x2 = pred.get("p1x2", {})
        choices = [
            ("1", f"{home} wygra", p1x2.get("home", 0.0)),
            ("X2", f"{away} lub remis (X2)", p1x2.get("away", 0.0) + p1x2.get("draw", 0.0)),
            ("1X", f"{home} lub remis (1X)", p1x2.get("home", 0.0) + p1x2.get("draw", 0.0)),
        ]

        markets = pred.get("markets", {})
        for t in markets.get("totals", []):
            line = t.get("line")
            choices.append(("U"+str(line), f"Poniżej {line:g} gola", t.get("under", 0.0)))
            choices.append(("O"+str(line), f"Powyżej {line:g} gola", t.get("over", 0.0)))

        btts = markets.get("btts", {})
        choices += [
            ("BTTS_NO", "Obie drużyny strzelą: Nie", btts.get("no", 0.0)),
            ("BTTS_YES", "Obie drużyny strzelą: Tak", btts.get("yes", 0.0)),
        ]

        tg = markets.get("team_goals", {})
        choices += [
            ("H05", f"{home} strzeli powyżej 0.5 gola", tg.get("home_over_0_5", 0.0)),
            ("A05", f"{away} strzeli powyżej 0.5 gola", tg.get("away_over_0_5", 0.0)),
            ("H15", f"{home} strzeli powyżej 1.5 gola", tg.get("home_over_1_5", 0.0)),
            ("A15", f"{away} strzeli powyżej 1.5 gola", tg.get("away_over_1_5", 0.0)),
        ]

        local = []
        seen_codes = set()
        for code, label, prob in choices:
            if code in seen_codes:
                continue
            seen_codes.add(code)
            c = _market_candidate(pred, label, prob, code)
            if c:
                local.append(c)

        # One market per fixture in the combined coupon. Prefer the strongest,
        # but don't allow a very high-probability market to hide all diversity.
        if local:
            local.sort(key=lambda x: (x["quality"], x["probability"], x["confidence"]), reverse=True)
            out.append(local[0])
        else:
            rejected += 1

    # One fixture = one leg. UEFA gets a priority bonus, then the strongest model probability.
    out.sort(key=lambda x: (x["quality"], x["probability"], x["confidence"]), reverse=True)
    return out, rejected


def _daily_market_candidates_test(history, upcoming_df):
    """TEST MODE: return the strongest model markets without the 72% admission gate."""
    out = []
    scanned = 0
    errors = 0
    if upcoming_df is None or upcoming_df.empty:
        return out, scanned, errors

    today = now().date()
    for _, r in upcoming_df.iterrows():
        status = str(r.get("status", "")).upper()
        if status in {"FT", "AET", "PEN", "CANC", "CANCELLED", "POSTPONED"}:
            continue

        dt = pd.to_datetime(r.get("date"), utc=True, errors="coerce")
        if pd.isna(dt) or dt.tz_convert(TZ).date() != today:
            continue

        home, away = str(r.get("home", "")), str(r.get("away", ""))
        league = str(r.get("league", ""))
        if not home or not away or home == "nan" or away == "nan":
            continue

        scanned += 1
        try:
            pred = predict(history, home, away, league=league, match_date=dt)
            pred["markets"] = market_options_from_prediction(pred)
        except Exception as exc:
            log.warning("TEST market prediction failed %s-%s: %s", home, away, exc)
            errors += 1
            continue

        p1x2 = pred.get("p1x2", {})
        choices = [
            ("1", f"{home} wygra", p1x2.get("home", 0.0)),
            ("X", "Remis", p1x2.get("draw", 0.0)),
            ("2", f"{away} wygra", p1x2.get("away", 0.0)),
            ("X2", f"{away} lub remis (X2)", p1x2.get("away", 0.0) + p1x2.get("draw", 0.0)),
            ("1X", f"{home} lub remis (1X)", p1x2.get("home", 0.0) + p1x2.get("draw", 0.0)),
        ]

        markets = pred.get("markets", {})
        for t in markets.get("totals", []):
            line = t.get("line")
            choices.append(("U"+str(line), f"Poniżej {line:g} gola", t.get("under", 0.0)))
            choices.append(("O"+str(line), f"Powyżej {line:g} gola", t.get("over", 0.0)))

        btts = markets.get("btts", {})
        choices += [
            ("BTTS_NO", "Obie drużyny strzelą: Nie", btts.get("no", 0.0)),
            ("BTTS_YES", "Obie drużyny strzelą: Tak", btts.get("yes", 0.0)),
        ]

        tg = markets.get("team_goals", {})
        choices += [
            ("H05", f"{home} strzeli powyżej 0.5 gola", tg.get("home_over_0_5", 0.0)),
            ("A05", f"{away} strzeli powyżej 0.5 gola", tg.get("away_over_0_5", 0.0)),
            ("H15", f"{home} strzeli powyżej 1.5 gola", tg.get("home_over_1_5", 0.0)),
            ("A15", f"{away} strzeli powyżej 1.5 gola", tg.get("away_over_1_5", 0.0)),
        ]

        # TEST MODE deliberately does not apply MARKET_MIN_PROB.
        # We still compute the same confidence/quality formula as production.
        for code, label, prob in choices:
            prob = float(prob or 0.0)
            base_conf = 0.72 * prob + 0.28 * float(pred.get("score_confidence", 0.0))
            priority = _league_priority_for_daily(league)
            quality = base_conf * priority
            out.append({
                "home": home,
                "away": away,
                "date": str(dt),
                "league": league,
                "competition_tier": competition_tier(league),
                "competition_priority": priority,
                "fixture_key": _fixture_key(pred),
                "market_code": code,
                "market": label,
                "probability": prob,
                "confidence": min(0.99, base_conf),
                "quality": quality,
                "source": str(r.get("source", "")),
                "score": pred.get("score", ""),
                "score_confidence": float(pred.get("score_confidence", 0.0)),
            })

    # For readability: first take the best market from each fixture,
    # then rank fixtures globally. This prevents one fixture flooding the report.
    best_by_fixture = {}
    for c in out:
        key = c["fixture_key"]
        if key not in best_by_fixture or (c["quality"], c["probability"]) > (
            best_by_fixture[key]["quality"], best_by_fixture[key]["probability"]
        ):
            best_by_fixture[key] = c

    ranked = list(best_by_fixture.values())
    ranked.sort(key=lambda x: (x["quality"], x["probability"], x["confidence"]), reverse=True)
    return ranked, scanned, errors


def _exact_watch_state_load():
    if not EXACT_WATCH_STATE.exists():
        return {}
    try:
        data = json.loads(EXACT_WATCH_STATE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _exact_watch_state_save(data):
    EXACT_WATCH_STATE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _exact_watch_predictions(history):
    """Analyze all valid upcoming fixed fixtures and rank them by exact-score strength."""
    history_rows = int(len(history))
    history_mode = "normal" if history_rows >= 30 else ("sparse-data" if history_rows > 0 else "prior-only")
    fixtures = _manual_exact_watch_fixtures()
    fixtures = _resolve_manual_fixture_dates(fixtures)
    result = []
    errors = 0
    skipped_past = 0
    current_utc = pd.Timestamp.now(tz="UTC")
    for _, r in fixtures.iterrows():
        try:
            dt = pd.to_datetime(r["date"], utc=True, errors="coerce")
            if pd.isna(dt):
                skipped_past += 1
                continue
            # Finished/started fixtures are not eligible for a new exact-score tip.
            if dt <= current_utc:
                skipped_past += 1
                continue
            pred = predict(history, r["home"], r["away"], league=r["league"], match_date=dt)
            pred["history_mode"] = history_mode
            pred["history_rows"] = history_rows
            ranked = exact_score_intelligence_rank(
                pred,
                context={
                    **(r.get("context") or {}),
                    "history_mode": history_mode,
                    "history_rows": history_rows,
                    "history_df": history,
                },
            )
            internal = ranked[:max(EXACT_WATCH_INTERNAL_SCORES, EXACT_WATCH_TOP_SCORES)]
            scores = internal[:EXACT_WATCH_TOP_SCORES]
            top = scores[0] if scores else {}
            second = scores[1] if len(scores) > 1 else {}
            # v13 fixture score: BASELINE + AI RANKING + AI LIFT.
            # The lift is secondary; it cannot rescue a weak absolute score.
            top_prob = float(top.get("rank_probability", top.get("prob", 0.0)))
            second_prob = float(second.get("rank_probability", second.get("prob", 0.0)))
            third_prob = float(scores[2].get("rank_probability", 0.0)) if len(scores) > 2 else 0.0
            stability = float(top.get("stability", 0.0))
            margin = max(0.0, top_prob - second_prob)
            top3_mass = min(1.0, top_prob + second_prob + third_prob)
            ai_lift = float(top.get("ai_lift", 1.0))
            baseline = float(top.get("baseline_probability", 0.0))
            lift_score = min(1.0, max(0.0, (ai_lift - 0.40) / max(0.01, EXACT_V13_LIFT_CAP - 0.40)))
            fixture_quality = (
                top_prob *
                (0.56 +
                 EXACT_V13_STABILITY_WEIGHT * stability +
                 0.10 * min(1.0, margin / 0.08) +
                 EXACT_V13_TOP3_MASS_WEIGHT * min(1.0, top3_mass / 0.45) +
                 EXACT_V13_AI_LIFT_WEIGHT * lift_score)
            )
            # Sparse history lowers confidence, not eligibility.
            if history_mode == "sparse-data":
                fixture_quality *= 0.72
            elif history_mode == "prior-only":
                fixture_quality *= 0.45
            result.append({
                "home": r["home"], "away": r["away"], "league": r["league"],
                "date": str(dt), "scores": scores, "internal_scores": internal,
                "best_score": scores[0]["score"] if scores else pred.get("score", ""),
                "score_confidence": float(pred.get("score_confidence", 0.0)),
                "baseline_probability": float(top.get("baseline_probability", 0.0)) if top else 0.0,
                "ai_lift": float(top.get("ai_lift", 1.0)) if top else 1.0,
                "base_prediction": pred, "context": r.get("context") or {},
                "history_mode": history_mode, "history_rows": history_rows,
                "fixture_quality": fixture_quality,
            })
        except Exception as exc:
            log.warning("Exact watch prediction failed %s-%s: %s", r["home"], r["away"], exc)
            errors += 1
    result.sort(key=lambda x: (x.get("fixture_quality", 0.0), x.get("score_confidence", 0.0)), reverse=True)
    return result, errors, skipped_past


def _exact_watch_combinations(predictions):
    """
    Build five independent-looking double exact-score coupons from the SAME
    two selected fixtures. The first is the mathematically strongest pair;
    later coupons add only small, plausible diversification. Odds are never used.
    """
    if len(predictions) != 2:
        return []

    left, right = predictions[0]["scores"], predictions[1]["scores"]
    raw = []
    for ia, a in enumerate(left):
        for ib, b in enumerate(right):
            pa = float(a.get("rank_probability", a.get("prob", 0.0)))
            pb = float(b.get("rank_probability", b.get("prob", 0.0)))
            if pa <= 0 or pb <= 0:
                continue
            combined = pa * pb
            stability = 0.5 * (
                float(a.get("stability", 0.0)) +
                float(b.get("stability", 0.0))
            )
            raw.append({
                "first": a["score"],
                "second": b["score"],
                "probability": combined,
                "base_probability": float(a.get("prob", 0.0)) * float(b.get("prob", 0.0)),
                "stability": stability,
                "score_quality": combined * (0.84 + 0.16 * stability),
                "_ia": ia,
                "_ib": ib,
            })

    raw.sort(key=lambda x: x["score_quality"], reverse=True)
    if not raw:
        return []

    selected = [raw[0]]
    # Diversify only when the alternative remains reasonably close to the best.
    # This avoids five near-identical tickets while refusing weak lottery scores.
    for candidate in raw[1:]:
        if len(selected) >= EXACT_WATCH_TOP_COMBINATIONS:
            break
        similarity = 0
        for chosen in selected:
            if candidate["first"] == chosen["first"]:
                similarity += 0.5
            if candidate["second"] == chosen["second"]:
                similarity += 0.5
        adjusted = candidate["score_quality"] * (
            1.0 - EXACT_PAIR_DIVERSITY_PENALTY * similarity
        )
        candidate["diversified_quality"] = adjusted
        if adjusted >= raw[0]["score_quality"] * 0.55:
            selected.append(candidate)

    # Fill any remaining slots with the strongest unused combinations.
    if len(selected) < min(EXACT_WATCH_TOP_COMBINATIONS, len(raw)):
        for candidate in raw:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= EXACT_WATCH_TOP_COMBINATIONS:
                break

    selected.sort(key=lambda x: x["score_quality"], reverse=True)
    return selected[:EXACT_WATCH_TOP_COMBINATIONS]

def _exact_watch_new_signals(previous_predictions, all_predictions, max_new=3):
    """Find genuinely new or materially strengthened exact-score signals.
    A signal is tied to fixture + exact score; it is new when it was not in the
    previous internal ranking, or materially stronger than before. Bookmaker odds
    are never considered.
    """
    old = {}
    for pred in previous_predictions or []:
        key = (norm_team(pred.get("home", "")), norm_team(pred.get("away", "")))
        for sc in pred.get("internal_scores", pred.get("scores", [])):
            score = sc.get("score")
            if score:
                old[(key, score)] = float(sc.get("rank_probability", sc.get("prob", 0.0)))

    candidates = []
    for pred in all_predictions:
        key = (norm_team(pred.get("home", "")), norm_team(pred.get("away", "")))
        for rank, sc in enumerate(pred.get("internal_scores", pred.get("scores", [])), 1):
            score = sc.get("score")
            if not score:
                continue
            current = float(sc.get("rank_probability", sc.get("prob", 0.0)))
            previous = old.get((key, score))
            is_new = previous is None
            strengthened = previous is not None and current >= previous * 1.12 and current - previous >= 0.015
            if is_new or strengthened:
                candidates.append({
                    "home": pred["home"], "away": pred["away"], "score": score,
                    "rank": rank, "probability": current,
                    "previous_probability": previous, "strengthened": strengthened
                })
    candidates.sort(key=lambda x: (x["probability"], x.get("strengthened", False)), reverse=True)
    return candidates[:max_new]


def _exact_watch_message(all_predictions, selected, combinations, update=False, errors=0, new_combinations=None, new_signals=None):
    title = "🔄 AKTUALIZACJA 30-MINUTOWA" if update else "🎯 EXACT DOUBLE INTELLIGENCE — 4 MECZE → 2 NAJLEPSZE → 5 AKO x2"
    cal = _exact_watch_calibration_load()
    resolved = int(cal.get("resolved", 0))
    hits = int(cal.get("hits", 0))
    cal_line = f"📈 Historyczna trafność TOP3 tego modułu: {hits}/{resolved} = {hits/resolved:.1%}" if resolved else "📈 Historyczna trafność TOP3: jeszcze brak prób"
    mode = all_predictions[0].get("history_mode", "unknown") if all_predictions else "unknown"
    rows = all_predictions[0].get("history_rows", 0) if all_predictions else 0
    mode_text = {
        "normal": "🟢 Dane historyczne: tryb normalny",
        "sparse-data": "🟡 Dane historyczne: tryb ograniczony (mała próbka)",
        "prior-only": "🔴 Brak historii drużyn — używam wyłącznie stabilnych priorytetów; sygnał ma niską wiarygodność",
    }.get(mode, f"⚪ Tryb danych: {mode}")
    lines = [
        title,
        f"📅 {now().strftime('%d.%m.%Y %H:%M')} ({TZ})",
        cal_line,
        mode_text + f" | rekordów: {rows}",
        "",
        "🔎 RANKING 4 MECZÓW — EXACT DOUBLE INTELLIGENCE (BEZ KURSÓW)"
    ]
    for i, pred in enumerate(all_predictions, 1):
        best = pred["scores"][0] if pred.get("scores") else {}
        lines.append(f"{i}. {pred['home']} — {pred['away']} → najlepszy {best.get('score','?')} | siła meczu {pred.get('fixture_quality',0.0):.2%}")
    lines += ["", "🏆 2 NAJPEWNIEJSZE MECZE DO DOKŁADNEGO WYNIKU"]
    for i, pred in enumerate(selected, 1):
        lines.append(f"{i}. {pred['home']} — {pred['away']}")
        for j, sc in enumerate(pred["scores"], 1):
            lines.append(
                f"   {j}. {sc['score']} — 🧠 AI {float(sc.get('rank_probability',0.0)):.2%} "
                f"| bazowy {float(sc.get('baseline_probability',sc.get('prob',0.0))):.2%} "
                f"| AI LIFT ×{float(sc.get('ai_lift',1.0)):.2f} "
                f"| MC {float(sc.get('monte_carlo_probability',0.0)):.2%}"
            )
    lines += ["", "🎫 5 NAJMOCNIEJSZYCH PODWÓJNYCH AKO — DOKŁADNE WYNIKI"]
    new_keys = {(c.get("first"), c.get("second")) for c in (new_combinations or [])}
    for i, c in enumerate(combinations, 1):
        marker = "🆕 " if (c.get("first"), c.get("second")) in new_keys else ""
        lines.append(f"{i}. {marker}{c['first']} + {c['second']} → modelowo {c['probability']:.4%}")
    if errors:
        lines += ["", f"⚠️ Błędy analizy: {errors}"]
    lines += [
        "",
        "🧠 V17: BASELINE → ENSEMBLE → MONTE CARLO → xG (jeśli dostępne) → AI LIFT → TOP 3 → TOP 5 AKO.",
        "🚫 KURS BUKMACHERA: 0% WPŁYWU NA WYBÓR DOKŁADNEGO WYNIKU.",
    ]
    if update and new_signals:
        lines += ["", "🚨 NOWE / WZMOCNIONE SYGNAŁY EXACT SCORE"]
        for i, sig in enumerate(new_signals, 1):
            if sig.get("previous_probability") is None:
                lines.append(f"{i}. 🆕 {sig['home']} — {sig['away']} → {sig['score']} | AI {sig['probability']:.2%}")
            else:
                lines.append(f"{i}. 🔥 WZMOCNIONY {sig['home']} — {sig['away']} → {sig['score']} | AI {sig['probability']:.2%} (wcześniej {sig['previous_probability']:.2%})")
    elif update:
        lines += ["", "ℹ️ Brak nowego lub wyraźnie wzmocnionego sygnału exact-score w tym skanie."]
    if update:
        lines += ["", "🧠 Bot ponownie przeliczył wszystkie 4 mecze i porównał wynik z poprzednim skanem.", "⏱️ Kolejna kontrola za około 30 minut."]
    else:
        lines += ["🛡️ To są prognozy modelu, nie gwarancje trafienia.", "⏱️ Bot będzie automatycznie kontrolował wszystkie 4 mecze co 30 minut."]
    return "\n".join(lines)


def _exact_watch_calibration_load():
    if not EXACT_WATCH_CALIBRATION_FILE.exists():
        return {"resolved": 0, "hits": 0, "samples": []}
    try:
        d = json.loads(EXACT_WATCH_CALIBRATION_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {"resolved": 0, "hits": 0, "samples": []}
    except Exception:
        return {"resolved": 0, "hits": 0, "samples": []}


def _exact_watch_calibration_save(data):
    try:
        EXACT_WATCH_CALIBRATION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Nie udało się zapisać kalibracji exact watch: %s", exc)


def _resolve_exact_watch_history(history):
    """After matches finish, compare the last 3-ranked predictions with actual score."""
    state = _exact_watch_state_load()
    if not state or state.get("resolved"):
        return state
    samples = []
    resolved_count = 0
    hits = 0
    for item in state.get("predictions", []):
        home, away = item.get("home"), item.get("away")
        dt = pd.to_datetime(item.get("date"), utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        x = history[(history.home_key == norm_team(home)) & (history.away_key == norm_team(away))].copy()
        if x.empty:
            continue
        x["date2"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
        x = x[x.date2 >= dt - pd.Timedelta(days=1)]
        x = x[x.date2 <= dt + pd.Timedelta(days=3)]
        if x.empty:
            continue
        r = x.sort_values("date2").iloc[-1]
        actual = f"{int(r.hg)}:{int(r.ag)}"
        predicted = [sc.get("score") for sc in item.get("scores", [])]
        hit = actual in predicted
        resolved_count += 1
        hits += int(hit)
        samples.append({"home": home, "away": away, "actual": actual, "predicted_top3": predicted, "hit_top3": hit})
    if resolved_count:
        state["resolved"] = True
        state["resolved_count"] = resolved_count
        state["hit_top3"] = hits
        state["samples"] = samples
        _exact_watch_state_save(state)
        cal = _exact_watch_calibration_load()
        cal["resolved"] = int(cal.get("resolved", 0)) + resolved_count
        cal["hits"] = int(cal.get("hits", 0)) + hits
        cal.setdefault("samples", []).extend(samples)
        cal["samples"] = cal["samples"][-1000:]
        _exact_watch_calibration_save(cal)
    return state


def process_exact_watch(chat_id=None, update=False):
    """
    Analyze all fixed fixtures and ALWAYS rank the best available exact-double pair.
    There is no absolute probability gate: exact scores are intrinsically low-probability,
    so the correct decision is relative ranking. Bookmaker odds are never used.
    """
    history = load_history()

    # v25 FIX:
    # The previous version incorrectly required >=200 historical rows before
    # it would even attempt the four manual fixtures. Around season changes,
    # failed/free data downloads, or narrow league coverage this can be false
    # even though the model has valid statistical priors and can still rank
    # candidates. We now use adaptive history:
    #   >=30 rows  -> normal mode
    #   1..29      -> sparse-data mode with stronger shrinkage
    #   0 rows     -> prior-only mode (clearly labeled; never pretends to have
    #                 team-specific evidence)
    history_mode = "normal"
    history_rows = int(len(history))

    if history.empty:
        history = pd.DataFrame(columns=[
            "date", "home", "away", "hg", "ag", "league",
            "home_key", "away_key"
        ])
        history_mode = "prior-only"
    elif history_rows < 30:
        history_mode = "sparse-data"

    old_state = _exact_watch_state_load()
    _FAST_STATE["last_scan_time"] = now()
    previous_predictions = old_state.get("all_predictions", []) if update else []

    if old_state.get("enabled") and old_state.get("date") != str(now().date()):
        try:
            _resolve_exact_watch_history(history)
        except Exception:
            log.exception("Nie udało się rozliczyć poprzedniego exact watch")

    all_predictions, errors, skipped_past = _exact_watch_predictions(history)
    if len(all_predictions) < 2:
        fixture_names = ", ".join(
            f"{r.get('home','?')}–{r.get('away','?')}" for r in _manual_exact_watch_fixtures().to_dict("records")
        )
        if skipped_past and errors == 0:
            send(
                "⏭️ Nie tworzę nowych typów, ponieważ mniej niż 2 wskazane mecze są jeszcze przed rozpoczęciem.\n"
                f"⚽ Lista kontrolna: {fixture_names}\n"
                f"⏭️ Pominięto mecze już rozpoczęte/zakończone: {skipped_past}\n"
                "🛡️ Bot nie będzie przewidywał wyniku meczu, który już się rozpoczął.",
                chat_id
            )
            return False
        send(
            "❌ Nie udało się przygotować co najmniej dwóch prognoz dokładnego wyniku.\n"
            f"📊 Historia wejściowa: {history_rows} rekordów | tryb: {history_mode}\n"
            f"⚽ Sprawdzane mecze: {fixture_names}\n"
            f"⚠️ Błędy pojedynczych analiz: {errors}/{len(fixtures)}\n"
            f"⏭️ Pominięto zakończone/rozpoczęte mecze: {skipped_past}\n"
            "➡️ Bot nie przerwie już analizy tylko dlatego, że historia ma mniej niż 200 rekordów. "
            "Jeśli nadal będzie błąd, powyższe dane wskażą źródło problemu.",
            chat_id
        )
        return False

    # Select the two fixtures by RELATIVE exact-score quality, not by bookmaker odds
    # and not by an absolute threshold.
    def fixture_rank(p):
        scores = p.get("scores") or []
        if not scores:
            return -1.0
        top = scores[0]
        prob = float(top.get("rank_probability", top.get("prob", 0.0)))
        conf = float(p.get("score_confidence", 0.0))
        stability = float(top.get("stability", 0.0))
        agreement = float(p.get("base_prediction", {}).get("score_agreement", 0.0))
        # Geometric blend prevents one metric from dominating the whole decision.
        vals = [max(1e-6, prob), max(1e-6, conf), max(1e-6, stability), max(1e-6, agreement)]
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    ranked = sorted(all_predictions, key=fixture_rank, reverse=True)
    selected = ranked[:2]

    # Rank all available score-pairs for the selected two fixtures. This is the
    # actual decision criterion. There is deliberately NO "minimum AKO strength".
    combinations = _exact_watch_combinations(selected)

    if not combinations:
        send("❌ Nie udało się zbudować żadnej kombinacji dokładnych wyników.", chat_id)
        return False

    best = combinations[0]
    pair_probability = float(best.get("probability", 0.0))

    # Human-readable relative confidence. This is NOT a claim that the real-world
    # chance equals this percentage; it only describes the model's internal ranking.
    alternatives = [float(c.get("score_quality", 0.0)) for c in combinations]
    best_q = max(alternatives) if alternatives else 0.0
    second_q = alternatives[1] if len(alternatives) > 1 else 0.0
    dominance = (best_q / second_q) if second_q > 0 else 999.0

    state = _exact_watch_state_load()
    previous = state.get("combinations", []) if update else []
    previous_keys = {(c.get("first"), c.get("second")) for c in previous}
    new_combinations = [
        c for c in combinations
        if (c.get("first"), c.get("second")) not in previous_keys
    ]
    new_signals = (
        _exact_watch_new_signals(previous_predictions, all_predictions, max_new=3)
        if update else []
    )

    # Never say "no signal" merely because an absolute threshold was missed.
    # If the model has candidates, report the strongest relative choice.
    message = _exact_watch_message(
        all_predictions,
        selected,
        combinations,
        update=update,
        errors=errors,
        new_combinations=new_combinations,
        new_signals=new_signals,
    )

    # Append a transparent decision block.
    decision = (
        "\n\n🧠 DECYZJA RELATYWNA MODELU\n"
        f"🎯 Najmocniejsza para: {best.get('first','?')} + {best.get('second','?')}\n"
        f"📊 Iloczyn wewnętrznych prawdopodobieństw: {pair_probability:.2%}\n"
        f"🧮 V17: baseline vs AI + ensemble/MC/xG jest pokazywane przy każdym wyniku.\n"
        f"📈 Przewaga #1 nad #2 w rankingu: "
        f"{dominance:.2f}×\n"
        "ℹ️ To ranking modelu, NIE gwarancja trafienia i NIE kurs bukmacherski."
    )
    send(message + decision, chat_id)

    state.update({
        "enabled": True,
        "resolved": False,
        "date": str(now().date()),
        "last_update": now().isoformat(),
        "combinations": combinations,
        "predictions": selected,
        "all_predictions": all_predictions,
    })
    _exact_watch_state_save(state)
    return True

def exact_watch_worker():
    """Background watcher: re-analyze the two exact-score fixtures every 30 minutes."""
    log.info("EXACT WATCH: co %s minut", EXACT_WATCH_INTERVAL_MINUTES)
    while True:
        try:
            state = _exact_watch_state_load()
            if not state.get("enabled") or state.get("date") != str(now().date()):
                time.sleep(30)
                continue
            current = now()
            last = state.get("last_update")
            due = True
            if last:
                try:
                    last_dt = datetime.fromisoformat(last)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=ZoneInfo(TZ))
                    due = (current - last_dt).total_seconds() >= EXACT_WATCH_INTERVAL_MINUTES * 60
                except Exception:
                    due = True
            if due:
                process_exact_watch(chat_id=TG_CHAT, update=True)
            time.sleep(30)
        except Exception as exc:
            log.exception("Błąd exact-score watch: %s", exc)
            time.sleep(30)


def _daily_test_message(markets, scanned, errors, akos=None):
    lines = [
        f"🧪 TEST MASTER OF AI — {now().strftime('%d.%m.%Y')}",
        "",
        f"🔎 Przeanalizowano dzisiejszych meczów: {scanned}",
        f"⚠️ Błędy analizy: {errors}",
        "",
        "🏆 NAJLEPSZE SYGNAŁY MODELU:",
        "⚠️ To są sygnały testowe — nie są sztucznie podbijane do progu AKO.",
        "",
    ]

    if not markets:
        lines.append("❌ Nie znaleziono żadnego meczu do testu.")
    else:
        for i, m in enumerate(markets[:TEST_TOP_MARKETS], 1):
            if m["probability"] >= MARKET_STRONG_PROB:
                tier = "🟢 MOCNY"
            elif m["probability"] >= MARKET_MIN_PROB:
                tier = "🟡 GRANICZNY"
            else:
                tier = "🔴 PONIŻEJ PROGU"
            lines.append(
                f"{i}. {tier} {m['home']} — {m['away']}\n"
                f"   🎯 {m['market']}\n"
                f"   📊 Prawdopodobieństwo modelu: {m['probability']:.1%}\n"
                f"   🧠 Pewność modelu: {m['confidence']:.1%}\n"
                f"   ⚽ Najlepszy dokładny wynik: {m['score']}"
            )

    if akos:
        lines += ["", "🎫 AKO PRODUKCYJNE ZNALEZIONE W TYM SKANIE:"]
        for ako in akos:
            lines.append(
                f"• AKO x{ako['size']} — modelowo {ako['combined_probability']:.2%}"
            )
    else:
        lines += [
            "",
            "🛡️ Brak AKO przy obecnych progach produkcyjnych.",
            "To NIE oznacza braku sygnałów — powyżej widzisz, co model wybrałby w trybie testowym.",
        ]

    return "\n".join(lines)


def _build_daily_akos(candidates):
    """Build several AKOs from distinct fixtures: x2, x3, x5 and optionally x7."""
    if not candidates:
        return []
    unique = []
    seen = set()
    for c in candidates:
        if c["fixture_key"] in seen:
            continue
        seen.add(c["fixture_key"])
        unique.append(c)

    strong = [x for x in unique if x["probability"] >= MARKET_STRONG_PROB]
    pool = strong if len(strong) >= 2 else [x for x in unique if x["probability"] >= MARKET_MIN_PROB]
    if len(pool) < 2:
        return []

    result = []
    for n in (2, 3, 5, MARKET_MAX_LEGS):
        if n < 2 or n > len(pool) or n > MARKET_MAX_LEGS:
            continue
        legs = pool[:n]
        combined = math.prod(float(x["probability"]) for x in legs)
        result.append({
            "size": n,
            "legs": legs,
            "combined_probability": combined,
        })
    return result


def _daily_ako_message(akos, scanned, rejected, hour):
    date_text = now().strftime("%d.%m.%Y")
    lines = [
        f"🎫 TYPY KOMBINOWANE — {date_text}",
        f"⏰ Automatyczna kontrola: {hour:02d}:00",
        "",
        f"🔎 Przeanalizowano dzisiejszych meczów: {scanned}",
        f"❌ Odrzucono słabe sygnały: {rejected}",
        "🏆 Priorytet: LM → LE → LKE → Serie A / La Liga / Bundesliga / Premier League / Ligue 1 / Ekstraklasa / Portugalia / Turcja / Chorwacja",
        "",
    ]
    for ako in akos:
        lines += [
            f"🎫 AKO x{ako['size']}",
            "➖➖➖➖",
        ]
        for leg in ako["legs"]:
            lines.append(
                f"⚽ {leg['home']} - {leg['away']} → {leg['market']} "
                f"(pewność modelu: {leg['probability']:.1%})"
            )
        lines += [
            f"🧠 NAJSŁABSZA NOGA: {min(x['probability'] for x in ako['legs']):.1%}",
            f"📊 Modelowe prawdopodobieństwo wszystkich nóg: {ako['combined_probability']:.2%}",
            "",
        ]
    lines += [
        "⚠️ Prawdopodobieństwa są wyliczeniami modelu, nie gwarancją trafienia.",
        "🚫 Kursów bukmacherskich nie używam do sztucznego podbijania pewności.",
    ]
    return "\n".join(lines)


def process_daily_combined_signal(chat_id=None):
    """Daily scan. In TEST_MODE, always exposes the strongest markets for validation."""
    history = load_history()
    if history.empty or len(history) < 200:
        send("🎫 TYPY KOMBINOWANE\n\n❌ Brak wystarczającej historii do analizy.", chat_id)
        return False

    upcoming_df = _manual_test_fixtures() if TEST_MODE else upcoming()
    candidates, rejected = _daily_market_candidates(history, upcoming_df)
    akos = _build_daily_akos(candidates)

    if TEST_MODE:
        test_markets, scanned, errors = _daily_market_candidates_test(history, upcoming_df)
        send(_daily_test_message(test_markets, scanned, errors, akos), chat_id)
        return bool(test_markets or akos)

    if not akos:
        send(
            f"🎫 TYPY KOMBINOWANE — {now().strftime('%d.%m.%Y')}\n\n"
            f"🔎 Przeanalizowano: {len(upcoming_df) if upcoming_df is not None else 0}\n"
            f"❌ Odrzucono: {rejected}\n\n"
            "🛡️ Nie znaleziono wystarczająco mocnych rynków do AKO. "
            "Bot nie dobiera słabych typów na siłę.",
            chat_id,
        )
        return False

    send(
        _daily_ako_message(akos, len(upcoming_df), rejected, now().hour),
        chat_id,
    )
    return True

def _load_auto_slot_state():
    if not AUTO_SLOT_STATE.exists():
        return {}
    try:
        data = json.loads(AUTO_SLOT_STATE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_auto_slot_state(slot):
    AUTO_SLOT_STATE.write_text(
        json.dumps({"last_slot": slot, "timestamp": now().isoformat()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _next_auto_run():
    current = now()
    for hour in AUTO_SEND_HOURS:
        candidate = current.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate > current:
            return candidate
    tomorrow = current + timedelta(days=1)
    return tomorrow.replace(
        hour=min(AUTO_SEND_HOURS), minute=0, second=0, microsecond=0
    )


def scheduled_worker():
    """Run automatically at exactly 09:00..20:00 Europe/Warsaw."""
    log.info("AUTO AKO: godziny %s, strefa %s", AUTO_SEND_HOURS, TZ)
    while True:
        try:
            if not AUTO_SCAN:
                time.sleep(30)
                continue

            current = now()
            slot = current.strftime("%Y-%m-%d-%H")
            if current.minute == 0 and current.hour in AUTO_SEND_HOURS:
                state = _load_auto_slot_state()
                if state.get("last_slot") != slot:
                    process_daily_combined_signal()
                    _save_auto_slot_state(slot)
                time.sleep(61)
                continue

            nxt = _next_auto_run()
            time.sleep(max(5, min(60, (nxt - now()).total_seconds())))
        except Exception as exc:
            log.exception("Błąd automatycznego AKO: %s", exc)
            time.sleep(30)


def process_hourly_signal(chat_id=None, force_first=False):
    """Hard hourly controller: always reports status, while keeping one active 2-match AKO."""
    history = load_history()

    if history.empty or len(history) < 200:
        send(
            _hourly_status_message(
                0, 0, extra="❌ Brak wystarczającej historii. Nie tworzę sztucznego sygnału."
            ),
            chat_id,
        )
        return False

    upcoming_df = upcoming()
    candidates, rejected = (
        _candidate_matches(history, upcoming_df, with_diagnostics=True)
        if not upcoming_df.empty else ([], [])
    )
    state = _load_pair_state()

    if state is None or _pair_is_expired(state):
        pair = _choose_initial_pair(candidates)
        if pair and len(pair) == 2:
            # First activation is one signal update for the day.
            if _daily_signal_allowed():
                _save_pair_state(pair, reason="first_uefa_priority_pair")
                save_last(pair, coupon_score(pair))
                save_predictions(pair, coupon_score(pair))
                _count_daily_signal_update("initial_pair", pair)
                send(prediction_message(pair, coupon_score(pair)), chat_id)
                send(
                    _hourly_status_message(
                        len(candidates) + len(rejected),
                        len(rejected),
                        current=pair,
                        extra="🟢 Utworzono aktywny duet 2 MECZE × 1 DOKŁADNY WYNIK. Maksymalnie 5 zmian sygnału dziennie."
                    ),
                    chat_id,
                )
                return True

            send(
                _hourly_status_message(
                    len(candidates) + len(rejected),
                    len(rejected),
                    extra="🛑 Dzisiejszy limit 5 nowych sygnałów został wykorzystany. Aktywny duet nie jest losowo zmieniany."
                ),
                chat_id,
            )
            return False

        send(
            _hourly_status_message(
                len(candidates) + len(rejected),
                len(rejected),
                extra="🔒 Nie znaleziono 2 wyników spełniających rygorystyczne warunki."
            ),
            chat_id,
        )
        return False

    current = state["matches"]
    fresh, improved, replacement = _improved_pair(current, candidates)

    if improved and _daily_signal_allowed():
        _save_pair_state(fresh, reason="stronger_uefa_pair")
        save_last(fresh, coupon_score(fresh))
        save_predictions(fresh, coupon_score(fresh))
        _count_daily_signal_update("stronger_pair", fresh)

        send(
            _pair_summary_message(
                fresh,
                "🚨 NOWY MOCNIEJSZY SYGNAŁ — AKTYWNY DUET ZAKTUALIZOWANY"
            ),
            chat_id,
        )
        send(
            _hourly_status_message(
                len(candidates) + len(rejected),
                len(rejected),
                current=fresh,
                extra="🟢 Zmiana zaakceptowana dopiero po spełnieniu rygoru jakości i priorytetu UEFA."
            ),
            chat_id,
        )
        return True

    if improved and not _daily_signal_allowed():
        reason = "🛑 Limit 5 nowych sygnałów na dziś został wykorzystany. Nie zmieniam aktywnego duetu."
    else:
        reason = (
            "🔒 NIE ZNALEZIONO LEPSZEGO TYPU. Aktywny duet pozostaje bez zmian."
        )

    send(
        _hourly_status_message(
            len(candidates) + len(rejected),
            len(rejected),
            current=current,
            extra=reason,
        ),
        chat_id,
    )
    return False


def diagnostics_message():
    """Explain why recent candidates were rejected without presenting them as tips."""
    if not REJECTION_LOG.exists():
        return "🔎 Brak zapisanej diagnostyki odrzuconych sygnałów."
    try:
        df = pd.read_csv(REJECTION_LOG).tail(200)
        if df.empty:
            return "🔎 Brak odrzuconych sygnałów w diagnostyce."
        reasons = {}
        for raw in df.get("reasons", pd.Series(dtype=str)).fillna(""):
            for reason in str(raw).split(" | "):
                if reason:
                    key = reason.split(" (")[0]
                    reasons[key] = reasons.get(key, 0) + 1
        top = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:DIAGNOSTIC_LIMIT]
        lines = [
            "🔎 DIAGNOSTYKA ODRZUCANYCH SYGNAŁÓW",
            "",
            f"Ostatnio odrzuconych analiz: {len(df)}",
            "",
        ]
        for reason, count in top:
            lines.append(f"❌ {count}× {reason}")
        lines += [
            "",
            "🧪 NAJBLIŻSZE ODRZUCONE MECZE (TYLKO DIAGNOSTYKA):",
        ]
        recent = df.sort_values("confidence", ascending=False).drop_duplicates(
            subset=["home", "away"]
        ).head(3)
        for _, row in recent.iterrows():
            reason_text = str(row.get("reasons", "")).split(" | ")[0]
            lines.append(
                f"• {row.get('home', '')} — {row.get('away', '')}: {reason_text}"
            )
        lines += [
            "",
            "⚠️ Odrzucone mecze NIE są typami i nie trafiają do AKO.",
            "🎯 Celem jest jakość, nie liczba wysyłanych sygnałów.",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"🔎 Diagnostyka niedostępna: {exc}"


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
        "🤖 MASTER OF AI v10 — EXACT DOUBLE INTELLIGENCE",
        "",
        "🎟️ AKTYWNY AKO: 2 MECZE × 1 DOKŁADNY WYNIK",
        f"📡 SIŁA CAŁEGO AKO: {pair_strength:.1%}",
        signal_level(pair_strength),
        "🏆 Priorytet: LM → LE → LKE",
        "",
    ]

    for i, x in enumerate(chosen, 1):
        patterns = x.get("historical_patterns", {})
        lines += [
            f"⚽ MECZ {i}: {x['home']} — {x['away']}",
            f"🏆 Rozgrywki: {x.get('league', '')}",
            f"🎯 DOKŁADNY WYNIK: {x.get('score', 'brak')}",
            f"📈 Prawdopodobieństwo modelu: {float(x.get('prob', 0.0)):.1%}",
            f"🧠 Siła sygnału: {float(x.get('score_confidence', 0.0)):.1%}",
            f"🤝 Zgodność modeli: {float(x.get('score_agreement', 0.0)):.0%}",
            f"📊 Forma: {float(x.get('form_home', 0)):.0%} vs {float(x.get('form_away', 0)):.0%}",
            f"📚 Historia: {x.get('matches_home', 0)} vs {x.get('matches_away', 0)} meczów",
            f"🔁 Powtarzalność wzorców: {float(x.get('pattern_quality', 0)):.0%}",
            f"🧩 Powtórzenie wyniku H2H: {patterns.get('h2h_repeated_score', 0)}× / {patterns.get('h2h_sample', 0)}",
            f"♟️ Elo: {x.get('elo_home', 1500):.0f} vs {x.get('elo_away', 1500):.0f}",
            f"⚽ Oczekiwane gole: {x.get('lh', 0):.2f} vs {x.get('la', 0):.2f}",
            "",
        ]

    lines += [
        "🔒 AI skupia się wyłącznie na tych 2 meczach.",
        "⏰ Co godzinę wykonuje pełną kontrolę.",
        f"📨 Maksymalnie {MAX_DAILY_SIGNAL_UPDATES} nowych aktualizacji sygnału dziennie.",
        "⚠️ Dokładny wynik pozostaje prognozą, a nie gwarancją.",
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
        f"🎯 Tryb: EXACT DOUBLE INTELLIGENCE\n"
        f"🎟️ Aktywny duet: 2 mecze × 1 dokładny wynik + 5 niezależnych wariantów\n"
        f"📊 Min. exact-score: {EXACT_SCORE_MIN_PROB:.0%}\n"
        f"🟢 Mocny exact-score od: {STRONG_SIGNAL_P:.0%}\n"
        f"🔥 Bardzo mocny exact-score od: {VERY_STRONG_SIGNAL_P:.0%}\n"
        f"🤯 Niesamowicie mocny exact-score od: {INCREDIBLE_SIGNAL_P:.0%}\n"
        f"⏰ Kontrola: co 30 minut — bez wyjątków\n"
        f"🏆 UEFA: LM → LE → LKE | ścisły priorytet\n"
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
                "/typy — dzisiejsze typy kombinowane AKO\n"
                "/wyniki — dzisiejsze mecze\n"
                "/statystyki — skuteczność zapisanych typów\n"
                "/status — status systemu\n"
                "/kontrola — natychmiastowa kontrola lepszego sygnału\n"
                "/diagnostyka — dlaczego sygnały są odrzucane\n"
                "/status — stan trybu UEFA i limitu dziennego\n"
                "/help — pomoc",
                chat_id,
            )

        elif cmd in ("/help", "/pomoc"):
            send(
                "📌 KOMENDY\n\n"
                "/typy — skanuje dzisiejsze mecze i buduje AKO\n"
                "/wyniki — pokazuje dostępne wyniki\n"
                "/statystyki — rozlicza zapisane typy\n"
                "/status — status źródeł i API\n"
                "/kontrola — sprawdza od razu, czy jest lepszy typ\n"
                "/diagnostyka — pokazuje najczęstsze powody odrzucenia",
                chat_id,
            )

        elif cmd == "/status":
            send(status_message(), chat_id)

        elif cmd == "/statystyki":
            send(evaluate_predictions(), chat_id)

        elif cmd == "/diagnostyka":
            send(diagnostics_message(), chat_id)

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
                "🧪 Master Of AI uruchamia TEST dokładnych wyników dla 2 wskazanych meczów...",
                chat_id,
            )
            process_exact_watch(chat_id=chat_id, update=False)

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
    if HISTORY_SEASONS < 1:
        raise RuntimeError("HISTORY_SEASONS musi być >= 1.")
    if MAX_DAILY_SIGNAL_UPDATES < 1:
        raise RuntimeError("MAX_DAILY_SIGNAL_UPDATES musi być >= 1.")
    if not AUTO_SEND_HOURS or any(h < 0 or h > 23 for h in AUTO_SEND_HOURS):
        raise RuntimeError("AUTO_SEND_HOURS musi zawierać godziny 0..23.")
    if MARKET_MIN_PROB <= 0 or MARKET_MIN_PROB > 1:
        raise RuntimeError("MARKET_MIN_PROB musi być w zakresie (0,1].")
    if MARKET_STRONG_PROB < MARKET_MIN_PROB or MARKET_STRONG_PROB > 1:
        raise RuntimeError("MARKET_STRONG_PROB musi być >= MARKET_MIN_PROB i <= 1.")


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
        "service": "Master Of AI v7 UEFA FOCUS",
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
    watcher = threading.Thread(target=exact_watch_worker, name="exact-watch", daemon=True)
    watcher.start()
    scheduled_worker()


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
        "Master Of AI v15 uruchomiony na porcie %s.",
        port,
    )

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()


def handle_check_command(send, chat_id):
    """Fast /check handler for projects using a custom dispatcher."""
    _fast_reply(send, chat_id, "⏱️ /check — sprawdzam ostatni stan…")
    try:
        payload = {
            "tracked_fixtures": _manual_exact_watch_fixtures().to_dict("records")
                if "_manual_exact_watch_fixtures" in globals() else [],
            "last_scan": _FAST_STATE.get("last_scan_time"),
        }
        changed, _ = _check_changes(payload)
        _fast_reply(
            send, chat_id,
            "🔄 ZMIANA WYKRYTA — stan danych/sygnałów różni się od poprzedniego sprawdzenia."
            if changed else
            "🟢 BRAK ZMIAN — nic nowego od poprzedniego /check."
        )
    except Exception as e:
        _fast_reply(send, chat_id, f"⚠️ /check: nie udało się odczytać stanu ({type(e).__name__}).")
