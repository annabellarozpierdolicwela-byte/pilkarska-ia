import os
import time
import json
import math
import logging
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

# Strength of the main 1X2 signal. Exact-score probability is reported
# separately because an exact score is naturally much less probable.
STRONG_SIGNAL_P = float(os.getenv("STRONG_SIGNAL_PROB", "0.70"))
VERY_STRONG_SIGNAL_P = float(os.getenv("VERY_STRONG_SIGNAL_PROB", "0.80"))

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

HISTORY_CACHE_HOURS = int(os.getenv("HISTORY_CACHE_HOURS", "12"))
FIXTURE_CACHE_MINUTES = int(os.getenv("FIXTURE_CACHE_MINUTES", "60"))

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
        raise RuntimeError("Brak API_FOOTBALL_KEY")

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

    # Keep upcoming games only.
    start = pd.Timestamp.now(tz="UTC")
    df = df[df["date"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT") >= start] \
        if df["date"].dt.tz is None else df[df["date"] >= start]

    # The conversion above can produce timezone-naive timestamps.
    # Normalize robustly.
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = dates
    df = df.dropna(subset=["date"])

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
                return cached.dropna(subset=["date"]).sort_values("date")
            except Exception:
                pass

    df = parse_fd_upcoming()

    # Use API only if the free CSV source has no upcoming matches.
    if df.empty or len(df) < 3:
        api_df = api_fallback_fixtures()
        if not api_df.empty:
            df = pd.concat([df, api_df], ignore_index=True)

    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=["date", "home", "away"],
        keep="first",
    ).sort_values("date")

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
        b = pd.Timestamp(before)
        if b.tzinfo is None:
            b = b.tz_localize("UTC")
        x = x[x.date < b]

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
        b = pd.Timestamp(before)
        if b.tzinfo is None:
            b = b.tz_localize("UTC")
        x = x[x.date < b]

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
        b = pd.Timestamp(before)
        if b.tzinfo is None:
            b = b.tz_localize("UTC")
        x = x[x.date < b]

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
        b = pd.Timestamp(before)
        if b.tzinfo is None:
            b = b.tz_localize("UTC")
        x = x[x.date < b]

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


def predict(df, home, away, league=None, match_date=None):
    """
    Main prediction:
      - recency weighted goals
      - venue-specific strength
      - Elo
      - form
      - H2H with very small weight
      - league scoring environment
      - conservative probability shrinkage

    No future data is used for the historical calculation.
    """
    hs = team_strength(df, home, before=match_date)
    aws = team_strength(df, away, before=match_date)
    elo = elo_ratings(df, before=match_date)

    eh = elo.get(norm_team(home), 1500)
    ea = elo.get(norm_team(away), 1500)

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

    # Pull extreme estimates slightly toward league average.
    lh = 0.78 * lh + 0.22 * lg_h
    la = 0.78 * la + 0.22 * lg_a

    # Elo effect.
    elo_diff = (eh + 55 - ea) / 400
    form_diff = hs["form"] - aws["form"]

    lh *= math.exp(0.16 * elo_diff + 0.07 * form_diff)
    la *= math.exp(-0.12 * elo_diff - 0.05 * form_diff)

    # H2H: tiny influence.
    h2h_h, h2h_a = h2h_adjustment(
        df, home, away, before=match_date
    )
    lh *= h2h_h
    la *= h2h_a

    # Reasonable bounds.
    lh = max(0.20, min(4.20, lh))
    la = max(0.15, min(3.80, la))

    dist = score_matrix(lh, la, 7)

    # We shrink raw Poisson probability slightly.
    # This prevents the bot from presenting an overconfident exact score.
    raw_p, best_h, best_a = dist[0]
    best_p = min(raw_p, 0.35)

    top_scores = []
    for p, h, a in dist[:8]:
        top_scores.append({
            "score": f"{h}:{a}",
            "prob": p,
        })

    # 1X2 probabilities from the matrix.
    p_home = sum(p for p, h, a in dist if h > a)
    p_draw = sum(p for p, h, a in dist if h == a)
    p_away = sum(p for p, h, a in dist if h < a)

    return {
        "home": home,
        "away": away,
        "score": f"{best_h}:{best_a}",
        "prob": best_p,
        "raw_prob": raw_p,
        "lh": lh,
        "la": la,
        "elo_home": eh,
        "elo_away": ea,
        "form_home": hs["form"],
        "form_away": aws["form"],
        "matches_home": hs["matches"],
        "matches_away": aws["matches"],
        "p1x2": {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        },
        "top_scores": top_scores,
    }


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


def label(p):
    if p >= VERY_STRONG_SIGNAL_P:
        return "🔥 BARDZO MOCNY SYGNAŁ AI"
    if p >= STRONG_SIGNAL_P:
        return "🟢 MOCNY SYGNAŁ AI"
    if p >= 0.60:
        return "🟡 DOBRY SYGNAŁ AI"
    return "⚪ SŁABSZY SYGNAŁ AI"


def coupon_score(items):
    if not items:
        return 0.0
    # Summary of main 1X2 signal strength, not a guaranteed coupon hit rate.
    return sum(x["signal_prob"] for x in items) / len(items)


def make_predictions():
    history = load_history()

    if history.empty or len(history) < 200:
        raise RuntimeError(
            f"Za mało historii z darmowych źródeł: {len(history)} meczów."
        )

    upcoming_df = upcoming()

    if upcoming_df.empty:
        return [], None

    cand = []

    for _, r in upcoming_df.iterrows():
        # Ignore games that already have a final result.
        if str(r.get("status", "")) in ("FT", "AET", "PEN", "CANC"):
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

        # Require a minimum amount of recent information.
        if p["matches_home"] < 3 or p["matches_away"] < 3:
            continue

        signal_code, signal_name, signal_prob = main_signal(p["p1x2"])

        p.update({
            "date": str(match_date),
            "league": league,
            "signal_code": signal_code,
            "signal": signal_name,
            "signal_prob": signal_prob,
            "label": label(signal_prob),
            "source": str(r.get("source", "football-data")),
        })

        # MINP remains a floor for the exact-score candidate, while ranking
        # is driven primarily by the much more meaningful 1X2 signal.
        if p["prob"] >= MINP:
            cand.append(p)

    cand.sort(key=lambda x: (x["signal_prob"], x["prob"]), reverse=True)

    chosen = cand[:MAXM]
    return chosen, coupon_score(chosen) if chosen else None


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
    lines = [
        "🤖 MASTER OF AI — ANALIZA MECZÓW",
        "",
        "📊 Model: forma + gole + dom/wyjazd + Elo + H2H + Poisson",
        "🗂️ Dane: darmowa baza Football-Data + lokalny cache",
        "",
    ]

    very_strong = [x for x in chosen if x["signal_prob"] >= VERY_STRONG_SIGNAL_P]
    if very_strong:
        lines += [
            "🔥 BARDZO MOCNY SYGNAŁ AI",
            "AI znalazło typ z bardzo wysoką szansą modelową.",
            "",
        ]
    else:
        best = chosen[0]
        lines += [
            "🟡 BRAK BARDZO MOCNEGO SYGNAŁU",
            f"Najlepszy dostępny typ: {best['signal']} — {best['signal_prob']:.1%}",
            "",
        ]

    for i, x in enumerate(chosen, 1):
        lines += [
            f"{i}. {x['home']} — {x['away']}",
            f"   {x['label']}",
            f"   🎯 TYP GŁÓWNY: {x['signal']} ({x['signal_prob']:.1%})",
            f"   ⚽ DOKŁADNY WYNIK: {x['score']} ({x['prob']:.1%})",
            f"   1X2: {x['p1x2']['home']:.0%} / {x['p1x2']['draw']:.0%} / {x['p1x2']['away']:.0%}",
        ]
        if x.get("top_scores"):
            top = " | ".join(f"{z['score']} {z['prob']:.1%}" for z in x["top_scores"][:3])
            lines.append(f"   🔎 TOP 3 DOKŁADNE WYNIKI: {top}")
        lines.append("")

    lines += [
        f"🧠 Średnia siła głównych typów: {score:.2%}",
        "",
        "ℹ️ Szansa typu głównego i szansa dokładnego wyniku są liczone osobno.",
        "⚠️ To predykcja statystyczna — nie gwarancja wyniku.",
    ]
    return "\n".join(lines)


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
        f"🎯 Maks. typów: {MAXM}\n"
        f"📊 Min. dokładnego wyniku: {MINP:.0%}\n"
        f"🟢 Mocny sygnał od: {STRONG_SIGNAL_P:.0%}\n"
        f"🔥 Bardzo mocny sygnał od: {VERY_STRONG_SIGNAL_P:.0%}\n"
        f"🛡️ API-Football dzienny budżet: "
        f"{usage['used']}/{API_DAILY_BUDGET}\n"
        f"⏱️ API odstęp: {API_MIN_INTERVAL:.1f}s\n"
        f"📅 Sezon: {CURRENT_SEASON}/{CURRENT_SEASON+1}\n"
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
                "/help — pomoc",
                chat_id,
            )

        elif cmd in ("/help", "/pomoc"):
            send(
                "📌 KOMENDY\n\n"
                "/typy — uruchamia analizę\n"
                "/wyniki — pokazuje dostępne wyniki\n"
                "/statystyki — rozlicza zapisane typy\n"
                "/status — status źródeł i API",
                chat_id,
            )

        elif cmd == "/status":
            send(status_message(), chat_id)

        elif cmd == "/statystyki":
            send(evaluate_predictions(), chat_id)

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
                "⏳ Master Of AI analizuje dane z lokalnej bazy...\n"
                "API-Football jest używane tylko jako awaryjne źródło.",
                chat_id,
            )

            chosen, score = make_predictions()

            if not chosen:
                send(
                    "🟡 Nie znaleziono meczu spełniającego minimalne "
                    "warunki danych modelu. Nie wymuszam losowego typu.",
                    chat_id,
                )
                return

            save_predictions(chosen, score)
            save_last(chosen, score)

            send(
                prediction_message(chosen, score),
                chat_id,
            )

        else:
            send("Nie znam tej komendy. Wpisz /help.", chat_id)

    except Exception as exc:
        log.exception("Błąd komendy")
        send(
            f"❌ Błąd: {exc}",
            chat_id,
        )


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
        "service": "Master Of AI v5",
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
    if not AUTO_SCAN:
        log.info("AUTO_SCAN=0 — automatyczny skaner wyłączony.")
        return

    log.info(
        "Automatyczny skaner uruchomiony co %s min.",
        SCAN,
    )

    while True:
        try:
            chosen, score = make_predictions()

            if chosen:
                save_predictions(chosen, score)
                save_last(chosen, score)
                send(prediction_message(chosen, score))

        except Exception as exc:
            log.exception("Błąd automatycznego skanera: %s", exc)

        time.sleep(max(30, SCAN * 60))


def main():
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
        "Master Of AI v5 uruchomiony na porcie %s.",
        port,
    )

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
