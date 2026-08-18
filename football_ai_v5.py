import os, time, json, math, threading, logging, random
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify, request
from dotenv import load_dotenv

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
LAST = DATA / "last_coupon.json"
HISTORY_LOG = DATA / "predictions.csv"
LOCAL_HISTORY = DATA / "history_openfootball.csv"

load_dotenv(ROOT / ".env")

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TZ = os.getenv("TIMEZONE", "Europe/Warsaw")

# API-Football is now only an OPTIONAL fallback. The bot can work without it.
API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
AF_BASE = "https://v3.football.api-sports.io"

# TheSportsDB free V1 is used for current/upcoming fixtures and today's results.
# Their documentation states that the free key is 123 and supports next league
# and day schedules. It is intentionally configurable in case the service changes.
TSDB_KEY = os.getenv("THESPORTSDB_API_KEY", "123").strip() or "123"
TSDB_BASE = "https://www.thesportsdb.com/api/v1/json"

# Main competitions. The IDs are TheSportsDB league IDs.
TSDB_LEAGUES = {
    "English Premier League": "4328",
    "Bundesliga": "4331",
    "Serie A": "4332",
    "La Liga": "4335",
    "Ligue 1": "4334",
}

# OpenFootball gives us public historical data for the model, without a key.
# 2025-26 is the last completed/mostly completed season available in the
# public football.json datasets used as the model's baseline.
OPENFOOTBALL_HISTORY = {
    "Premier League": "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/en.1.json",
    "Bundesliga": "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/de.1.json",
    "Serie A": "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/it.1.json",
    "La Liga": "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/es.1.json",
    "Ligue 1": "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/fr.1.json",
}

# Optional API-Football fallback for current fixtures only.
API_FOOTBALL_LEAGUES = [39, 78, 135, 140, 61]

MAXM = max(1, int(os.getenv("MAX_MATCHES", "2")))
MIN_SIGNAL_PROB = float(os.getenv("MIN_SIGNAL_PROB", "0.57"))
MIN_SIGNAL_EDGE = float(os.getenv("MIN_SIGNAL_EDGE", "0.10"))
MIN_HISTORY = max(20, int(os.getenv("MIN_HISTORY", "50")))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
CACHE_MINUTES = max(1, int(os.getenv("CACHE_MINUTES", "15")))
SCAN = int(os.getenv("SCAN_MINUTES", "60"))
AUTO_SCAN = os.getenv("AUTO_SCAN", "0").strip().lower() in ("1", "true", "yes", "on")

log = logging.getLogger("pilkarska_ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

_http_lock = threading.Lock()
_last_http = 0.0


def now():
    return datetime.now(ZoneInfo(TZ))


def safe_get(url, params=None, headers=None, timeout=None):
    """Never lets a network error crash a command."""
    global _last_http
    timeout = timeout or REQUEST_TIMEOUT
    try:
        with _http_lock:
            gap = 0.35 - (time.monotonic() - _last_http)
            if gap > 0:
                time.sleep(gap)
            _last_http = time.monotonic()
        r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
        if r.status_code == 429:
            log.warning("429 from %s", url)
            return None
        if not r.ok:
            log.warning("HTTP %s from %s", r.status_code, url)
            return None
        return r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("Network/data error for %s: %s", url, e)
        return None


def tsdb(endpoint, params=None):
    url = f"{TSDB_BASE}/{TSDB_KEY}/{endpoint}"
    return safe_get(url, params=params)


def parse_tsdb_event(e, league_name):
    try:
        home = (e.get("strHomeTeam") or "").strip()
        away = (e.get("strAwayTeam") or "").strip()
        if not home or not away:
            return None
        date = e.get("dateEvent") or ""
        time_part = e.get("strTime") or "00:00:00"
        if len(time_part) == 5:
            time_part += ":00"
        dt = pd.to_datetime(f"{date} {time_part}", errors="coerce", utc=False)
        if pd.isna(dt):
            dt = pd.to_datetime(date, errors="coerce")
        status = (e.get("strStatus") or "").strip().lower()
        hg = e.get("intHomeScore")
        ag = e.get("intAwayScore")
        try:
            hg = float(hg) if hg not in (None, "", "null") else None
            ag = float(ag) if ag not in (None, "", "null") else None
        except Exception:
            hg, ag = None, None
        return {
            "id": str(e.get("idEvent") or f"{date}-{home}-{away}"),
            "date": dt,
            "status": status,
            "league": league_name,
            "home": home,
            "away": away,
            "hg": hg,
            "ag": ag,
        }
    except Exception:
        return None


def tsdb_upcoming():
    """Current/upcoming matches. No API-Football key is required."""
    cache = DATA / "upcoming_tsdb.csv"
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_MINUTES * 60:
        try:
            df = pd.read_csv(cache)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.dropna(subset=["date"])
        except Exception:
            pass

    out = []
    for name, lid in TSDB_LEAGUES.items():
        j = tsdb("eventsnextleague.php", {"id": lid})
        events = (j or {}).get("events") or []
        for e in events:
            r = parse_tsdb_event(e, name)
            if r:
                out.append(r)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.drop_duplicates("id").sort_values("date")
    try:
        df.to_csv(cache, index=False)
    except Exception:
        pass
    return df


def tsdb_today():
    cache = DATA / "today_tsdb.csv"
    if cache.exists() and time.time() - cache.stat().st_mtime < 5 * 60:
        try:
            df = pd.read_csv(cache)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.dropna(subset=["date"])
        except Exception:
            pass

    out = []
    d = now().strftime("%Y-%m-%d")
    for name, lid in TSDB_LEAGUES.items():
        j = tsdb("eventsday.php", {"d": d, "l": lid})
        events = (j or {}).get("events") or []
        for e in events:
            r = parse_tsdb_event(e, name)
            if r:
                out.append(r)
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.drop_duplicates("id").sort_values("date")
        try:
            df.to_csv(cache, index=False)
        except Exception:
            pass
    return df


def api_football(path, params):
    if not API_KEY:
        return []
    j = safe_get(
        AF_BASE + path,
        params=params,
        headers={"x-apisports-key": API_KEY},
        timeout=REQUEST_TIMEOUT,
    )
    if not j:
        return []
    if j.get("errors"):
        log.warning("API-Football error: %s", j.get("errors"))
        return []
    return j.get("response") or []


def af_row(f):
    try:
        return {
            "id": str(f["fixture"]["id"]),
            "date": pd.to_datetime(f["fixture"]["date"], utc=True, errors="coerce"),
            "status": f["fixture"]["status"]["short"],
            "league": f["league"]["name"],
            "home": f["teams"]["home"]["name"],
            "away": f["teams"]["away"]["name"],
            "hg": f["goals"].get("home"),
            "ag": f["goals"].get("away"),
        }
    except Exception:
        return None


def fallback_upcoming():
    """Optional API-Football fallback. Failure here is harmless."""
    out = []
    d1 = now().date()
    d2 = d1 + timedelta(days=7)
    for lid in API_FOOTBALL_LEAGUES:
        data = api_football("/fixtures", {
            "league": lid,
            "from": d1.isoformat(),
            "to": d2.isoformat(),
            "timezone": TZ,
        })
        for f in data:
            r = af_row(f)
            if r:
                out.append(r)
    return pd.DataFrame(out).drop_duplicates("id") if out else pd.DataFrame()


def upcoming():
    try:
        df = tsdb_upcoming()
        if not df.empty:
            return df
    except Exception:
        log.exception("TheSportsDB upcoming failed")
    try:
        df = fallback_upcoming()
        if not df.empty:
            return df
    except Exception:
        log.exception("API-Football fallback failed")
    return pd.DataFrame()


def load_openfootball(url, league):
    j = safe_get(url, timeout=REQUEST_TIMEOUT)
    if not isinstance(j, dict):
        return []
    out = []
    for m in j.get("matches") or []:
        try:
            score = m.get("score") or {}
            ft = score.get("ft") if isinstance(score, dict) else score
            if not isinstance(ft, list) or len(ft) < 2:
                continue
            out.append({
                "id": f"{league}-{m.get('date')}-{m.get('team1')}-{m.get('team2')}",
                "date": pd.to_datetime(m.get("date"), errors="coerce"),
                "status": "FT",
                "league": league,
                "home": str(m.get("team1") or "").strip(),
                "away": str(m.get("team2") or "").strip(),
                "hg": float(ft[0]),
                "ag": float(ft[1]),
            })
        except Exception:
            continue
    return out


def get_model_history():
    if LOCAL_HISTORY.exists():
        try:
            age = time.time() - LOCAL_HISTORY.stat().st_mtime
            if age < 24 * 3600:
                df = pd.read_csv(LOCAL_HISTORY)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date", "hg", "ag"])
                if len(df) >= MIN_HISTORY:
                    return df.sort_values("date")
        except Exception:
            pass

    out = []
    for league, url in OPENFOOTBALL_HISTORY.items():
        out.extend(load_openfootball(url, league))
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.dropna(subset=["date", "hg", "ag"]).drop_duplicates("id").sort_values("date")
        try:
            df.to_csv(LOCAL_HISTORY, index=False)
        except Exception:
            pass
    return df


def team_stats(df, team):
    if df.empty:
        return None
    h = df[df.home == team].copy()
    a = df[df.away == team].copy()
    if h.empty and a.empty:
        # tolerate small naming differences from data providers
        norm = team.lower().replace(" fc", "").replace(" afc", "").strip()
        maskh = df.home.astype(str).str.lower().str.replace(" fc", "", regex=False).str.replace(" afc", "", regex=False).str.strip() == norm
        maska = df.away.astype(str).str.lower().str.replace(" fc", "", regex=False).str.replace(" afc", "", regex=False).str.strip() == norm
        h, a = df[maskh].copy(), df[maska].copy()
    h["gf"], h["ga"] = h.hg, h.ag
    a["gf"], a["ga"] = a.ag, a.hg
    x = pd.concat([h, a]).sort_values("date").tail(10)
    if len(x) < 3:
        return None
    pts = sum(3 if r.gf > r.ga else 1 if r.gf == r.ga else 0 for _, r in x.iterrows())
    return float(x.gf.mean()), float(x.ga.mean()), pts / (3 * len(x)), len(x)


def elo(df):
    e = {}
    for _, r in df.sort_values("date").iterrows():
        h, a = r.home, r.away
        rh, ra = e.get(h, 1500), e.get(a, 1500)
        eh = 1 / (1 + 10 ** ((ra - rh - 55) / 400))
        actual = 1 if r.hg > r.ag else .5 if r.hg == r.ag else 0
        margin = max(1, math.log1p(abs(r.hg - r.ag)) * 1.6)
        e[h] = rh + 24 * margin * (actual - eh)
        e[a] = ra + 24 * margin * ((1 - actual) - (1 - eh))
    return e


def poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def outcome_probs(df, home, away):
    hs, aas = team_stats(df, home), team_stats(df, away)
    if not hs or not aas:
        return None

    hg, hga, hform, hn = hs
    ag, aga, aform, an = aas
    er = elo(df)
    eh, ea = er.get(home, 1500), er.get(away, 1500)

    lh = max(0.15, min(3.8, (hg + aga) / 2))
    la = max(0.15, min(3.8, (ag + hga) / 2))

    strength = (eh + 55 - ea) / 400
    form = hform - aform
    lh *= math.exp(0.18 * strength + 0.08 * form)
    la *= math.exp(-0.18 * strength - 0.08 * form)

    p1 = px = p2 = 0.0
    best_score, best_score_p = "0:0", 0.0
    for h in range(8):
        for a in range(8):
            p = poisson(h, lh) * poisson(a, la)
            if p > best_score_p:
                best_score_p = p
                best_score = f"{h}:{a}"
            if h > a:
                p1 += p
            elif h == a:
                px += p
            else:
                p2 += p

    probs = {"1": p1, "X": px, "2": p2}
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    best, bestp = ordered[0]
    secondp = ordered[1][1]

    # Confidence requires both a decent probability and a meaningful gap.
    signal = bestp >= MIN_SIGNAL_PROB and (bestp - secondp) >= MIN_SIGNAL_EDGE

    return {
        "home": home, "away": away,
        "pick": best, "prob": bestp, "second_prob": secondp,
        "edge": bestp - secondp,
        "p1": p1, "px": px, "p2": p2,
        "score": best_score, "score_prob": best_score_p,
        "history_games": min(hn, an), "strong": signal,
    }


def make_predictions():
    history = get_model_history()
    if len(history) < MIN_HISTORY:
        return [], {"reason": f"Za mało wiarygodnej historii ({len(history)}/{MIN_HISTORY})."}

    u = upcoming()
    if u.empty:
        return [], {"reason": "Nie udało się pobrać aktualnych meczów ze źródeł danych."}

    cand, weak = [], []
    for _, r in u.iterrows():
        if pd.isna(r.date) or r.date < pd.Timestamp.now():
            continue
        p = outcome_probs(history, str(r.home), str(r.away))
        if not p:
            continue
        p.update({
            "id": str(r.id), "date": str(r.date),
            "league": r.league,
        })
        if p["strong"]:
            cand.append(p)
        else:
            weak.append(p)

    cand.sort(key=lambda x: (x["prob"], x["edge"]), reverse=True)
    return cand[:MAXM], {"weak_checked": len(weak), "all_checked": len(cand) + len(weak)}


def pick_label(p):
    if p >= 0.67:
        return "🔥 BARDZO MOCNY SYGNAŁ"
    if p >= 0.61:
        return "🟢 MOCNY SYGNAŁ"
    return "🟡 UMIARKOWANY"


def friendly_no_signal(info):
    checked = (info or {}).get("all_checked", 0)
    reason = (info or {}).get("reason")
    if reason:
        return (
            "🟡 BRAK MOCNEGO SYGNAŁU NA TERAZ\n\n"
            f"{reason}\n\n"
            "Nie będę na siłę podawał typu. Lepiej poczekać na mecz, "
            "dla którego dane dają wyraźniejszą przewagę."
        )
    return (
        "🟡 BRAK MOCNEGO SYGNAŁU NA TERAZ\n\n"
        f"Sprawdziłem {checked} dostępnych meczów, ale żaden nie spełnił "
        f"progu bezpieczeństwa ({MIN_SIGNAL_PROB:.0%} prawdopodobieństwa "
        f"i min. {MIN_SIGNAL_EDGE:.0%} przewagi nad drugim wynikiem).\n\n"
        "Nie będę na siłę generował typu. Spróbuj ponownie później."
    )


def format_predictions(chosen):
    lines = ["🎯 PIŁKARSKA AI — MOCNE SYGNAŁY", ""]
    for i, x in enumerate(chosen, 1):
        pick_name = {"1": "Gospodarze (1)", "X": "Remis (X)", "2": "Goście (2)"}[x["pick"]]
        lines += [
            f"{i}. {x['home']} - {x['away']}",
            f"   🎯 Typ: {pick_name}",
            f"   {pick_label(x['prob'])} — {x['prob']:.1%}",
            f"   📐 Przewaga nad 2. opcją: {x['edge']:.1%}",
            f"   ⚽ Modelowany wynik: {x['score']}",
            f"   🏟️ Liga: {x['league']}",
            "",
        ]
    lines += ["⚠️ To analiza statystyczna, nie gwarancja wyniku."]
    return "\n".join(lines)


def tg(method, data=None):
    if not TG_TOKEN:
        raise RuntimeError("Brak TELEGRAM_BOT_TOKEN")
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
        data=data or {},
        timeout=25,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(str(j))
    return j


def send(msg, chat_id=None):
    chat_id = chat_id or TG_CHAT
    if not TG_TOKEN or not chat_id:
        print(msg, flush=True)
        return
    try:
        tg("sendMessage", {"chat_id": chat_id, "text": msg})
    except Exception:
        log.exception("Telegram send failed")


def results_message():
    try:
        df = tsdb_today()
        if df.empty:
            return "📅 Dzisiaj nie znaleziono meczów w obsługiwanych ligach."
        lines = [f"📅 MECZE DZISIAJ — {now().strftime('%d.%m.%Y')}", ""]
        for _, r in df.sort_values("date").iterrows():
            score = ""
            if pd.notna(r.hg) and pd.notna(r.ag):
                score = f"  ⚽ {int(r.hg)}:{int(r.ag)}"
            status = r.status or "planowany"
            lines.append(f"{r.home} - {r.away}{score} [{status}]")
        return "\n".join(lines[:101])
    except Exception:
        log.exception("results_message failed")
        return "🟡 Nie udało się teraz pobrać wyników. Spróbuj ponownie za chwilę."


def status_message():
    return (
        "🤖 PIŁKARSKA AI działa.\n\n"
        "🛡️ Główne dane: TheSportsDB (bez Twojego klucza API-Football)\n"
        f"🔁 Fallback: {'API-Football włączony' if API_KEY else 'lokalny/cache — API-Football nie ustawiony'}\n"
        f"🎯 Próg mocnego sygnału: {MIN_SIGNAL_PROB:.0%}\n"
        f"📐 Minimalna przewaga: {MIN_SIGNAL_EDGE:.0%}\n"
        f"📚 Minimum historii: {MIN_HISTORY} meczów\n"
        f"⏱ Auto-skaner: {'ON' if AUTO_SCAN else 'OFF'}\n"
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
                "🤖 Piłkarska AI działa.\n\n"
                "/typy — szuka tylko mocnych sygnałów\n"
                "/wyniki — dzisiejsze mecze i wyniki\n"
                "/status — stan systemu\n"
                "/help — pomoc",
                chat_id,
            )
        elif cmd in ("/help", "/pomoc"):
            send(
                "📌 KOMENDY\n"
                "/typy — analiza mocnych sygnałów\n"
                "/wyniki — mecze i wyniki dzisiaj\n"
                "/status — status bota",
                chat_id,
            )
        elif cmd == "/status":
            send(status_message(), chat_id)
        elif cmd == "/wyniki":
            send(results_message(), chat_id)
        elif cmd == "/typy":
            send("⏳ Analizuję mecze. Nie będę podawał typu na siłę...", chat_id)
            chosen, info = make_predictions()
            if not chosen:
                send(friendly_no_signal(info), chat_id)
            else:
                send(format_predictions(chosen), chat_id)
        else:
            send("Nie znam tej komendy. Wpisz /help.", chat_id)
    except Exception:
        log.exception("Błąd komendy")
        send(
            "🟡 Nie udało się teraz wykonać analizy.\n"
            "Dane źródłowe są chwilowo niedostępne albo nie ma wystarczającej historii. "
            "Spróbuj ponownie za chwilę.",
            chat_id,
        )


def configure_telegram_webhook():
    public_url = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
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
        log.exception("Webhook setup failed")
        return False


def command_loop():
    if os.getenv("RENDER_EXTERNAL_URL"):
        return
    if not TG_TOKEN:
        return
    offset = None
    while True:
        try:
            params = {"timeout": 25, "allowed_updates": json.dumps(["message"])}
            if offset is not None:
                params["offset"] = offset
            updates = tg("getUpdates", params).get("result", [])
            for u in updates:
                offset = u["update_id"] + 1
                handle_update(u)
        except Exception:
            log.exception("Telegram polling error")
            time.sleep(10)


app = Flask(__name__)

@app.post("/telegram/webhook")
def telegram_webhook():
    u = request.get_json(silent=True) or {}
    threading.Thread(target=handle_update, args=(u,), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "Pilkarska AI — TEST SAFE SOURCES"})


@app.get("/health")
def health_check():
    return jsonify({"status": "healthy"})


def worker():
    if not AUTO_SCAN:
        return
    while True:
        try:
            chosen, info = make_predictions()
            if chosen:
                send(format_predictions(chosen))
            else:
                log.info("Brak mocnego sygnału: %s", info)
        except Exception:
            log.exception("Automatyczny skaner nie wykonał analizy")
        time.sleep(max(1, SCAN) * 60)


def main():
    if not TG_TOKEN:
        log.warning("Brak TELEGRAM_BOT_TOKEN.")
    port = int(os.getenv("PORT", "10000"))
    configure_telegram_webhook()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=command_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=port, use_reloader=False)


if __name__ == "__main__":
    main()
