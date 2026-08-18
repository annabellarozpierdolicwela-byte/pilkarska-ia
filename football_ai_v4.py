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
HISTORY_CACHE = DATA / "history_2024.csv"
HISTORY_CACHE_HOURS = 24
HISTORY_SEASON = int(os.getenv("HISTORY_SEASON", "2024"))
# API-Football requires season whenever league is supplied to /fixtures.
# Domestic/European competitions use the season start year (e.g. 2026 for 2026/27).
CURRENT_SEASON = int(os.getenv("CURRENT_SEASON", "2026"))

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TZ = os.getenv("TIMEZONE", "Europe/Warsaw")
SCAN = int(os.getenv("SCAN_MINUTES", "60"))
AUTO_SCAN = os.getenv("AUTO_SCAN", "0").strip().lower() in ("1", "true", "yes", "on")
API_MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "7.0"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
_api_lock = threading.Lock()
_last_api_call = 0.0

MAXM = int(os.getenv("MAX_MATCHES", "2"))
MINP = float(os.getenv("MIN_SCORE_PROB", "0.08"))
MINIMP = float(os.getenv("MIN_COUPON_IMPROVEMENT", "0.02"))
LEAGUES = [int(x) for x in os.getenv("LEAGUE_IDS", "39,140,135,78,106,2").split(",") if x.strip().isdigit()]
BASE = "https://v3.football.api-sports.io"
log = logging.getLogger("pilkarska_ai")


def now():
    return datetime.now(ZoneInfo(TZ))


def api(path, params):
    """Bezpieczne API-Football: serializacja, odstęp i retry dla 429."""
    global _last_api_call
    if not API_KEY:
        raise RuntimeError("Brak API_FOOTBALL_KEY")
    for attempt in range(API_MAX_RETRIES + 1):
        with _api_lock:
            wait = API_MIN_INTERVAL - (time.monotonic() - _last_api_call)
            if wait > 0:
                time.sleep(wait)
            _last_api_call = time.monotonic()
            try:
                r = requests.get(
                    BASE + path,
                    headers={"x-apisports-key": API_KEY},
                    params=params,
                    timeout=30,
                )
            except requests.RequestException:
                if attempt >= API_MAX_RETRIES:
                    raise
                time.sleep(min(30, 5 * (2 ** attempt)))
                continue
        if r.status_code == 429:
            if attempt >= API_MAX_RETRIES:
                raise RuntimeError("API-Football: limit zapytań (429). Spróbuj ponownie za chwilę.")
            delay = 15 * (2 ** attempt) + random.uniform(0, 2)
            log.warning("API-Football 429 — czekam %.1f s przed ponowieniem.", delay)
            time.sleep(delay)
            continue
        r.raise_for_status()
        remaining = r.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None:
            log.info("API-Football: %s %s | pozostało dziś: %s", path, params, remaining)
        j = r.json()
        if j.get("errors"):
            raise RuntimeError(str(j["errors"]))
        return j.get("response", [])
    raise RuntimeError("Nie udało się pobrać danych z API-Football.")


def row(f):
    return {
        "id": f["fixture"]["id"],
        "date": f["fixture"]["date"],
        "status": f["fixture"]["status"]["short"],
        "league": f["league"]["name"],
        "lid": f["league"]["id"],
        "home": f["teams"]["home"]["name"],
        "away": f["teams"]["away"]["name"],
        "hg": f["goals"].get("home"),
        "ag": f["goals"].get("away"),
    }


# Testowa wersja dla planu Free API-Football.
# Historia modelu korzysta z sezonu HISTORY_SEASON.
# Aktualne mecze zawsze przekazują CURRENT_SEASON, ponieważ API-Football
# wymaga pola season przy zapytaniach /fixtures z parametrem league.

def history():
    out = []
    for lid in LEAGUES:
        data = api("/fixtures", {"league": lid, "season": HISTORY_SEASON, "status": "FT"})
        out += [row(x) for x in data]
    df = pd.DataFrame(out).drop_duplicates("id")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["hg", "ag"]).sort_values("date")


UPCOMING_CACHE = DATA / f"upcoming_{CURRENT_SEASON}.csv"
UPCOMING_CACHE_MINUTES = 30
_upcoming_lock = threading.Lock()
_history_lock = threading.Lock()

def _upcoming_uncached():
    """Pobiera aktualne i najbliższe mecze po konkretnym sezonie i dacie.

    API-Football na planie Free blokuje parametr `next`, dlatego pobieramy
    mecze po konkretnej dacie. Sprawdzamy dzisiaj oraz kolejne 3 dni.
    Wynik jest cachowany na 30 minut, żeby niepotrzebnie nie zużywać limitu.
    """
    if UPCOMING_CACHE.exists():
        age = time.time() - UPCOMING_CACHE.stat().st_mtime
        if age < UPCOMING_CACHE_MINUTES * 60:
            try:
                cached = pd.read_csv(UPCOMING_CACHE)
                if not cached.empty:
                    cached["date"] = pd.to_datetime(
                        cached["date"], utc=True, errors="coerce"
                    )
                    return cached.dropna(subset=["date"]).sort_values("date")
            except Exception:
                pass

    # 1 request per league for 4 days instead of 4 requests per league.
    out = []
    start = now().date()
    end = start + timedelta(days=3)
    for lid in LEAGUES:
        data = api(
            "/fixtures",
            {"league": lid, "season": CURRENT_SEASON, "from": start.isoformat(), "to": end.isoformat(), "timezone": TZ},
        )
        out += [row(x) for x in data]

    df = pd.DataFrame(out).drop_duplicates("id")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    try:
        df.to_csv(UPCOMING_CACHE, index=False)
    except Exception:
        pass
    return df


def upcoming():
    with _upcoming_lock:
        return _upcoming_uncached()

TODAY_CACHE = DATA / f"today_{CURRENT_SEASON}.csv"
TODAY_CACHE_MINUTES = 5

def today_fixtures():
    if TODAY_CACHE.exists() and time.time() - TODAY_CACHE.stat().st_mtime < TODAY_CACHE_MINUTES * 60:
        try:
            cached = pd.read_csv(TODAY_CACHE)
            if not cached.empty:
                cached["date"] = pd.to_datetime(cached["date"], utc=True, errors="coerce")
                return cached.dropna(subset=["date"]).sort_values("date")
        except Exception:
            pass
    out = []
    d = now().strftime("%Y-%m-%d")
    for lid in LEAGUES:
        data = api("/fixtures", {"league": lid, "season": CURRENT_SEASON, "date": d, "timezone": TZ})
        out += [row(x) for x in data]
    df = pd.DataFrame(out).drop_duplicates("id")
    if not df.empty:
        try:
            df.to_csv(TODAY_CACHE, index=False)
        except Exception:
            pass
    return df


def elo(df):
    e = {}
    for _, r in df.sort_values("date").iterrows():
        h, a = r.home, r.away
        rh, ra = e.get(h, 1500), e.get(a, 1500)
        eh = 1 / (1 + 10 ** ((ra - rh - 55) / 400))
        actual = 1 if r.hg > r.ag else .5 if r.hg == r.ag else 0
        m = max(1, math.log1p(abs(r.hg - r.ag)) * 1.6)
        e[h] = rh + 24 * m * (actual - eh)
        e[a] = ra + 24 * m * ((1 - actual) - (1 - eh))
    return e


def team_stats(df, team):
    h = df[df.home == team].copy()
    a = df[df.away == team].copy()
    h["gf"], h["ga"] = h.hg, h.ag
    a["gf"], a["ga"] = a.ag, a.hg
    x = pd.concat([h, a]).sort_values("date").tail(10)
    if x.empty:
        return 1.2, 1.2, .5
    pts = sum(3 if r.gf > r.ga else 1 if r.gf == r.ga else 0 for _, r in x.iterrows())
    return float(x.gf.mean()), float(x.ga.mean()), pts / (3 * len(x))


def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def predict(df, home, away):
    hg, hga, hform = team_stats(df, home)
    ag, aga, aform = team_stats(df, away)
    er = elo(df)
    eh, ea = er.get(home, 1500), er.get(away, 1500)
    lh = (hg + aga) / 2
    la = (ag + hga) / 2
    ed = (eh + 55 - ea) / 400
    fd = hform - aform
    lh *= math.exp(.18 * ed + .08 * fd)
    la *= math.exp(-.18 * ed - .08 * fd)
    lh = max(.05, min(4.5, lh))
    la = max(.05, min(4.5, la))
    dist = sorted(
        (pois(h, lh) * pois(a, la), h, a)
        for h in range(7) for a in range(7)
    )[::-1]
    p, h, a = dist[0]
    return {
        "home": home, "away": away, "score": f"{h}:{a}",
        "prob": p, "lh": lh, "la": la
    }


def label(p):
    if p >= .12:
        return "🔥 BARDZO WYSOKI SYGNAŁ"
    if p >= .08:
        return "🟢 WYSOKI SYGNAŁ"
    return "🟡 NISKI SYGNAŁ"


def coupon_score(items):
    ps = sorted([x["prob"] for x in items], reverse=True)
    return ps[0] if len(ps) == 1 else .6 * ps[1] + .4 * ps[0]


def load_last():
    if not LAST.exists():
        return None
    try:
        return json.loads(LAST.read_text(encoding="utf-8"))
    except Exception:
        return None


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
        log.warning("Telegram nie jest skonfigurowany.")
        print(msg, flush=True)
        return
    tg("sendMessage", {"chat_id": chat_id, "text": msg})


def get_model_history():
    with _history_lock:
        if HISTORY_CACHE.exists():
            age = time.time() - HISTORY_CACHE.stat().st_mtime
            if age < HISTORY_CACHE_HOURS * 3600:
                try:
                    df = pd.read_csv(HISTORY_CACHE)
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                        return df.dropna(subset=["hg", "ag"]).sort_values("date")
                except Exception:
                    pass
        h = history()
        if not h.empty:
            h.to_csv(HISTORY_CACHE, index=False)
        return h


def make_predictions():
    h = get_model_history()
    if len(h) < 50:
        raise RuntimeError(f"Za mało historii do analizy: {len(h)} meczów.")
    u = upcoming()
    cand = []
    if u.empty:
        return [], None
    for _, r in u.iterrows():
        if r.status not in ("NS", "TBD", ""):
            continue
        p = predict(h, r.home, r.away)
        if p["prob"] >= MINP:
            p.update({
                "id": int(r.id), "date": str(r.date),
                "league": r.league, "label": label(p["prob"])
            })
            cand.append(p)
    cand.sort(key=lambda x: x["prob"], reverse=True)
    chosen = cand[:MAXM]
    return chosen, coupon_score(chosen) if chosen else None


def scan(force=False):
    chosen, score = make_predictions()
    if not chosen:
        log.info("Brak odpowiednio mocnych sygnałów.")
        return False

    last = load_last()
    old = last["score"] if last else None

    if not force and old is not None and score <= old + MINIMP:
        log.info("Nie wysyłam: nowy %.4f, poprzedni %.4f", score, old)
        return False

    lines = ["🎫 PIŁKARSKA AI — NOWY KUPON", ""]
    for i, x in enumerate(chosen, 1):
        lines += [
            f"{i}. {x['home']} - {x['away']}",
            f"   🎯 Dokładny wynik: {x['score']}",
            f"   {x['label']}",
            f"   📊 prawdopodobieństwo: {x['prob']:.2%}",
            ""
        ]
    lines += [f"📈 Ocena kuponu: {score:.4f}"]
    if old is not None:
        lines.append(f"⬆️ Poprzednio: {old:.4f} | poprawa: {score-old:+.4f}")
    lines += ["", "To prognoza statystyczna, nie gwarancja wyniku."]
    msg = "\n".join(lines)
    print(msg, flush=True)
    send(msg)

    LAST.write_text(json.dumps({
        "timestamp": now().isoformat(), "score": score, "items": chosen
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [{
        "timestamp": now().isoformat(), "fixture_id": x["id"],
        "home": x["home"], "away": x["away"], "score": x["score"],
        "prob": x["prob"], "label": x["label"], "coupon_score": score
    } for x in chosen]
    nd = pd.DataFrame(rows)
    if HISTORY_LOG.exists():
        try:
            nd = pd.concat([pd.read_csv(HISTORY_LOG), nd], ignore_index=True)
        except Exception:
            pass
    nd.to_csv(HISTORY_LOG, index=False)
    return True


def results_message():
    df = today_fixtures()
    if df.empty:
        return "📅 Dzisiaj nie znaleziono meczów w ustawionych ligach."
    df = df.sort_values("date")
    lines = [f"📅 MECZE DZISIAJ — {now().strftime('%d.%m.%Y')}", ""]
    for _, r in df.iterrows():
        status = r.status
        score = ""
        if pd.notna(r.hg) and pd.notna(r.ag):
            score = f"  ⚽ {int(r.hg)}:{int(r.ag)}"
        lines.append(f"{r.home} - {r.away}{score} [{status}]")
    return "\n".join(lines[:101])


def status_message():
    return (
        "🤖 PIŁKARSKA AI działa.\n\n"
        f"⏱ Auto-skaner: {'ON' if AUTO_SCAN else 'OFF'}" + (f" (co {SCAN} min)\n" if AUTO_SCAN else "\n") +
        f"🎯 Maks. typów: {MAXM}\n"
        f"📊 Minimalne prawdopodobieństwo: {MINP:.0%}\n"
        f"🛡️ Odstęp API: {API_MIN_INTERVAL:.1f}s\n"
        f"🌍 Strefa: {TZ}\n"
        f"📅 Sezon API: {CURRENT_SEASON}"
    )


def handle_update(u):
    """Obsługuje pojedynczy update Telegrama."""
    msg = u.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = str((msg.get("chat") or {}).get("id", ""))
    if not text or not chat_id:
        return

    cmd = text.split()[0].split("@")[0].lower()
    try:
        if cmd == "/start":
            send(
                "🤖 Witaj! Piłkarska AI działa.\n\n"
                "Komendy:\n"
                "/typy — uruchamia analizę teraz\n"
                "/wyniki — pokazuje dzisiejsze mecze i wyniki\n"
                "/status — sprawdza działanie bota\n"
                "/help — pomoc",
                chat_id,
            )
        elif cmd in ("/help", "/pomoc"):
            send(
                "📌 KOMENDY\n"
                "/typy — analiza i najlepsze dokładne wyniki\n"
                "/wyniki — mecze z dzisiaj\n"
                "/status — status bota\n",
                chat_id,
            )
        elif cmd == "/status":
            send(status_message(), chat_id)
        elif cmd == "/wyniki":
            send(results_message(), chat_id)
        elif cmd == "/typy":
            send("⏳ Analizuję mecze. Chwilę to potrwa...", chat_id)
            chosen, score = make_predictions()
            if not chosen:
                send("🟡 Nie znalazłem teraz wystarczająco mocnych typów.", chat_id)
            else:
                lines = ["🎯 PIŁKARSKA AI — ANALIZA NA ŻĄDANIE", ""]
                for i, x in enumerate(chosen, 1):
                    lines += [
                        f"{i}. {x['home']} - {x['away']}",
                        f"   🎯 {x['score']} | {x['prob']:.2%}",
                        f"   {x['label']}", ""
                    ]
                lines += [
                    f"📈 Ocena: {score:.4f}", "",
                    "To prognoza statystyczna, nie gwarancja."
                ]
                send("\n".join(lines), chat_id)
        else:
            send("Nie znam tej komendy. Wpisz /help.", chat_id)
    except Exception as e:
        log.exception("Błąd komendy")
        send(f"❌ Nie udało się wykonać polecenia: {e}", chat_id)


def configure_telegram_webhook():
    """Na Renderze używamy webhooka zamiast getUpdates.
    Eliminuje to błąd 409 Conflict powodowany drugim procesem pollującym."""
    public_url = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if not TG_TOKEN or not public_url:
        return False
    webhook_url = public_url + "/telegram/webhook"
    try:
        result = tg("setWebhook", {
            "url": webhook_url,
            "allowed_updates": json.dumps(["message"]),
            "drop_pending_updates": "false",
        })
        log.info("Telegram webhook ustawiony: %s", webhook_url)
        return bool(result.get("ok"))
    except Exception:
        log.exception("Nie udało się ustawić webhooka Telegrama")
        return False


def command_loop():
    """Tryb lokalny: polling. Na Renderze webhook zastępuje polling."""
    if os.getenv("RENDER_EXTERNAL_URL"):
        log.info("Render wykryty — używam Telegram webhook, nie getUpdates.")
        return

    if not TG_TOKEN:
        log.warning("Brak TELEGRAM_BOT_TOKEN — pomijam obsługę komend.")
        return

    offset = None
    log.info("Obsługa Telegrama w trybie lokalnego pollingu uruchomiona.")
    while True:
        try:
            params = {"timeout": 25, "allowed_updates": json.dumps(["message"])}
            if offset is not None:
                params["offset"] = offset
            updates = tg("getUpdates", params).get("result", [])
            for u in updates:
                offset = u["update_id"] + 1
                handle_update(u)
        except Exception as e:
            log.exception("Błąd pętli Telegrama")
            time.sleep(10)


app = Flask(__name__)

@app.post("/telegram/webhook")
def telegram_webhook():
    u = request.get_json(silent=True) or {}
    threading.Thread(target=handle_update, args=(u,), daemon=True).start()
    return jsonify({"ok": True})




@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "Pilkarska AI — TEST FREE"})


@app.get("/health")
def health_check():
    return jsonify({"status": "healthy"})


def worker():
    if not AUTO_SCAN:
        log.info("Automatyczny skaner WYŁĄCZONY (AUTO_SCAN=0). /typy działa na żądanie.")
        return
    log.info("PIŁKARSKA AI — automatyczny skaner uruchomiony co %s min", SCAN)
    while True:
        try:
            scan()
        except Exception as e:
            log.exception("BŁĄD skanera: %r", e)
        time.sleep(max(1, SCAN) * 60)


def main():
    if not API_KEY:
        log.warning("Brak API_FOOTBALL_KEY.")
    if not TG_TOKEN:
        log.warning("Brak TELEGRAM_BOT_TOKEN.")
    port = int(os.getenv("PORT", "10000"))

    configure_telegram_webhook()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=command_loop, daemon=True).start()

    log.info("Serwer HTTP nasłuchuje na 0.0.0.0:%s", port)
    app.run(host="0.0.0.0", port=port, use_reloader=False)


if __name__ == "__main__":
    main()
