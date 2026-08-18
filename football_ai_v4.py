import os, time, json, math, threading, logging
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify
from dotenv import load_dotenv

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
LAST = DATA / "last_coupon.json"
HISTORY_LOG = DATA / "predictions.csv"
HISTORY_CACHE = DATA / "history_2024.csv"
HISTORY_CACHE_HOURS = 12

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TZ = os.getenv("TIMEZONE", "Europe/Warsaw")
SCAN = int(os.getenv("SCAN_MINUTES", "60"))
MAXM = int(os.getenv("MAX_MATCHES", "2"))
MINP = float(os.getenv("MIN_SCORE_PROB", "0.08"))
MINIMP = float(os.getenv("MIN_COUPON_IMPROVEMENT", "0.02"))
LEAGUES = [int(x) for x in os.getenv("LEAGUE_IDS", "39,140,135,78,106,2").split(",") if x.strip().isdigit()]
BASE = "https://v3.football.api-sports.io"
log = logging.getLogger("pilkarska_ai")


def now():
    return datetime.now(ZoneInfo(TZ))


def api(path, params):
    if not API_KEY:
        raise RuntimeError("Brak API_FOOTBALL_KEY")
    r = requests.get(
        BASE + path,
        headers={"x-apisports-key": API_KEY},
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        raise RuntimeError(str(j["errors"]))
    return j.get("response", [])


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


# API-Football Free plan currently limits historical seasons. 2024 is
# intentionally used for the model history; current fixtures are requested
# by DATE without a season parameter.
HISTORY_SEASON = 2024

def history():

    out = []
    for lid in LEAGUES:
        data = api("/fixtures", {"league": lid, "season": season_for(lid), "status": "FT"})
        if not data:
            data = api("/fixtures", {"league": lid, "season": now().year - 1, "status": "FT"})
        out += [row(x) for x in data]
    df = pd.DataFrame(out).drop_duplicates("id")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["hg", "ag"]).sort_values("date")


def upcoming():
    out = []
    for lid in LEAGUES:
        out += [row(x) for x in api(
            "/fixtures",
            {"league": lid, "season": season_for(lid), "next": 30, "timezone": TZ},
        )]
    df = pd.DataFrame(out).drop_duplicates("id")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df


def today_fixtures():
    out = []
    d = now().strftime("%Y-%m-%d")
    for lid in LEAGUES:
        out += [row(x) for x in api(
            "/fixtures",
            {"league": lid, "season": season_for(lid), "date": d, "timezone": TZ},
        )]
    return pd.DataFrame(out).drop_duplicates("id")


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
    # Cache the 2024 history so /typy and the automatic scanner do not
    # consume the Free plan request quota over and over.
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
        f"⏱ Skanowanie: co {SCAN} min\n"
        f"🎯 Maks. typów: {MAXM}\n"
        f"📊 Minimalne prawdopodobieństwo: {MINP:.0%}\n"
        f"🌍 Strefa: {TZ}"
    )


def command_loop():
    if not TG_TOKEN:
        log.warning("Brak TELEGRAM_BOT_TOKEN — pomijam obsługę komend.")
        return

    offset = None
    log.info("Obsługa Telegrama uruchomiona.")
    while True:
        try:
            params = {"timeout": 25, "allowed_updates": json.dumps(["message"])}
            if offset is not None:
                params["offset"] = offset
            updates = tg("getUpdates", params).get("result", [])
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = str((msg.get("chat") or {}).get("id", ""))
                if not text or not chat_id:
                    continue

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
                            lines += [f"📈 Ocena: {score:.4f}", "",
                                      "To prognoza statystyczna, nie gwarancja."]
                            send("\n".join(lines), chat_id)
                    else:
                        send("Nie znam tej komendy. Wpisz /help.", chat_id)
                except Exception as e:
                    log.exception("Błąd komendy")
                    send(f"❌ Nie udało się wykonać polecenia: {e}", chat_id)
        except Exception as e:
            log.exception("Błąd pętli Telegrama")
            time.sleep(10)


app = Flask(__name__)


@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "Pilkarska AI v5"})


@app.get("/health")
def health_check():
    return jsonify({"status": "healthy"})


def worker():
    log.info("PIŁKARSKA AI v5 — skaner uruchomiony")
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

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=command_loop, daemon=True).start()

    log.info("Serwer HTTP nasłuchuje na 0.0.0.0:%s", port)
    app.run(host="0.0.0.0", port=port, use_reloader=False)


if __name__ == "__main__":
    main()
