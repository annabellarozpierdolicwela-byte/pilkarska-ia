import os, time, json, math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT=Path(__file__).parent
DATA=ROOT/"data"; DATA.mkdir(exist_ok=True)
LAST=DATA/"last_coupon.json"
HISTORY_LOG=DATA/"predictions.csv"
load_dotenv(ROOT/".env")

API_KEY=os.getenv("API_FOOTBALL_KEY","").strip()
TG_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TG_CHAT=os.getenv("TELEGRAM_CHAT_ID","").strip()
TZ=os.getenv("TIMEZONE","Europe/Warsaw")
SCAN=int(os.getenv("SCAN_MINUTES","60"))
MAXM=int(os.getenv("MAX_MATCHES","2"))
MINP=float(os.getenv("MIN_SCORE_PROB","0.08"))
MINIMP=float(os.getenv("MIN_COUPON_IMPROVEMENT","0.02"))
LEAGUES=[int(x) for x in os.getenv("LEAGUE_IDS","39,140,135,78,106,2").split(",") if x.strip().isdigit()]
BASE="https://v3.football.api-sports.io"

def now(): return datetime.now(ZoneInfo(TZ))

def api(path, params):
    if not API_KEY: raise RuntimeError("Brak API_FOOTBALL_KEY")
    r=requests.get(BASE+path,headers={"x-apisports-key":API_KEY},params=params,timeout=30)
    r.raise_for_status()
    j=r.json()
    if j.get("errors"): raise RuntimeError(str(j["errors"]))
    return j.get("response",[])

def row(f):
    return {
      "id":f["fixture"]["id"],"date":f["fixture"]["date"],
      "status":f["fixture"]["status"]["short"],
      "league":f["league"]["name"],"lid":f["league"]["id"],
      "home":f["teams"]["home"]["name"],"away":f["teams"]["away"]["name"],
      "hg":f["goals"].get("home"),"ag":f["goals"].get("away")
    }

def history():
    out=[]
    for lid in LEAGUES:
        out += [row(x) for x in api("/fixtures",{"league":lid,"season":now().year,"status":"FT"})]
    df=pd.DataFrame(out).drop_duplicates("id")
    if df.empty: return df
    df["date"]=pd.to_datetime(df["date"],utc=True,errors="coerce")
    return df.dropna(subset=["hg","ag"]).sort_values("date")

def upcoming():
    out=[]
    for lid in LEAGUES:
        out += [row(x) for x in api("/fixtures",{"league":lid,"season":now().year,"next":30,"timezone":TZ})]
    df=pd.DataFrame(out).drop_duplicates("id")
    if df.empty: return df
    df["date"]=pd.to_datetime(df["date"],utc=True,errors="coerce")
    return df

def elo(df):
    e={}
    for _,r in df.sort_values("date").iterrows():
        h,a=r.home,r.away; rh=e.get(h,1500); ra=e.get(a,1500)
        eh=1/(1+10**((ra-rh-55)/400))
        actual=1 if r.hg>r.ag else .5 if r.hg==r.ag else 0
        m=max(1,math.log1p(abs(r.hg-r.ag))*1.6)
        e[h]=rh+24*m*(actual-eh); e[a]=ra+24*m*((1-actual)-(1-eh))
    return e

def team_stats(df,team):
    h=df[df.home==team].copy(); a=df[df.away==team].copy()
    h["gf"],h["ga"]=h.hg,h.ag; a["gf"],a["ga"]=a.ag,a.hg
    x=pd.concat([h,a]).sort_values("date").tail(10)
    if x.empty: return 1.2,1.2,.5
    pts=sum(3 if r.gf>r.ga else 1 if r.gf==r.ga else 0 for _,r in x.iterrows())
    return float(x.gf.mean()),float(x.ga.mean()),pts/(3*len(x))

def pois(k,l): return math.exp(-l)*l**k/math.factorial(k)

def predict(df,home,away):
    hg,hga,hform=team_stats(df,home); ag,aga,aform=team_stats(df,away)
    er=elo(df); eh,ea=er.get(home,1500),er.get(away,1500)
    lh=(hg+aga)/2; la=(ag+hga)/2
    ed=(eh+55-ea)/400; fd=hform-aform
    lh*=math.exp(.18*ed+.08*fd); la*=math.exp(-.18*ed-.08*fd)
    lh=max(.05,min(4.5,lh)); la=max(.05,min(4.5,la))
    dist=sorted((pois(h,lh)*pois(a,la),h,a) for h in range(7) for a in range(7))[::-1]
    p,h,a=dist[0]
    return {"home":home,"away":away,"score":f"{h}:{a}","prob":p,"lh":lh,"la":la}

def label(p):
    if p>=.12: return "🔥 BARDZO WYSOKI SYGNAŁ"
    if p>=.08: return "🟢 WYSOKI SYGNAŁ"
    return "🟡 NISKI SYGNAŁ"

def coupon_score(items):
    ps=sorted([x["prob"] for x in items],reverse=True)
    return ps[0] if len(ps)==1 else .6*ps[1]+.4*ps[0]

def load_last():
    if not LAST.exists(): return None
    try: return json.loads(LAST.read_text(encoding="utf-8"))
    except: return None

def send(msg):
    if not TG_TOKEN or not TG_CHAT:
        print("Telegram nie jest skonfigurowany. Wiadomość:")
        print(msg); return
    r=requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                    data={"chat_id":TG_CHAT,"text":msg},timeout=20)
    r.raise_for_status()

def scan():
    h=history()
    if len(h)<50:
        print("Za mało historii:",len(h)); return
    u=upcoming()
    cand=[]
    for _,r in u.iterrows():
        if r.status not in ("NS","TBD",""): continue
        p=predict(h,r.home,r.away)
        if p["prob"]>=MINP:
            p.update({"id":int(r.id),"date":str(r.date),"league":r.league,"label":label(p["prob"])})
            cand.append(p)
    cand.sort(key=lambda x:x["prob"],reverse=True)
    chosen=cand[:MAXM]
    if not chosen: print("Brak odpowiednio mocnych sygnałów."); return

    score=coupon_score(chosen)
    last=load_last()
    old=last["score"] if last else None

    # Tylko lepszy kupon może zostać wysłany.
    if old is not None and score <= old + MINIMP:
        print(f"Nie wysyłam: nowy {score:.4f}, poprzedni {old:.4f}")
        return

    lines=["🎫 PIŁKARSKA AI — NOWY KUPON",""]
    for i,x in enumerate(chosen,1):
        lines += [f"{i}. {x['home']} - {x['away']}",
                  f"   🎯 {x['score']}",
                  f"   {x['label']}",
                  f"   📊 exact-score: {x['prob']:.2%}",""]
    lines += [f"📈 Ocena kuponu: {score:.4f}"]
    if old is not None: lines.append(f"⬆️ Poprzednio: {old:.4f} | poprawa: {score-old:+.4f}")
    lines += ["","To prognoza statystyczna, nie gwarancja wyniku."]
    msg="\n".join(lines)
    print(msg); send(msg)

    LAST.write_text(json.dumps({"timestamp":now().isoformat(),"score":score,"items":chosen},ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[{"timestamp":now().isoformat(),"fixture_id":x["id"],"home":x["home"],"away":x["away"],
           "score":x["score"],"prob":x["prob"],"label":x["label"],"coupon_score":score} for x in chosen]
    nd=pd.DataFrame(rows)
    if HISTORY_LOG.exists(): nd=pd.concat([pd.read_csv(HISTORY_LOG),nd],ignore_index=True)
    nd.to_csv(HISTORY_LOG,index=False)

def main():
    print("PIŁKARSKA AI v4 uruchomiona")
    while True:
        try: scan()
        except Exception as e: print("BŁĄD:",repr(e))
        time.sleep(SCAN*60)

if __name__=="__main__": main()
