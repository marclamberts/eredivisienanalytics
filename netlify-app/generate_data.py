import csv, glob, json, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return 0.0

team_names = {}
xt_teams = rows(os.path.join(ROOT, "xT", "xt_team_summary.csv"))
for r in xt_teams: team_names[r["contestant_id"]] = r["team_name"]

clubs = defaultdict(lambda: {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"xt":0,"actions":0,"gda":0,"danger":0,"shots":0})
matches = []
for path in sorted(glob.glob(os.path.join(ROOT, "*.json"))):
    stem = os.path.basename(path)[:-5]
    date, fixture = stem.split("_", 1)
    home, away = fixture.split(" - ", 1)
    try:
        raw = json.load(open(path, encoding="utf-8"))
        score = raw.get("matchDetails", {}).get("scores", {})
        # Opta feeds vary: try common final-score shapes, then goals in events.
        h = score.get("home", score.get("total", {}).get("home", score.get("ft", {}).get("home"))) if isinstance(score, dict) else None
        a = score.get("away", score.get("total", {}).get("away", score.get("ft", {}).get("away"))) if isinstance(score, dict) else None
        if h is None or a is None:
            goals = [e for e in raw.get("event",[]) if e.get("typeId")==16 and e.get("outcome")==1]
            ids = []
            for e in raw.get("event",[]):
                cid=e.get("contestantId")
                if cid and cid not in ids: ids.append(cid)
            h = sum(1 for g in goals if ids and g.get("contestantId")==ids[0])
            a = sum(1 for g in goals if len(ids)>1 and g.get("contestantId")==ids[1])
        h,a=int(h),int(a)
    except Exception:
        h=a=0
    matches.append({"date":date,"home":home,"away":away,"hs":h,"as":a})
    for name,gf,ga in ((home,h,a),(away,a,h)):
        c=clubs[name]; c["mp"]+=1;c["gf"]+=gf;c["ga"]+=ga
        if gf>ga:c["w"]+=1
        elif gf==ga:c["d"]+=1
        else:c["l"]+=1

for r in xt_teams:
    c=clubs[r["team_name"]]; c["xt"]=num(r["xT_added"]); c["actions"]=int(num(r["actions"]))

gda_players = rows(os.path.join(ROOT,"GDA","gda_player_summary.csv"))
for r in gda_players:
    name=team_names.get(r["contestant_id"])
    if name: clubs[name]["gda"] += num(r["goal_difference_added"])

danger_path=os.path.join(ROOT,"Danger","all_eredivisie_danger_models.csv")
danger_rows=rows(danger_path)
for r in danger_rows:
    name=team_names.get(r["contestant_id"])
    if name:
        clubs[name]["danger"] += num(r["danger_score"]); clubs[name]["shots"] += 1

teams=[]
for name,c in clubs.items():
    c.update({"name":name,"pts":c["w"]*3+c["d"],"gd":c["gf"]-c["ga"]})
    teams.append(c)
teams.sort(key=lambda c:(c["pts"],c["gd"],c["gf"]), reverse=True)

xt_players=[]
for r in rows(os.path.join(ROOT,"xT","xt_player_summary.csv")):
    xt_players.append({"name":r["player_name"],"team":r["team_name"],"actions":int(num(r["actions"])),"passes":int(num(r["passes"])),"carries":int(num(r["carries"])),"xt":num(r["xT_added"]),"xt100":num(r["xT_per_100_actions"])})
xt_players.sort(key=lambda x:x["xt"], reverse=True)

gda=[]
for r in gda_players:
    if num(r["minutes"]) >= 450:
        gda.append({"name":r["player_name"],"team":team_names.get(r["contestant_id"],"—"),"minutes":round(num(r["minutes"])),"gda":num(r["goal_difference_added"]),"gda90":num(r["goal_difference_added_per90"])})
gda.sort(key=lambda x:x["gda90"], reverse=True)

dp=defaultdict(lambda:{"name":"","team":"—","shots":0,"danger":0,"xg":0,"goals":0})
for r in danger_rows:
    pid=r["player_id"] or r["player_name"]
    p=dp[pid];p["name"]=r["player_name"] or "Unknown";p["team"]=team_names.get(r["contestant_id"],"—");p["shots"]+=1;p["danger"]+=num(r["danger_score"]);p["xg"]+=num(r["xg"]);p["goals"]+=int(num(r["is_goal"]))
danger_players=sorted(dp.values(),key=lambda x:x["danger"],reverse=True)

shots=[{"team":team_names.get(r["contestant_id"],"—"),"player":r["player_name"] or "Unknown","x":round(num(r["x"]),1),"y":round(num(r["y"]),1),"xg":round(num(r["xg"]),4),"danger":round(num(r["danger_score"]),4),"goal":int(num(r["is_goal"])),"target":int(num(r["is_on_target"])),"date":r["match_file"][:10]} for r in danger_rows]

trends={}
for team in clubs:
    played=pts=gf=ga=0; series=[]
    for m in matches:
        if team not in (m["home"],m["away"]): continue
        home=m["home"]==team; f=m["hs"] if home else m["as"]; a=m["as"] if home else m["hs"]
        played+=1;gf+=f;ga+=a;pts+=3 if f>a else 1 if f==a else 0
        series.append({"date":m["date"],"mp":played,"pts":pts,"ppg":round(pts/played,3),"gd":gf-ga})
    trends[team]=series

# xG player totals from the shot model.
xg_players=[]
for p in dp.values():
    if p["shots"]:
        xg_players.append({"name":p["name"],"team":p["team"],"shots":p["shots"],"goals":p["goals"],"xg":round(p["xg"],4),"xg_per_shot":round(p["xg"]/p["shots"],4),"goals_minus_xg":round(p["goals"]-p["xg"],4)})
xg_players.sort(key=lambda p:p["xg"],reverse=True)

# Link each shot to the nearest preceding completed same-team pass. The shot's
# xG is credited to that passer as xA, regardless of whether the shot is scored.
xa=defaultdict(lambda:{"name":"","team":"—","key_passes":0,"assists":0,"xa":0.0})
shots_by_match=defaultdict(list)
for r in danger_rows: shots_by_match[r["match_file"]].append(r)
passes=[]
for path in sorted(glob.glob(os.path.join(ROOT,"*.json"))):
    raw=json.load(open(path,encoding="utf-8")); events=raw.get("event",[]); fn=os.path.basename(path)
    for sr in shots_by_match.get(fn,[]):
        idx=int(num(sr["event_index"])); cid=sr["contestant_id"]
        for e in reversed(events[max(0,idx-4):idx]):
            if e.get("contestantId")==cid and e.get("typeId")==1 and e.get("outcome")==1:
                pid=e.get("playerId") or e.get("playerName","Unknown"); q=xa[pid];q["name"]=e.get("playerName","Unknown");q["team"]=team_names.get(cid,"—");q["key_passes"]+=1;q["assists"]+=int(num(sr["is_goal"]));q["xa"]+=num(sr["xg"]);break
    for e in events:
        if e.get("typeId")!=1 or not e.get("playerId"): continue
        q={z.get("qualifierId"):z.get("value") for z in e.get("qualifier",[])}
        ex,ey=num(q.get(140)),num(q.get(141)); dist=((ex-num(e.get("x")))**2+(ey-num(e.get("y")))**2)**.5
        zone=min(2,int(num(e.get("x"))//33.34)); bucket=min(5,int(dist//10))
        passes.append({"player":e.get("playerName","Unknown"),"pid":e.get("playerId"),"team":team_names.get(e.get("contestantId"),"—"),"outcome":int(e.get("outcome",0)==1),"bucket":bucket,"zone":zone})
xa_players=[{"name":v["name"],"team":v["team"],"key_passes":v["key_passes"],"assists":v["assists"],"xa":round(v["xa"],4),"xa_per_key_pass":round(v["xa"]/v["key_passes"],4)} for v in xa.values() if v["key_passes"]]
xa_players.sort(key=lambda p:p["xa"],reverse=True)

# Empirical xPass: smoothed completion rate by 10m distance bucket and pitch third.
bucket_stats=defaultdict(lambda:[0,0])
for p in passes: bucket_stats[(p["bucket"],p["zone"])][0]+=p["outcome"];bucket_stats[(p["bucket"],p["zone"])][1]+=1
xp=defaultdict(lambda:{"name":"","team":"—","passes":0,"completed":0,"expected":0.0})
for p in passes:
    made,total=bucket_stats[(p["bucket"],p["zone"])]; prob=(made+20*.78)/(total+20)
    z=xp[p["pid"]];z["name"]=p["player"];z["team"]=p["team"];z["passes"]+=1;z["completed"]+=p["outcome"];z["expected"]+=prob
xpass_players=[{"name":v["name"],"team":v["team"],"passes":v["passes"],"completed":v["completed"],"completion":round(v["completed"]/v["passes"]*100,2),"xpass":round(v["expected"],2),"completed_above_expected":round(v["completed"]-v["expected"],2)} for v in xp.values() if v["passes"]>=100]
xpass_players.sort(key=lambda p:p["completed_above_expected"],reverse=True)

# One wide, FBref-style player record used by the grouped player tables.
master={}
def merge_players(source, fields):
    for p in source:
        key=(p["name"],p["team"]); row=master.setdefault(key,{"name":p["name"],"team":p["team"]})
        for field in fields: row[field]=p.get(field,0)
merge_players(xt_players,["actions","passes","carries","xt","xt100"])
merge_players(gda,["minutes","gda","gda90"])
merge_players(xg_players,["shots","goals","xg","xg_per_shot","goals_minus_xg"])
merge_players(xa_players,["key_passes","assists","xa","xa_per_key_pass"])
merge_players(xpass_players,["passes","completed","completion","xpass","completed_above_expected"])
merge_players(danger_players,["danger"])
player_master=list(master.values())
player_master.sort(key=lambda p:(p.get("minutes",0),p.get("actions",0)),reverse=True)

payload={"generated":"2026-07-18","matches":matches,"teams":teams,"playerMaster":player_master,"xtPlayers":xt_players[:100],"gdaPlayers":gda[:100],"dangerPlayers":danger_players[:100],"xgPlayers":xg_players[:100],"xaPlayers":xa_players[:100],"xpassPlayers":xpass_players[:100],"shots":shots,"trends":trends,"totals":{"matches":len(matches),"events":sum(1 for _ in glob.glob(os.path.join(ROOT,"*.json"))),"players":len({p['name'] for p in xt_players}),"actions":sum(t['actions'] for t in teams),"shots":len(shots),"goals":sum(s["goal"] for s in shots),"passes":len(passes)}}
with open(OUT,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,separators=(",",":"))
print(OUT)
