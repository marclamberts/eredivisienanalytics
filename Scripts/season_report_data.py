"""
Shared data loading + stat computation for the season analysis report
(season_report.py). All numbers here are derived directly from:

  - Danger/all_eredivisie_danger_models.csv   shot-level data (xG, PSxG, on
    target, penalty flag, goal flag) for every match this season
  - Events/*.json                             full Opta-style event stream
    per match (passes, take-ons, tackles, GK actions, ...)

Event qualifier IDs used below were verified empirically against this
specific data (not assumed from generic Opta documentation), by checking
that events tagged with a given qualifier show the expected spatial
pattern - e.g. qualifier 2 ("cross") only fires on passes that start in a
wide channel near the byline and end centrally inside the box:
  - 1   = long ball
  - 2   = cross            (verified: wide-origin -> central/box-end passes)
  - 5   = free kick taken  (verified: scattered pitch locations)
  - 6   = corner taken     (verified: originates from the corner arc, x~100/0, y~0/100)
  - 140/141 = pass end x/y (already used throughout buildup_patterns.py)

There is no reliable "key pass" or "assist" qualifier in this feed, so both
are derived structurally instead: the pass immediately preceding a shot (in
the same uninterrupted possession) is that shot's key pass; if the shot is
a goal, that same pass is the assist. This reuses the exact sequence logic
already validated in buildup_patterns.py.
"""
import glob
import json
import re
import statistics
import collections

import numpy as np
import pandas as pd

DANGER_CSV = 'Danger/all_eredivisie_danger_models.csv'
TEAM_SUMMARY_CSV = 'xT/xt_team_summary.csv'
EVENTS_DIR = 'Events'

SHOT_TYPES = {13, 14, 15, 16}
GOAL_TYPE = 16
PASS_TYPE = 1
TAKEON_TYPE = 3
GK_TYPES = {10, 11, 41, 52, 59}  # save, claim, punch, keeper pick-up, keeper sweeper
REGAIN_TYPES = {7, 8, 49}  # tackle, interception, ball recovery

FINAL_THIRD_X = 200 / 3
BOX_X, BOX_Y_LO, BOX_Y_HI = 83.0, 21.1, 78.9


def minute_value(e):
    return float(e.get('timeMin') or 0) + float(e.get('timeSec') or 0) / 60.0


def team_id_map():
    df = pd.read_csv(TEAM_SUMMARY_CSV)[['contestant_id', 'team_name']].drop_duplicates()
    return dict(zip(df.contestant_id, df.team_name)), dict(zip(df.team_name, df.contestant_id))


def resolve_team(team_query):
    id2name, name2id = team_id_map()
    match = next((nm for nm in name2id if team_query.lower() in nm.lower()), None)
    if match is None:
        raise SystemExit(f"Team '{team_query}' not found. Options: {list(name2id)}")
    return match, name2id[match]


def match_files_for_team(team_name):
    files = sorted(glob.glob(f'{EVENTS_DIR}/*.json'))
    return [f for f in files if team_name in f]


def load_shots(team_name):
    """Full-season shot-level rows for every match this team played, tagged
    with whether the shot was taken BY this team ('for') or AGAINST it
    ('against'), using the Danger models (already used by top3_ratings.py)."""
    id2name, name2id = team_id_map()
    cid = name2id[team_name]
    df = pd.read_csv(DANGER_CSV)
    files = match_files_for_team(team_name)
    fnames = {f.split('/')[-1] for f in files}
    sub = df[df['match_file'].isin(fnames)].copy()
    sub['team_name'] = sub['contestant_id'].map(id2name)
    sub['side'] = np.where(sub['team_name'] == team_name, 'for', 'against')
    return sub


def load_match_events(team_name):
    """Yield (match_file, sorted_events, cid, opp_cid) for every match."""
    id2name, name2id = team_id_map()
    cid = name2id[team_name]
    for fn in match_files_for_team(team_name):
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data['event'] if e.get('periodId') in (1, 2) and e.get('contestantId')]
        evs.sort(key=lambda e: (e['periodId'], minute_value(e), e.get('eventId', 0)))
        cids = {e['contestantId'] for e in evs}
        opp_cid = next((c for c in cids if c != cid), None)
        yield fn.split('/')[-1], evs, cid, opp_cid


def qids(e):
    return {q['qualifierId'] for q in e.get('qualifier', [])}


def pass_end_xy(e):
    qmap = {q['qualifierId']: q.get('value') for q in e.get('qualifier', [])}
    x1 = qmap.get(140)
    y1 = qmap.get(141)
    if x1 is None or y1 is None:
        return None
    return float(x1), float(y1)


def sequences_by_contestant(evs):
    """Split one match's events into maximal same-contestantId runs."""
    seq, cur = [], None
    runs = []
    for e in evs:
        c = e['contestantId']
        if c != cur:
            if seq:
                runs.append((cur, seq))
            seq, cur = [e], c
        else:
            seq.append(e)
    if seq:
        runs.append((cur, seq))
    return runs


# ── passing & crossing ──────────────────────────────────────────────────────
def compute_passing(team_name):
    rows = []
    n_matches = 0
    for fn, evs, cid, opp in load_match_events(team_name):
        n_matches += 1
        for e in evs:
            if e['contestantId'] != cid or e['typeId'] != PASS_TYPE:
                continue
            end = pass_end_xy(e)
            completed = e.get('outcome') == 1
            prog = False
            if completed and end is not None:
                start_rem = 100 - e['x']
                end_rem = 100 - end[0]
                prog = start_rem > 0 and end_rem <= start_rem * 0.75
            ft_entry = completed and end is not None and e['x'] < FINAL_THIRD_X <= end[0]
            box_entry = (completed and end is not None and end[0] >= BOX_X
                         and BOX_Y_LO <= end[1] <= BOX_Y_HI and not (e['x'] >= BOX_X and BOX_Y_LO <= e['y'] <= BOX_Y_HI))
            q = qids(e)
            rows.append(dict(match=fn, completed=completed, is_cross=2 in q, is_long=1 in q,
                              progressive=prog, final_third_entry=ft_entry, box_entry=box_entry,
                              x=e['x'], y=e['y'], end_x=end[0] if end else None, end_y=end[1] if end else None))
    df = pd.DataFrame(rows)
    return df, n_matches


# ── key passes & assists (structural, not qualifier-based) ─────────────────
def compute_key_passes_assists(team_name):
    records = []
    for fn, evs, cid, opp in load_match_events(team_name):
        for c, seq in sequences_by_contestant(evs):
            if c != cid or len(seq) < 2:
                continue
            last = seq[-1]
            if last['typeId'] not in SHOT_TYPES:
                continue
            prev = seq[-2]
            if prev['typeId'] != PASS_TYPE or prev.get('outcome') != 1:
                continue
            q = qids(prev)
            records.append(dict(match=fn, is_goal=last['typeId'] == GOAL_TYPE,
                                 is_cross=2 in q, passer=prev.get('playerName'),
                                 shooter=last.get('playerName')))
    return pd.DataFrame(records)


# ── build-up zone profile (volume by starting third, all qualifying shots-ending sequences) ─
def compute_buildup_profile(team_name):
    from buildup_patterns import ZONES
    records = []
    for fn, evs, cid, opp in load_match_events(team_name):
        for c, seq in sequences_by_contestant(evs):
            if c != cid or len(seq) < 2:
                continue
            first = seq[0]
            last = seq[-1]
            if last['typeId'] not in SHOT_TYPES or first.get('x') is None:
                continue
            n_passes = sum(1 for e in seq if e['typeId'] == PASS_TYPE)
            for zname, (lo, hi, label) in ZONES.items():
                if lo <= first['x'] < hi:
                    records.append(dict(match=fn, zone=label, n_passes=n_passes,
                                         is_goal=last['typeId'] == GOAL_TYPE))
                    break
    return pd.DataFrame(records)


# ── transitions (regain -> shot, fast) ──────────────────────────────────────
TRANSITION_MAX_SECONDS = 15


def compute_transitions(team_name):
    records = []
    for fn, evs, cid, opp in load_match_events(team_name):
        for c, seq in sequences_by_contestant(evs):
            if c != cid or len(seq) < 1:
                continue
            first, last = seq[0], seq[-1]
            if first['typeId'] not in REGAIN_TYPES:
                continue
            if last['typeId'] not in SHOT_TYPES:
                continue
            if first['periodId'] != last['periodId']:
                continue
            elapsed = (minute_value(last) - minute_value(first)) * 60
            if elapsed < 0 or elapsed > TRANSITION_MAX_SECONDS:
                continue
            n_passes = sum(1 for e in seq if e['typeId'] == PASS_TYPE)
            records.append(dict(match=fn, elapsed=elapsed, n_passes=n_passes,
                                 is_goal=last['typeId'] == GOAL_TYPE, seq=seq, shot=last))
    return records


# ── set pieces (corner/free kick delivery starts the sequence) ─────────────
# The dead-ball "awarded" event (e.g. typeId 6, Corner Awarded) is attributed
# to the taking team and lands as the sequence's first event, one step
# *before* the actual delivery pass that carries the qualifier - so the
# qualifier is looked up across the first few events, not strictly seq[0].
def compute_setpieces(team_name):
    records = []
    for fn, evs, cid, opp in load_match_events(team_name):
        for c, seq in sequences_by_contestant(evs):
            if c != cid or not seq:
                continue
            last = seq[-1]
            head_q = set().union(*(qids(e) for e in seq[:3])) if seq else set()
            kind = None
            if 6 in head_q:
                kind = 'Corner'
            elif 5 in head_q:
                kind = 'Free kick'
            if kind is None:
                continue
            has_shot = last['typeId'] in SHOT_TYPES
            records.append(dict(match=fn, kind=kind, has_shot=has_shot,
                                 is_goal=has_shot and last['typeId'] == GOAL_TYPE))
    return pd.DataFrame(records)


# ── goalkeeping ──────────────────────────────────────────────────────────────
def identify_keeper(team_name):
    counts = collections.Counter()
    for fn, evs, cid, opp in load_match_events(team_name):
        for e in evs:
            if e['contestantId'] == cid and e['typeId'] == 52:  # keeper pick-up
                counts[e.get('playerName')] += 1
    return counts.most_common(1)[0][0] if counts else None


def compute_gk_distribution(team_name, keeper_name):
    rows = []
    for fn, evs, cid, opp in load_match_events(team_name):
        for e in evs:
            if e['contestantId'] == cid and e['typeId'] == PASS_TYPE and e.get('playerName') == keeper_name:
                q = qids(e)
                rows.append(dict(completed=e.get('outcome') == 1, is_long=1 in q))
    return pd.DataFrame(rows)
