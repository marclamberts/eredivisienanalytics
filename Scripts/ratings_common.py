"""
Shared rating-engine pipeline for Eredivisie attacking/defensive ratings.

Match results are reconstructed from the shot-level Danger models (goals via
the is_goal flag), since no ready-made results table is checked into the repo.
Four independent rating systems are then tracked match-by-match for every
side:

  - PI        simple per-match goal rate (no opponent adjustment)
  - ELO       classic Elo, opponent-adjusted, unbounded
  - Glicko-2  Elo extension with rating deviation / volatility (uncertainty
              shrinks as more matches are played)
  - Base-70   bounded scale anchored on 70, opponent-unadjusted with mild
              mean-reversion (a simple "current level" gauge)

Two output pipelines are exposed:
  - compute_all_systems()        4 systems rescaled onto a shared 0-100 index
                                  (needed to overlay them on one axis) — used
                                  by top3_ratings.py (trend charts).
  - compute_all_systems_native()  each system in its own real unit, no
                                  rescaling — used by ratings_table.py
                                  (per-system league tables).
"""
import re
import numpy as np
import pandas as pd

DANGER_CSV = 'Danger/all_eredivisie_danger_models.csv'
TEAM_SUMMARY_CSV = 'xT/xt_team_summary.csv'

MA_WINDOW = 5
METHOD_ORDER = ['PI', 'ELO', 'Glicko-2', 'Base-70']


# ── 1. reconstruct match results from shot-event data ───────────────────────
def load_matches():
    df = pd.read_csv(DANGER_CSV, usecols=['match_file', 'contestant_id', 'is_goal'])
    id2name = dict(pd.read_csv(TEAM_SUMMARY_CSV)[['contestant_id', 'team_name']].values)

    goals = df.groupby(['match_file', 'contestant_id'])['is_goal'].sum().reset_index()

    rows = []
    for mf, grp in goals.groupby('match_file'):
        m = re.match(r'(\d{4}-\d{2}-\d{2})_(.+) - (.+)\.json', mf)
        if not m or len(grp) != 2:
            continue
        date, home_name, away_name = m.group(1), m.group(2), m.group(3)
        g = grp.set_index('contestant_id')['is_goal'].to_dict()
        names = {cid: id2name.get(cid, cid) for cid in g}
        home_id = [cid for cid, nm in names.items() if nm == home_name]
        away_id = [cid for cid, nm in names.items() if nm == away_name]
        if len(home_id) != 1 or len(away_id) != 1:
            continue
        rows.append(dict(date=date, home=home_name, away=away_name,
                          home_goals=g[home_id[0]], away_goals=g[away_id[0]]))

    matches = pd.DataFrame(rows)
    matches['date'] = pd.to_datetime(matches['date'])
    return matches.sort_values(['date', 'home']).reset_index(drop=True)


def continuous_score(goals, avg):
    """Smooth 0-1 performance score: 0.5 at the league-average scoring rate."""
    return float(np.clip(0.5 + (goals - avg) / (2 * avg), 0, 1))


# ── 2a. PI: raw per-match goal rate, no opponent adjustment ─────────────────
def compute_pi(matches, teams, invert_defense=False):
    """PI's native unit is goals per match: attack = goals scored, defense =
    goals conceded (so LOWER is better for defense here, unlike the other
    three systems, which are constructed so higher is always better).

    invert_defense=True negates the defensive value instead, so higher also
    means better here — used only when this feeds a shared 0-100 index next
    to systems that already use that convention (see normalise()/compute_all_
    systems() for the chart pipeline). Native/table consumers should leave it
    False and simply note that PI's defensive column reads low-is-better.
    """
    sign = -1 if invert_defense else 1
    hist = {t: [] for t in teams}
    for row in matches.itertuples():
        hist[row.home].append((row.date, row.home_goals, sign * row.away_goals))
        hist[row.away].append((row.date, row.away_goals, sign * row.home_goals))
    return hist  # {team: [(date, att_raw, def_raw), ...]}


# ── 2b. ELO: opponent-adjusted attack/defense ratings ───────────────────────
def elo_expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def compute_elo(matches, teams, league_avg, K=32, home_adv=30):
    att = {t: 1500.0 for t in teams}
    deff = {t: 1500.0 for t in teams}
    hist = {t: [] for t in teams}

    for row in matches.itertuples():
        h, a, hg, ag, date = row.home, row.away, row.home_goals, row.away_goals, row.date

        e1 = elo_expected(att[h] + home_adv, deff[a])
        s1 = continuous_score(hg, league_avg)
        att[h] += K * (s1 - e1)
        deff[a] += K * ((1 - s1) - (1 - e1))

        e2 = elo_expected(att[a], deff[h])
        s2 = continuous_score(ag, league_avg)
        att[a] += K * (s2 - e2)
        deff[h] += K * ((1 - s2) - (1 - e2))

        hist[h].append((date, att[h], deff[h]))
        hist[a].append((date, att[a], deff[a]))
    return hist


# ── 2c. Glicko-2: opponent-adjusted + uncertainty-weighted ──────────────────
PHI0, SIGMA0, TAU = 350, 0.06, 0.5
SCALE = 173.7178


def g_fn(phi):
    return 1 / np.sqrt(1 + 3 * phi ** 2 / np.pi ** 2)


def e_fn(mu, mu_j, phi_j):
    return 1 / (1 + np.exp(-g_fn(phi_j) * (mu - mu_j)))


def new_sigma(sigma, delta, phi, v, tau=TAU):
    a = np.log(sigma ** 2)
    eps = 1e-6

    def f(x):
        ex = np.exp(x)
        return (ex * (delta ** 2 - phi ** 2 - v - ex)) / (2 * (phi ** 2 + v + ex) ** 2) - (x - a) / tau ** 2

    A = a
    if delta ** 2 > phi ** 2 + v:
        B = np.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau
    fA, fB = f(A), f(B)
    for _ in range(100):
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0:
            A, fA = B, fB
        else:
            fA = fA / 2
        B, fB = C, fC
        if abs(B - A) < eps:
            break
    return np.exp(A / 2)


def update_glicko2(mu, phi, sigma, opponents, scores):
    if not opponents:
        phi_star = np.sqrt(phi ** 2 + sigma ** 2)
        return mu, min(phi_star, PHI0 / SCALE), sigma
    v = sum(g_fn(pj) ** 2 * e_fn(mu, mj, pj) * (1 - e_fn(mu, mj, pj)) for mj, pj in opponents) ** -1
    delta = v * sum(g_fn(pj) * (s - e_fn(mu, mj, pj)) for (mj, pj), s in zip(opponents, scores))
    sigma_new = new_sigma(sigma, delta, phi, v)
    phi_star = np.sqrt(phi ** 2 + sigma_new ** 2)
    phi_new = 1 / np.sqrt(1 / phi_star ** 2 + 1 / v)
    mu_new = mu + phi_new ** 2 * sum(g_fn(pj) * (s - e_fn(mu, mj, pj)) for (mj, pj), s in zip(opponents, scores))
    return mu_new, phi_new, sigma_new


def compute_glicko2(matches, teams, league_avg):
    mu_att = {t: 0.0 for t in teams}
    phi_att = {t: PHI0 / SCALE for t in teams}
    sig_att = {t: SIGMA0 for t in teams}
    mu_def = {t: 0.0 for t in teams}
    phi_def = {t: PHI0 / SCALE for t in teams}
    sig_def = {t: SIGMA0 for t in teams}
    hist = {t: [] for t in teams}

    for date, day in matches.groupby('date'):
        opp_att = {t: [] for t in teams}
        sc_att = {t: [] for t in teams}
        opp_def = {t: [] for t in teams}
        sc_def = {t: [] for t in teams}

        for row in day.itertuples():
            h, a, hg, ag = row.home, row.away, row.home_goals, row.away_goals
            s1 = continuous_score(hg, league_avg)
            opp_att[h].append((mu_def[a], phi_def[a])); sc_att[h].append(s1)
            opp_def[a].append((mu_att[h], phi_att[h])); sc_def[a].append(1 - s1)
            s2 = continuous_score(ag, league_avg)
            opp_att[a].append((mu_def[h], phi_def[h])); sc_att[a].append(s2)
            opp_def[h].append((mu_att[a], phi_att[a])); sc_def[h].append(1 - s2)

        new_mu_att, new_phi_att, new_sig_att = {}, {}, {}
        new_mu_def, new_phi_def, new_sig_def = {}, {}, {}
        for t in teams:
            new_mu_att[t], new_phi_att[t], new_sig_att[t] = update_glicko2(mu_att[t], phi_att[t], sig_att[t], opp_att[t], sc_att[t])
            new_mu_def[t], new_phi_def[t], new_sig_def[t] = update_glicko2(mu_def[t], phi_def[t], sig_def[t], opp_def[t], sc_def[t])
        mu_att, phi_att, sig_att = new_mu_att, new_phi_att, new_sig_att
        mu_def, phi_def, sig_def = new_mu_def, new_phi_def, new_sig_def

        for t in teams:
            if opp_att[t] or opp_def[t]:
                hist[t].append((date, mu_att[t] * SCALE + 1500, mu_def[t] * SCALE + 1500))
    return hist


# ── 2d. Base-70: bounded, opponent-unadjusted, mean-reverting ───────────────
def compute_base70(matches, teams, league_avg, K=2.2, revert=0.04, lo=35, hi=99):
    att = {t: 70.0 for t in teams}
    deff = {t: 70.0 for t in teams}
    hist = {t: [] for t in teams}

    for row in matches.itertuples():
        for team, gf, ga in [(row.home, row.home_goals, row.away_goals),
                              (row.away, row.away_goals, row.home_goals)]:
            att[team] += K * (gf - league_avg) - revert * (att[team] - 70)
            deff[team] += K * (league_avg - ga) - revert * (deff[team] - 70)
            att[team] = float(np.clip(att[team], lo, hi))
            deff[team] = float(np.clip(deff[team], lo, hi))
        hist[row.home].append((row.date, att[row.home], deff[row.home]))
        hist[row.away].append((row.date, att[row.away], deff[row.away]))
    return hist


# ── 3a. smooth each raw trajectory (native units, no rescaling) ─────────────
# a rolling mean absorbs single-match outliers (e.g. a 6-0 rout) without
# altering what the numbers actually mean — goals/match stay goals/match,
# Elo points stay Elo points, etc.
def smooth_native(hist):
    out = {}
    for t, series in hist.items():
        df_t = pd.DataFrame(series, columns=['date', 'att', 'def'])
        df_t['att'] = df_t['att'].rolling(MA_WINDOW, min_periods=1).mean()
        df_t['def'] = df_t['def'].rolling(MA_WINDOW, min_periods=1).mean()
        out[t] = df_t
    return out


# ── 3b. smooth, then normalise onto a shared 0-100 index ────────────────────
# only used where 4 differently-scaled systems must share one axis (the
# trend charts in top3_ratings.py); table/native consumers use smooth_native
# directly and keep each system's real units.
def normalise(hist):
    smoothed = smooth_native(hist)

    all_att = np.concatenate([df_t['att'].values for df_t in smoothed.values()])
    all_def = np.concatenate([df_t['def'].values for df_t in smoothed.values()])
    att_lo, att_hi = all_att.min(), all_att.max()
    def_lo, def_hi = all_def.min(), all_def.max()

    out = {}
    for t, df_t in smoothed.items():
        out[t] = pd.DataFrame({
            'date': df_t['date'],
            'att': 100 * (df_t['att'] - att_lo) / (att_hi - att_lo),
            'def': 100 * (df_t['def'] - def_lo) / (def_hi - def_lo),
        })
    return out


def _load_and_compute(pi_invert_defense):
    matches = load_matches()
    teams = sorted(set(matches.home) | set(matches.away))
    league_avg = (matches.home_goals.sum() + matches.away_goals.sum()) / (2 * len(matches))
    print(f'Matches reconstructed: {len(matches)}  |  Teams: {len(teams)}  |  league avg goals/match: {league_avg:.3f}')

    systems_raw = {
        'PI':       compute_pi(matches, teams, invert_defense=pi_invert_defense),
        'ELO':      compute_elo(matches, teams, league_avg),
        'Glicko-2': compute_glicko2(matches, teams, league_avg),
        'Base-70':  compute_base70(matches, teams, league_avg),
    }
    return systems_raw, teams, league_avg


def compute_all_systems():
    """Load matches and return (systems, teams, league_avg), normalised to a
    shared 0-100 index (higher = better for every system/dimension).

    systems: {method_name: {team: DataFrame[date, att, def]}}, covering every
    Eredivisie side (callers filter down to whichever teams they need).
    Used by top3_ratings.py, where 4 differently-scaled systems need to share
    one axis per panel.
    """
    systems_raw, teams, league_avg = _load_and_compute(pi_invert_defense=True)
    systems = {name: normalise(hist) for name, hist in systems_raw.items()}
    return systems, teams, league_avg


def compute_all_systems_native():
    """Same as compute_all_systems(), but keeps every system in its own real
    unit (goals/match, Elo points, Glicko-2 rating, Base-70 score) — smoothed
    (5-match rolling mean) but NOT rescaled onto a shared index.

    Note PI's defensive column is goals conceded per match: LOWER is better
    there, unlike every other column in every system, where higher is better.
    Used by ratings_table.py.
    """
    systems_raw, teams, league_avg = _load_and_compute(pi_invert_defense=False)
    systems = {name: smooth_native(hist) for name, hist in systems_raw.items()}
    return systems, teams, league_avg


# ── 3c. whole-season value per team (not a snapshot of recent form) ────────
# ELO/Glicko-2/Base-70 already accumulate every match played, so their
# "whole season" value is simply their rating after the last match. PI has
# no memory between matches, so its whole-season value is the mean goal
# rate across every match played, not just the most recent ones.
def season_native(hist, method):
    out = {}
    for t, series in hist.items():
        df_t = pd.DataFrame(series, columns=['date', 'att', 'def'])
        if method == 'PI':
            out[t] = (df_t['att'].mean(), df_t['def'].mean())
        else:
            out[t] = (df_t['att'].iloc[-1], df_t['def'].iloc[-1])
    return out


def compute_all_systems_native_season():
    """Whole-2025/26-season value per team/system, in native units (no
    rescaling, no recent-form windowing). Returns (systems, teams, league_avg)
    where systems: {method_name: {team: (att_value, def_value)}}.
    """
    systems_raw, teams, league_avg = _load_and_compute(pi_invert_defense=False)
    systems = {name: season_native(hist, name) for name, hist in systems_raw.items()}
    return systems, teams, league_avg
