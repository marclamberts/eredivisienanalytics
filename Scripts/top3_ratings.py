"""
Attacking & defensive rating trends for Ajax, Feyenoord and PSV — 2025/26.

Four independent rating systems are tracked match-by-match for every
Eredivisie side, then normalised onto a common 0-100 index so their shapes
are comparable on one axis:

  - PI        simple per-match goal rate (no opponent adjustment)
  - ELO       classic Elo, opponent-adjusted, unbounded
  - Glicko-2  Elo extension with rating deviation / volatility (uncertainty
              shrinks as more matches are played)
  - Base-70   bounded scale anchored on 70, opponent-unadjusted with mild
              mean-reversion (a simple "current level" gauge)

Match results are reconstructed from the shot-level Danger models (goals via
the is_goal flag), since no ready-made results table is checked into the repo.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

DANGER_CSV = 'Danger/all_eredivisie_danger_models.csv'
TEAM_SUMMARY_CSV = 'xT/xt_team_summary.csv'
OUT_PATH = 'Visuals/Rating/top3rating.png'
OUT_DIR_INDIVIDUAL = 'Visuals/Rating'

HIGHLIGHT_TEAMS = ['AFC Ajax', 'Feyenoord Rotterdam', 'PSV Eindhoven']
TEAM_FILE_SLUG = {
    'AFC Ajax':            'ajax',
    'Feyenoord Rotterdam': 'feyenoord',
    'PSV Eindhoven':       'psv',
}
MA_WINDOW = 5

# ── housestyle, light variant (mirrors Q_LIGHT_* in Scripts/coach_profiling.py) ──
Q_INK     = '#2b2b2b'
Q_MUTED   = '#6b6b6b'
Q_GRID    = '#ddd9cd'
Q_SURFACE = '#f4f1ea'

# Team identity colors from the glasbey-safe palette; Feyenoord's assigned
# lavender (#8287ff) reads fine on a dark surface but fails contrast on this
# light one (2.7:1), so it's swapped for a darker indigo of the same hue here.
TEAM_COLORS = {
    'AFC Ajax':                     '#d60000',
    'Feyenoord Rotterdam':          '#4650c9',
    'PSV Eindhoven':                '#008069',
}
TEAM_SHORT = {
    'AFC Ajax':            'Ajax',
    'Feyenoord Rotterdam': 'Feyenoord',
    'PSV Eindhoven':       'PSV',
}

# Darker, higher-contrast method palette for the light surface (the dark-theme
# pastel set fell to ~2:1 contrast on a cream background).
METHOD_COLORS = {
    'PI':        '#2f6fb0',
    'ELO':       '#a6620a',
    'Glicko-2':  '#a52a52',
    'Base-70':   '#1a7a52',
}
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
def compute_pi(matches, teams):
    hist = {t: [] for t in teams}
    for row in matches.itertuples():
        hist[row.home].append((row.date, row.home_goals, row.away_goals))
        hist[row.away].append((row.date, row.away_goals, row.home_goals))
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


# ── 3. smooth each raw trajectory, then normalise onto a shared 0-100 index ─
# (smoothing happens BEFORE normalisation so a single-match outlier — e.g. a
# 6-0 rout — can't compress an entire noisy system like PI toward the bottom
# of the shared axis; ELO/Glicko-2/Base-70 are already smooth so this barely
# moves them.)
def normalise(hist):
    smoothed = {}
    for t, series in hist.items():
        df_t = pd.DataFrame(series, columns=['date', 'att', 'def'])
        df_t['att'] = df_t['att'].rolling(MA_WINDOW, min_periods=1).mean()
        df_t['def'] = df_t['def'].rolling(MA_WINDOW, min_periods=1).mean()
        smoothed[t] = df_t

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


def draw_panel(ax, systems, team, dim, row_is_top):
    ax.set_facecolor(Q_SURFACE)
    ax.grid(axis='y', color=Q_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(50, color=Q_MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)

    for method in METHOD_ORDER:
        df_t = systems[method][team]
        ax.plot(df_t['date'], df_t[dim], color=METHOD_COLORS[method],
                 linewidth=2.2, solid_capstyle='round', label=method, zorder=3)

    # right margin + end-of-line value labels, so the current standing per
    # system is readable at a glance instead of only the trend shape
    dates_all = systems[METHOD_ORDER[0]][team]['date']
    span = (dates_all.iloc[-1] - dates_all.iloc[0])
    ax.set_xlim(dates_all.iloc[0] - span * 0.02, dates_all.iloc[-1] + span * 0.12)

    # end-of-line dots + value labels, with the labels nudged apart in y so
    # two systems finishing at a near-identical index don't print on top of
    # each other
    ends = []
    for method in METHOD_ORDER:
        df_t = systems[method][team]
        ends.append((method, df_t['date'].iloc[-1], df_t[dim].iloc[-1]))
    for method, last_x, last_y in ends:
        ax.scatter([last_x], [last_y], color=METHOD_COLORS[method], s=22, zorder=4,
                   edgecolor=Q_SURFACE, linewidth=1.0)

    min_sep = 4.0
    ends_sorted = sorted(ends, key=lambda e: e[2])
    label_ys = []
    for _, _, val in ends_sorted:
        y = val if not label_ys else max(val, label_ys[-1] + min_sep)
        label_ys.append(y)
    for (method, last_x, last_y), label_y in zip(ends_sorted, label_ys):
        ax.annotate(f'{last_y:.0f}', (last_x, label_y), xytext=(6, 0), textcoords='offset points',
                    va='center', ha='left', color=METHOD_COLORS[method], fontsize=8, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2.6, foreground=Q_SURFACE)], zorder=5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_edgecolor(Q_GRID)
    ax.tick_params(colors=Q_INK, labelsize=7.5)
    ax.set_ylim(0, 104)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(axis='x', rotation=25)


def plot_combined(systems):
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.5), facecolor=Q_SURFACE, sharex=False)
    fig.suptitle('Eredivisie 2025/26 — Attacking & Defensive Rating Trends',
                 color=Q_INK, fontsize=17, fontweight='bold', y=0.995)
    fig.text(0.5, 0.955,
             f'Top 3  ·  {MA_WINDOW}-match rolling average  ·  4 rating systems, each normalised to a 0–100 index',
             ha='center', color=Q_MUTED, fontsize=9.5)

    dim_config = [('att', 'Attacking Rating Index', 0), ('def', 'Defensive Rating Index', 1)]

    for dim, dim_label, row_i in dim_config:
        for col_i, team in enumerate(HIGHLIGHT_TEAMS):
            ax = axes[row_i, col_i]
            draw_panel(ax, systems, team, dim, row_i == 0)

            if row_i == 0:
                short = TEAM_SHORT[team]
                ax.set_title(short, color=TEAM_COLORS[team], fontsize=13, fontweight='bold', pad=10)
            if col_i == 0:
                ax.set_ylabel(dim_label, color=Q_INK, fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, facecolor=Q_SURFACE,
               edgecolor=Q_GRID, labelcolor=Q_INK, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))

    fig.text(0.02, 0.015,
             'Data: Opta (shot-event goals)  ·  ELO & Glicko-2 are opponent-adjusted, PI & Base-70 are not  ·  50 = league-average line',
             ha='left', color=Q_MUTED, fontsize=7.5)

    plt.tight_layout(rect=(0, 0.055, 1, 0.945))
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print('Saved:', OUT_PATH)


def plot_individual(systems, team):
    short = TEAM_SHORT[team]
    color = TEAM_COLORS[team]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=Q_SURFACE)
    fig.suptitle(f'{short} — Attacking & Defensive Rating Trend, 2025/26',
                 color=Q_INK, fontsize=15.5, fontweight='bold', y=1.02)
    fig.text(0.5, 0.955,
             f'{MA_WINDOW}-match rolling average  ·  4 rating systems, each normalised to a 0–100 index (vs. the full Eredivisie field)',
             ha='center', color=Q_MUTED, fontsize=9)

    dim_config = [('att', 'Attacking Rating Index'), ('def', 'Defensive Rating Index')]
    for ax, (dim, dim_label) in zip(axes, dim_config):
        draw_panel(ax, systems, team, dim, True)
        ax.set_title(dim_label, color=color, fontsize=12, fontweight='bold', pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, facecolor=Q_SURFACE,
               edgecolor=Q_GRID, labelcolor=Q_INK, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

    fig.text(0.02, -0.06,
             'Data: Opta (shot-event goals)  ·  ELO & Glicko-2 are opponent-adjusted, PI & Base-70 are not  ·  50 = league-average line',
             ha='left', color=Q_MUTED, fontsize=7.5)

    out_path = f'{OUT_DIR_INDIVIDUAL}/{TEAM_FILE_SLUG[team]}_rating.png'
    plt.tight_layout(rect=(0, 0.08, 1, 0.92))
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print('Saved:', out_path)


def main():
    matches = load_matches()
    teams = sorted(set(matches.home) | set(matches.away))
    league_avg = (matches.home_goals.sum() + matches.away_goals.sum()) / (2 * len(matches))
    print(f'Matches reconstructed: {len(matches)}  |  Teams: {len(teams)}  |  league avg goals/match: {league_avg:.3f}')

    systems_raw = {
        'PI':       compute_pi(matches, teams),
        'ELO':      compute_elo(matches, teams, league_avg),
        'Glicko-2': compute_glicko2(matches, teams, league_avg),
        'Base-70':  compute_base70(matches, teams, league_avg),
    }
    systems = {name: normalise(hist) for name, hist in systems_raw.items()}

    plot_combined(systems)
    for team in HIGHLIGHT_TEAMS:
        plot_individual(systems, team)


if __name__ == '__main__':
    main()
