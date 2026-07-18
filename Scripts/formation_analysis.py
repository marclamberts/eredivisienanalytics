"""
Formation analysis: identify 3/4/5-at-the-back per team per game,
then overlay on style radars grouped by defensive structure.
"""

import json, os, glob, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from scipy.stats import percentileofscore
import warnings
warnings.filterwarnings('ignore')

FOLDER = '/Users/marclamberts/Event data/Eredivisie'
OUTPUT_DIR = os.path.join(FOLDER, 'Analysis', 'Formation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Opta formation_id -> (name, n_defenders)
# Derived from Opta documentation and observed DEF counts
FORMATION_MAP = {
    1:  ('4-4-2',    4),
    2:  ('4-4-2 D',  4),
    3:  ('4-3-3',    4),
    4:  ('4-3-3',    4),
    5:  ('4-1-2-1-2',4),
    6:  ('3-4-3',    3),
    7:  ('4-3-2-1',  4),
    8:  ('4-2-3-1',  4),
    9:  ('4-1-4-1',  4),
    10: ('3-5-2',    3),
    11: ('5-4-1',    5),
    12: ('5-3-2',    5),
    13: ('3-4-3',    3),
    14: ('3-4-1-2',  3),
    15: ('4-4-2',    4),
    16: ('4-3-2-1',  4),
    17: ('3-4-3',    3),
    18: ('4-2-2-2',  4),
    19: ('5-3-2',    5),
    20: ('4-1-3-2',  4),
    21: ('3-3-3-1',  3),
    22: ('4-2-4',    4),
    23: ('3-4-2-1',  3),
    24: ('3-5-2 WB', 5),  # wing-backs as DEF
    25: ('5-2-3',    5),
}

TEAM_COLORS = {
    'AFC Ajax':                    '#DC0714',
    'Alkmaar Zaanstreek':          '#E60012',
    'FC Groningen':                '#00943A',
    'FC Twente':                   '#CC0000',
    'FC Utrecht':                  '#E2001A',
    'FC Volendam':                 '#FF6600',
    'Feyenoord Rotterdam':         '#CC0000',
    'Fortuna Sittard':             '#FFCC00',
    'Go Ahead Eagles':             '#1B6B2F',
    'Heracles Almelo':             '#555555',
    'NAC Breda':                   '#D4AF37',
    'Nijmegen Eendracht Combinatie': '#CC0000',
    'PEC Zwolle':                  '#003DA5',
    'PSV Eindhoven':               '#FF0000',
    'SBV Excelsior':               '#CC0000',
    'SC Heerenveen':               '#003DA5',
    'SC Telstar':                  '#CC2200',
    'Sparta Rotterdam':            '#CC0000',
}

TEAM_SHORT = {
    'AFC Ajax': 'Ajax',
    'Alkmaar Zaanstreek': 'AZ',
    'FC Groningen': 'Groningen',
    'FC Twente': 'Twente',
    'FC Utrecht': 'Utrecht',
    'FC Volendam': 'Volendam',
    'Feyenoord Rotterdam': 'Feyenoord',
    'Fortuna Sittard': 'Fortuna',
    'Go Ahead Eagles': 'Go Ahead',
    'Heracles Almelo': 'Heracles',
    'NAC Breda': 'NAC',
    'Nijmegen Eendracht Combinatie': 'NEC',
    'PEC Zwolle': 'PEC',
    'PSV Eindhoven': 'PSV',
    'SBV Excelsior': 'Excelsior',
    'SC Heerenveen': 'Heerenveen',
    'SC Telstar': 'Telstar',
    'Sparta Rotterdam': 'Sparta',
}

BACK_LINE_COLORS = {3: '#FF6B35', 4: '#4FC3F7', 5: '#A8E063'}
BACK_LINE_LABELS = {3: '3 at the back', 4: '4 at the back', 5: '5 at the back'}

METRICS_CONFIG = [
    ('territory',      'Territory',     True),
    ('long_ball_pct',  'Long Balls',    False),
    ('deep_circ_pct',  'Deep Circ.',    True),
    ('wing_pct',       'Wing Play',     True),
    ('cross_pct',      'Crossing',      True),
    ('ppda',           'High Press',    False),
    ('counters_per90', 'Counters',      True),
    ('low_block',      'Low Block',     True),
    ('gk_short_pct',   'GK Build Up',  True),
    ('sp_shots_pct',   'Set Pieces',   True),
]


# ── Data extraction ────────────────────────────────────────────────────────────

def has_qual(event, qid):
    return any(q['qualifierId'] == qid for q in event.get('qualifier', []))

def get_qual(event, qid, default=None):
    for q in event.get('qualifier', []):
        if q['qualifierId'] == qid:
            return q.get('value', default)
    return default

def get_attack_direction(events, team_id):
    direction = {}
    for period in [1, 2]:
        passes = [e for e in events
                  if e['typeId'] == 1 and e['contestantId'] == team_id
                  and e['periodId'] == period
                  and not has_qual(e, 72) and not has_qual(e, 6) and not has_qual(e, 74)]
        direction[period] = 1 if (not passes or np.mean([p['x'] for p in passes]) < 50) else -1
    return direction

def norm_x(x, period, direction):
    return x if direction.get(period, 1) == 1 else 100 - x


def extract_lineup_formation(events, team_id):
    """Extract formation info from lineup event (typeId=34)."""
    lineup_events = [e for e in events if e['typeId'] == 34 and e['contestantId'] == team_id]
    if not lineup_events:
        return None, None, 4

    e = lineup_events[0]
    form_id_str = get_qual(e, 130)
    form_id = int(form_id_str) if form_id_str else None

    pos_str  = get_qual(e, 44, '')
    slot_str = get_qual(e, 131, '')

    if pos_str and slot_str:
        positions = [int(x) for x in pos_str.split(',') if x.strip()]
        slots     = [int(x) for x in slot_str.split(',') if x.strip()]
        starter_pos = [positions[i] for i, s in enumerate(slots) if s != 0][:11]
        n_def = sum(1 for p in starter_pos if p == 2)
    else:
        n_def = 4

    # Map n_def to back-line category
    # If form_id known, trust its n_defenders, else use n_def
    if form_id and form_id in FORMATION_MAP:
        form_name, back_line = FORMATION_MAP[form_id]
    else:
        back_line = n_def
        form_name = f'?-{n_def}def'

    return form_id, form_name, back_line


def process_match(filepath):
    with open(filepath) as f:
        data = json.load(f)

    match_details = data['matchDetails']
    events = data['event']
    basename = os.path.basename(filepath).replace('.json', '')
    date_str, teams_str = basename.split('_', 1)
    home_name, away_name = teams_str.split(' - ')

    contestant_order = []
    for e in events:
        cid = e.get('contestantId')
        if cid and cid not in contestant_order:
            contestant_order.append(cid)
    if len(contestant_order) < 2:
        return []

    home_id, away_id = contestant_order[0], contestant_order[1]
    team_map = {home_id: home_name, away_id: away_name}
    total_min = match_details.get('matchLengthMin', 90)

    results = []
    for team_id, team_name in team_map.items():
        opp_id = away_id if team_id == home_id else home_id
        direction     = get_attack_direction(events, team_id)
        opp_direction = get_attack_direction(events, opp_id)

        form_id, form_name, back_line = extract_lineup_formation(events, team_id)

        # Passes
        all_passes = [e for e in events if e['typeId'] == 1 and e['contestantId'] == team_id]
        open_play_passes = [p for p in all_passes
                            if not has_qual(p, 5) and not has_qual(p, 6)
                            and not has_qual(p, 72) and not has_qual(p, 74)]
        n_op = len(open_play_passes)

        long_ball_pct = len([p for p in open_play_passes if has_qual(p, 107)]) / n_op if n_op else 0
        deep_circ_pct = len([p for p in open_play_passes
                              if norm_x(p['x'], p['periodId'], direction) < 33]) / n_op if n_op else 0
        wing_pct      = len([p for p in open_play_passes
                              if p['y'] < 25 or p['y'] > 75]) / n_op if n_op else 0
        territory     = np.mean([norm_x(p['x'], p['periodId'], direction)
                                 for p in open_play_passes]) if n_op else 50
        cross_pct     = len([p for p in open_play_passes if has_qual(p, 2)]) / n_op if n_op else 0

        # PPDA
        opp_passes_low = [e for e in events if e['typeId'] == 1 and e['contestantId'] == opp_id
                          and norm_x(e['x'], e['periodId'], opp_direction) < 50]
        def_high = [e for e in events if e['contestantId'] == team_id
                    and e['typeId'] in [7, 8, 4]
                    and norm_x(e['x'], e['periodId'], direction) > 50]
        ppda = len(opp_passes_low) / len(def_high) if def_high else 999

        # Low block
        def_all = [e for e in events if e['contestantId'] == team_id
                   and e['typeId'] in [7, 8, 4, 12]]
        def_own = [e for e in def_all if norm_x(e['x'], e['periodId'], direction) < 33]
        low_block = len(def_own) / len(def_all) if def_all else 0

        # Counters
        counter_passes = []
        for p in open_play_passes:
            nx = norm_x(p['x'], p['periodId'], direction)
            end_x_val = get_qual(p, 141)
            if end_x_val and nx < 40:
                try:
                    end_x = float(end_x_val)
                    end_nx = end_x if direction.get(p['periodId'], 1) == 1 else 100 - end_x
                    if end_nx > 66:
                        counter_passes.append(p)
                except:
                    pass
        counters_per90 = len(counter_passes) / (total_min / 90)

        # GK build up
        gk_passes = [p for p in all_passes if has_qual(p, 72)]
        gk_short  = [p for p in gk_passes if (lambda v: float(v) <= 25 if v else False)(get_qual(p, 212))]
        gk_short_pct     = len(gk_short) / len(gk_passes) if gk_passes else 0
        gk_passes_per90  = len(gk_passes) / (total_min / 90)

        # Set pieces
        sp_passes = [p for p in all_passes if has_qual(p, 5) or has_qual(p, 6)]
        shots = [e for e in events if e['contestantId'] == team_id
                 and e['typeId'] in [13, 15, 16, 65]]
        sp_shots = 0
        for shot in shots:
            st = shot['timeMin'] * 60 + shot['timeSec']
            for sp in sp_passes:
                if sp['periodId'] == shot['periodId']:
                    spt = sp['timeMin'] * 60 + sp['timeSec']
                    if 0 <= st - spt <= 5:
                        sp_shots += 1
                        break
        sp_shots_pct = sp_shots / len(shots) if shots else 0

        results.append({
            'team': team_name, 'date': date_str,
            'form_id': form_id, 'form_name': form_name, 'back_line': back_line,
            'long_ball_pct': long_ball_pct, 'deep_circ_pct': deep_circ_pct,
            'wing_pct': wing_pct, 'territory': territory, 'cross_pct': cross_pct,
            'ppda': ppda, 'low_block': low_block, 'counters_per90': counters_per90,
            'gk_short_pct': gk_short_pct, 'gk_passes_per90': gk_passes_per90,
            'sp_pct': len(sp_passes) / len(all_passes) if all_passes else 0,
            'sp_shots_pct': sp_shots_pct,
            'total_min': total_min,
        })
    return results


def load_all():
    files = sorted(glob.glob(os.path.join(FOLDER, '*.json')))
    print(f'Processing {len(files)} files...')
    rows = []
    for i, f in enumerate(files):
        try:
            rows.extend(process_match(f))
        except Exception as ex:
            print(f'  Error {os.path.basename(f)}: {ex}')
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(files)}')
    print(f'Done. {len(rows)} rows.')
    return pd.DataFrame(rows)


# ── Aggregation ────────────────────────────────────────────────────────────────

METRIC_COLS = [c for c, _, _ in METRICS_CONFIG]

def team_agg(df):
    agg = df.groupby('team')[METRIC_COLS].mean().reset_index()
    agg['n_games'] = df.groupby('team').size().values

    # Dominant back-line per team (most common)
    bl = df.groupby('team')['back_line'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    bl.columns = ['team', 'dominant_back_line']
    agg = agg.merge(bl, on='team')

    # Formation breakdown per team
    form_counts = df.groupby(['team', 'form_name']).size().reset_index(name='n')
    top_form = form_counts.sort_values('n', ascending=False).groupby('team').first().reset_index()
    top_form = top_form.rename(columns={'form_name': 'top_formation'})
    agg = agg.merge(top_form[['team', 'top_formation']], on='team')

    return agg


def backline_group_agg(df):
    """Average metrics per back-line category (3/4/5)."""
    return df.groupby('back_line')[METRIC_COLS].mean().reset_index()


# ── Helpers ────────────────────────────────────────────────────────────────────

def pct_rank(series, value, hib=True):
    p = percentileofscore(series, value)
    return p if hib else 100 - p


def make_radar(ax, values, labels, color, title, fill_alpha=0.2, lw=2):
    N = len(values)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals   = values + [values[0]]
    angs   = angles + [angles[0]]
    ax.set_facecolor('#0e1117')
    ax.plot(angs, vals, color=color, linewidth=lw)
    ax.fill(angs, vals, color=color, alpha=fill_alpha)
    ax.set_ylim(0, 100)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=7, color='white')
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#444')
    ax.grid(color='#333', linewidth=0.5)
    ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=8)


# ── Plot 1: Formation frequency heatmap ───────────────────────────────────────

def plot_formation_heatmap(df, path):
    form_team = df.groupby(['team', 'form_name']).size().unstack(fill_value=0)
    form_team = form_team.reindex(sorted(form_team.index,
                                         key=lambda t: TEAM_SHORT.get(t, t)))
    # Order columns by total usage
    col_order = form_team.sum().sort_values(ascending=False).index
    form_team = form_team[col_order]

    fig, ax = plt.subplots(figsize=(max(12, len(col_order)*1.4), 9), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    fig.suptitle('Eredivisie 2025/26 — Formation Usage per Team',
                 color='white', fontsize=14, fontweight='bold')

    teams_short = [TEAM_SHORT.get(t, t) for t in form_team.index]
    im = ax.imshow(form_team.values, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(len(form_team.columns)))
    ax.set_xticklabels(form_team.columns, color='white', fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(teams_short)))
    ax.set_yticklabels(teams_short, color='white', fontsize=9)

    for i in range(len(form_team.index)):
        for j in range(len(form_team.columns)):
            v = form_team.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha='center', va='center',
                        color='black' if v > form_team.values.max()*0.5 else 'white',
                        fontsize=7.5, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Games', color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'Saved: {path}')


# ── Plot 2: Back-line distribution per team ───────────────────────────────────

def plot_backline_bars(df, path):
    bl_counts = (df.groupby(['team', 'back_line'])
                   .size().unstack(fill_value=0)
                   .reindex(columns=[3, 4, 5], fill_value=0))
    bl_pct = bl_counts.div(bl_counts.sum(axis=1), axis=0) * 100
    bl_pct = bl_pct.sort_values([3, 5, 4], ascending=False)
    teams_short = [TEAM_SHORT.get(t, t) for t in bl_pct.index]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    fig.suptitle('Eredivisie 2025/26 — Defensive Structure Usage (%)',
                 color='white', fontsize=14, fontweight='bold')

    bottom = np.zeros(len(bl_pct))
    for back_n in [3, 4, 5]:
        if back_n in bl_pct.columns:
            vals = bl_pct[back_n].values
            ax.barh(teams_short, vals, left=bottom,
                    color=BACK_LINE_COLORS[back_n], label=BACK_LINE_LABELS[back_n],
                    height=0.7)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 3:
                    ax.text(b + v / 2, i, f'{v:.0f}%',
                            ha='center', va='center', color='black', fontsize=7.5, fontweight='bold')
            bottom += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel('% of games', color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.legend(loc='lower right', fontsize=8, facecolor='#1a1a2e', labelcolor='white',
              edgecolor='#444')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'Saved: {path}')


# ── Plot 3: Grouped radars (3/4/5 at back) ───────────────────────────────────

def plot_grouped_radars(agg, path):
    """One radar per back-line group showing average percentile profiles."""
    labels = [lbl for _, lbl, _ in METRICS_CONFIG]

    fig = plt.figure(figsize=(18, 7), facecolor='#0e1117')
    fig.suptitle('Eredivisie 2025/26 — Style Profile by Defensive Structure',
                 color='white', fontsize=15, fontweight='bold', y=1.02)

    for idx, bl in enumerate([3, 4, 5]):
        subset = agg[agg['dominant_back_line'] == bl]
        if subset.empty:
            continue
        ax = fig.add_subplot(1, 3, idx + 1, polar=True)
        color = BACK_LINE_COLORS[bl]
        label = BACK_LINE_LABELS[bl]
        teams_in = ', '.join(TEAM_SHORT.get(t, t) for t in subset['team'])

        # Compute percentile ranks vs whole league
        percentiles = []
        for col, _, hib in METRICS_CONFIG:
            avg_val = subset[col].mean()
            p = pct_rank(agg[col], avg_val, hib=hib)
            percentiles.append(p)

        make_radar(ax, percentiles, labels, color,
                   f'{label}\n({len(subset)} teams)', fill_alpha=0.3, lw=2.5)

        # Add team names below
        fig.text(0.17 + idx * 0.33, -0.01, teams_in,
                 ha='center', color='#aaa', fontsize=6.5, wrap=True)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'Saved: {path}')


# ── Plot 4: All-team radars coloured by back-line ─────────────────────────────

def plot_all_radars_by_backline(agg, path):
    teams = sorted(agg['team'].unique())
    n = len(teams)
    ncols, nrows = 6, math.ceil(n / 6)
    labels = [lbl for _, lbl, _ in METRICS_CONFIG]

    fig = plt.figure(figsize=(ncols * 4.2, nrows * 4.4), facecolor='#0e1117')
    fig.suptitle('Eredivisie 2025/26 — Team Style Radars  (coloured by defensive structure)',
                 color='white', fontsize=16, fontweight='bold', y=1.01)

    for i, team in enumerate(teams):
        row_data = agg[agg['team'] == team].iloc[0]
        bl = row_data['dominant_back_line']
        color = BACK_LINE_COLORS.get(bl, '#888')
        short = TEAM_SHORT.get(team, team)

        percentiles = [pct_rank(agg[col], row_data[col], hib=hib)
                       for col, _, hib in METRICS_CONFIG]

        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        make_radar(ax, percentiles, labels, color,
                   f'{short}\n({BACK_LINE_LABELS.get(bl, f"{bl}b")} | {row_data["top_formation"]})')

    # Legend
    handles = [mpatches.Patch(color=BACK_LINE_COLORS[b], label=BACK_LINE_LABELS[b])
               for b in [3, 4, 5]]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               facecolor='#1a1a2e', labelcolor='white', edgecolor='#444',
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'Saved: {path}')


# ── Plot 5: Metric comparison across back-line groups (bar chart) ─────────────

def plot_metric_comparison(df, path):
    """For each metric, show average per back-line group."""
    metric_labels = {col: lbl for col, lbl, _ in METRICS_CONFIG}
    metrics_to_show = [c for c, _, _ in METRICS_CONFIG]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), facecolor='#0e1117')
    fig.suptitle('Eredivisie 2025/26 — Style Metrics by Defensive Structure',
                 color='white', fontsize=14, fontweight='bold')

    for ax, col in zip(axes.flatten(), metrics_to_show):
        ax.set_facecolor('#0e1117')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')

        for bl in [3, 4, 5]:
            grp = df[df['back_line'] == bl][col]
            mean, sem = grp.mean(), grp.sem()
            if col == 'ppda':
                mean_disp = mean
            elif col in ('territory', 'counters_per90', 'gk_passes_per90'):
                mean_disp = mean
            else:
                mean_disp = mean * 100

            ax.bar(str(bl), mean_disp, color=BACK_LINE_COLORS[bl],
                   width=0.6, edgecolor='none',
                   yerr=sem * (100 if col not in ('ppda', 'territory', 'counters_per90', 'gk_passes_per90') else 1),
                   capsize=4, error_kw={'color': 'white', 'linewidth': 1})
            ax.text(str(bl), mean_disp + sem * (100 if col not in ('ppda', 'territory', 'counters_per90', 'gk_passes_per90') else 1) + 0.2,
                    f'{mean_disp:.1f}', ha='center', va='bottom', color='white', fontsize=7)

        ax.set_title(metric_labels.get(col, col), color='white', fontsize=8.5, fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('Defenders', color='#aaa', fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'Saved: {path}')


# ── Plot 6: Per-team radar with back-line overlay ─────────────────────────────

def plot_team_radar_with_backline(agg, df, team_name, path):
    """Single team radar showing their style vs back-line group average."""
    row = agg[agg['team'] == team_name].iloc[0]
    bl  = row['dominant_back_line']
    bl_group = agg[agg['dominant_back_line'] == bl]
    color_team = TEAM_COLORS.get(team_name, '#4FC3F7')
    color_group = BACK_LINE_COLORS.get(bl, '#888')
    short = TEAM_SHORT.get(team_name, team_name)
    labels = [lbl for _, lbl, _ in METRICS_CONFIG]

    pcts_team  = [pct_rank(agg[col], row[col], hib=hib) for col, _, hib in METRICS_CONFIG]
    pcts_group = [pct_rank(agg[col], bl_group[col].mean(), hib=hib) for col, _, hib in METRICS_CONFIG]

    fig = plt.figure(figsize=(7, 6.5), facecolor='#0e1117')
    ax = fig.add_subplot(111, polar=True)
    make_radar(ax, pcts_group, labels, color_group,
               f'{short} — Style vs {BACK_LINE_LABELS.get(bl,"?")} Average\n({row["top_formation"]})',
               fill_alpha=0.12, lw=1.5)
    make_radar(ax, pcts_team, labels, color_team, '', fill_alpha=0.25, lw=2.5)

    handles = [
        mpatches.Patch(color=color_team,  label=f'{short}'),
        mpatches.Patch(color=color_group, label=f'{BACK_LINE_LABELS.get(bl,"?")} avg'),
    ]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.35, 1.1),
              facecolor='#1a1a2e', labelcolor='white', edgecolor='#444', fontsize=8)
    fig.text(0.5, 0.02, 'Percentile ranks vs Eredivisie 25/26 | darker = group avg',
             ha='center', color='#888', fontsize=6.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = load_all()
    df.to_csv(os.path.join(OUTPUT_DIR, 'formation_metrics_raw.csv'), index=False)

    agg = team_agg(df)
    agg.to_csv(os.path.join(OUTPUT_DIR, 'formation_metrics_agg.csv'), index=False)

    # Print summary
    print('\n── Team Formation Summary ──')
    summary = agg[['team', 'dominant_back_line', 'top_formation', 'n_games']].copy()
    summary['team_short'] = summary['team'].map(TEAM_SHORT)
    summary = summary.sort_values(['dominant_back_line', 'team_short'])
    print(summary[['team_short', 'dominant_back_line', 'top_formation', 'n_games']].to_string(index=False))

    # Back-line group counts
    print('\n── Back-line group distribution (game-level) ──')
    print(df['back_line'].value_counts().sort_index())

    # Formation frequency per team
    print('\n── Formation frequency ──')
    fc = df.groupby(['team', 'form_name']).size().reset_index(name='n')
    fc['team_short'] = fc['team'].map(TEAM_SHORT)
    print(fc.sort_values(['team_short', 'n'], ascending=[True, False]).to_string(index=False))

    # Plots
    plot_formation_heatmap(df, os.path.join(OUTPUT_DIR, 'formation_heatmap.png'))
    plot_backline_bars(df, os.path.join(OUTPUT_DIR, 'backline_distribution.png'))
    plot_grouped_radars(agg, os.path.join(OUTPUT_DIR, 'grouped_radars_by_backline.png'))
    plot_all_radars_by_backline(agg, os.path.join(OUTPUT_DIR, 'all_radars_coloured_by_backline.png'))
    plot_metric_comparison(df, os.path.join(OUTPUT_DIR, 'metric_comparison_by_backline.png'))

    # Individual team radars vs group
    for team in agg['team']:
        short = TEAM_SHORT.get(team, team).replace(' ', '_')
        plot_team_radar_with_backline(
            agg, df, team,
            os.path.join(OUTPUT_DIR, f'team_radar_{short}.png')
        )
    print('All individual radars saved.')
    print('Done!')


if __name__ == '__main__':
    main()
