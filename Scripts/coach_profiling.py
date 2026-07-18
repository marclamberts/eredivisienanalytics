"""
Eredivisie Team Profiling - inspired by analyticsfc.co.uk/blog/2021/03/22/profiling-coaches-with-data/
Metrics: Long Balls, Deep Circulation, Wing Play, Territory, Crossing, High Press,
         Counters, Low Block, GK Build Up, Set Pieces
"""

import json
import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

FOLDER = '/Users/marclamberts/Event data/Eredivisie'
OUTPUT_DIR = os.path.join(FOLDER, 'Analysis', 'Coach Profiling')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Each radar style gets its own subfolder; dashboards and raw CSVs stay at the top level.
SPIDER_DIR = os.path.join(OUTPUT_DIR, 'Spider Radar')
RADIAL_BAR_DIR = os.path.join(OUTPUT_DIR, 'Radial Bar Radar')
RADIAL_HEATMAP_DIR = os.path.join(OUTPUT_DIR, 'Radial Heatmap Radar')
STYLE_SIMILARITY_DIR = os.path.join(OUTPUT_DIR, 'Style Similarity Map')
for _d in (SPIDER_DIR, RADIAL_BAR_DIR, RADIAL_HEATMAP_DIR, STYLE_SIMILARITY_DIR):
    os.makedirs(_d, exist_ok=True)

# Opta qualifier IDs
Q_LONG_BALL   = 107
Q_CROSS       = 2
Q_HEAD        = 1
Q_THROUGH     = 3
Q_FREE_KICK   = 5
Q_CORNER      = 6
Q_GOAL_KICK   = 72
Q_KICK_OFF    = 74
Q_INSWING     = 123
Q_OUTSWING    = 124
Q_PASS_LEN    = 212   # metres
Q_END_X       = 141
Q_END_Y       = 140

# Opta event type IDs
T_PASS        = 1
T_TAKE_ON     = 3
T_FOUL        = 4
T_TACKLE      = 7
T_INTERCEPTION= 8
T_SAVE        = 10
T_CLEARANCE   = 12
T_SHOT_MISS   = 13
T_SHOT_POST   = 15
T_SHOT_SAVED  = 16
T_SHOT_OG     = 17
T_YELLOW      = 18
T_RED         = 19
T_PERIOD_START= 27
T_PERIOD_END  = 28
T_SUB         = 40
T_BALL_TOUCH  = 43
T_RECOVERY    = 49
T_GOAL        = 65


def has_qual(event, qid):
    return any(q['qualifierId'] == qid for q in event.get('qualifier', []))

def get_qual(event, qid, default=None):
    for q in event.get('qualifier', []):
        if q['qualifierId'] == qid:
            return q.get('value', default)
    return default


def get_attack_direction(events, team_id):
    """Return {period: 1 or -1}, 1 means team attacks toward higher x (normal)."""
    direction = {}
    for period in [1, 2]:
        passes = [e for e in events
                  if e['typeId'] == T_PASS
                  and e['contestantId'] == team_id
                  and e['periodId'] == period
                  and not has_qual(e, Q_GOAL_KICK)
                  and not has_qual(e, Q_CORNER)
                  and not has_qual(e, Q_KICK_OFF)]
        if passes:
            avg_x = np.mean([p['x'] for p in passes])
            direction[period] = 1 if avg_x < 50 else -1
        else:
            direction[period] = 1
    return direction


def norm_x(x, period, direction):
    """Normalize x so that 0=own goal, 100=opponent goal."""
    d = direction.get(period, 1)
    return x if d == 1 else 100 - x


def process_match(filepath):
    """Extract per-team metrics from a single match file."""
    with open(filepath) as f:
        data = json.load(f)

    match_details = data['matchDetails']
    events = data['event']

    basename = os.path.basename(filepath).replace('.json', '')
    date_str, teams_str = basename.split('_', 1)
    home_name, away_name = teams_str.split(' - ')

    # Get contestant IDs from start events (first two unique contestant IDs)
    contestant_order = []
    for e in events:
        cid = e.get('contestantId')
        if cid and cid not in contestant_order:
            contestant_order.append(cid)

    if len(contestant_order) < 2:
        return []

    home_id, away_id = contestant_order[0], contestant_order[1]
    team_map = {home_id: home_name, away_id: away_name}

    # Match duration (minutes)
    total_min = match_details.get('matchLengthMin', 90)

    results = []
    for team_id, team_name in team_map.items():
        opp_id = away_id if team_id == home_id else home_id

        direction = get_attack_direction(events, team_id)
        opp_direction = get_attack_direction(events, opp_id)

        # --- PASSES ---
        all_passes = [e for e in events if e['typeId'] == T_PASS and e['contestantId'] == team_id]
        open_play_passes = [p for p in all_passes
                            if not has_qual(p, Q_FREE_KICK)
                            and not has_qual(p, Q_CORNER)
                            and not has_qual(p, Q_GOAL_KICK)
                            and not has_qual(p, Q_KICK_OFF)]

        n_passes = len(all_passes)
        n_op_passes = len(open_play_passes)

        # 1. Long Balls
        long_ball_passes = [p for p in open_play_passes if has_qual(p, Q_LONG_BALL)]
        long_ball_pct = len(long_ball_passes) / n_op_passes if n_op_passes > 0 else 0

        # 2. Deep Circulation - own-third passes (normalized x < 33)
        deep_passes = [p for p in open_play_passes
                       if norm_x(p['x'], p['periodId'], direction) < 33]
        deep_circ_pct = len(deep_passes) / n_op_passes if n_op_passes > 0 else 0

        # 3. Wing Play - passes in wide areas (y < 25 or y > 75)
        wing_passes = [p for p in open_play_passes if p['y'] < 25 or p['y'] > 75]
        wing_pct = len(wing_passes) / n_op_passes if n_op_passes > 0 else 0

        # 4. Territory - average normalized x position of open play passes
        if n_op_passes > 0:
            territory = np.mean([norm_x(p['x'], p['periodId'], direction) for p in open_play_passes])
        else:
            territory = 50

        # 5. Crossing
        crosses = [p for p in open_play_passes if has_qual(p, Q_CROSS)]
        cross_pct = len(crosses) / n_op_passes if n_op_passes > 0 else 0

        # 6. High Press (PPDA variant)
        # Opponent passes in own+mid third (opp normalized x < 67) / defensive actions in opp half
        opp_passes_low = [e for e in events
                          if e['typeId'] == T_PASS
                          and e['contestantId'] == opp_id
                          and norm_x(e['x'], e['periodId'], opp_direction) < 50]
        # Defensive actions by this team in opponent's half (normalized x > 50)
        def_actions_high = [e for e in events
                            if e['contestantId'] == team_id
                            and e['typeId'] in [T_TACKLE, T_INTERCEPTION, T_FOUL]
                            and norm_x(e['x'], e['periodId'], direction) > 50]
        ppda = len(opp_passes_low) / len(def_actions_high) if len(def_actions_high) > 0 else 999

        # 7. Low Block - % of defensive actions in own third
        def_actions_all = [e for e in events
                           if e['contestantId'] == team_id
                           and e['typeId'] in [T_TACKLE, T_INTERCEPTION, T_FOUL, T_CLEARANCE]]
        def_actions_own = [e for e in def_actions_all
                           if norm_x(e['x'], e['periodId'], direction) < 33]
        low_block = len(def_actions_own) / len(def_actions_all) if len(def_actions_all) > 0 else 0

        # 8. Counters - open play attacks that start in own half and reach final third within 10 seconds
        # Proxy: recoveries in own half followed by shots or final-third passes quickly
        # Simple proxy: passes where start is own half and end is final third within same sequence
        counter_passes = []
        for p in open_play_passes:
            nx = norm_x(p['x'], p['periodId'], direction)
            end_x_val = get_qual(p, Q_END_X)
            if end_x_val and nx < 40:
                try:
                    end_x = float(end_x_val)
                    end_nx = end_x if direction.get(p['periodId'], 1) == 1 else 100 - end_x
                    if end_nx > 66:
                        counter_passes.append(p)
                except:
                    pass
        counters_per90 = len(counter_passes) / (total_min / 90)

        # 9. GK Build Up - goal kick passes, % that are short (<= 25m)
        gk_passes = [p for p in all_passes if has_qual(p, Q_GOAL_KICK)]
        gk_short = []
        for p in gk_passes:
            length = get_qual(p, Q_PASS_LEN)
            if length:
                try:
                    if float(length) <= 25:
                        gk_short.append(p)
                except:
                    pass
        gk_short_pct = len(gk_short) / len(gk_passes) if len(gk_passes) > 0 else 0
        gk_passes_per90 = len(gk_passes) / (total_min / 90)

        # 10. Set Pieces - attacks from set pieces (FK + corner passes leading to shots)
        sp_passes = [p for p in all_passes
                     if has_qual(p, Q_FREE_KICK) or has_qual(p, Q_CORNER)]
        sp_pct = len(sp_passes) / n_passes if n_passes > 0 else 0

        # Set piece shots: shots where preceding event was a set piece pass within 5 seconds
        shots = [e for e in events
                 if e['contestantId'] == team_id
                 and e['typeId'] in [T_SHOT_MISS, T_SHOT_POST, T_SHOT_SAVED, T_GOAL]]
        sp_shots = 0
        sp_pass_ids = set(p.get('eventId') for p in sp_passes)
        for shot in shots:
            shot_time = shot['timeMin'] * 60 + shot['timeSec']
            shot_period = shot['periodId']
            # Look for a set piece pass from same team in same period within last 5 seconds
            for sp in sp_passes:
                if sp['periodId'] == shot_period:
                    sp_time = sp['timeMin'] * 60 + sp['timeSec']
                    if 0 <= shot_time - sp_time <= 5:
                        sp_shots += 1
                        break

        sp_shots_pct = sp_shots / len(shots) if len(shots) > 0 else 0

        results.append({
            'team': team_name,
            'date': date_str,
            'is_home': team_id == home_id,
            'total_min': total_min,
            'n_passes': n_passes,
            'long_ball_pct': long_ball_pct,
            'deep_circ_pct': deep_circ_pct,
            'wing_pct': wing_pct,
            'territory': territory,
            'cross_pct': cross_pct,
            'ppda': ppda,
            'low_block': low_block,
            'counters_per90': counters_per90,
            'gk_short_pct': gk_short_pct,
            'gk_passes_per90': gk_passes_per90,
            'sp_pct': sp_pct,
            'sp_shots_pct': sp_shots_pct,
        })

    return results


def load_all_data():
    files = sorted(glob.glob(os.path.join(FOLDER, '*.json')))
    print(f'Processing {len(files)} match files...')
    all_rows = []
    for i, f in enumerate(files):
        try:
            rows = process_match(f)
            all_rows.extend(rows)
        except Exception as ex:
            print(f'  Error in {os.path.basename(f)}: {ex}')
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(files)} done')

    print(f'Total rows: {len(all_rows)}')
    return pd.DataFrame(all_rows)


def aggregate_by_team(df):
    metrics = [
        'long_ball_pct', 'deep_circ_pct', 'wing_pct', 'territory',
        'cross_pct', 'ppda', 'low_block', 'counters_per90',
        'gk_short_pct', 'gk_passes_per90', 'sp_pct', 'sp_shots_pct'
    ]
    # For ppda, lower = more pressing, so we'll invert it later for visualization
    agg = df.groupby('team')[metrics].mean().reset_index()
    agg['n_games'] = df.groupby('team').size().values
    return agg


# ── Visualization ──────────────────────────────────────────────────────────────

Q_INK        = '#c3c2b7'   # secondary ink (labels, axes)
Q_MUTED      = '#6a6f7b'   # muted gridlines / reference lines
Q_GRID       = '#23262e'   # recessive gridlines
Q_SURFACE    = '#0e1117'   # chart surface (dark)

# Fixed, colorblind-checked 18-color categorical palette (glasbey_dark,
# max hue separation, all slots >=3.3:1 contrast on Q_SURFACE). Assigned in a
# fixed order (alphabetical by team) so a team keeps the same color everywhere.
TEAM_COLORS = {
    'AFC Ajax':                     '#d60000',
    'Alkmaar Zaanstreek':           '#03dae6',
    'FC Groningen':                 '#8956af',
    'FC Twente':                    '#6b9700',
    'FC Utrecht':                   '#2db152',
    'FC Volendam':                  '#ff7ed1',
    'Feyenoord Rotterdam':          '#8287ff',
    'Fortuna Sittard':              '#e6a500',
    'Go Ahead Eagles':              '#0774d8',
    'Heracles Almelo':              '#eb4d00',
    'NAC Breda':                    '#bf03b8',
    'Nijmegen Eendracht Combinatie': '#15e18c',
    'PEC Zwolle':                   '#018700',
    'PSV Eindhoven':                '#008069',
    'SBV Excelsior':                '#c4668e',
    'SC Heerenveen':                '#ff6989',
    'SC Telstar':                   '#e252ff',
    'Sparta Rotterdam':             '#c89a69',
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


# (column, label, higher_is_better) — shared by every radar/radial-bar chart
RADAR_METRICS_CONFIG = [
    ('territory',      'Territory',     True),
    ('long_ball_pct',  'Long Balls',    False),
    ('deep_circ_pct',  'Deep Circ.',    True),
    ('wing_pct',       'Wing Play',     True),
    ('cross_pct',      'Crossing',      True),
    ('ppda',           'High Press',    False),  # lower PPDA = more pressing
    ('counters_per90', 'Counters',      True),
    ('low_block',      'Low Block',     True),
    ('gk_short_pct',   'GK Build Up',   True),
    ('sp_shots_pct',   'Set Pieces',    True),
]

# Diverging heatmap for percentile bars: below league average -> at average -> above.
_DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'below_above_avg', ['#3987e5', '#383835', '#e66767'], N=256)


def percentile_color(value):
    return _DIVERGING_CMAP(np.clip(value, 0, 100) / 100.0)


# --- Light-theme "radial heatmap" gauge style (equal-size wedges; a sequential
# yellow->red gauge coloring fills each wedge up to its percentile, the rest
# left blank) ------------------------------------------------------------------
Q_LIGHT_SURFACE = '#f4f1ea'
Q_LIGHT_INK     = '#2b2b2b'
Q_LIGHT_MUTED   = '#6b6b6b'
Q_LIGHT_GRID    = '#ddd9cd'
_HEATMAP_CMAP = plt.get_cmap('YlOrRd')


def percentile_rank(series, value, higher_is_better=True):
    """Return percentile rank 0-100."""
    pct = percentileofscore(series, value)
    if not higher_is_better:
        pct = 100 - pct
    return pct


def make_radar(ax, values, labels, color, title, fill_alpha=0.32, show_values=False, title_size=9.5):
    N = len(values)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    ax.set_facecolor(Q_SURFACE)

    # League-average reference ring — every metric is a percentile rank, so the
    # 50-mark is the league average for all axes simultaneously.
    ax.plot(angles_plot, [50] * (N + 1), color=Q_MUTED, linewidth=1,
            linestyle=(0, (2, 2)), zorder=1)

    ax.fill(angles_plot, values_plot, color=color, alpha=fill_alpha, zorder=2)
    ax.plot(angles_plot, values_plot, color=color, linewidth=2.4, zorder=3,
            solid_capstyle='round', solid_joinstyle='round')
    ax.scatter(angles, values, color=color, s=24, zorder=4,
               edgecolor=Q_SURFACE, linewidth=1.1)

    if show_values:
        for ang, val in zip(angles, values):
            # Keep the badge clear of the outer axis-label ring: push inward
            # for high values, outward (but never past r=88) for low ones.
            r = val - 11 if val >= 55 else min(val + 11, 88)
            r = max(r, 8)
            ax.text(ang, r, f'{val:.0f}', color='white', fontsize=6.6, ha='center',
                    va='center', fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.16', facecolor=color,
                              edgecolor=Q_SURFACE, linewidth=0.8, alpha=0.95))

    ax.set_ylim(0, 100)
    ax.set_rlabel_position(360 / N / 2)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=7.5, color='#e8e8e8')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=5.3, color=Q_MUTED)
    ax.tick_params(colors=Q_MUTED, pad=3)
    ax.spines['polar'].set_color('#2a2d35')
    ax.grid(color='#2a2d35', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color='white', fontsize=title_size, fontweight='bold', pad=14)


def make_radial_bar(ax, values, labels, edge_color, title, title_size=9.5, show_values=False):
    """Polar bar chart: bar length AND a diverging heatmap fill both encode the
    percentile, avoiding the area-distortion that a filled spider/line radar has.
    Bar edge keeps the team's categorical color as an identity accent."""
    N = len(values)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = (2 * np.pi / N) * 0.80

    ax.set_facecolor(Q_SURFACE)

    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ring, [50] * len(ring), color=Q_MUTED, linewidth=1,
            linestyle=(0, (2, 2)), zorder=1)

    colors = [percentile_color(v) for v in values]
    ax.bar(angles, values, width=width, bottom=0, color=colors, zorder=3,
           edgecolor=edge_color, linewidth=1.8, align='center')

    if show_values:
        for ang, val in zip(angles, values):
            # Keep the badge clear of the outer axis-label ring, same rule as make_radar.
            r = val - 11 if val >= 55 else min(val + 11, 88)
            r = max(r, 8)
            ax.text(ang, r, f'{val:.0f}', color='white', fontsize=6.8, ha='center',
                    va='center', fontweight='bold', zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground=Q_SURFACE)])

    ax.set_ylim(0, 100)
    ax.set_rlabel_position(360 / N / 2)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=7.5, color='#e8e8e8')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=5.3, color=Q_MUTED)
    ax.tick_params(colors=Q_MUTED, pad=3)
    ax.spines['polar'].set_color('#2a2d35')
    ax.grid(color='#2a2d35', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color='white', fontsize=title_size, fontweight='bold', pad=14)


def make_radial_heatmap(ax, values, labels, title, title_size=13, n_bins=50, gap_frac=0.16,
                        label_pad=32, value_r=104, value_size=8.2):
    """Equal-size wedge 'gauge' radar: every wedge is the same full sector, and
    a fixed sequential yellow->red colormap fills it from the center out to the
    percentile value — like a battery gauge. The color at the fill edge always
    matches the shared colorbar, and unfilled capacity stays visible as an
    outlined but empty wedge, which shows how far below 100 a metric sits
    without redrawing text for every value."""
    N = len(values)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = (2 * np.pi / N) * (1 - gap_frac)
    r_edges = np.linspace(0, 100, n_bins + 1)

    ax.set_facecolor(Q_LIGHT_SURFACE)

    for ang, val in zip(angles, values):
        val = np.clip(val, 0, 100)
        for i in range(n_bins):
            r0, r1 = r_edges[i], r_edges[i + 1]
            if r0 >= val:
                break
            r1 = min(r1, val)
            color = _HEATMAP_CMAP(((r0 + r1) / 2) / 100.0)
            ax.bar(ang, r1 - r0, bottom=r0, width=width, color=color,
                   edgecolor='none', zorder=2, align='center')
        # full-capacity outline so the "empty" remainder is still legible
        ax.bar(ang, 100, bottom=0, width=width, facecolor='none',
               edgecolor=Q_LIGHT_INK, linewidth=1.1, zorder=4, align='center')

    for ang, val in zip(angles, values):
        ax.text(ang, value_r, f'{val:.0f}', ha='center', va='center', color=Q_LIGHT_INK,
                 fontsize=value_size, fontweight='bold', zorder=6)

    ax.set_ylim(0, 100)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=8, color=Q_LIGHT_INK)
    ax.set_yticklabels([])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.tick_params(axis='x', colors=Q_LIGHT_MUTED, pad=label_pad)
    ax.tick_params(axis='y', length=0)
    ax.spines['polar'].set_visible(False)
    ax.grid(color=Q_LIGHT_GRID, linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title(title, color=Q_LIGHT_INK, fontsize=title_size, fontweight='bold', pad=22)


def make_bar_chart(ax, agg, metric, title, higher_is_better=True, fmt='{:.1f}%', scale=100):
    agg_sorted = agg.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    teams = [TEAM_SHORT.get(t, t) for t in agg_sorted['team']]
    values = agg_sorted[metric].values * scale
    colors = [TEAM_COLORS.get(t, '#888') for t in agg_sorted['team']]
    mean_val = agg[metric].mean() * scale

    ax.set_facecolor(Q_SURFACE)
    for i in range(len(teams)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='white', alpha=0.025, zorder=0)

    ax.grid(axis='x', color=Q_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(mean_val, color=Q_MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=1)

    bars = ax.barh(teams, values, color=colors, edgecolor='none', height=0.62, zorder=2)
    ax.axvline(0, color='#383835', linewidth=0.9, zorder=1)
    ax.set_xlabel(title, color=Q_INK, fontsize=8)
    ax.tick_params(colors=Q_INK, labelsize=7.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(Q_INK)
    ax.set_xlim(0, values.max() * 1.16 if values.max() > 0 else 1)
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + values.max() * 0.015, bar.get_y() + bar.get_height()/2,
                fmt.format(val), va='center', ha='left', color='white', fontsize=6.6)
    ax.invert_yaxis()


def make_scatter(ax, agg, x_metric, y_metric, x_label, y_label,
                 x_scale=100, y_scale=100, x_fmt='{:.0f}%', y_fmt='{:.0f}%',
                 quadrant_labels=None):
    ax.set_facecolor(Q_SURFACE)
    ax.grid(color=Q_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    x_mean = agg[x_metric].mean() * x_scale
    y_mean = agg[y_metric].mean() * y_scale

    if quadrant_labels:
        xlo, xhi = agg[x_metric].min() * x_scale, agg[x_metric].max() * x_scale
        ylo, yhi = agg[y_metric].min() * y_scale, agg[y_metric].max() * y_scale
        pad_x, pad_y = (xhi - xlo) * 0.06, (yhi - ylo) * 0.08
        positions = [(xhi, yhi, 'right', 'top'), (xlo, yhi, 'left', 'top'),
                     (xlo, ylo, 'left', 'bottom'), (xhi, ylo, 'right', 'bottom')]
        for (qx, qy, ha, va), label in zip(positions, quadrant_labels):
            if not label:
                continue
            qx = qx - pad_x if ha == 'right' else qx + pad_x
            qy = qy - pad_y if va == 'top' else qy + pad_y
            ax.text(qx, qy, label, ha=ha, va=va, fontsize=6.3, color=Q_MUTED,
                    style='italic', zorder=1)

    for _, row in agg.iterrows():
        color = TEAM_COLORS.get(row['team'], '#888')
        short = TEAM_SHORT.get(row['team'], row['team'])
        xv = row[x_metric] * x_scale
        yv = row[y_metric] * y_scale
        ax.scatter(xv, yv, color=color, s=85, zorder=3, edgecolor=Q_SURFACE, linewidth=1.2)
        ax.annotate(short, (xv, yv), textcoords='offset points', xytext=(5, 4),
                    fontsize=6.4, color='white', fontweight='medium', zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=Q_SURFACE)])

    ax.axhline(y_mean, color=Q_MUTED, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(x_mean, color=Q_MUTED, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlabel(x_label, color=Q_INK, fontsize=8)
    ax.set_ylabel(y_label, color=Q_INK, fontsize=8)
    ax.tick_params(colors=Q_INK, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(Q_GRID)


def plot_team_radar(agg, team_name, output_path):
    """Full radar for one team."""
    row = agg[agg['team'] == team_name].iloc[0]

    labels = [m[1] for m in RADAR_METRICS_CONFIG]
    percentiles = [percentile_rank(agg[col], row[col], higher_is_better=hib)
                   for col, lbl, hib in RADAR_METRICS_CONFIG]

    color = TEAM_COLORS.get(team_name, '#4FC3F7')
    short = TEAM_SHORT.get(team_name, team_name)
    n_games = int(row.get('n_games', 0))

    fig = plt.figure(figsize=(6.4, 6.8), facecolor=Q_SURFACE)
    ax = fig.add_subplot(111, polar=True)
    make_radar(ax, percentiles, labels, color, f'{short} — Style Profile 25/26',
               show_values=True, title_size=13)
    fig.text(0.5, 0.035,
             f'Percentile rank vs Eredivisie 25/26 field  ·  {n_games} games  ·  dashed ring = league average',
             ha='center', color=Q_MUTED, fontsize=7.3)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()


def plot_team_radial_bar(agg, team_name, output_path):
    """Percentile radar as a radial heatmap bar chart — bar length AND a
    diverging blue/red heatmap fill both encode the percentile, which avoids
    the area-distortion of a filled spider radar (see plot_team_radar)."""
    row = agg[agg['team'] == team_name].iloc[0]

    labels = [m[1] for m in RADAR_METRICS_CONFIG]
    percentiles = [percentile_rank(agg[col], row[col], higher_is_better=hib)
                   for col, lbl, hib in RADAR_METRICS_CONFIG]

    color = TEAM_COLORS.get(team_name, '#4FC3F7')
    short = TEAM_SHORT.get(team_name, team_name)
    n_games = int(row.get('n_games', 0))

    fig = plt.figure(figsize=(6.4, 6.8), facecolor=Q_SURFACE)
    ax = fig.add_subplot(111, polar=True)
    make_radial_bar(ax, percentiles, labels, color, f'{short} — Percentile Radar 25/26',
                     show_values=True, title_size=13)
    fig.text(0.5, 0.035,
             f'Percentile rank vs Eredivisie 25/26 field  ·  {n_games} games  ·  '
             'blue = below average, red = above average',
             ha='center', color=Q_MUTED, fontsize=7.3)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()


def plot_team_radial_heatmap(agg, team_name, output_path):
    """Light-theme 'radial heatmap' percentile gauge: every metric is the same
    full-size wedge, filled from the center with a yellow->red heatmap up to
    the percentile value, with a shared colorbar giving the color scale."""
    row = agg[agg['team'] == team_name].iloc[0]

    labels = [m[1] for m in RADAR_METRICS_CONFIG]
    percentiles = [percentile_rank(agg[col], row[col], higher_is_better=hib)
                   for col, lbl, hib in RADAR_METRICS_CONFIG]

    short = TEAM_SHORT.get(team_name, team_name)
    n_games = int(row.get('n_games', 0))

    fig = plt.figure(figsize=(9, 8), facecolor=Q_LIGHT_SURFACE)
    ax = fig.add_subplot(111, polar=True)
    make_radial_heatmap(ax, percentiles, labels,
                        f'{short} — Eredivisie 25/26\nRadial Heatmap Percentile Radar')

    sm = mcm.ScalarMappable(norm=mcolors.Normalize(0, 100), cmap=_HEATMAP_CMAP)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.14, shrink=0.75)
    cbar.set_label('Percentile intensity', color=Q_LIGHT_INK, fontsize=9)
    cbar.ax.tick_params(colors=Q_LIGHT_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(Q_LIGHT_GRID)

    fig.text(0.02, -0.02, f'Data: Opta  ·  {n_games} games  ·  percentile rank vs Eredivisie 25/26 field',
             ha='left', color=Q_LIGHT_MUTED, fontsize=8)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=Q_LIGHT_SURFACE)
    plt.close()


def plot_overview_dashboard(agg, output_path):
    """League-wide overview with scatter plots and bar charts."""
    total_matches = int(agg['n_games'].sum() // 2)
    fig = plt.figure(figsize=(22, 26.5), facecolor=Q_SURFACE)
    fig.suptitle('Eredivisie 2025/26 — Team Style Profiles',
                 color='white', fontsize=19, fontweight='bold', y=0.995)
    fig.text(0.5, 0.978, f'{total_matches} matches  ·  18 teams  ·  dashed lines mark the league average',
             ha='center', color=Q_MUTED, fontsize=9.5)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35,
                           top=0.955, bottom=0.02, left=0.06, right=0.97)

    subplot_title_kw = dict(color='white', fontsize=10, fontweight='bold', pad=8)

    # Row 0: Scatter plots
    ax00 = fig.add_subplot(gs[0, 0])
    make_scatter(ax00, agg, 'territory', 'ppda', 'Territory (avg x)', 'PPDA',
                 x_scale=1, y_scale=1, x_fmt='{:.0f}', y_fmt='{:.1f}',
                 quadrant_labels=('high territory, low press', 'low territory, low press',
                                   'low territory, heavy press', 'high territory, heavy press'))
    ax00.set_title('Territory vs High Press (PPDA)', **subplot_title_kw)
    ax00.invert_yaxis()  # lower PPDA = more pressure

    ax01 = fig.add_subplot(gs[0, 1])
    make_scatter(ax01, agg, 'long_ball_pct', 'deep_circ_pct', 'Long Ball %', 'Deep Circulation %')
    ax01.set_title('Long Balls vs Deep Circulation', **subplot_title_kw)

    ax02 = fig.add_subplot(gs[0, 2])
    make_scatter(ax02, agg, 'wing_pct', 'cross_pct', 'Wing Play %', 'Crossing %')
    ax02.set_title('Wing Play vs Crossing', **subplot_title_kw)

    # Row 1: Bar charts
    ax10 = fig.add_subplot(gs[1, 0])
    make_bar_chart(ax10, agg, 'territory', 'Territory (avg normalized x)', higher_is_better=True, fmt='{:.1f}', scale=1)
    ax10.set_title('Territory', **subplot_title_kw)

    ax11 = fig.add_subplot(gs[1, 1])
    make_bar_chart(ax11, agg, 'ppda', 'PPDA (lower = more pressing)', higher_is_better=False, fmt='{:.1f}', scale=1)
    ax11.set_title('High Press (PPDA)', **subplot_title_kw)

    ax12 = fig.add_subplot(gs[1, 2])
    make_bar_chart(ax12, agg, 'long_ball_pct', 'Long Ball %', fmt='{:.1f}%')
    ax12.set_title('Long Balls', **subplot_title_kw)

    # Row 2: Bar charts
    ax20 = fig.add_subplot(gs[2, 0])
    make_bar_chart(ax20, agg, 'deep_circ_pct', 'Deep Circulation %')
    ax20.set_title('Deep Circulation', **subplot_title_kw)

    ax21 = fig.add_subplot(gs[2, 1])
    make_bar_chart(ax21, agg, 'wing_pct', 'Wing Play %')
    ax21.set_title('Wing Play', **subplot_title_kw)

    ax22 = fig.add_subplot(gs[2, 2])
    make_bar_chart(ax22, agg, 'cross_pct', 'Crossing %')
    ax22.set_title('Crossing', **subplot_title_kw)

    # Row 3: GK build up, Set Pieces, Low Block + Counters scatter
    ax30 = fig.add_subplot(gs[3, 0])
    make_bar_chart(ax30, agg, 'gk_short_pct', 'GK Short Pass %')
    ax30.set_title('GK Build Up (short %)', **subplot_title_kw)

    ax31 = fig.add_subplot(gs[3, 1])
    make_bar_chart(ax31, agg, 'sp_shots_pct', 'Set Piece Shot %')
    ax31.set_title('Set Piece Shots', **subplot_title_kw)

    ax32 = fig.add_subplot(gs[3, 2])
    make_scatter(ax32, agg, 'low_block', 'counters_per90', 'Low Block %', 'Counters p90',
                 y_scale=1, y_fmt='{:.1f}')
    ax32.set_title('Low Block vs Counters', **subplot_title_kw)

    plt.savefig(output_path, dpi=140, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print(f'Saved: {output_path}')


def plot_all_radars(agg, output_path):
    """Grid of radars for all 18 teams."""
    teams = sorted(agg['team'].unique())
    n = len(teams)
    ncols = 6
    nrows = math.ceil(n / ncols)
    labels = [m[1] for m in RADAR_METRICS_CONFIG]

    fig = plt.figure(figsize=(ncols * 4.6, nrows * 4.8), facecolor=Q_SURFACE)
    fig.suptitle('Eredivisie 2025/26 — Team Style Profiles',
                 color='white', fontsize=21, fontweight='bold', y=1.035)
    fig.text(0.5, 0.998, 'Percentile rank per metric vs the Eredivisie 25/26 field  ·  dashed ring = league average',
             ha='center', color=Q_MUTED, fontsize=10)

    for i, team in enumerate(teams):
        row_data = agg[agg['team'] == team].iloc[0]
        percentiles = [percentile_rank(agg[col], row_data[col], higher_is_better=hib)
                       for col, lbl, hib in RADAR_METRICS_CONFIG]

        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        color = TEAM_COLORS.get(team, '#4FC3F7')
        short = TEAM_SHORT.get(team, team)
        make_radar(ax, percentiles, labels, color, short, title_size=11)

    fig.subplots_adjust(hspace=0.55, wspace=0.45, top=0.94, bottom=0.02, left=0.02, right=0.98)
    plt.savefig(output_path, dpi=130, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print(f'Saved: {output_path}')


def plot_all_radial_bars(agg, output_path):
    """Grid of diverging (blue/red) percentile bar radars for all 18 teams."""
    teams = sorted(agg['team'].unique())
    n = len(teams)
    ncols = 6
    nrows = math.ceil(n / ncols)
    labels = [m[1] for m in RADAR_METRICS_CONFIG]

    fig = plt.figure(figsize=(ncols * 4.6, nrows * 4.8), facecolor=Q_SURFACE)
    fig.suptitle('Eredivisie 2025/26 — Percentile Radars',
                 color='white', fontsize=21, fontweight='bold', y=1.035)
    fig.text(0.5, 0.998,
             'Bar length & color = percentile vs the field  ·  blue = below average, red = above average',
             ha='center', color=Q_MUTED, fontsize=10)

    for i, team in enumerate(teams):
        row_data = agg[agg['team'] == team].iloc[0]
        percentiles = [percentile_rank(agg[col], row_data[col], higher_is_better=hib)
                       for col, lbl, hib in RADAR_METRICS_CONFIG]

        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        color = TEAM_COLORS.get(team, '#4FC3F7')
        short = TEAM_SHORT.get(team, team)
        make_radial_bar(ax, percentiles, labels, color, short, title_size=11)

    fig.subplots_adjust(hspace=0.55, wspace=0.45, top=0.94, bottom=0.02, left=0.02, right=0.98)
    plt.savefig(output_path, dpi=130, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print(f'Saved: {output_path}')


def plot_all_radial_heatmaps(agg, output_path):
    """Grid of light-theme radial heatmap gauge radars for all 18 teams."""
    teams = sorted(agg['team'].unique())
    n = len(teams)
    ncols = 6
    nrows = math.ceil(n / ncols)
    labels = [m[1] for m in RADAR_METRICS_CONFIG]

    fig = plt.figure(figsize=(ncols * 4.8, nrows * 5.0), facecolor=Q_LIGHT_SURFACE)
    fig.suptitle('Eredivisie 2025/26 — Radial Heatmap Percentile Radars',
                 color=Q_LIGHT_INK, fontsize=22, fontweight='bold', y=1.02)
    fig.text(0.5, 0.998, 'Percentile rank vs the field, filled center-out as a yellow-to-red gauge',
             ha='center', color=Q_LIGHT_MUTED, fontsize=10.5)

    for i, team in enumerate(teams):
        row_data = agg[agg['team'] == team].iloc[0]
        percentiles = [percentile_rank(agg[col], row_data[col], higher_is_better=hib)
                       for col, lbl, hib in RADAR_METRICS_CONFIG]

        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        short = TEAM_SHORT.get(team, team)
        make_radial_heatmap(ax, percentiles, labels, short, title_size=12, n_bins=25,
                            label_pad=20, value_r=100.5, value_size=7.2)

    fig.subplots_adjust(hspace=0.9, wspace=0.75, top=0.93, bottom=0.05, left=0.02, right=0.93)
    sm = mcm.ScalarMappable(norm=mcolors.Normalize(0, 100), cmap=_HEATMAP_CMAP)
    cbar_ax = fig.add_axes((0.955, 0.15, 0.012, 0.7))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Percentile intensity', color=Q_LIGHT_INK, fontsize=11)
    cbar.ax.tick_params(colors=Q_LIGHT_MUTED, labelsize=9)
    cbar.outline.set_edgecolor(Q_LIGHT_GRID)

    plt.savefig(output_path, dpi=130, bbox_inches='tight', facecolor=Q_LIGHT_SURFACE)
    plt.close()
    print(f'Saved: {output_path}')


def plot_gk_buildup(agg, output_path):
    """Detailed GK build-up chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), facecolor=Q_SURFACE)
    fig.suptitle('Eredivisie 2025/26 — GK Build Up & Set Pieces',
                 color='white', fontsize=15, fontweight='bold', y=0.99)
    fig.text(0.5, 0.945, 'Dashed line marks the league average',
             ha='center', color=Q_MUTED, fontsize=8.5)

    ax1, ax2 = axes
    make_bar_chart(ax1, agg, 'gk_short_pct', 'Short GK Pass %')
    ax1.set_title('GK Build Up — Short Pass %\n(short = ≤25m)', color='white', fontsize=10.5, fontweight='bold')

    make_bar_chart(ax2, agg, 'sp_shots_pct', 'Set Piece Shot %')
    ax2.set_title('Set Pieces — % of shots from SP', color='white', fontsize=10.5, fontweight='bold')

    for ax in axes:
        ax.set_facecolor(Q_SURFACE)
    fig.patch.set_facecolor(Q_SURFACE)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print(f'Saved: {output_path}')


def plot_style_similarity_map(agg, output_path):
    """PCA biplot of all 18 teams over their 10 raw style metrics (z-scored).
    Teams that play alike land close together; arrows show which raw metrics
    drive each axis, so the map is readable, not just a cloud of dots."""
    metric_cols = [col for col, lbl, hib in RADAR_METRICS_CONFIG]
    metric_labels = {col: lbl for col, lbl, hib in RADAR_METRICS_CONFIG}

    X_std = StandardScaler().fit_transform(agg[metric_cols].values)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_std)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    var_pct = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(11.5, 9.5), facecolor=Q_SURFACE)
    ax.set_facecolor(Q_SURFACE)
    ax.grid(color=Q_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=Q_MUTED, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(0, color=Q_MUTED, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)

    # scale loading arrows to sit comfortably alongside the team score cloud
    arrow_scale = np.abs(scores).max() / np.abs(loadings).max() * 0.8

    for j, col in enumerate(metric_cols):
        lx, ly = loadings[j, 0] * arrow_scale, loadings[j, 1] * arrow_scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0), zorder=2,
                    arrowprops=dict(arrowstyle='-|>', color=Q_MUTED, linewidth=1.2, alpha=0.85))
        ax.text(lx * 1.14, ly * 1.14, metric_labels[col], color=Q_MUTED, fontsize=8.3,
                ha='center', va='center', style='italic', zorder=2)

    for i, row in agg.reset_index(drop=True).iterrows():
        team = row['team']
        color = TEAM_COLORS.get(team, '#888')
        short = TEAM_SHORT.get(team, team)
        x, y = scores[i, 0], scores[i, 1]
        ax.scatter(x, y, color=color, s=120, zorder=4, edgecolor=Q_SURFACE, linewidth=1.3)
        ax.annotate(short, (x, y), textcoords='offset points', xytext=(6, 5),
                    fontsize=8.8, color='white', fontweight='medium', zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground=Q_SURFACE)])

    ax.set_xlabel(f'PC1  ({var_pct[0]:.0f}% of style variance)', color=Q_INK, fontsize=10.5)
    ax.set_ylabel(f'PC2  ({var_pct[1]:.0f}% of style variance)', color=Q_INK, fontsize=10.5)
    ax.tick_params(colors=Q_INK, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(Q_GRID)

    fig.suptitle('Eredivisie 2025/26 — Style Similarity Map', color='white',
                 fontsize=18, fontweight='bold', y=0.99)
    fig.text(0.5, 0.945,
             'Teams close together play a similar style  ·  arrows show which raw metrics pull each axis',
             ha='center', color=Q_MUTED, fontsize=9.5)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print(f'Saved: {output_path}')


def main():
    df = load_all_data()
    df.to_csv(os.path.join(OUTPUT_DIR, 'team_metrics_raw.csv'), index=False)
    print('Saved raw CSV.')

    agg = aggregate_by_team(df)
    agg.to_csv(os.path.join(OUTPUT_DIR, 'team_metrics_aggregated.csv'), index=False)
    print('Saved aggregated CSV.')
    print(agg.to_string(index=False))

    # Plots
    plot_overview_dashboard(agg, os.path.join(OUTPUT_DIR, 'overview_dashboard.png'))
    plot_all_radars(agg, os.path.join(SPIDER_DIR, 'all_team_radars.png'))
    plot_all_radial_bars(agg, os.path.join(RADIAL_BAR_DIR, 'all_radial_bars.png'))
    plot_all_radial_heatmaps(agg, os.path.join(RADIAL_HEATMAP_DIR, 'all_radial_heatmaps.png'))
    plot_gk_buildup(agg, os.path.join(OUTPUT_DIR, 'gk_setpieces.png'))
    plot_style_similarity_map(agg, os.path.join(STYLE_SIMILARITY_DIR, 'style_similarity_map.png'))

    # Individual radars
    for team in agg['team']:
        short = TEAM_SHORT.get(team, team).replace(' ', '_')
        plot_team_radar(agg, team, os.path.join(SPIDER_DIR, f'radar_{short}.png'))
        plot_team_radial_bar(agg, team, os.path.join(RADIAL_BAR_DIR, f'radial_bar_{short}.png'))
        plot_team_radial_heatmap(agg, team, os.path.join(RADIAL_HEATMAP_DIR, f'radial_heatmap_{short}.png'))
    print('All individual radars saved.')
    print('Done!')


if __name__ == '__main__':
    main()
