"""
League-wide attacking/defensive ratings table — all 18 Eredivisie teams, 2025/26.

Reuses the same 4 rating engines as top3_ratings.py (ratings_common.py), reads
off each team's latest (5-match rolling average) index per system/dimension,
and renders one sorted table image instead of a trend chart.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from ratings_common import compute_all_systems, METHOD_ORDER

OUT_PATH = 'Visuals/Rating/ratings_table.png'

# ── housestyle, light variant (mirrors Q_LIGHT_* in Scripts/coach_profiling.py) ──
Q_INK     = '#2b2b2b'
Q_MUTED   = '#6b6b6b'
Q_GRID    = '#ddd9cd'
Q_SURFACE = '#f4f1ea'
Q_BAND    = '#eae6dc'

# Same glasbey-safe team identity palette as coach_profiling.py / top3_ratings.py
TEAM_COLORS = {
    'AFC Ajax':                     '#d60000',
    'Alkmaar Zaanstreek':           '#03dae6',
    'FC Groningen':                 '#8956af',
    'FC Twente':                    '#6b9700',
    'FC Utrecht':                   '#2db152',
    'FC Volendam':                  '#ff7ed1',
    'Feyenoord Rotterdam':          '#4650c9',  # darkened for contrast, see top3_ratings.py
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

METHOD_LABEL = {'PI': 'PI', 'ELO': 'ELO', 'Glicko-2': 'G2', 'Base-70': 'B70'}

# value -> cell color: red (weak) through cream (average) to green (strong)
CELL_CMAP = LinearSegmentedColormap.from_list('ratings_diverging', ['#c0392b', '#f4f1ea', '#1a7a52'], N=256)


def cell_color(v, alpha=0.75):
    r, g, b, _ = CELL_CMAP(np.clip(v, 0, 100) / 100.0)
    return (r, g, b, alpha)


def build_table(systems, teams):
    rows = []
    for t in teams:
        vals = {}
        for method in METHOD_ORDER:
            vals[f'att_{method}'] = systems[method][t]['att'].iloc[-1]
            vals[f'def_{method}'] = systems[method][t]['def'].iloc[-1]
        att_avg = np.mean([vals[f'att_{m}'] for m in METHOD_ORDER])
        def_avg = np.mean([vals[f'def_{m}'] for m in METHOD_ORDER])
        rows.append(dict(team=t, att_avg=att_avg, def_avg=def_avg,
                          overall=(att_avg + def_avg) / 2, **vals))
    df = pd.DataFrame(rows).sort_values('overall', ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# column key, header label, relative width
COLUMNS = (
    [('rank', 'Rk', 0.9), ('team', 'Team', 4.6), ('gap1', '', 0.3)] +
    [(f'att_{m}', METHOD_LABEL[m], 1.15) for m in METHOD_ORDER] +
    [('att_avg', 'Avg', 1.3), ('gap2', '', 0.3)] +
    [(f'def_{m}', METHOD_LABEL[m], 1.15) for m in METHOD_ORDER] +
    [('def_avg', 'Avg', 1.3), ('gap3', '', 0.3), ('overall', 'Overall', 1.6)]
)


def col_edges():
    widths = np.array([w for _, _, w in COLUMNS], dtype=float)
    widths = widths / widths.sum()
    edges = np.concatenate([[0], np.cumsum(widths)])
    return {key: (edges[i], edges[i + 1]) for i, (key, _, _) in enumerate(COLUMNS)}


def plot_table(df):
    n = len(df)
    row_h = 0.85
    header_h = 1.7
    title_h = 1.85
    footer_h = 0.55
    fig_h = title_h + header_h + n * row_h + footer_h
    fig, ax = plt.subplots(figsize=(16, fig_h * 0.34), facecolor=Q_SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')

    # ── title block (drawn in the same data coordinates as the table, so its
    # spacing scales correctly regardless of this figure's unusually short,
    # wide aspect ratio) ──
    ax.text(0.005, title_h * 0.38, 'Eredivisie 2025/26 — Attacking & Defensive Ratings, League Table',
            ha='left', va='center', color=Q_INK, fontsize=16.5, fontweight='bold')
    ax.text(0.005, title_h * 0.78,
            '4 rating systems (PI, ELO, Glicko-2, Base-70), each a 0–100 index (higher = better)  ·  '
            'sorted by Overall = mean of all 8',
            ha='left', va='center', color=Q_MUTED, fontsize=9.5)

    edges = col_edges()

    def xmid(key):
        l, r = edges[key]
        return (l + r) / 2

    # ── group headers ("Attacking" / "Defensive") ──
    att_l = edges[f'att_{METHOD_ORDER[0]}'][0]
    att_r = edges['att_avg'][1]
    def_l = edges[f'def_{METHOD_ORDER[0]}'][0]
    def_r = edges['def_avg'][1]
    grp_y = title_h + 0.15
    grp_line_y = title_h + 0.62
    ax.text((att_l + att_r) / 2, grp_y, 'ATTACKING', ha='center', va='top',
            color='#1a5f9e', fontsize=11.5, fontweight='bold')
    ax.plot([att_l, att_r], [grp_line_y, grp_line_y], color='#1a5f9e', linewidth=2.2, solid_capstyle='round')
    ax.text((def_l + def_r) / 2, grp_y, 'DEFENSIVE', ha='center', va='top',
            color='#1a7a52', fontsize=11.5, fontweight='bold')
    ax.plot([def_l, def_r], [grp_line_y, grp_line_y], color='#1a7a52', linewidth=2.2, solid_capstyle='round')

    # ── sub-column headers ──
    header_y = title_h + 1.15
    for key, label, _ in COLUMNS:
        if key.startswith('gap') or not label:
            continue
        ha = 'left' if key == 'team' else 'center'
        x = edges[key][0] + 0.004 if key == 'team' else xmid(key)
        ax.text(x, header_y, label, ha=ha, va='center', color=Q_MUTED, fontsize=8.7, fontweight='bold')
    ax.plot([0, 1], [title_h + header_h - 0.05, title_h + header_h - 0.05], color=Q_GRID, linewidth=1.2)

    metric_keys = [f'att_{m}' for m in METHOD_ORDER] + ['att_avg'] + [f'def_{m}' for m in METHOD_ORDER] + ['def_avg']

    for i, (_, row) in enumerate(df.iterrows()):
        y0 = title_h + header_h + i * row_h
        y1 = y0 + row_h
        yc = (y0 + y1) / 2

        if i % 2 == 1:
            ax.add_patch(mpatches.Rectangle((0, y0), 1, row_h, facecolor=Q_BAND, edgecolor='none', zorder=0))

        rank = df.index[i]
        ax.text(xmid('rank'), yc, str(rank), ha='center', va='center', color=Q_MUTED, fontsize=9, zorder=2)

        team = row['team']
        swatch_x = edges['team'][0] + 0.004
        ax.add_patch(mpatches.Rectangle((swatch_x, yc - 0.19), 0.010, 0.38,
                                         facecolor=TEAM_COLORS.get(team, '#888'), edgecolor='none', zorder=2))
        ax.text(swatch_x + 0.018, yc, TEAM_SHORT.get(team, team), ha='left', va='center',
                color=Q_INK, fontsize=9.3, fontweight='bold', zorder=2)

        for key in metric_keys:
            val = row[key]
            l, r = edges[key]
            pad = 0.003
            ax.add_patch(mpatches.FancyBboxPatch((l + pad, yc - 0.18), (r - l) - 2 * pad, 0.36,
                                                  boxstyle='round,pad=0,rounding_size=0.006',
                                                  facecolor=cell_color(val), edgecolor='none', zorder=1))
            ax.text(xmid(key), yc, f'{val:.0f}', ha='center', va='center',
                    color=Q_INK, fontsize=8.4, zorder=2)

        ax.text(xmid('overall'), yc, f"{row['overall']:.1f}", ha='center', va='center',
                color=Q_INK, fontsize=10, fontweight='bold', zorder=2)

    ax.text(0.005, fig_h - 0.12,
            'Data: Opta (shot-event goals)  ·  ELO & Glicko-2 are opponent-adjusted, PI & Base-70 are not  ·  '
            'ratings are the latest 5-match rolling average',
            ha='left', va='center', color=Q_MUTED, fontsize=8)

    plt.tight_layout(pad=0.4)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print('Saved:', OUT_PATH)


def main():
    systems, teams, league_avg = compute_all_systems()
    df = build_table(systems, teams)
    plot_table(df)


if __name__ == '__main__':
    main()
