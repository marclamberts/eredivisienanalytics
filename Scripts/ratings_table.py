"""
League-wide attacking/defensive ratings tables — all 18 Eredivisie teams, 2025/26.

One table per rating system (PI, ELO, Glicko-2, Base-70), each showing that
system's own real units — no cross-system 0-100 normalisation. Attacking and
Defensive are on the same native scale within a single system, so they can be
combined into one sortable Composite column without rescaling anything:

  - PI        Composite = Att − Def (goals/match, i.e. goal difference/match)
  - ELO       Composite = mean(Att, Def)               (Elo points)
  - Glicko-2  Composite = mean(Att, Def)                (Glicko-2 rating)
  - Base-70   Composite = mean(Att, Def)                 (bounded 0-100 score)

PI is the one system where the defensive column is not "higher is better":
it's goals conceded per match, so lower is better there — flagged directly in
that table's column header and footnote instead of being hidden by rescaling.

Cell background shading is a per-column visual aid only (relative strength
within that column); the printed number is always the real, un-rescaled value.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from ratings_common import compute_all_systems_native, METHOD_ORDER

OUT_DIR = 'Visuals/Rating'

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

CELL_CMAP = LinearSegmentedColormap.from_list('ratings_diverging', ['#c0392b', '#f4f1ea', '#1a7a52'], N=256)

# per-system display config
SYSTEM_CONFIG = {
    'PI': dict(
        file='pi_table.png', title='PI Ratings', unit='goals / match',
        fmt='{:.2f}', att_label='Attacking\n(goals/match)',
        def_label='Defensive\n(goals/match, ↓ better)',
        def_higher_is_better=False,
        composite_label='Net\n(Att − Def)',
        composite_fn=lambda att, deff: att - deff,
        footnote='PI is a simple, opponent-unadjusted per-match goal rate. Its defensive column is goals '
                 'conceded/match, so LOWER is better there — the only column in these 4 tables where that is true.',
    ),
    'ELO': dict(
        file='elo_table.png', title='ELO Ratings', unit='Elo points',
        fmt='{:.0f}', att_label='Attacking ELO', def_label='Defensive ELO',
        def_higher_is_better=True,
        composite_label='Combined ELO\n(mean of Att, Def)',
        composite_fn=lambda att, deff: (att + deff) / 2,
        footnote='Classic Elo, opponent-adjusted; both columns start every team at 1500 and higher is always better.',
    ),
    'Glicko-2': dict(
        file='glicko2_table.png', title='Glicko-2 Ratings', unit='Glicko-2 rating',
        fmt='{:.0f}', att_label='Attacking Glicko-2', def_label='Defensive Glicko-2',
        def_higher_is_better=True,
        composite_label='Combined Glicko-2\n(mean of Att, Def)',
        composite_fn=lambda att, deff: (att + deff) / 2,
        footnote='Elo extension with rating deviation/volatility (both start at 1500); higher is always better.',
    ),
    'Base-70': dict(
        file='base70_table.png', title='Base-70 Ratings', unit='0-100 score',
        fmt='{:.0f}', att_label='Attacking Base-70', def_label='Defensive Base-70',
        def_higher_is_better=True,
        composite_label='Combined Base-70\n(mean of Att, Def)',
        composite_fn=lambda att, deff: (att + deff) / 2,
        footnote='Bounded scale anchored on a neutral 70, opponent-unadjusted, mildly mean-reverting; higher is always better.',
    ),
}


def cell_color(val, lo, hi, invert=False, alpha=0.75):
    if hi == lo:
        frac = 0.5
    else:
        frac = (val - lo) / (hi - lo)
    if invert:
        frac = 1 - frac
    r, g, b, _ = CELL_CMAP(np.clip(frac, 0, 1))
    return (r, g, b, alpha)


def build_table(systems, teams, method):
    cfg = SYSTEM_CONFIG[method]
    rows = []
    for t in teams:
        att = systems[method][t]['att'].iloc[-1]
        deff = systems[method][t]['def'].iloc[-1]
        rows.append(dict(team=t, att=att, deff=deff, composite=cfg['composite_fn'](att, deff)))
    df = pd.DataFrame(rows).sort_values('composite', ascending=False).reset_index(drop=True)
    df.index += 1
    return df


COLUMNS = [('rank', 'Rk', 0.6), ('team', 'Team', 2.9), ('gap1', '', 0.2),
           ('att', 'att_label', 2.6), ('deff', 'def_label', 2.6), ('gap2', '', 0.2),
           ('composite', 'composite_label', 2.4)]


def col_edges():
    widths = np.array([w for _, _, w in COLUMNS], dtype=float)
    widths = widths / widths.sum()
    edges = np.concatenate([[0], np.cumsum(widths)])
    return {key: (edges[i], edges[i + 1]) for i, (key, _, _) in enumerate(COLUMNS)}


def plot_table(df, method):
    cfg = SYSTEM_CONFIG[method]
    n = len(df)
    row_h = 0.85
    header_h = 1.5
    title_h = 1.6
    footer_h = 0.75
    fig_h = title_h + header_h + n * row_h + footer_h
    fig, ax = plt.subplots(figsize=(11, fig_h * 0.34), facecolor=Q_SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')

    ax.text(0.005, title_h * 0.35, f"Eredivisie 2025/26 — {cfg['title']}",
            ha='left', va='center', color=Q_INK, fontsize=16, fontweight='bold')
    ax.text(0.005, title_h * 0.75,
            f"Latest 5-match rolling average, in native units ({cfg['unit']})  ·  sorted by {cfg['composite_label'].splitlines()[0]}",
            ha='left', va='center', color=Q_MUTED, fontsize=9)

    edges = col_edges()

    def xmid(key):
        l, r = edges[key]
        return (l + r) / 2

    label_map = {'att': cfg['att_label'], 'deff': cfg['def_label'], 'composite': cfg['composite_label']}
    line1_y = title_h + header_h * 0.42
    line2_y = title_h + header_h * 0.78
    for key, label, _ in COLUMNS:
        if key.startswith('gap'):
            continue
        text = label_map.get(key, label)
        parts = text.split('\n')
        ha = 'left' if key == 'team' else 'center'
        x = edges[key][0] + 0.006 if key == 'team' else xmid(key)
        ax.text(x, line1_y, parts[0], ha=ha, va='center', color=Q_MUTED, fontsize=8.8, fontweight='bold')
        if len(parts) > 1:
            ax.text(x, line2_y, parts[1], ha=ha, va='center', color=Q_MUTED, fontsize=7.3, style='italic')
    ax.plot([0, 1], [title_h + header_h - 0.05, title_h + header_h - 0.05], color=Q_GRID, linewidth=1.2)

    att_lo, att_hi = df['att'].min(), df['att'].max()
    def_lo, def_hi = df['deff'].min(), df['deff'].max()
    comp_lo, comp_hi = df['composite'].min(), df['composite'].max()

    for i, (_, row) in enumerate(df.iterrows()):
        y0 = title_h + header_h + i * row_h
        yc = y0 + row_h / 2

        if i % 2 == 1:
            ax.add_patch(mpatches.Rectangle((0, y0), 1, row_h, facecolor=Q_BAND, edgecolor='none', zorder=0))

        rank = df.index[i]
        ax.text(xmid('rank'), yc, str(rank), ha='center', va='center', color=Q_MUTED, fontsize=9, zorder=2)

        team = row['team']
        swatch_x = edges['team'][0] + 0.006
        ax.add_patch(mpatches.Rectangle((swatch_x, yc - 0.19), 0.014, 0.38,
                                         facecolor=TEAM_COLORS.get(team, '#888'), edgecolor='none', zorder=2))
        ax.text(swatch_x + 0.026, yc, TEAM_SHORT.get(team, team), ha='left', va='center',
                color=Q_INK, fontsize=9.5, fontweight='bold', zorder=2)

        for key, val, lo, hi, invert in [
            ('att', row['att'], att_lo, att_hi, False),
            ('deff', row['deff'], def_lo, def_hi, not cfg['def_higher_is_better']),
            ('composite', row['composite'], comp_lo, comp_hi, False),
        ]:
            l, r = edges[key]
            pad = 0.004
            ax.add_patch(mpatches.FancyBboxPatch((l + pad, yc - 0.19), (r - l) - 2 * pad, 0.38,
                                                  boxstyle='round,pad=0,rounding_size=0.007',
                                                  facecolor=cell_color(val, lo, hi, invert=invert), edgecolor='none', zorder=1))
            weight = 'bold' if key == 'composite' else 'normal'
            fs = 9.5 if key == 'composite' else 9
            ax.text(xmid(key), yc, cfg['fmt'].format(val), ha='center', va='center',
                    color=Q_INK, fontsize=fs, fontweight=weight, zorder=2)

    ax.text(0.005, fig_h - footer_h * 0.55, cfg['footnote'],
            ha='left', va='center', color=Q_MUTED, fontsize=7.8, wrap=True)
    ax.text(0.005, fig_h - footer_h * 0.15,
            'Data: Opta (shot-event goals)  ·  cell shading is a per-column visual aid only — the printed number is the real value',
            ha='left', va='center', color=Q_MUTED, fontsize=7.5)

    plt.tight_layout(pad=0.4)
    out_path = f"{OUT_DIR}/{cfg['file']}"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=Q_SURFACE)
    plt.close()
    print('Saved:', out_path)


def main():
    systems, teams, league_avg = compute_all_systems_native()
    for method in METHOD_ORDER:
        df = build_table(systems, teams, method)
        plot_table(df, method)


if __name__ == '__main__':
    main()
