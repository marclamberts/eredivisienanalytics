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

Each metric cell shows the real value with a proportional data-bar underneath
(length + shade = that team's rank within the column, in the system's own
accent color — the same accent used for that system's line in top3_ratings.py).

Values are whole-season (all matches played in 2025/26), not a recent-form
snapshot: ELO/Glicko-2/Base-70 already accumulate every match, so their
season value is just the rating after the last one; PI has no memory between
matches, so its season value is the mean goal rate across every match played.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from ratings_common import compute_all_systems_native_season, METHOD_ORDER

OUT_DIR = 'Visuals/Rating'

# ── housestyle, light variant (mirrors Q_LIGHT_* in Scripts/coach_profiling.py) ──
Q_INK     = '#2b2b2b'
Q_MUTED   = '#6b6b6b'
Q_GRID    = '#ddd9cd'
Q_SURFACE = '#f4f1ea'
Q_BAND    = '#eae6dc'
Q_CARD    = '#faf8f3'
Q_TRACK   = '#e4e0d5'

MEDALS = {1: '#c9962b', 2: '#9aa0a6', 3: '#ad6a37'}

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

# per-system display config — accent matches that system's line color in top3_ratings.py
SYSTEM_CONFIG = {
    'PI': dict(
        file='pi_table.png', title='PI Ratings', unit='goals / match', accent='#2f6fb0',
        fmt='{:.2f}', att_label='Attacking', att_sub='goals/match',
        def_label='Defensive', def_sub='goals/match, ↓ better',
        def_higher_is_better=False,
        composite_label='Net', composite_sub='Att − Def',
        composite_fn=lambda att, deff: att - deff,
        footnote='PI is a simple, opponent-unadjusted per-match goal rate. Its defensive column is goals '
                 'conceded/match, so LOWER is better there — the only column in these 4 tables where that is true.',
    ),
    'ELO': dict(
        file='elo_table.png', title='ELO Ratings', unit='Elo points', accent='#a6620a',
        fmt='{:.0f}', att_label='Attacking', att_sub='Elo points',
        def_label='Defensive', def_sub='Elo points',
        def_higher_is_better=True,
        composite_label='Combined', composite_sub='mean of Att, Def',
        composite_fn=lambda att, deff: (att + deff) / 2,
        footnote='Classic Elo, opponent-adjusted; both columns start every team at 1500 and higher is always better.',
    ),
    'Glicko-2': dict(
        file='glicko2_table.png', title='Glicko-2 Ratings', unit='Glicko-2 rating', accent='#a52a52',
        fmt='{:.0f}', att_label='Attacking', att_sub='Glicko-2 rating',
        def_label='Defensive', def_sub='Glicko-2 rating',
        def_higher_is_better=True,
        composite_label='Combined', composite_sub='mean of Att, Def',
        composite_fn=lambda att, deff: (att + deff) / 2,
        footnote='Elo extension with rating deviation/volatility (both start at 1500); higher is always better.',
    ),
    'Base-70': dict(
        file='base70_table.png', title='Base-70 Ratings', unit='0-100 score', accent='#1a7a52',
        fmt='{:.0f}', att_label='Attacking', att_sub='0-100 score',
        def_label='Defensive', def_sub='0-100 score',
        def_higher_is_better=True,
        composite_label='Combined', composite_sub='mean of Att, Def',
        composite_fn=lambda att, deff: (att + deff) / 2,
        footnote='Bounded scale anchored on a neutral 70, opponent-unadjusted, mildly mean-reverting; higher is always better.',
    ),
}


def sequential_color(accent, frac, light=(0.925, 0.910, 0.855), alpha=1.0):
    """Blend from a light neutral tint up to the system's full accent color."""
    frac = np.clip(frac, 0, 1)
    ar, ag, ab = to_rgb(accent)
    lr, lg, lb = light
    return (lr + (ar - lr) * frac, lg + (ag - lg) * frac, lb + (ab - lb) * frac, alpha)


def col_frac(val, lo, hi, invert=False):
    frac = 0.5 if hi == lo else (val - lo) / (hi - lo)
    return 1 - frac if invert else frac


def build_table(systems, teams, method):
    cfg = SYSTEM_CONFIG[method]
    rows = []
    for t in teams:
        att, deff = systems[method][t]
        rows.append(dict(team=t, att=att, deff=deff, composite=cfg['composite_fn'](att, deff)))
    df = pd.DataFrame(rows).sort_values('composite', ascending=False).reset_index(drop=True)
    df.index += 1
    return df


COLUMNS = [('rank', 0.75), ('team', 2.9), ('gap1', 0.2),
           ('att', 2.6), ('deff', 2.6), ('gap2', 0.2), ('composite', 2.5)]


def col_edges():
    widths = np.array([w for _, w in COLUMNS], dtype=float)
    widths = widths / widths.sum()
    edges = np.concatenate([[0], np.cumsum(widths)])
    return {key: (edges[i], edges[i + 1]) for i, (key, _) in enumerate(COLUMNS)}


MARGIN = 0.018


def plot_table(df, method):
    cfg = SYSTEM_CONFIG[method]
    accent = cfg['accent']
    n = len(df)
    row_h = 0.86
    header_h = 1.55
    title_h = 1.55
    footer_h = 0.7
    fig_h = title_h + header_h + n * row_h + footer_h
    fig, ax = plt.subplots(figsize=(12.6, fig_h * 0.34), facecolor=Q_SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')

    # ── card background + accent header banner ──
    ax.add_patch(mpatches.FancyBboxPatch((MARGIN, 0.03), 1 - 2 * MARGIN, fig_h - 0.06,
                                          boxstyle='round,pad=0,rounding_size=0.012',
                                          facecolor=Q_CARD, edgecolor=Q_GRID, linewidth=1.1, zorder=0))
    ax.add_patch(mpatches.FancyBboxPatch((MARGIN, 0.03), 1 - 2 * MARGIN, title_h,
                                          boxstyle='round,pad=0,rounding_size=0.012',
                                          facecolor=sequential_color(accent, 0.14, light=to_rgb(Q_CARD)),
                                          edgecolor='none', zorder=0))
    ax.add_patch(mpatches.Rectangle((MARGIN, title_h - 0.05), 1 - 2 * MARGIN, 0.05,
                                     facecolor=accent, edgecolor='none', zorder=0))

    pad_x = MARGIN + 0.016
    ax.text(pad_x, title_h * 0.36, f"Eredivisie 2025/26 — {cfg['title']}",
            ha='left', va='center', color=Q_INK, fontsize=17, fontweight='bold', zorder=2)
    ax.text(pad_x, title_h * 0.74,
            f"Full 2025/26 season, in native units ({cfg['unit']})  ·  sorted by {cfg['composite_label']}",
            ha='left', va='center', color=Q_MUTED, fontsize=9.2, zorder=2)

    edges = col_edges()

    def xmid(key):
        l, r = edges[key]
        return (l + r) / 2

    def xscale(key, frac_in_margin):
        # rescale a 0..1 fraction from full-width coords into the card's inset
        return MARGIN + frac_in_margin * (1 - 2 * MARGIN)

    label_map = {
        'att': (cfg['att_label'], cfg['att_sub']),
        'deff': (cfg['def_label'], cfg['def_sub']),
        'composite': (cfg['composite_label'], cfg['composite_sub']),
    }
    line1_y = title_h + header_h * 0.40
    line2_y = title_h + header_h * 0.76
    for key, _ in COLUMNS:
        if key.startswith('gap'):
            continue
        ha = 'left' if key == 'team' else 'center'
        x = xscale('team', edges[key][0]) + 0.014 if key == 'team' else xscale(key, xmid(key))
        if key in label_map:
            l1, l2 = label_map[key]
            ax.text(x, line1_y, l1, ha=ha, va='center', color=Q_INK, fontsize=9.3, fontweight='bold', zorder=2)
            ax.text(x, line2_y, l2, ha=ha, va='center', color=Q_MUTED, fontsize=7.3, style='italic', zorder=2)
        else:
            ax.text(x, (line1_y + line2_y) / 2, 'Rk' if key == 'rank' else 'Team',
                    ha=ha, va='center', color=Q_INK, fontsize=9.3, fontweight='bold', zorder=2)
    ax.plot([MARGIN + 0.01, 1 - MARGIN - 0.01],
            [title_h + header_h - 0.06, title_h + header_h - 0.06], color=Q_GRID, linewidth=1.3, zorder=1)

    att_lo, att_hi = df['att'].min(), df['att'].max()
    def_lo, def_hi = df['deff'].min(), df['deff'].max()
    comp_lo, comp_hi = df['composite'].min(), df['composite'].max()

    bar_h = 0.16
    for i, (_, row) in enumerate(df.iterrows()):
        y0 = title_h + header_h + i * row_h
        yc = y0 + row_h / 2

        if i % 2 == 1:
            ax.add_patch(mpatches.Rectangle((MARGIN + 0.004, y0), 1 - 2 * MARGIN - 0.008, row_h,
                                             facecolor=Q_BAND, edgecolor='none', zorder=0))

        rank = df.index[i]
        rx = xscale('rank', xmid('rank'))
        if rank in MEDALS:
            ax.scatter([rx], [yc], s=340, marker='o', color=MEDALS[rank], zorder=2, edgecolor='none')
            ax.text(rx, yc, str(rank), ha='center', va='center', color='white', fontsize=8.6,
                    fontweight='bold', zorder=3)
        else:
            ax.text(rx, yc, str(rank), ha='center', va='center', color=Q_MUTED, fontsize=9, zorder=2)

        team = row['team']
        swatch_x = xscale('team', edges['team'][0]) + 0.014
        ax.add_patch(mpatches.Rectangle((swatch_x, yc - 0.19), 0.013, 0.38,
                                         facecolor=TEAM_COLORS.get(team, '#888'), edgecolor='none', zorder=2))
        ax.text(swatch_x + 0.026, yc, TEAM_SHORT.get(team, team), ha='left', va='center',
                color=Q_INK, fontsize=9.6, fontweight='bold', zorder=2)

        for key, val, lo, hi, invert in [
            ('att', row['att'], att_lo, att_hi, False),
            ('deff', row['deff'], def_lo, def_hi, not cfg['def_higher_is_better']),
            ('composite', row['composite'], comp_lo, comp_hi, False),
        ]:
            l, r = edges[key]
            l, r = xscale(key, l), xscale(key, r)
            frac = col_frac(val, lo, hi, invert=invert)
            bold = key == 'composite'

            # value, top of cell
            ax.text((l + r) / 2, yc - 0.15, cfg['fmt'].format(val), ha='center', va='center',
                    color=Q_INK, fontsize=9.8 if bold else 9.2, fontweight='bold' if bold else 'normal', zorder=2)

            # proportional data-bar, bottom of cell
            bar_pad = 0.01
            track_l, track_r = l + bar_pad, r - bar_pad
            ax.add_patch(mpatches.FancyBboxPatch((track_l, yc + 0.12), track_r - track_l, bar_h,
                                                  boxstyle='round,pad=0,rounding_size=0.008',
                                                  facecolor=Q_TRACK, edgecolor='none', zorder=1))
            bar_w = max((track_r - track_l) * frac, 0.006)
            ax.add_patch(mpatches.FancyBboxPatch((track_l, yc + 0.12), bar_w, bar_h,
                                                  boxstyle='round,pad=0,rounding_size=0.008',
                                                  facecolor=sequential_color(accent, frac), edgecolor='none', zorder=2))

    ax.text(pad_x, fig_h - footer_h * 0.62, cfg['footnote'],
            ha='left', va='center', color=Q_MUTED, fontsize=7.9)
    ax.text(pad_x, fig_h - footer_h * 0.2,
            'Data: Opta (shot-event goals)  ·  bar length/shade = rank within this column — the printed number is the real value',
            ha='left', va='center', color=Q_MUTED, fontsize=7.5)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = f"{OUT_DIR}/{cfg['file']}"
    plt.savefig(out_path, dpi=150, facecolor=Q_SURFACE)
    plt.close()
    print('Saved:', out_path)


def main():
    systems, teams, league_avg = compute_all_systems_native_season()
    for method in METHOD_ORDER:
        df = build_table(systems, teams, method)
        plot_table(df, method)


if __name__ == '__main__':
    main()
