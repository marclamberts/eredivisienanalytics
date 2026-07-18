"""
Attacking & defensive rating trend charts for Ajax, Feyenoord and PSV — 2025/26.

Rating-engine details live in ratings_common.py (shared with ratings_table.py).
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

from ratings_common import compute_all_systems, MA_WINDOW, METHOD_ORDER

OUT_PATH = 'Visuals/Rating/top3rating.png'
OUT_DIR_INDIVIDUAL = 'Visuals/Rating'

HIGHLIGHT_TEAMS = ['AFC Ajax', 'Feyenoord Rotterdam', 'PSV Eindhoven']
TEAM_FILE_SLUG = {
    'AFC Ajax':            'ajax',
    'Feyenoord Rotterdam': 'feyenoord',
    'PSV Eindhoven':       'psv',
}

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
    systems, teams, league_avg = compute_all_systems()

    plot_combined(systems)
    for team in HIGHLIGHT_TEAMS:
        plot_individual(systems, team)


if __name__ == '__main__':
    main()
