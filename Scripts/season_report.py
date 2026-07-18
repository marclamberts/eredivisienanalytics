"""
Season analysis report: shooting, passing, key passing, crossing, build-up,
transition, set pieces, goalkeeping, entries - one team, one PDF.

Data + methodology notes live in season_report_data.py. Every number in this
report is derived from Events/*.json (Opta-style event stream) and
Danger/all_eredivisie_danger_models.csv (shot xG/PSxG model), both already
used elsewhere in this repo (ratings, build-up pattern charts). Nothing here
is a "black box" stat - see that module's docstring for exactly which event
qualifiers were used and how they were verified.

Usage: python3 season_report.py "PSV Eindhoven" [out.pdf]
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mplsoccer import VerticalPitch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from buildup_patterns import BG, PITCH_LINE, C_AMBER, C_GREEN, add_logo, clean_name
from ratings_common import load_matches
from season_report_data import (
    resolve_team, load_shots, compute_passing, compute_key_passes_assists,
    compute_buildup_profile, compute_transitions, compute_setpieces,
    identify_keeper, compute_gk_distribution, FINAL_THIRD_X,
)

INK = '#e6e9ee'
MUTED = '#9aa4b2'
GRID = '#232a36'
ACCENT2 = '#f06fa3'
PAGE_SIZE = (16.5, 11.7)  # landscape A3-ish


def new_page():
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    return fig


def page_header(fig, title, subtitle):
    fig.text(0.035, 0.955, title, fontsize=25, fontweight='bold', color='white')
    fig.text(0.035, 0.917, subtitle, fontsize=12.5, color=MUTED)
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.895, 0.895], color=GRID, linewidth=1.2, transform=fig.transFigure))
    add_logo(fig, y=0.955)


def page_footer(fig, note):
    fig.text(0.035, 0.02, note, fontsize=8, color=MUTED)
    fig.text(0.965, 0.02, 'Marc Lamberts · Waltzing Analytics', fontsize=9, ha='right',
             color=MUTED, style='italic')


def stat_cards(fig, x0, x1, y, cards, value_size=27, label_size=10):
    """cards: list of (value_str, label_str[, color])"""
    n = len(cards)
    w = (x1 - x0) / n
    for i, card in enumerate(cards):
        val, label = card[0], card[1]
        color = card[2] if len(card) > 2 else 'white'
        cx = x0 + w * (i + 0.5)
        fig.text(cx, y, val, fontsize=value_size, fontweight='bold', ha='center', va='center', color=color)
        fig.text(cx, y - 0.038, label, fontsize=label_size, ha='center', va='center', color=MUTED)
        if i > 0:
            fig.add_artist(plt.Line2D([x0 + w * i, x0 + w * i], [y - 0.06, y + 0.03],
                                       color=GRID, linewidth=1, transform=fig.transFigure))


def hbar(ax, labels, values, color=C_AMBER, fmt='{:.0f}', unit=''):
    ax.set_facecolor(BG)
    y = range(len(labels))
    ax.barh(list(y), values, color=color, height=0.55, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    vmax = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(v + vmax * 0.02, i, fmt.format(v) + unit, va='center', color='white', fontsize=9.5)
    ax.set_xlim(0, vmax * 1.2 if vmax else 1)


# ── page 1: cover ────────────────────────────────────────────────────────────
def page_cover(team_short, season_row):
    fig = new_page()
    fig.text(0.5, 0.62, season_row['team_full'], fontsize=44, fontweight='bold', ha='center', color='white')
    fig.text(0.5, 0.555, 'Season Analysis Report  ·  Eredivisie 2025/26', fontsize=17, ha='center', color=MUTED)
    fig.add_artist(plt.Line2D([0.30, 0.70], [0.50, 0.50], color=C_AMBER, linewidth=2.4, transform=fig.transFigure))

    cards = [
        (str(season_row['mp']), 'Played'),
        (f"{season_row['w']}-{season_row['d']}-{season_row['l']}", 'W-D-L'),
        (str(season_row['pts']), 'Points'),
        (str(season_row['gf']), 'Goals For'),
        (str(season_row['ga']), 'Goals Against'),
        (f"{season_row['gd']:+d}", 'Goal Diff', C_GREEN if season_row['gd'] >= 0 else ACCENT2),
    ]
    stat_cards(fig, 0.12, 0.88, 0.36, cards, value_size=30, label_size=11)

    fig.text(0.5, 0.10, 'Shooting  ·  Passing  ·  Key Passing  ·  Crossing  ·  Build-Up  ·  '
             'Transition  ·  Set Pieces  ·  Goalkeeping  ·  Entries', fontsize=11, ha='center', color=MUTED)
    add_logo(fig, y=0.955)
    fig.text(0.035, 0.02, 'Data: Opta event stream + shot xG/PSxG model, reconstructed from '
             'Events/*.json and Danger/all_eredivisie_danger_models.csv', fontsize=8, color=MUTED)
    fig.text(0.965, 0.02, 'Marc Lamberts · Waltzing Analytics', fontsize=9, ha='right', color=MUTED, style='italic')
    return fig


# ── page 2: shooting ─────────────────────────────────────────────────────────
def page_shooting(team, shots):
    fig = new_page()
    page_header(fig, 'Shooting', f'{team} · every shot taken this season, sized by xG')

    for_ = shots[shots['side'] == 'for']
    n_shots = len(for_)
    goals = int(for_['is_goal'].sum())
    on_target = int(for_['is_on_target'].sum())
    xg = for_['xg'].sum()
    npxg = for_[for_['is_penalty'] == 0]['xg'].sum()

    cards = [
        (str(n_shots), 'Shots'),
        (f"{on_target/n_shots*100:.0f}%", 'On Target'),
        (str(goals), 'Goals', C_GREEN),
        (f"{xg:.1f}", 'xG'),
        (f"{xg/n_shots:.2f}", 'xG / Shot'),
        (f"{goals-xg:+.1f}", 'Goals vs xG', C_GREEN if goals - xg >= 0 else ACCENT2),
    ]
    stat_cards(fig, 0.06, 0.62, 0.885, cards, value_size=22, label_size=9)

    ax = fig.add_axes([0.06, 0.06, 0.55, 0.72])
    pitch = VerticalPitch(pitch_type='opta', pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1, half=True)
    pitch.draw(ax=ax)
    goals_df = for_[for_['is_goal'] == 1]
    miss_df = for_[for_['is_goal'] == 0]
    pitch.scatter(miss_df['x'], miss_df['y'], ax=ax, s=miss_df['xg'] * 900 + 25, color=C_AMBER,
                  alpha=0.55, edgecolors=BG, linewidths=0.8, zorder=2)
    pitch.scatter(goals_df['x'], goals_df['y'], ax=ax, s=goals_df['xg'] * 900 + 40, color=C_GREEN,
                  alpha=0.9, edgecolors=BG, linewidths=1.0, marker='*', zorder=3)
    fig.text(0.06 + 0.55 / 2, 0.795, 'Shot Map  (● miss/save/post, ★ goal — size = xG)',
             fontsize=11, ha='center', color=INK, transform=fig.transFigure)

    ax2 = fig.add_axes([0.68, 0.50, 0.28, 0.30])
    hbar(ax2, ['Goals', 'On target\n(no goal)', 'Off target /\nblocked'],
         [goals, on_target - goals, n_shots - on_target], color=C_AMBER)
    ax2.set_title('Shot Outcomes', color=INK, fontsize=12, loc='left')

    npen = int(for_['is_penalty'].sum())
    ax3 = fig.add_axes([0.68, 0.10, 0.28, 0.30])
    ax3.axis('off')
    ax3.text(0, 0.9, f'Penalties: {npen} taken', color=INK, fontsize=11, transform=ax3.transAxes)
    ax3.text(0, 0.7, f'Penalty goals: {int(for_[for_["is_penalty"]==1]["is_goal"].sum())}',
             color=INK, fontsize=11, transform=ax3.transAxes)
    ax3.text(0, 0.5, f'npxG: {npxg:.1f}', color=INK, fontsize=11, transform=ax3.transAxes)
    ax3.text(0, 0.25, '* set-piece-originated shots are broken out separately\non the Set Pieces page.',
             color=MUTED, fontsize=8.5, style='italic', transform=ax3.transAxes)

    page_footer(fig, 'Data: Opta shot-event xG/PSxG model · xG = expected goals per shot, summed')
    return fig


# ── page 3: passing ──────────────────────────────────────────────────────────
def page_passing(team, pdf, n_matches):
    fig = new_page()
    page_header(fig, 'Passing', f'{team} · {n_matches} matches · all completed + attempted passes')

    n = len(pdf)
    completed = int(pdf['completed'].sum())
    prog = int(pdf['progressive'].sum())
    cards = [
        (f"{n/n_matches:.0f}", 'Passes / Match'),
        (f"{completed/n*100:.1f}%", 'Completion'),
        (f"{prog/n_matches:.1f}", 'Progressive* / Match'),
        (f"{int(pdf['is_long'].sum())/n_matches:.1f}", 'Long Balls / Match'),
        (f"{int(pdf['final_third_entry'].sum())/n_matches:.1f}", 'Final-3rd Entries / Match'),
        (f"{int(pdf['box_entry'].sum())/n_matches:.1f}", 'Box Entries / Match'),
    ]
    stat_cards(fig, 0.06, 0.94, 0.83, cards, value_size=21, label_size=9)

    zones = [(0, 100 / 3, 'Def. Third'), (100 / 3, FINAL_THIRD_X, 'Middle Third'),
             (FINAL_THIRD_X, 100, 'Att. Third')]
    labels, comp_pct, vols = [], [], []
    for lo, hi, name in zones:
        sub = pdf[(pdf['x'] >= lo) & (pdf['x'] < hi)]
        labels.append(name)
        comp_pct.append(sub['completed'].mean() * 100 if len(sub) else 0)
        vols.append(len(sub))

    ax1 = fig.add_axes([0.06, 0.12, 0.40, 0.60])
    hbar(ax1, labels, comp_pct, color=C_AMBER, fmt='{:.0f}', unit='%')
    ax1.set_title('Completion % by Starting Third', color=INK, fontsize=13, loc='left')

    ax2 = fig.add_axes([0.55, 0.12, 0.40, 0.60])
    hbar(ax2, labels, vols, color=ACCENT2, fmt='{:.0f}')
    ax2.set_title('Pass Volume by Starting Third', color=INK, fontsize=13, loc='left')

    page_footer(fig, '* Progressive = completed pass that cuts the remaining distance to the byline by >=25%. '
                'Final-third/box entries = completed passes starting outside and landing inside that zone.')
    return fig


# ── page 4: key passing & assists ───────────────────────────────────────────
def page_keypasses(team, kpa, goals_total):
    fig = new_page()
    page_header(fig, 'Key Passing & Assists', f'{team} · the pass immediately before each shot, in the same '
                 'uninterrupted possession')

    n_kp = len(kpa)
    n_assists = int(kpa['is_goal'].sum())
    n_cross_assist = int(kpa[kpa['is_goal']]['is_cross'].sum()) if n_assists else 0
    cards = [
        (str(n_kp), 'Key Passes'),
        (str(n_assists), 'Assists', C_GREEN),
        (f"{n_assists/goals_total*100:.0f}%", 'Goals Assisted'),
        (f"{n_cross_assist}", 'Assists from a Cross'),
        (f"{n_kp/34:.1f}", 'Key Passes / Match'),
    ]
    stat_cards(fig, 0.06, 0.94, 0.83, cards, value_size=23, label_size=9.5)

    top = kpa.groupby('passer').size().sort_values(ascending=False).head(8)
    ax1 = fig.add_axes([0.06, 0.12, 0.40, 0.62])
    hbar(ax1, list(top.index)[::-1], list(top.values)[::-1], color=C_AMBER)
    ax1.set_title('Most Key Passes', color=INK, fontsize=13, loc='left')

    top_a = kpa[kpa['is_goal']].groupby('passer').size().sort_values(ascending=False).head(8)
    ax2 = fig.add_axes([0.55, 0.12, 0.40, 0.62])
    if len(top_a):
        hbar(ax2, list(top_a.index)[::-1], list(top_a.values)[::-1], color=C_GREEN)
    ax2.set_title('Most Assists', color=INK, fontsize=13, loc='left')

    page_footer(fig, 'No "key pass"/"assist" tag exists in this event feed - both are derived structurally: the '
                'completed pass directly preceding a shot (goal, for assists) in the same unbroken possession.')
    return fig


# ── page 5: crossing ─────────────────────────────────────────────────────────
def page_crossing(team, pdf, kpa, n_matches):
    fig = new_page()
    page_header(fig, 'Crossing', f'{team} · passes tagged as crosses (wide-channel origin, near-post/box end)')

    cx = pdf[pdf['is_cross']]
    n = len(cx)
    completed = int(cx['completed'].sum())
    assists = int(kpa[kpa['is_cross'] & kpa['is_goal']].shape[0])
    left = int((cx['y'] > 66.7).sum())
    right = int((cx['y'] < 33.3).sum())
    cards = [
        (f"{n/n_matches:.1f}", 'Crosses / Match'),
        (f"{completed/n*100:.0f}%", 'Completion'),
        (str(assists), 'Assists from Crosses', C_GREEN),
        (f"{left}", 'From Left'),
        (f"{right}", 'From Right'),
    ]
    stat_cards(fig, 0.06, 0.94, 0.885, cards, value_size=22, label_size=9.5)

    ax = fig.add_axes([0.30, 0.06, 0.40, 0.72])
    pitch = VerticalPitch(pitch_type='opta', pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1, half=True)
    pitch.draw(ax=ax)
    good = cx[cx['completed']]
    bad = cx[~cx['completed']]
    for _, r in bad.iterrows():
        if r['end_x'] is not None:
            pitch.lines(r['x'], r['y'], r['end_x'], r['end_y'], ax=ax, color=MUTED, lw=1.0, alpha=0.35, zorder=1)
    for _, r in good.iterrows():
        if r['end_x'] is not None:
            pitch.lines(r['x'], r['y'], r['end_x'], r['end_y'], ax=ax, color=C_AMBER, lw=1.3, alpha=0.75, zorder=2,
                        comet=True)
    fig.text(0.30 + 0.20, 0.795, 'Cross Origins & Destinations  (bright = completed, dim = incomplete)',
             fontsize=11, ha='center', color=INK, transform=fig.transFigure)

    page_footer(fig, 'Cross = pass qualifier verified empirically (wide origin near the byline, ending centrally '
                'near/in the box) · left/right measured by the cross\'s starting y-coordinate')
    return fig


# ── page 6: build-up profile ─────────────────────────────────────────────────
def page_buildup(team, bp_df, style_summary):
    fig = new_page()
    page_header(fig, 'Build-Up', f'{team} · shot-ending possessions, by the third where they started')

    zone_order = ['Defensive Third', 'Middle Third', 'Attacking Third']
    counts = bp_df.groupby('zone').size().reindex(zone_order).fillna(0)
    goal_counts = bp_df.groupby('zone')['is_goal'].sum().reindex(zone_order).fillna(0)
    total = counts.sum()

    cards = [(str(int(counts[z])), z.replace(' Third', '')) for z in zone_order]
    cards.append((str(int(goal_counts.sum())), 'Resulting Goals', C_GREEN))
    stat_cards(fig, 0.06, 0.94, 0.845, cards, value_size=24, label_size=9.5)

    ax1 = fig.add_axes([0.06, 0.12, 0.38, 0.60])
    hbar(ax1, [z.replace(' Third', '') for z in zone_order], list(counts.values), color=C_AMBER)
    ax1.set_title('Shot-Ending Sequences by Starting Third', color=INK, fontsize=12.5, loc='left')

    ax2 = fig.add_axes([0.52, 0.12, 0.42, 0.60])
    ax2.axis('off')
    ax2.text(0, 0.97, 'Positionist vs Relationist', color='white', fontsize=15, fontweight='bold',
             va='top', transform=ax2.transAxes)
    ax2.text(0, 0.80, style_summary, color=INK, fontsize=11, va='top', transform=ax2.transAxes, linespacing=1.7)
    ax2.text(0, 0.30, 'Full pitch maps for each starting third, and for the Positionist/Relationist\n'
             'style split, are in Visuals/BuildUp/ - this page summarises the volumes only.',
             color=MUTED, fontsize=9.5, style='italic', va='top', transform=ax2.transAxes)

    page_footer(fig, 'A build-up sequence = uninterrupted possession (>=1 pass) ending in a shot; classified by the '
                'pitch third where the FIRST touch of that possession occurred.')
    return fig


# ── page 7: transitions ─────────────────────────────────────────────────────
def page_transitions(team, transitions):
    fig = new_page()
    page_header(fig, 'Transition', f'{team} · regain (tackle/interception/recovery) to shot, within '
                 f'{15} seconds')

    n = len(transitions)
    goals = sum(t['is_goal'] for t in transitions)
    avg_elapsed = sum(t['elapsed'] for t in transitions) / n if n else 0
    avg_passes = sum(t['n_passes'] for t in transitions) / n if n else 0
    cards = [
        (str(n), 'Transitions -> Shot'),
        (str(goals), 'Goals', C_GREEN),
        (f"{goals/n*100:.0f}%" if n else '-', 'Conversion'),
        (f"{avg_elapsed:.1f}s", 'Avg. Regain -> Shot'),
        (f"{avg_passes:.1f}", 'Avg. Passes'),
    ]
    stat_cards(fig, 0.06, 0.94, 0.885, cards, value_size=22, label_size=9.5)

    ax = fig.add_axes([0.30, 0.06, 0.40, 0.72])
    pitch = VerticalPitch(pitch_type='opta', pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1, half=False)
    pitch.draw(ax=ax)
    for t in transitions:
        x, y = t['seq'][0]['x'], t['seq'][0]['y']
        color = C_GREEN if t['is_goal'] else C_AMBER
        pitch.scatter(x, y, ax=ax, s=140, color=color, alpha=0.8, edgecolors=BG, linewidths=1.0, zorder=2)
    fig.text(0.30 + 0.20, 0.795, 'Regain Locations  (green = led to a goal)', fontsize=11, ha='center',
             color=INK, transform=fig.transFigure)

    page_footer(fig, 'Transition = a possession that both (a) begins with a tackle, interception, or ball recovery '
                'and (b) reaches a shot within 15 seconds, without the ball changing hands.')
    return fig


# ── page 8: set pieces ───────────────────────────────────────────────────────
def page_setpieces(team, sp_df, pen_taken, pen_goals):
    fig = new_page()
    page_header(fig, 'Set Pieces', f'{team} · corner and free-kick restarts that stayed with the team')

    corners = sp_df[sp_df['kind'] == 'Corner']
    fks = sp_df[sp_df['kind'] == 'Free kick']
    total_sp_goals = int(corners['is_goal'].sum() + fks['is_goal'].sum()) + pen_goals

    cards = [
        (str(len(corners)), 'Corners Taken'),
        (str(int(corners['has_shot'].sum())), 'Corners -> Shot'),
        (str(len(fks)), 'Free Kicks Taken'),
        (str(int(fks['has_shot'].sum())), 'Free Kicks -> Shot'),
        (str(pen_taken), 'Penalties Taken'),
        (str(total_sp_goals), 'Set-Piece Goals*', C_GREEN),
    ]
    stat_cards(fig, 0.06, 0.94, 0.845, cards, value_size=20, label_size=9)

    labels = ['Corner', 'Free Kick', 'Penalty']
    goals_by_kind = [int(corners['is_goal'].sum()), int(fks['is_goal'].sum()), pen_goals]
    ax1 = fig.add_axes([0.10, 0.12, 0.36, 0.60])
    hbar(ax1, labels, goals_by_kind, color=C_GREEN)
    ax1.set_title('Goals by Set-Piece Type', color=INK, fontsize=13, loc='left')

    shots_by_kind = [int(corners['has_shot'].sum()), int(fks['has_shot'].sum()), pen_taken]
    ax2 = fig.add_axes([0.56, 0.12, 0.36, 0.60])
    hbar(ax2, labels, shots_by_kind, color=C_AMBER)
    ax2.set_title('Shots by Set-Piece Type', color=INK, fontsize=13, loc='left')

    page_footer(fig, "* Corner/free-kick goals only count the delivery's own immediate possession (no rebounds "
                "recycled through a second dead-ball restart). Penalties from the shot xG model directly.")
    return fig


# ── page 9: goalkeeping ──────────────────────────────────────────────────────
def page_goalkeeping(team, keeper, shots, gkdf):
    fig = new_page()
    page_header(fig, 'Goalkeeping', f'{team} · {keeper}')

    against = shots[shots['side'] == 'against']
    on_target = against[against['is_on_target'] == 1]
    saves = len(on_target) - int(against['is_goal'].sum())
    psxg = on_target['psxg'].sum()
    ga = int(against['is_goal'].sum())

    cards = [
        (str(len(against)), 'Shots Faced'),
        (str(len(on_target)), 'On Target Faced'),
        (str(ga), 'Goals Conceded', ACCENT2),
        (str(saves), 'Saves', C_GREEN),
        (f"{saves/len(on_target)*100:.0f}%", 'Save %'),
        (f"{psxg-ga:+.1f}", 'Goals Prevented (PSxG-GA)', C_GREEN if psxg - ga >= 0 else ACCENT2),
    ]
    stat_cards(fig, 0.06, 0.94, 0.845, cards, value_size=19, label_size=8.7)

    ax = fig.add_axes([0.10, 0.12, 0.36, 0.60])
    ax.axis('off')
    ax.text(0, 0.9, 'Distribution', color='white', fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.text(0, 0.72, f"{len(gkdf)} passes attempted", color=INK, fontsize=12, transform=ax.transAxes)
    ax.text(0, 0.58, f"{gkdf['completed'].mean()*100:.0f}% completion", color=INK, fontsize=12, transform=ax.transAxes)
    ax.text(0, 0.44, f"{gkdf['is_long'].mean()*100:.0f}% long balls", color=INK, fontsize=12, transform=ax.transAxes)
    ax.text(0, 0.30, f"{len(gkdf)/34:.1f} passes / match", color=INK, fontsize=12, transform=ax.transAxes)

    ax2 = fig.add_axes([0.56, 0.12, 0.36, 0.60])
    hbar(ax2, ['On target,\nsaved', 'On target,\nconceded', 'Blocked/\noff target'],
         [saves, ga, len(against) - len(on_target)], color=C_AMBER)
    ax2.set_title('Shots Faced, Breakdown', color=INK, fontsize=12.5, loc='left')

    page_footer(fig, 'PSxG = post-shot xG of on-target shots faced; Goals Prevented = PSxG minus actual goals '
                'conceded (positive = better than expected shot-stopping, team-level, not isolating defense).')
    return fig


# ── page 10: entries ─────────────────────────────────────────────────────────
def page_entries(team, pdf, n_matches):
    fig = new_page()
    page_header(fig, 'Entries', f'{team} · completed passes into the final third and the penalty box')

    ft = pdf[pdf['final_third_entry']]
    box = pdf[pdf['box_entry']]
    cards = [
        (f"{len(ft)/n_matches:.1f}", 'Final-3rd Entries / Match'),
        (f"{len(box)/n_matches:.1f}", 'Box Entries / Match'),
        (f"{len(box)/len(ft)*100:.0f}%", 'Entries Reaching the Box'),
        (str(int(ft['is_cross'].sum())), 'Via a Cross'),
        (str(int(ft['is_long'].sum())), 'Via a Long Ball'),
    ]
    stat_cards(fig, 0.06, 0.94, 0.885, cards, value_size=21, label_size=9)

    ax = fig.add_axes([0.30, 0.06, 0.40, 0.72])
    pitch = VerticalPitch(pitch_type='opta', pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1, half=True)
    pitch.draw(ax=ax)
    pitch.scatter(box['end_x'], box['end_y'], ax=ax, s=40, color=C_GREEN, alpha=0.6, edgecolors=BG,
                  linewidths=0.5, zorder=3, label='Box entry')
    other_ft = ft[~ft['box_entry']]
    pitch.scatter(other_ft['end_x'], other_ft['end_y'], ax=ax, s=22, color=C_AMBER, alpha=0.35, edgecolors='none',
                  zorder=2)
    fig.text(0.30 + 0.20, 0.795, 'Entry Endpoints  (green = into the box, amber = final third only)',
             fontsize=11, ha='center', color=INK, transform=fig.transFigure)

    page_footer(fig, 'Final-third entry: completed pass starting outside, landing inside x>=66.7. Box entry: '
                'completed pass landing inside the penalty area (x>=83, 21<=y<=79), not already there.')
    return fig


def main():
    team_query = sys.argv[1] if len(sys.argv) > 1 else 'PSV Eindhoven'
    team, cid = resolve_team(team_query)
    short = clean_name(team)
    out = sys.argv[2] if len(sys.argv) > 2 else f"Visuals/SeasonReport/{team.replace(' ', '_')}_Season_Report.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f'Building season report for {team}...')

    matches = load_matches()
    tm = matches[(matches.home == team) | (matches.away == team)]
    w = d = l = gf = ga = 0
    for r in tm.itertuples():
        is_home = r.home == team
        gf_m, ga_m = (r.home_goals, r.away_goals) if is_home else (r.away_goals, r.home_goals)
        gf += gf_m
        ga += ga_m
        if gf_m > ga_m:
            w += 1
        elif gf_m == ga_m:
            d += 1
        else:
            l += 1
    season_row = dict(team_full=short, mp=len(tm), w=w, d=d, l=l, gf=int(gf), ga=int(ga),
                       gd=int(gf - ga), pts=w * 3 + d)
    n_matches = len(tm)

    shots = load_shots(team)
    pdf_passes, _ = compute_passing(team)
    kpa = compute_key_passes_assists(team)
    bp = compute_buildup_profile(team)
    transitions = compute_transitions(team)
    sp = compute_setpieces(team)
    keeper = identify_keeper(team)
    gkdf = compute_gk_distribution(team, keeper)

    for_shots = shots[shots['side'] == 'for']
    pen_taken = int(for_shots['is_penalty'].sum())
    pen_goals = int(for_shots[for_shots['is_penalty'] == 1]['is_goal'].sum())

    try:
        from style_patterns import classify_by_style
        from buildup_patterns import collect_sequences
        import glob
        files = sorted(glob.glob('Events/*.json'))
        qualifying = collect_sequences(files, cid, zone_lo=0, zone_hi=100)
        relationist, positionist, median_dist = classify_by_style(qualifying)
        style_summary = (f"{len(qualifying)} qualifying build-up sequences this season.\n"
                          f"{len(positionist)} lean Positionist (longer, zone-to-zone passing),\n"
                          f"{len(relationist)} lean Relationist (tighter combination play),\n"
                          f"split at this team's own median action distance\n"
                          f"of {median_dist:.1f} (Opta pitch units).")
    except Exception as ex:
        style_summary = f"(style split unavailable: {ex})"

    print('Rendering pages...')
    with PdfPages(out) as pdf_out:
        for fig in [
            page_cover(short, season_row),
            page_shooting(short, shots),
            page_passing(short, pdf_passes, n_matches),
            page_keypasses(short, kpa, season_row['gf']),
            page_crossing(short, pdf_passes, kpa, n_matches),
            page_buildup(short, bp, style_summary),
            page_transitions(short, transitions),
            page_setpieces(short, sp, pen_taken, pen_goals),
            page_goalkeeping(short, keeper, shots, gkdf),
            page_entries(short, pdf_passes, n_matches),
        ]:
            pdf_out.savefig(fig, facecolor=BG)
            plt.close(fig)

    print('Saved:', out)


if __name__ == '__main__':
    main()
