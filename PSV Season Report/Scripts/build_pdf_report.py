"""
Assemble the PSV Eindhoven 2025/26 Season Report PDF: a cover page,
section divider pages, and every generated chart as a full-bleed
landscape page. Grammar mirrors match_report_addons.py's
build_match_pdf (Netherlands-Sweden report style).

Usage: python3 build_pdf_report.py [out.pdf]
"""
import sys
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from PIL import Image

VIS = Path("/Users/marclamberts/Event data/Eredivisie/PSV Season Report/Visuals")
MATCH = VIS / "Match Style"
METRIC = VIS / "Metric Pitches"
EXP = VIS / "Expansion"

BG = "#0d1117"
PANEL = "#161b22"
GRID = "#26384d"
TEXT = "#f5f7fb"
MUTED = "#8ca0b3"
GOLD = "#ffd166"
PSV_COLOR = "#ff8a3d"

SEASON_SUMMARY = "34 games · 27W 3D 4L · 101 GF – 45 GA (GD +56) · 84 points"

SECTIONS = [
    ("Season Story",
     "How the 2025/26 title race unfolded: results, cumulative points, "
     "PI-rating strength versus the rest of the Eredivisie, and the "
     "underlying xG trend behind the record.",
     [
        MATCH / "06_results_strip_psv.png",
        MATCH / "04_points_trajectory_psv.png",
        MATCH / "05_home_away_scoutcard_psv.png",
        VIS / "pi_rating_progression_eredivisie.png",
        VIS / "pi_rating_progression_raw_eredivisie.png",
        VIS / "pi_rating_vs_points_eredivisie.png",
        VIS / "home_away_rating_gap_eredivisie.png",
        VIS / "schedule_difficulty_eredivisie.png",
        VIS / "xg_for_against_trend_psv.png",
        VIS / "rolling_xgd_form_guide_eredivisie.png",
        VIS / "weighted_xg_goal_tiers_eredivisie.png",
        VIS / "attack_defense_quadrant_eredivisie.png",
     ]),
    ("Shooting & Chance Quality",
     "Where PSV's 597 shots and 101 goals came from this season, how "
     "dangerous those chances were, and how the attack compares to the "
     "league average across the board. Individual pitch breakdowns by "
     "xG, danger score, and body part follow the season summary views.",
     [
        MATCH / "01_shot_map_season_psv.png",
        MATCH / "02_danger_heatmap_season_psv.png",
        MATCH / "03_radar_vs_league_psv.png",
        METRIC / "01_shotmap_xg_psv.png",
        METRIC / "02_shotmap_danger_psv.png",
        METRIC / "03_shotmap_types_psv.png",
        VIS / "shot_assists_by_body_part_psv.png",
        VIS / "crosses_to_shots_psv.png",
        VIS / "xt_map_psv.png",
     ]),
    ("Chance Creation: Individual Pitch Maps",
     "Every pass type that builds a PSV attack, broken out onto its own "
     "pitch: who created shots, who supplied the assist and the pass "
     "before it, and where progressive, through, half-space, Zone 14, "
     "and cutback passes actually happened.",
     [
        METRIC / "04_key_passes_psv.png",
        METRIC / "05_shot_assists_psv.png",
        METRIC / "06_second_assists_psv.png",
        METRIC / "07_progressive_passes_psv.png",
        METRIC / "08_through_passes_psv.png",
        METRIC / "09_halfspace_passes_psv.png",
        METRIC / "10_zone14_passes_psv.png",
        METRIC / "11_cutbacks_psv.png",
     ]),
    ("Buildup, Progression & Chance Creation",
     "How PSV moved the ball from back to front: pass networks, "
     "buildup patterns, progressive carries, box entries, and the "
     "wide/half-space delivery mix.",
     [
        VIS / "pass_network_psv.png",
        VIS / "pass_network_clusters_psv.png",
        VIS / "buildup_patterns_psv.png",
        VIS / "goalkeeper_buildup_eredivisie.png",
        VIS / "key_passes_into_box_psv.png",
        VIS / "linebreaking_passes_psv.png",
        VIS / "long_passes_final_third_psv.png",
        VIS / "switches_psv.png",
        VIS / "box_entries_pass_vs_carry_psv.png",
        VIS / "carries_wide_to_box_psv.png",
        VIS / "progressive_carries_final_third_psv.png",
        VIS / "crosses_wide_psv.png",
        VIS / "crosses_halfspace_psv.png",
     ]),
    ("Team Identity & Territory",
     "PSV's positional and stylistic profile relative to the rest of "
     "the league: directness, relationism, territorial control, and "
     "field tilt over the season.",
     [
        VIS / "team_directness_eredivisie.png",
        VIS / "relationism_index_eredivisie.png",
        VIS / "relation_position_quadrant_eredivisie.png",
        VIS / "zone_control_eredivisie.png",
        VIS / "field_tilt_trend_psv.png",
        EXP / "04_width_of_play_psv.png",
        EXP / "05_line_height_trend_psv.png",
        EXP / "02_pass_completion_matrix_psv.png",
        EXP / "03_forward_pass_matrix_psv.png",
     ]),
    ("Chance Creation, Part II: More Individual Pitches",
     "Further breakdowns mined from the Netherlands-Sweden report style, "
     "rebuilt as season aggregates: goal-mouth placement, shot types by "
     "situation, long balls, carries, individual player heatmaps, and "
     "touches inside the box.",
     [
        EXP / "01_goal_mouth_psv.png",
        EXP / "17_shot_types_situation_psv.png",
        EXP / "08_player_heatmaps_psv.png",
        EXP / "09_long_ball_map_psv.png",
        EXP / "10_carry_map_psv.png",
        EXP / "14_touches_in_box_psv.png",
        EXP / "15_free_kick_map_psv.png",
     ]),
    ("Defending, Pressing & Set Pieces",
     "The other side of the ball: defensive actions, pressing "
     "intensity by zone, and how PSV attacked and defended set-piece "
     "second balls.",
     [
        VIS / "defensive_actions_psv.png",
        VIS / "ppda_trend_psv.png",
        VIS / "pressures_by_third_psv.png",
        VIS / "second_balls_league_eredivisie.png",
        VIS / "set_piece_second_balls_psv.png",
        VIS / "corner_routines_eredivisie.png",
        VIS / "long_throwins_psv.png",
        EXP / "11_duels_map_psv.png",
        EXP / "12_press_map_psv.png",
        EXP / "13_ball_losses_psv.png",
        EXP / "16_box_defending_psv.png",
     ]),
    ("Shots Against: The Defensive Mirror",
     "Everything the attacking sections show, flipped: every shot PSV "
     "faced this season, where the danger came from, mapped from PSV's "
     "own defensive perspective.",
     [
        EXP / "06_xg_conceded_psv.png",
        EXP / "07_danger_against_psv.png",
     ]),
]


def draw_cover(c, width, height):
    c.setFillColor(colors.HexColor(BG))
    c.rect(0, 0, width, height, fill=True, stroke=False)

    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(24 * mm, height - 40 * mm, "WALTZING ANALYTICS · SEASON REPORT")

    c.setFillColor(colors.HexColor(TEXT))
    c.setFont("Helvetica-Bold", 42)
    c.drawString(24 * mm, height - 66 * mm, "PSV Eindhoven")

    c.setFillColor(colors.HexColor(GOLD))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(24 * mm, height - 80 * mm, "2025/26 Eredivisie Season Report")

    c.setFillColor(colors.HexColor(PSV_COLOR))
    c.setFont("Helvetica-Bold", 15)
    c.drawString(24 * mm, height - 96 * mm, SEASON_SUMMARY)

    c.setStrokeColor(colors.HexColor(GRID))
    c.setLineWidth(1.0)
    c.line(24 * mm, height - 106 * mm, width - 24 * mm, height - 106 * mm)

    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Helvetica", 11)
    blurb = (
        "Full-season aggregate report combining a league-wide team-viz "
        "chart set (pass networks, PI ratings, zone control, PPDA, "
        "xG trends) with match-report-style visuals (season shot map, "
        "danger heatmap, radar, points trajectory) rolled up across all "
        "34 Eredivisie fixtures."
    )
    y = height - 120 * mm
    for line in textwrap.wrap(blurb, width=100):
        c.drawString(24 * mm, y, line)
        y -= 5.6 * mm

    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Helvetica", 9)
    c.drawString(24 * mm, 16 * mm, "Data: Opta event data · Eredivisie 2025/26")
    c.drawRightString(width - 24 * mm, 16 * mm, "Marc Lamberts · Waltzing Analytics")


def draw_section_page(c, width, height, section, desc, page_no, total):
    c.setFillColor(colors.HexColor(BG))
    c.rect(0, 0, width, height, fill=True, stroke=False)
    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(24 * mm, height - 38 * mm, "SEASON REPORT SECTION")
    c.setFillColor(colors.HexColor(TEXT))
    c.setFont("Helvetica-Bold", 30)
    c.drawString(24 * mm, height - 54 * mm, section)
    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Helvetica", 11)
    y = height - 70 * mm
    for line in textwrap.wrap(desc, width=96):
        c.drawString(24 * mm, y, line)
        y -= 6 * mm
    c.setStrokeColor(colors.HexColor(GRID))
    c.setLineWidth(1.0)
    c.line(24 * mm, 28 * mm, width - 24 * mm, 28 * mm)
    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Helvetica", 7)
    c.drawRightString(width - 10 * mm, 6 * mm, f"{page_no}/{total}")


def draw_image_page(c, width, height, image_path, page_no, total):
    margin = 9 * mm
    c.setFillColor(colors.HexColor(BG))
    c.rect(0, 0, width, height, fill=True, stroke=False)
    with Image.open(image_path) as img:
        iw, ih = img.size
    max_w = width - 2 * margin
    max_h = height - 2 * margin
    scale = min(max_w / iw, max_h / ih)
    draw_w, draw_h = iw * scale, ih * scale
    x = (width - draw_w) / 2
    y = (height - draw_h) / 2
    c.drawImage(str(image_path), x, y, draw_w, draw_h, preserveAspectRatio=True, mask="auto")
    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Helvetica", 7)
    c.drawRightString(width - margin, 6 * mm, f"{page_no}/{total}")


def build(out_path):
    missing = []
    items = []
    for section, desc, paths in SECTIONS:
        items.append((section, desc, None))
        for p in paths:
            if p.exists():
                items.append((section, "", p))
            else:
                missing.append(str(p))
    if missing:
        print("WARNING missing files:")
        for m in missing:
            print(" -", m)

    page_size = landscape(A4)
    c = canvas.Canvas(str(out_path), pagesize=page_size)
    width, height = page_size

    draw_cover(c, width, height)
    c.showPage()

    total = len(items)
    for idx, (section, desc, image_path) in enumerate(items, 1):
        if image_path is None:
            draw_section_page(c, width, height, section, desc, idx, total)
        else:
            draw_image_page(c, width, height, image_path, idx, total)
        c.showPage()
    c.save()
    print("Saved:", out_path)
    print(f"Total pages: {total + 1} (cover + {len(SECTIONS)} sections + "
          f"{sum(len(p) for _, _, p in SECTIONS)} charts)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/marclamberts/Event data/Eredivisie/PSV Season Report/PSV Eindhoven 2025-26 Season Report.pdf")
    build(out)
