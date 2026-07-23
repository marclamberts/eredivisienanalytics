#!/bin/bash
# Ensures the meridian-housestyle skill (Marc Lamberts' Waltzing Analytics
# house style for charts/graphs/dashboards) is available in this Claude
# Code web session, even though this repo doesn't carry it directly.
#
# Claude Code web spins up a fresh container per session and only
# auto-discovers skills already committed to this repo's own
# .claude/skills/ folder — there's no cross-repo global for web sessions
# the way ~/.claude/skills/ is for local CLI installs. So this hook fetches
# the skill from its home repo (marclamberts/housestyle) at the start of
# every session here, rather than requiring it to be copied in by hand.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

SKILL_DIR="$CLAUDE_PROJECT_DIR/.claude/skills/meridian-housestyle"

# Idempotent: nothing to do if it's already here (e.g. copied in directly,
# or a previous run in the same session already fetched it).
if [ -d "$SKILL_DIR" ]; then
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if git clone --depth 1 --filter=blob:none --sparse -q \
    https://github.com/marclamberts/housestyle.git "$TMP_DIR" 2>/dev/null; then
  git -C "$TMP_DIR" sparse-checkout set .claude/skills/meridian-housestyle -q 2>/dev/null || true
  if [ -d "$TMP_DIR/.claude/skills/meridian-housestyle" ]; then
    mkdir -p "$CLAUDE_PROJECT_DIR/.claude/skills"
    cp -r "$TMP_DIR/.claude/skills/meridian-housestyle" "$CLAUDE_PROJECT_DIR/.claude/skills/"
  fi
fi
# Best-effort: if the fetch fails (offline, network policy blocks GitHub),
# the session still starts normally — it just won't have the skill this run.
exit 0
