#!/usr/bin/env bash
# =============================================================================
#  argos-runner.sh
# =============================================================================
#  Runs Argos in a loop against the project this script lives in.
#
#  Each iteration sends a prompt to Argos via the ACP (Agent Client Protocol)
#  using the Windsurf-bundled credentials, then sleeps for INTERVAL seconds
#  and repeats.
#
#  Session continuation: by default a fresh session is used each run. Set
#  SESSION_CONTINUATION (or --session-continuation) to N to keep the same
#  session for N consecutive runs before starting a new one. This lets Argos
#  remember the conversation context between runs (e.g. "any new issues since
#  your last check?"). The devin acp process stays alive for the entire loop
#  so sessions remain accessible.
#
#  This script is a single self-contained file. It embeds a Python ACP
#  client (extracted to a temp file at runtime) that talks to Devin via
#  `devin acp`, bypassing the `devin -p` auth requirement by using the
#  windsurf-api-key authentication method from ~/.local/share/devin/
#  credentials.toml.
#
#  Portable: copy this ONE file into any project root and run it there --
#  it auto-detects its own directory as the workspace.
#
#  Runner registry: every running instance registers itself in a shared state
#  directory (STATE_DIR below). On startup this script lists all OTHER live
#  argos-runner runners on the machine so you can see what else is running,
#  and cleans up stale entries left by crashed instances.
#
#  Usage:
#    ./argos-runner.sh                  # run with defaults (every 10 min)
#    ./argos-runner.sh --once           # single run, then exit
#    ./argos-runner.sh --interval 300   # custom interval (seconds)
#    ./argos-runner.sh --timeout 120    # kill runs that take >2 min
#    ./argos-runner.sh --loops 5        # stop after 5 runs (default: infinite)
#    ./argos-runner.sh --session-continuation 3  # same session for 3 runs
#    ./argos-runner.sh --prompt "..."   # override the prompt for this run
#    ./argos-runner.sh --model glm      # override the model for this run
#    ./argos-runner.sh --log /path      # write run output to a log file
#    ./argos-runner.sh --dry-run        # show what would run, don't call Argos
#    ./argos-runner.sh --no-color       # disable colored output
#    ./argos-runner.sh --check-auth     # verify credentials and exit
#    ./argos-runner.sh --status         # show all running argos-runner instances
#    ./argos-runner.sh --help           # show this header
#
#  Stop a running loop with Ctrl+C.
#
#  All tunable settings live in the CONFIGURATION block below.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# How long to wait between runs, in seconds. 5 = quick turnaround.
INTERVAL="${INTERVAL:-5}"

# Maximum number of runs before stopping. 0 = infinite (loop forever).
MAX_LOOPS="${MAX_LOOPS:-0}"

# Timeout per Argos run, in seconds. 0 = no timeout. If a run takes longer
# than this, it is killed and the loop continues. 1800 = 30 minutes.
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"

# The prompt sent to Argos on each run.
# By default, read from .devin/argos-prompt.txt (Argos can edit this file to
# change its own prompt). Override with --prompt or the PROMPT env var.
# (Actual loading happens after SCRIPT_DIR is resolved below.)
DEFAULT_PROMPT_FALLBACK="deep scan for inconsistencies across the project; ensure theoretical consistency with wider corpus @manuscripts."

# AI model to use. The ACP client accepts model IDs like "glm-5-2", "claude-opus-5-high",
# "swe-1-7-medium", etc. Common aliases are mapped automatically:
#   "glm-5.2-high" / "glm"     -> "glm-5-2"
#   "glm-5.2-max"              -> "glm-5-2-max"
#   "opus" / "claude-opus"     -> "claude-opus-5-high"
#   "sonnet" / "claude-sonnet" -> "claude-sonnet-5-high"
#   "swe-1.7" / "swe"          -> "swe-1-7-medium"
# Leave empty to use your Argos default.
MODEL="${MODEL:-glm-5-2}"

# Permission mode for non-interactive runs. ACP accepts:
#   "bypass"       - auto-approves ALL tool calls (reads, writes, shell)
#   "accept-edits" - auto-approves file reads/edits, asks for shell commands
#   "smart"        - auto-runs actions a fast model judges safe
#   "ask"          - read-only, answers questions without code changes
#   "plan"         - plans changes before implementing
# "bypass" is required for the runner to actually fix issues autonomously.
# Aliases: "dangerous", "yolo" -> "bypass"
PERMISSION_MODE="${PERMISSION_MODE:-dangerous}"

# Session continuation: how many consecutive runs share the same Argos session
# before starting a new one. 1 = new session every run (no memory). 5 = Argos
# remembers the previous 4 runs' context on the 5th run, then starts fresh.
# 0 = never start a new session (continue forever, Argos remembers everything).
SESSION_CONTINUATION="${SESSION_CONTINUATION:-0}"

# Path to the Devin CLI binary. Leave empty to auto-detect (PATH, then the
# bundled binary inside Devin.app on macOS).
DEVIN_BIN="${DEVIN_BIN:-}"

# Output formatting: "auto" uses colors when stdout is a TTY, "always" forces
# color, "never" disables it. The formatter renders markdown headers/bullets/
# code blocks with color and adds visual structure.
COLOR="${COLOR:-auto}"

# Log file for run output. If set, each run's raw Argos output is appended
# to this file with a timestamp header. Leave empty to disable file logging.
LOG_FILE="${LOG_FILE:-}"

# Show agent thinking (reasoning) output on stderr. 0 = hide, 1 = show.
SHOW_THINKING="${SHOW_THINKING:-0}"

# Shared state directory where each runner registers itself so other runners
# can discover it. Override with ARGOS_RUNNER_STATE_DIR if needed.
STATE_DIR="${ARGOS_RUNNER_STATE_DIR:-/tmp/argos-runner}"
# -----------------------------------------------------------------------------

# Don't use set -e here because we handle errors explicitly in the loop.
# pipefail is useful for catching exit codes through pipes.
set -uo pipefail

# --- Resolve project root: the directory this script lives in ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SELF_PID=$$

# --- Load prompt from file (after SCRIPT_DIR is resolved) -------------------
# Argos can edit .devin/argos-prompt.txt to change its own prompt.
# Override with --prompt or the PROMPT env var.
if [[ -z "${PROMPT:-}" ]]; then
  _DEFAULT_PROMPT_FILE="$SCRIPT_DIR/.devin/argos-prompt.txt"
  if [[ -f "$_DEFAULT_PROMPT_FILE" ]]; then
    PROMPT="$(cat "$_DEFAULT_PROMPT_FILE")"
  else
    PROMPT="$DEFAULT_PROMPT_FALLBACK"
  fi
fi

# --- Temp file for the embedded Python ACP client ---------------------------
ACP_CLIENT="/tmp/argos-runner-acp-${SELF_PID}.py"

# --- Color setup -------------------------------------------------------------
USE_COLOR=0
if [[ "$COLOR" == "always" ]]; then
  USE_COLOR=1
elif [[ "$COLOR" == "auto" && -t 1 ]]; then
  USE_COLOR=1
fi

if [[ "$USE_COLOR" -eq 1 ]]; then
  C_RESET='\033[0m'    C_BOLD='\033[1m'     C_DIM='\033[2m'
  C_RED='\033[31m'     C_GREEN='\033[32m'   C_YELLOW='\033[33m'
  C_BLUE='\033[34m'    C_MAGENTA='\033[35m' C_CYAN='\033[36m'
  C_GRAY='\033[90m'
else
  C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW=''
  C_BLUE='' C_MAGENTA='' C_CYAN='' C_GRAY=''
fi

# --- Locate the Devin binary ------------------------------------------------
find_devin_bin() {
  if [[ -n "${DEVIN_BIN:-}" && -x "$DEVIN_BIN" ]]; then
    return 0
  fi
  if command -v devin >/dev/null 2>&1; then
    DEVIN_BIN="$(command -v devin)"
    return 0
  fi
  local mac_app="/Applications/Devin.app/Contents/Resources/app/extensions/windsurf/devin/bin/devin"
  if [[ -x "$mac_app" ]]; then
    DEVIN_BIN="$mac_app"
    return 0
  fi
  printf "${C_BOLD}${C_RED}error: could not find the 'devin' binary.${C_RESET}\n" >&2
  printf "       set DEVIN_BIN=/path/to/devin or install devin on your PATH.\n" >&2
  return 1
}

# --- Extract the embedded Python ACP client ---------------------------------
extract_acp_client() {
  local marker="# --- BEGIN EMBEDDED PYTHON ACP CLIENT ---"
  local end_marker="# --- END EMBEDDED PYTHON ACP CLIENT ---"
  local script="$0"
  awk -v m="$marker" -v e="$end_marker" '
    $0 == m { found=1; next }
    $0 == e { found=0 }
    found && /^: <</ { next }
    found && /^__ACP_PYTHON_EOF__$/ { next }
    found { print }
  ' "$script" > "$ACP_CLIENT"
  if [[ ! -s "$ACP_CLIENT" ]]; then
    printf "${C_BOLD}${C_RED}error: failed to extract embedded ACP client.${C_RESET}\n" >&2
    return 1
  fi
  return 0
}

# --- Model name mapping ------------------------------------------------------
map_model() {
  local input="$1"
  [[ -z "$input" ]] && { echo ""; return; }
  case "$input" in
    glm-5.2-high|glm-5.2|glm)           echo "glm-5-2";           return ;;
    glm-5.2-max|glm-max)                echo "glm-5-2-max";        return ;;
    glm-5.2-1m|glm-1m)                  echo "glm-5-2-1m";         return ;;
    glm-5.2-max-1m)                     echo "glm-5-2-max-1m";     return ;;
    glm-5.2-none|glm-none)              echo "glm-5-2-none";       return ;;
    glm-5.2-none-1m)                    echo "glm-5-2-none-1m";    return ;;
    opus|claude-opus|opus-5|claude-opus-5)         echo "claude-opus-5-high";    return ;;
    claude-opus-5-high|opus-high)                  echo "claude-opus-5-high";    return ;;
    claude-opus-5-medium|opus-medium)              echo "claude-opus-5-medium";  return ;;
    claude-opus-5-low|opus-low)                    echo "claude-opus-5-low";     return ;;
    claude-opus-5-max|opus-max)                    echo "claude-opus-5-max";     return ;;
    claude-opus-5-xhigh|opus-xhigh)                echo "claude-opus-5-xhigh";   return ;;
    sonnet|claude-sonnet|sonnet-5)                 echo "claude-sonnet-5-high";  return ;;
    claude-sonnet-5-high|sonnet-high)              echo "claude-sonnet-5-high";  return ;;
    claude-sonnet-5-medium|sonnet-medium)          echo "claude-sonnet-5-medium"; return ;;
    claude-sonnet-5-low|sonnet-low)                echo "claude-sonnet-5-low";   return ;;
    swe-1.7|swe|swe-1.7-medium|swe-medium)         echo "swe-1-7-medium";        return ;;
    swe-1.7-max|swe-max)                           echo "swe-1-7";               return ;;
    swe-1.7-lightning|swe-lightning)               echo "swe-1-7-lightning";     return ;;
    gemini|gemini-3.7|gemini-flash)                echo "gemini-3-7-flash-medium"; return ;;
    gpt-5.6|gpt-sol|gpt-5.6-sol)                   echo "gpt-5-6-sol-medium";    return ;;
    gpt-5.6-luna|gpt-luna)                         echo "gpt-5-6-luna-medium";   return ;;
    kimi|kimi-k3|kimi-k3-high)                     echo "kimi-k3-high";          return ;;
    deepseek|deepseek-v4|deepseek-flash|ds|ds-v4|deepseek-4) echo "deepseek-v4-flash";     return ;;
    adaptive|auto)                                 echo "";                      return ;;
  esac
  echo "$input"
}

# --- Permission mode mapping -------------------------------------------------
map_mode() {
  local input="$1"
  case "$input" in
    dangerous|yolo|bypass)    echo "bypass" ;;
    accept-edits|code)        echo "accept-edits" ;;
    smart)                    echo "smart" ;;
    ask|read-only)            echo "ask" ;;
    plan)                     echo "plan" ;;
    "")                       echo "bypass" ;;
    *)                        echo "$input" ;;
  esac
}

# --- Auth check --------------------------------------------------------------
check_auth() {
  local cred_file
  cred_file="$HOME/.local/share/devin/credentials.toml"
  if [[ ! -f "$cred_file" ]]; then
    printf "${C_BOLD}${C_YELLOW}Warning: No credentials file found at %s${C_RESET}\n" "$cred_file" >&2
    printf "${C_DIM}  The runner will not work. To fix:${C_RESET}\n" >&2
    printf "    Log in to Devin via the Windsurf IDE, or run:${C_RESET}\n" >&2
    printf "    %s auth login${C_RESET}\n" "$DEVIN_BIN" >&2
    return 1
  fi
  if ! grep -q 'windsurf_api_key' "$cred_file" 2>/dev/null; then
    printf "${C_BOLD}${C_YELLOW}Warning: No windsurf_api_key in credentials file.${C_RESET}\n" >&2
    printf "${C_DIM}  Log in to Devin via the Windsurf IDE, or run:${C_RESET}\n" >&2
    printf "    %s auth login${C_RESET}\n" "$DEVIN_BIN" >&2
    return 1
  fi
  local key
  key="$(grep 'windsurf_api_key' "$cred_file" | head -1 | sed 's/.*= *//; s/^"//; s/"$//')"
  local key_preview
  if [[ ${#key} -gt 30 ]]; then
    key_preview="${key:0:15}...${key: -10}"
  else
    key_preview="$key"
  fi
  printf "${C_DIM}  auth    : %s${C_RESET}\n" "$key_preview"
  return 0
}

# --- Output formatter --------------------------------------------------------
format_devin_output() {
  awk -v USE_COLOR="$USE_COLOR" '
  BEGIN {
    in_code = 0
    prev_blank = 0
    if (USE_COLOR) {
      reset = "\033[0m"; bold = "\033[1m"; dim = "\033[2m"
      red = "\033[31m"; green = "\033[32m"; yellow = "\033[33m"
      blue = "\033[34m"; magenta = "\033[35m"; cyan = "\033[36m"
      gray = "\033[90m"
    } else {
      reset = ""; bold = ""; dim = ""; red = ""; green = ""; yellow = ""
      blue = ""; magenta = ""; cyan = ""; gray = ""
    }
  }

  {
    line = $0

    # --- Strip any stray ANSI escape sequences ---
    while (match(line, /\033\[[0-9;]*[a-zA-Z]/)) {
      line = substr(line, 1, RSTART-1) substr(line, RSTART+RLENGTH)
    }
    gsub(/\033/, "", line)
    gsub(/[\001-\010\013-\037]/, "", line)

    # --- Skip empty lines (collapse consecutive blanks into one) ---
    stripped = line
    gsub(/[ \t]+/, "", stripped)
    if (stripped == "" && !in_code) {
      if (prev_blank == 0) print ""
      prev_blank = 1
      next
    }
    prev_blank = 0

    # --- Code block fences ---
    if (line ~ /^```/ || line ~ /^~~~/) {
      if (in_code) {
        in_code = 0
        printf "%s    +----%s\n", gray, reset
      } else {
        in_code = 1
        lang = line
        sub(/^[`~]+/, "", lang)
        gsub(/^[ \t]+|[ \t]+$/, "", lang)
        if (lang == "") lang = "code"
        printf "%s    +--- %s ---%s\n", gray, lang, reset
      }
      next
    }

    if (in_code) {
      printf "%s    | %s%s\n", gray, line, reset
      next
    }

    # --- Markdown headers ---
    if (line ~ /^##### /) { printf "%s%s       %s%s\n", bold, cyan, substr(line, 7), reset; next }
    if (line ~ /^#### /)  { printf "%s%s     %s%s\n",   bold, cyan, substr(line, 6), reset; next }
    if (line ~ /^### /)   { printf "%s%s   %s%s\n",     bold, cyan, substr(line, 5), reset; next }
    if (line ~ /^## /)    { printf "\n%s%s  %s%s\n",    bold, cyan, substr(line, 4), reset; next }
    if (line ~ /^# /)     { printf "\n%s%s %s%s\n",     bold, magenta, substr(line, 3), reset; next }

    # --- Bullet points ---
    if (line ~ /^[ \t]*[-*] /) {
      content = line
      sub(/^[ \t]*[-*][ \t]+/, "", content)
      printf "%s  - %s%s\n", dim, content, reset
      next
    }

    # --- Numbered lists ---
    if (line ~ /^[ \t]*[0-9]+\.[ \t]+/) {
      content = line
      sub(/^[ \t]*[0-9]+\.[ \t]+/, "", content)
      num = line
      sub(/^[ \t]*/, "", num)
      sub(/\..*/, "", num)
      printf "%s  %s. %s%s\n", dim, num, content, reset
      next
    }

    # --- Blockquotes ---
    if (line ~ /^>/) {
      content = line
      sub(/^>[ \t]*/, "", content)
      printf "%s  | %s%s\n", gray, content, reset
      next
    }

    # --- Horizontal rules ---
    if (line ~ /^---+$/ || line ~ /^===+$/) {
      printf "%s  ----------------------------------------------------------------%s\n", gray, reset
      next
    }

    # --- Regular text ---
    printf "  %s\n", line
  }
  '
}

# --- Runner registry --------------------------------------------------------
state_file_path() {
  local key; key="$(printf '%s' "$PROJECT_ROOT" | tr '/: ' '___')"
  printf '%s/%s.state\n' "$STATE_DIR" "$key"
}

register_self() {
  mkdir -p "$STATE_DIR"
  local f; f="$(state_file_path)"
  local started; started="$(date '+%Y-%m-%d %H:%M:%S')"
  local short_prompt; short_prompt="$(printf '%s' "$PROMPT" | head -c 120)"
  {
    printf 'pid=%s\n'           "$SELF_PID"
    printf 'project=%s\n'       "$PROJECT_ROOT"
    printf 'started=%s\n'       "$started"
    printf 'interval=%s\n'      "$INTERVAL"
    printf 'continuation=%s\n'  "$SESSION_CONTINUATION"
    printf 'prompt=%s\n'        "$short_prompt"
  } > "$f"
}

unregister_self() {
  rm -f "$(state_file_path)"
}

pid_alive() {
  [[ -n "${1:-}" ]] || return 1
  kill -0 "$1" 2>/dev/null
}

list_other_runners() {
  OTHER_RUNNERS_COUNT=0
  [[ -d "$STATE_DIR" ]] || return 0
  local others=()
  local f pid proj started intvl cont pr
  for f in "$STATE_DIR"/*.state; do
    [[ -e "$f" ]] || continue
    pid=""; proj=""; started=""; intvl=""; cont=""; pr=""
    while IFS='=' read -r k v; do
      case "$k" in
        pid) pid="$v" ;;  project) proj="$v" ;;
        started) started="$v" ;;  interval) intvl="$v" ;;
        continuation) cont="$v" ;;  prompt) pr="$v" ;;
      esac
    done < "$f"
    [[ "$pid" == "$SELF_PID" ]] && continue
    if ! pid_alive "$pid"; then
      rm -f "$f"; continue
    fi
    others+=("$pid|$proj|$started|$intvl|$cont|$pr")
  done

  OTHER_RUNNERS_COUNT=${#others[@]}
  if [[ "$OTHER_RUNNERS_COUNT" -eq 0 ]]; then
    printf "${C_DIM}Other argos-runner runners: none${C_RESET}\n"
    return 0
  fi
  printf "${C_BOLD}${C_YELLOW}Other argos-runner runners (%s):${C_RESET}\n" "$OTHER_RUNNERS_COUNT"
  local entry
  for entry in "${others[@]}"; do
    IFS='|' read -r pid proj started intvl cont pr <<< "$entry"
    local proj_short; proj_short="$(basename "$proj")"
    printf "  %b-%b pid=%-7s  %-20s  every %ss  cont=%s  since %s\n" "$C_CYAN" "$C_RESET" "$pid" "$proj_short" "$intvl" "${cont:-1}" "$started"
    printf "    %bproject :%b %s\n" "$C_DIM" "$C_RESET" "$proj"
    printf "    %bprompt  :%b %s\n" "$C_DIM" "$C_RESET" "$pr"
  done
}

# --- Show all runners (--status) --------------------------------------------
show_status() {
  [[ -d "$STATE_DIR" ]] || { echo "No runners registered."; exit 0; }
  local found=0
  local f pid proj started intvl cont pr
  for f in "$STATE_DIR"/*.state; do
    [[ -e "$f" ]] || continue
    pid=""; proj=""; started=""; intvl=""; cont=""; pr=""
    while IFS='=' read -r k v; do
      case "$k" in
        pid) pid="$v" ;;  project) proj="$v" ;;
        started) started="$v" ;;  interval) intvl="$v" ;;
        continuation) cont="$v" ;;  prompt) pr="$v" ;;
      esac
    done < "$f"
    if ! pid_alive "$pid"; then
      rm -f "$f"; continue
    fi
    found=$((found + 1))
    if [[ "$found" -eq 1 ]]; then
      printf "%b${C_BOLD}${C_MAGENTA}Active argos-runner instances:${C_RESET}\n" ""
    fi
    local proj_short; proj_short="$(basename "$proj")"
    local age
    age="$(( $(date +%s) - $(date -j -f '%Y-%m-%d %H:%M:%S' "$started" +%s 2>/dev/null || echo 0) ))"
    local age_str
    if [[ "$age" -gt 3600 ]]; then
      age_str="$((age / 3600))h$((age % 3600 / 60))m"
    elif [[ "$age" -gt 60 ]]; then
      age_str="$((age / 60))m"
    else
      age_str="${age}s"
    fi
    printf "  %b#%b pid=%-7s  %-20s  every %ss  cont=%s  uptime %s\n" "$C_CYAN" "$C_RESET" "$pid" "$proj_short" "$intvl" "${cont:-1}" "$age_str"
    printf "    %bproject :%b %s\n" "$C_DIM" "$C_RESET" "$proj"
    printf "    %bprompt  :%b %s\n" "$C_DIM" "$C_RESET" "$pr"
  done
  [[ "$found" -eq 0 ]] && echo "No active argos-runner instances."
  exit 0
}

# --- Build and run the ACP client -------------------------------------------
run_devin() {
  # Build the ACP client command.
  # The Python client handles looping, session management, and timing.
  # stdout = agent response text -> piped through formatter
  # stderr = banners, thinking, tool calls, usage, summaries -> direct to terminal
  local cmd=(python3 "$ACP_CLIENT"
    --cwd "$PROJECT_ROOT"
    --prompt "$PROMPT"
    --mode "$ACP_MODE"
    --loops "$MAX_LOOPS"
    --interval "$INTERVAL"
    --session-continuation "$SESSION_CONTINUATION"
    --script-path "$0"
  )
  if [[ -n "$ACP_MODEL" ]]; then
    cmd+=(--model "$ACP_MODEL")
  fi
  if [[ "${RUN_TIMEOUT:-0}" != "0" ]]; then
    cmd+=(--timeout "$RUN_TIMEOUT")
  fi
  if [[ "${SHOW_THINKING:-0}" == "1" ]]; then
    cmd+=(--show-thinking)
  fi

  # Dry run: show the command and exit
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf "${C_DIM}  [dry-run] command:%s\n" "$C_RESET"
    printf "    %s" "python3"
    local arg
    for arg in "${cmd[@]:1}"; do
      if [[ "$arg" == *" "* ]]; then
        printf ' "%s"' "$arg"
      else
        printf ' %s' "$arg"
      fi
    done
    printf "\n"
    printf "${C_DIM}  [dry-run] no Argos invocation performed.${C_RESET}\n"
    return 0
  fi

  # Run the ACP client.
  # The Python client manages the full loop: session creation, prompts,
  # intervals, and session continuation. It prints banners/summaries to
  # stderr and agent response text to stdout.
  if [[ -n "${LOG_FILE:-}" ]]; then
    "${cmd[@]}" 2> >(tee -a "$LOG_FILE" >&2) | format_devin_output
    return ${PIPESTATUS[0]}
  else
    "${cmd[@]}" 2>&2 | format_devin_output
    return ${PIPESTATUS[0]}
  fi
}

# --- Argument parsing -------------------------------------------------------
ONCE=0
CHECK_AUTH_ONLY=0
STATUS_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --loops)    MAX_LOOPS="$2"; shift 2 ;;
    --timeout)  RUN_TIMEOUT="$2"; shift 2 ;;
    --prompt)   PROMPT="$2";   shift 2 ;;
    --model)    MODEL="$2";    shift 2 ;;
    --session-continuation) SESSION_CONTINUATION="$2"; shift 2 ;;
    --log)      LOG_FILE="$2"; shift 2 ;;
    --color)    COLOR="$2";    shift 2 ;;
    --no-color)  COLOR="never"; shift   ;;
    --once)      ONCE=1; MAX_LOOPS=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --check-auth) CHECK_AUTH_ONLY=1; shift ;;
    --status)    STATUS_ONLY=1; shift ;;
    --show-thinking) SHOW_THINKING=1; shift ;;
    -h|--help)   sed -n '2,44p' "$0"; exit 0 ;;
    *) printf "unknown argument: %s (try --help)\n" "$1" >&2; exit 1 ;;
  esac
done

# --- Resolve dependencies ---------------------------------------------------
find_devin_bin || exit 1
extract_acp_client || exit 1

# Map model and permission mode to ACP IDs
ACP_MODEL="$(map_model "${MODEL:-}")"
ACP_MODE="$(map_mode "${PERMISSION_MODE:-}")"

# --status: show all runners and exit
if [[ "$STATUS_ONLY" == "1" ]]; then
  show_status
fi

# --check-auth: just verify credentials and exit
if [[ "$CHECK_AUTH_ONLY" == "1" ]]; then
  check_auth && exit 0 || exit 1
fi

# Banner
printf "${C_BOLD}${C_MAGENTA}argos-runner${C_RESET}\n"
printf "${C_DIM}  project  : %s${C_RESET}\n" "$PROJECT_ROOT"
printf "${C_DIM}  devin    : %s${C_RESET}\n" "$DEVIN_BIN"
printf "${C_DIM}  interval : %s${C_RESET}\n" "${INTERVAL}s"
printf "${C_DIM}  timeout  : %s${C_RESET}\n" "${RUN_TIMEOUT:-0} ($([[ "${RUN_TIMEOUT:-0}" == "0" ]] && echo "none" || echo "${RUN_TIMEOUT}s per run"))"
printf "${C_DIM}  loops    : %s${C_RESET}\n" "${MAX_LOOPS:-0} ($([[ "${MAX_LOOPS:-0}" == "0" ]] && echo "infinite" || echo "max ${MAX_LOOPS}"))"
printf "${C_DIM}  model    : %s${C_RESET}\n" "${MODEL:-<default>}$( [[ -n "$ACP_MODEL" && "$ACP_MODEL" != "$MODEL" ]] && printf " -> %s" "$ACP_MODEL" )"
printf "${C_DIM}  mode     : %s${C_RESET}\n" "${PERMISSION_MODE:-auto}$( [[ -n "$ACP_MODE" && "$ACP_MODE" != "${PERMISSION_MODE:-}" ]] && printf " -> %s" "$ACP_MODE" )"
if [[ "${SESSION_CONTINUATION:-1}" == "1" ]]; then
  printf "${C_DIM}  session  : new each run${C_RESET}\n"
elif [[ "${SESSION_CONTINUATION:-1}" == "0" ]]; then
  printf "${C_DIM}  session  : continue forever (no new sessions)${C_RESET}\n"
else
  printf "${C_DIM}  session  : new every %s runs${C_RESET}\n" "$SESSION_CONTINUATION"
fi
printf "${C_DIM}  color    : %s${C_RESET}\n" "$COLOR"
[[ -n "${LOG_FILE:-}" ]] && printf "${C_DIM}  log      : %s${C_RESET}\n" "$LOG_FILE"
[[ "${DRY_RUN:-0}" == "1" ]] && printf "${C_BOLD}${C_YELLOW}  DRY RUN -- no Argos calls will be made${C_RESET}\n"
check_auth || true
printf "${C_DIM}  prompt   : %s${C_RESET}\n" "$PROMPT"
echo ""

# Register this runner, then show what else is running.
register_self
list_other_runners
echo ""

printf "${C_DIM}Press Ctrl+C to stop.${C_RESET}\n"

# Cleanup on exit (Ctrl+C, TERM, or normal exit).
cleanup() {
  unregister_self
  rm -f "$ACP_CLIENT"
  echo ""
  printf "${C_BOLD}${C_YELLOW}Stopping argos-runner (pid=%s).${C_RESET}\n" "$SELF_PID"
}
trap 'cleanup; exit 0' INT TERM
trap 'cleanup' EXIT

# --- Run Argos (the Python client handles the loop) -------------------------
run_devin

# =============================================================================
# --- BEGIN EMBEDDED PYTHON ACP CLIENT ---
# =============================================================================
# The Python code below is extracted at runtime by extract_acp_client().
# It is wrapped in a : <<'__ACP_PYTHON_EOF__' heredoc so that bash skips
# it during parsing (the Python syntax would otherwise cause bash errors).
# The awk extractor strips the heredoc delimiter lines before writing the
# temp .py file.
#
# The client implements a minimal ACP (Agent Client Protocol) that:
#   - Reads Windsurf credentials from ~/.local/share/devin/credentials.toml
#   - Connects to `devin acp` with ACP_BACKEND=windsurf
#   - Authenticates via the windsurf-api-key method
#   - Creates sessions, sets mode/model, sends prompts
#   - Supports session continuation (reuse same session for N runs)
#   - Handles looping with configurable interval
#   - Streams agent thinking + tool calls to stderr
#   - Streams the final response text to stdout
#   - Returns exit code 0 (ok), 1 (error), 124 (timeout), 130 (interrupted)
: <<'__ACP_PYTHON_EOF__'
import argparse
import json
import os
import subprocess
import sys
import time


def find_devin_bin():
    devin_bin = os.environ.get("DEVIN_BIN", "")
    if devin_bin and os.path.isfile(devin_bin) and os.access(devin_bin, os.X_OK):
        return devin_bin
    import shutil
    path_bin = shutil.which("devin")
    if path_bin:
        return path_bin
    mac_app = "/Applications/Devin.app/Contents/Resources/app/extensions/windsurf/devin/bin/devin"
    if os.path.isfile(mac_app) and os.access(mac_app, os.X_OK):
        return mac_app
    return None


def load_credentials():
    cred_path = os.path.expanduser("~/.local/share/devin/credentials.toml")
    api_key = None
    api_server_url = "https://server.codeium.com"
    try:
        with open(cred_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("windsurf_api_key") and "=" in line:
                    val = line.split("=", 1)[1].strip()
                    api_key = val.strip('"').strip("'")
                elif line.startswith("api_server_url") and "=" in line:
                    val = line.split("=", 1)[1].strip()
                    api_server_url = val.strip('"').strip("'")
    except FileNotFoundError:
        pass
    return api_key, api_server_url


class AcpClient:
    def __init__(self, devin_bin, api_key, api_server_url, cwd, prompt,
                 model=None, mode="bypass", timeout=300, show_thinking=False,
                 loops=1, interval=0, session_continuation=1):
        self.devin_bin = devin_bin
        self.api_key = api_key
        self.api_server_url = api_server_url
        self.cwd = cwd
        self.prompt = prompt
        self.default_prompt = prompt  # fallback if script re-read fails
        self.script_path = None       # path to argos-runner.sh for live prompt re-reading
        self.model = model
        self.mode = mode
        self.timeout = timeout
        self.show_thinking = show_thinking
        self.loops = loops  # 0 = infinite
        self.interval = interval
        self.session_continuation = session_continuation  # 0 = never start new

        self.proc = None
        self.msg_id = 0
        self.session_id = None
        self.finished = False
        self.error = None
        self.start_time = time.time()
        self.run_start_time = None

        # State machine
        self.auth_done = False
        self.steps = []           # queue of steps to execute
        self.current_step = None
        self._prompt_id = None

        # Stats
        self.stats = {"ok": 0, "fail": 0, "timeout": 0}
        self._last_usage = None
        self._agent_stats = None
        self._response_started = False
        self._thinking_started = False
        self._last_section = None  # "thinking", "tool", "response", or None

    def _next_id(self):
        self.msg_id += 1
        return self.msg_id

    def _send(self, method, params=None, id=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if id is not None:
            msg["id"] = id
        data = json.dumps(msg) + "\n"
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _emit(self, channel, text, end="\n"):
        if not text:
            return
        # Color-code by channel for visual hierarchy.
        # Brightness order: response (stdout, white) > thinking (white) >
        # tool (dim cyan) > usage/info (dim gray) > activity (dim gray)
        c = {
            "TOOL":      "\033[2;36m",    # dim cyan
            "USAGE":     "\033[90m",      # bright gray
            "INFO":      "\033[35m",      # magenta
            "ERROR":     "\033[31m",      # red
            "ACTIVITY":  "\033[90m",      # bright gray
            "THINK":     "\033[37m",      # white (brighter than tool)
            "PROMPT":    "\033[33m",      # yellow
        }.get(channel, "")
        reset = "\033[0m" if c else ""
        if channel:
            line = f"{c}[{channel}]{reset} {text}"
        else:
            line = text
        sys.stderr.write(line + end)
        sys.stderr.flush()

    # --- State machine: step queue ------------------------------------------

    def _begin_run(self, run_num, need_new_session):
        """Set up the step queue for the next run."""
        self.finished = False
        self.error = None
        self._prompt_id = None
        self._response_started = False
        self._thinking_started = False
        self._last_usage = None
        self._agent_stats = None
        self._last_section = None
        self.run_start_time = time.time()

        steps = []

        # First run ever: need to initialize and authenticate
        if not self.auth_done:
            steps.append("initialize")
            steps.append("authenticate")

        # New session needed: create it, then set mode/model
        if need_new_session:
            steps.append("session_new")
            if self.mode:
                steps.append("set_mode")
            if self.model:
                steps.append("set_model")

        # Always end with sending the prompt
        steps.append("prompt")

        self.steps = steps
        self._execute_next_step()

    def _execute_next_step(self):
        """Send the next message in the step queue."""
        if not self.steps:
            return  # waiting for prompt response

        step = self.steps.pop(0)
        self.current_step = step

        if step == "initialize":
            self._send("initialize", {
                "protocolVersion": "2025-03-26",
                "client": {"name": "argos-runner", "version": "1.0"},
                "clientCapabilities": {},
            }, id=self._next_id())

        elif step == "authenticate":
            self._send("authenticate", {
                "methodId": "windsurf-api-key",
                "_meta": {
                    "api_key": self.api_key,
                    "api_server_url": self.api_server_url,
                }
            }, id=self._next_id())

        elif step == "session_new":
            self._send("session/new", {
                "cwd": self.cwd,
                "mcpServers": [],
            }, id=self._next_id())

        elif step == "set_mode":
            self._send("session/set_mode", {
                "modeId": self.mode,
                "sessionId": self.session_id,
            }, id=self._next_id())

        elif step == "set_model":
            self._send("session/set_config_option", {
                "configId": "model",
                "value": self.model,
                "sessionId": self.session_id,
            }, id=self._next_id())

        elif step == "prompt":
            self._prompt_id = self._next_id()
            self._send("session/prompt", {
                "prompt": [{"type": "text", "text": self.prompt}],
                "sessionId": self.session_id,
            }, id=self._prompt_id)

    # --- Response handler ---------------------------------------------------

    def _handle_response(self, msg):
        if msg.get("error"):
            self.error = msg["error"].get("message", "Unknown error")
            return

        result = msg.get("result", {})

        if self.current_step == "initialize":
            self._execute_next_step()  # -> authenticate

        elif self.current_step == "authenticate":
            self.auth_done = True
            self._execute_next_step()  # -> session_new or prompt

        elif self.current_step == "session_new":
            self.session_id = result.get("sessionId", "")
            self._execute_next_step()  # -> set_mode / set_model / prompt

        elif self.current_step == "set_mode":
            self._execute_next_step()  # -> set_model / prompt

        elif self.current_step == "set_model":
            self._execute_next_step()  # -> prompt

        elif self.current_step == "prompt":
            stop_reason = result.get("stopReason", "")
            if stop_reason in ("end_turn", "max_tokens", "max_turn_requests",
                               "refusal", "cancelled"):
                self.finished = True

    # --- Notification handler -----------------------------------------------

    def _handle_notification(self, msg):
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "session/update":
            update = params.get("update", {})
            session_update = update.get("sessionUpdate", "")

            if session_update == "agent_message_chunk":
                content = update.get("content", {})
                if isinstance(content, dict):
                    text = content.get("text", "")
                    if text and text.strip():
                        # Close thinking if still open
                        if self._thinking_started:
                            sys.stderr.write("\n")
                            sys.stderr.flush()
                            self._thinking_started = False
                        # Add blank line before response when transitioning from tools or thinking
                        if not self._response_started and self._last_section in ("tool", "thinking"):
                            sys.stderr.write("\n")
                            sys.stderr.flush()
                        self._response_started = True
                        self._last_section = "response"
                        sys.stdout.write(text)
                        sys.stdout.flush()

            elif session_update == "agent_message":
                message = update.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "")
                                if text and text.strip():
                                    if self._thinking_started:
                                        sys.stderr.write("\n")
                                        sys.stderr.flush()
                                        self._thinking_started = False
                                    if not self._response_started and self._last_section in ("tool", "thinking"):
                                        sys.stderr.write("\n")
                                        sys.stderr.flush()
                                    self._response_started = True
                                    self._last_section = "response"
                                    sys.stdout.write(text)
                                    sys.stdout.flush()

            elif session_update == "agent_thought_chunk":
                content = update.get("content", {})
                if isinstance(content, dict):
                    text = content.get("text", "")
                    if text:
                        # Thinking at normal brightness (brighter than tool calls).
                        # Add blank line before thinking when transitioning from
                        # tools or response, then print the marker.
                        if not self._thinking_started:
                            if self._last_section in ("tool", "response"):
                                sys.stderr.write("\n")
                            sys.stderr.write(f"\033[37m\033[1mthinking:\033[0m ")
                            sys.stderr.flush()
                            self._thinking_started = True
                            self._last_section = "thinking"
                        sys.stderr.write(text)
                        sys.stderr.flush()

            elif session_update == "tool_call":
                # Close thinking if still open (some flows skip thinking_complete)
                if self._thinking_started:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    self._thinking_started = False
                # Add blank line before tools when transitioning from thinking or response
                if self._last_section in ("thinking", "response"):
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                self._last_section = "tool"
                title = update.get("title", "")
                kind = update.get("kind", "")
                raw_input = update.get("rawInput", {})
                detail = ""
                if isinstance(raw_input, dict):
                    for key in ("command", "pattern", "path", "prompt", "url"):
                        if key in raw_input:
                            detail = str(raw_input[key])
                            break
                output_text = ""
                content_parts = update.get("content", [])
                if isinstance(content_parts, list):
                    for part in content_parts:
                        if isinstance(part, dict):
                            inner = part.get("content", {})
                            if isinstance(inner, dict):
                                resource = inner.get("resource", {})
                                if isinstance(resource, dict):
                                    output_text = resource.get("text", "")
                                    break
                # If the title already contains the detail, don't repeat it
                label = title or kind or "tool"
                show_detail = detail
                if show_detail and label:
                    # Check if label ends with the detail or contains it
                    label_stripped = label.rstrip()
                    if label_stripped.endswith(show_detail.strip()) or \
                       show_detail.strip() in label_stripped:
                        show_detail = ""
                # Truncate long commands (e.g. multi-line python -c scripts)
                if show_detail:
                    detail_one = " ".join(show_detail.split("\n")).strip()
                    if len(detail_one) > 120:
                        detail_one = detail_one[:117] + "..."
                    self._emit("TOOL", f"{label}: {detail_one}")
                else:
                    self._emit("TOOL", label)
                # Show output compactly: first 3 lines, indented.
                # Skip if output just echoes the command (ACP sometimes does this).
                if output_text:
                    out_stripped = output_text.strip()
                    cmd_stripped = (show_detail or "").strip()
                    # Skip if output is empty, equals the command, or is a
                    # substring of the command (e.g. command had | head -50
                    # but output echoes without it)
                    skip = (not out_stripped
                            or out_stripped == cmd_stripped
                            or (cmd_stripped and out_stripped in cmd_stripped)
                            or (out_stripped and len(out_stripped) < len(cmd_stripped)
                                and out_stripped in cmd_stripped))
                    if not skip:
                        lines = [l for l in out_stripped.split("\n") if l.strip()]
                        for line in lines[:3]:
                            if len(line) > 150:
                                line = line[:147] + "..."
                            self._emit("TOOL", f"  {line}")
                        if len(lines) > 3:
                            self._emit("TOOL", f"  ... +{len(lines) - 3} more lines")

            elif session_update == "activity":
                # Suppress activity notifications (too noisy, tool calls cover this)
                pass

            elif session_update == "usage_update":
                # Store usage but don't print every chunk — too noisy.
                # Will be printed once at the end by _print_run_summary.
                used = update.get("used", 0)
                size = update.get("size", 0)
                meta = update.get("_meta", {})
                self._last_usage = {
                    "used": used, "size": size,
                    "in": meta.get("cognition.ai/inputTokens", 0),
                    "out": meta.get("cognition.ai/outputTokens", 0),
                }

        elif method == "_cognition.ai/thinking_complete":
            # Just end the thinking line; the next section (tool/response)
            # will add the blank line separator via _last_section transition.
            if self._thinking_started:
                sys.stderr.write("\n")
                self._thinking_started = False
            sys.stderr.flush()

        elif method == "_cognition.ai/agent_stopped":
            cause = params.get("cause", "")
            stats = params.get("stats", {})
            if stats:
                # Store stats for the run summary; don't print a separate
                # INFO line (the summary already shows duration + status).
                self._agent_stats = {
                    "cause": cause,
                    "tool_calls": stats.get("toolCalls", 0),
                    "commands": stats.get("commandsRun", 0),
                    "model": stats.get("modelLabel", ""),
                    "total_s": stats.get("totalTimeMs", 0) / 1000,
                }
            # Only use as completion signal if we're actively waiting for
            # a prompt response AND haven't already finished. This prevents
            # leftover notifications from a previous run from prematurely
            # ending the current run.
            if self._prompt_id is not None and self.current_step == "prompt" and not self.finished:
                self.finished = True

        elif method == "_cognition.ai/output":
            level = params.get("level", "")
            if level == "error":
                channel = params.get("channel", "")
                message = params.get("message", "")
                self._emit("ERROR", f"{channel}: {message}")

    # --- Event loop for a single run ----------------------------------------

    def _event_loop(self):
        """Process messages until the current run finishes or times out."""
        deadline = self.run_start_time + self.timeout if self.timeout > 0 else 0

        while True:
            if deadline > 0 and time.time() > deadline:
                if self.session_id:
                    self._send("session/cancel",
                               {"sessionId": self.session_id},
                               id=self._next_id())
                    time.sleep(1)
                return 124

            if self.proc.poll() is not None:
                return 1 if not self.finished else 0

            if self.finished:
                return 0

            if self.error:
                self._emit("ERROR", self.error)
                return 1

            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "id" in msg:
                self._handle_response(msg)
            elif "method" in msg:
                self._handle_notification(msg)

    # --- Run banner / summary ------------------------------------------------

    def _print_run_banner(self, run_num, total, need_new_session, session_run):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        B = "\033[1m"   # bold
        D = "\033[2m"   # dim
        C = "\033[36m"  # cyan
        G = "\033[32m"  # green
        Y = "\033[33m"  # yellow
        R = "\033[0m"   # reset
        if total > 0:
            title = f"Argos run #{run_num}/{total}"
        else:
            title = f"Argos run #{run_num}"
        if need_new_session:
            if self.session_continuation == 1:
                sess = "new"
            elif self.session_continuation == 0:
                sess = "new (continue forever)"
            else:
                sess = f"new (run {session_run}/{self.session_continuation})"
        else:
            sess = f"continuing (run {session_run}/{self.session_continuation})"
        prompt_preview = self.prompt[:80] + ("..." if len(self.prompt) > 80 else "")
        print(f"\n{C}{'=' * 64}{R}", file=sys.stderr)
        print(f"  {B}{title}{R}  {D}{ts}{R}", file=sys.stderr)
        print(f"  {D}project :{R} {self.cwd}", file=sys.stderr)
        print(f"  {D}model   :{R} {self.model or '<default>'}  {D}mode:{R} {self.mode}", file=sys.stderr)
        print(f"  {D}session :{R} {sess}", file=sys.stderr)
        print(f"  {D}prompt  :{R} {Y}{prompt_preview}{R}", file=sys.stderr)
        print(f"{C}{'=' * 64}{R}", file=sys.stderr)
        sys.stderr.flush()

    def _print_run_summary(self, rc, run_num):
        ts = time.strftime("%H:%M:%S")
        duration = time.time() - self.run_start_time
        D = "\033[2m"   # dim
        G = "\033[32m"  # green
        R = "\033[31m"  # red
        Y = "\033[33m"  # yellow
        C = "\033[36m"  # cyan
        X = "\033[0m"   # reset
        # Build usage string from last stored usage
        usage_str = ""
        if self._last_usage:
            u = self._last_usage
            usage_str = f"tokens: {u['used']}/{u['size']} (in:{u['in']} out:{u['out']})"
        # Build stats string from agent_stopped
        stats_str = ""
        if self._agent_stats:
            a = self._agent_stats
            stats_str = f"{a['tool_calls']} tools, {a['commands']} cmds"
            if a['model']:
                stats_str += f", {a['model']}"
        if rc == 0:
            self.stats["ok"] += 1
            status = f"{G}complete{X}"
        elif rc == 124:
            self.stats["timeout"] += 1
            status = f"{Y}TIMED OUT{X} after {self.timeout}s"
        else:
            self.stats["fail"] += 1
            status = f"{R}FAILED{X} (exit {rc})"
        print(f"\n{C}{'=' * 64}{X}", file=sys.stderr)
        print(f"  {D}[{ts}]{X} run #{run_num} {status} ({duration:.1f}s)", file=sys.stderr)
        if stats_str or usage_str:
            parts = []
            if stats_str: parts.append(stats_str)
            if usage_str: parts.append(usage_str)
            print(f"  {D}{'  |  '.join(parts)}{X}", file=sys.stderr)
        print(f"{C}{'=' * 64}{X}", file=sys.stderr)
        sys.stderr.flush()

    def _print_final_summary(self):
        total = sum(self.stats.values())
        if total == 0:
            return
        G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; B = "\033[1m"; X = "\033[0m"
        ok = self.stats['ok']; fail = self.stats['fail']; to = self.stats['timeout']
        print(f"\n{B}Summary:{X} {total} runs -- "
              f"{G}{ok} ok{X}, {R}{fail} failed{X}, {Y}{to} timed out{X}", file=sys.stderr)
        sys.stderr.flush()

    # --- Drain pending messages between runs ---------------------------------

    def _reread_prompt(self):
        """Re-read the PROMPT from the bash script file before each run.

        This allows live steering: edit the PROMPT line in argos-runner.sh
        while the runner is running, and the next run will pick up the new
        prompt without restarting. Falls back to the original prompt if the
        file can't be read or the PROMPT line isn't found.
        """
        if not self.script_path or not os.path.isfile(self.script_path):
            return
        try:
            with open(self.script_path, "r") as f:
                for line in f:
                    # Match: PROMPT="${PROMPT:-...}"
                    # The prompt text is everything between ${PROMPT:- and }
                    stripped = line.strip()
                    if stripped.startswith("PROMPT=") and "${PROMPT:-" in stripped:
                        idx = stripped.index("${PROMPT:-")
                        start = idx + len("${PROMPT:-")
                        end = stripped.rindex("}", start)
                        new_prompt = stripped[start:end]
                        if new_prompt and new_prompt != self.prompt:
                            old_preview = self.prompt[:60]
                            new_preview = new_prompt[:60]
                            self._emit("INFO",
                                       f"prompt changed (live edit): "
                                       f"'{old_preview}...' -> '{new_preview}...'")
                            self.prompt = new_prompt
                        return
        except Exception:
            pass  # fall back to existing prompt

    def _drain_messages(self):
        """Consume any pending messages from the buffer to prevent
        leftover notifications from a previous run interfering with the
        next run's event loop."""
        import fcntl
        # Set stdout to non-blocking
        fd = self.proc.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                try:
                    line = self.proc.stdout.readline()
                except (IOError, OSError):
                    break
                if not line:
                    break
        finally:
            # Restore blocking mode
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    # --- Main run loop -------------------------------------------------------

    def run(self):
        if not self.api_key:
            print("Error: No windsurf_api_key found in credentials.toml",
                  file=sys.stderr)
            return 1

        env = os.environ.copy()
        env["ACP_BACKEND"] = "windsurf"
        self.proc = subprocess.Popen(
            [self.devin_bin, "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            bufsize=1,
        )

        overall_rc = 0
        run_num = 0

        try:
            while True:
                run_num += 1

                # Determine if we need a new session
                if self.session_continuation == 0:
                    # Never start new session (except first run)
                    need_new_session = (run_num == 1)
                else:
                    need_new_session = ((run_num - 1) % self.session_continuation == 0)

                # Track which run within the current session
                if self.session_continuation == 0:
                    session_run = run_num
                else:
                    session_run = ((run_num - 1) % self.session_continuation) + 1

                # Print banner
                self._print_run_banner(run_num, self.loops, need_new_session, session_run)

                # Re-read the prompt from the script file (live steering)
                if run_num > 1:
                    self._reread_prompt()

                # Drain any leftover messages from the previous run
                if run_num > 1:
                    self._drain_messages()

                # Start the run (sets up step queue and sends first message)
                self._begin_run(run_num, need_new_session)

                # Run the event loop until this run completes
                rc = self._event_loop()
                if rc != 0:
                    overall_rc = rc

                # Flush a newline to stdout so the formatter can process
                # the response text from this run (agent_message_chunk writes
                # without trailing newlines, so the line-based formatter needs
                # this to delimit runs).
                sys.stdout.write("\n")
                sys.stdout.flush()

                # Print summary
                self._print_run_summary(rc, run_num)

                # Check if we've hit the loop limit
                if self.loops > 0 and run_num >= self.loops:
                    break

                # Sleep between runs
                if self.interval > 0:
                    self._emit("INFO", f"sleeping {self.interval}s until next run...")
                    time.sleep(self.interval)

        except KeyboardInterrupt:
            if self.session_id:
                self._send("session/cancel",
                           {"sessionId": self.session_id},
                           id=self._next_id())
            overall_rc = 130
        finally:
            self._print_final_summary()
            try:
                if self.proc.poll() is None:
                    if self.session_id:
                        self._send("session/close",
                                   {"sessionId": self.session_id},
                                   id=self._next_id())
                        time.sleep(0.5)
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

        return overall_rc


def main():
    parser = argparse.ArgumentParser(description="ACP client for argos-runner")
    parser.add_argument("--prompt", required=True, help="Prompt to send")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory")
    parser.add_argument("--model", default=None, help="Model to use")
    parser.add_argument("--mode", default="bypass",
                        help="Permission mode (bypass, accept-edits, ask, plan)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per run in seconds")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Show agent thinking on stderr")
    parser.add_argument("--loops", type=int, default=1,
                        help="Number of runs (0 = infinite)")
    parser.add_argument("--interval", type=int, default=0,
                        help="Seconds between runs")
    parser.add_argument("--session-continuation", type=int, default=1,
                        help="Runs per session (1 = new each run, 0 = never new)")
    parser.add_argument("--script-path", default=None,
                        help="Path to argos-runner.sh for live prompt re-reading")
    args = parser.parse_args()

    devin_bin = find_devin_bin()
    if not devin_bin:
        print("Error: Could not find devin binary", file=sys.stderr)
        return 1

    api_key, api_server_url = load_credentials()
    if not api_key:
        print("Error: No windsurf_api_key in credentials.toml", file=sys.stderr)
        return 1

    client = AcpClient(
        devin_bin=devin_bin,
        api_key=api_key,
        api_server_url=api_server_url,
        cwd=args.cwd,
        prompt=args.prompt,
        model=args.model,
        mode=args.mode,
        timeout=args.timeout,
        show_thinking=args.show_thinking,
        loops=args.loops,
        interval=args.interval,
        session_continuation=args.session_continuation,
    )
    client.script_path = args.script_path
    return client.run()


if __name__ == "__main__":
    sys.exit(main())
__ACP_PYTHON_EOF__
# --- END EMBEDDED PYTHON ACP CLIENT ---
