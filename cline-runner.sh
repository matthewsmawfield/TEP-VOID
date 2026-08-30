#!/usr/bin/env bash
# =============================================================================
#  cline-runner.sh
# =============================================================================
#  Runs Cline CLI in a loop against the project this script lives in.
#
#  Unlike argos-runner.sh (which embeds a Python ACP client and reimplements
#  output formatting), this script delegates entirely to the `cline` CLI:
#    - Cline renders its own markdown, thinking/reasoning, and tool output
#    - Cline manages sessions, checkpoints, and history natively
#    - This script just handles the loop, timing, registry, and config
#
#  Session continuation: by default a fresh session is used each run. Set
#  SESSION_CONTINUATION (or --session-continuation) to N to keep the same
#  cline session for N consecutive runs before starting a new one. Cline
#  remembers the conversation context between runs (e.g. "any new issues
#  since your last check?").
#
#  Portable: copy this ONE file into any project root and run it there --
#  it auto-detects its own directory as the workspace.
#
#  Runner registry: every running instance registers itself in a shared state
#  directory (STATE_DIR below). On startup this script lists all OTHER live
#  cline-runner runners on the machine so you can see what else is running,
#  and cleans up stale entries left by crashed instances.
#
#  Usage:
#    ./cline-runner.sh                       # run with defaults (every 5 min)
#    ./cline-runner.sh --once                # single run, then exit
#    ./cline-runner.sh --interval 300        # custom interval (seconds)
#    ./cline-runner.sh --timeout 120         # kill runs that take >2 min
#    ./cline-runner.sh --loops 5             # stop after 5 runs (default: infinite)
#    ./cline-runner.sh --session-continuation 3   # same session for 3 runs
#    ./cline-runner.sh --prompt "..."        # override the prompt for this run
#    ./cline-runner.sh --model deepseek-v4   # override the model for this run
#    ./cline-runner.sh --provider openrouter # override the provider
#    ./cline-runner.sh --thinking high       # set reasoning effort
#    ./cline-runner.sh --plan                # run in plan mode (read-only)
#    ./cline-runner.sh --log /path           # write run output to a log file
#    ./cline-runner.sh --dry-run             # show what would run, don't call Cline
#    ./cline-runner.sh --no-color            # disable colored output
#    ./cline-runner.sh --check-auth          # verify credentials and exit
#    ./cline-runner.sh --status              # show all running cline-runner instances
#    ./cline-runner.sh --list-models         # show all known models and exit
#    ./cline-runner.sh --help                # show this header
#
#  Stop a running loop with Ctrl+C.
#
#  All tunable settings live in the CONFIGURATION block below.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION  —  edit these defaults to suit your project
# -----------------------------------------------------------------------------
# How long to wait between runs, in seconds. 300 = 5 minutes.
INTERVAL="${INTERVAL:-300}"

# Maximum number of runs before stopping. 0 = infinite (loop forever).
MAX_LOOPS="${MAX_LOOPS:-0}"

# Timeout per Cline run, in seconds. 0 = no timeout. If a run takes longer
# than this, it is killed and the loop continues. 1800 = 30 minutes.
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"

# The prompt sent to Cline on each run.
# By default, read from .cline/runner-prompt.txt (Cline can edit this file to
# change its own prompt). Override with --prompt or the PROMPT env var.
# (Actual loading happens after SCRIPT_DIR is resolved below.)
DEFAULT_PROMPT_FALLBACK="Deep scan and fix all issues; keep investigating for bugs and inconsistencies. Make sure its all correct and genuine. Investigate, explore, debug and fix all issues. We shouldn't have to force or fudge it, actually if you understand TEP properly it should all fall into place quite simply. Ensure consistency with wider corpus @manuscripts."

# AI provider to use. Cline supports: anthropic, openai-native, openrouter,
# gemini, deepseek, moonshot, xai, bedrock, cerebras, ollama, lmstudio, etc.
# Leave empty to use your Cline default (last authenticated provider).
PROVIDER="${PROVIDER:-cline-pass}"

# AI model to use. See --list-models for all known models. Common aliases
# are mapped automatically (e.g. "deepseek-v4" -> "deepseek/deepseek-v4-flash").
# Leave empty to use your provider's default model.
MODEL="${MODEL:-cline-pass/kimi-k3}"

# Reasoning effort level: none|low|medium|high|xhigh. Bare --thinking uses
# medium. Omitted leaves the provider default. Higher = more thinking tokens.
THINKING="${THINKING:-high}"

# Auto-approve tool calls for autonomous runs. true = Cline runs without
# asking for permission (required for the runner to actually fix issues).
# false = Cline will ask for approval (won't work in non-interactive mode).
AUTO_APPROVE="${AUTO_APPROVE:-true}"

# Run in plan mode (read-only analysis, no code changes). Equivalent to
# --plan. Set to 1 to enable.
PLAN_MODE="${PLAN_MODE:-0}"

# Session continuation: how many consecutive runs share the same Cline session
# before starting a new one. 1 = new session every run (no memory). 5 = Cline
# remembers the previous 4 runs' context on the 5th run, then starts fresh.
# 0 = never start a new session (continue forever, Cline remembers everything).
SESSION_CONTINUATION="${SESSION_CONTINUATION:-0}"

# Path to the Cline CLI binary. Leave empty to auto-detect (PATH, then
# homebrew, then npm global).
CLINE_BIN="${CLINE_BIN:-}"

# Output formatting: "auto" uses colors when stdout is a TTY, "always" forces
# color, "never" disables it. Cline handles its own markdown rendering; this
# only controls the runner's own status messages.
COLOR="${COLOR:-auto}"

# Log file for run output. If set, each run's raw Cline output is appended
# to this file with a timestamp header. Leave empty to disable file logging.
LOG_FILE="${LOG_FILE:-}"

# Show agent thinking (reasoning) output. Cline handles this natively via
# --thinking; this flag is for the runner's own verbose status messages.
# 0 = quiet, 1 = show runner internals.
VERBOSE="${VERBOSE:-0}"

# Shared state directory where each runner registers itself so other runners
# can discover it. Override with CLINE_RUNNER_STATE_DIR if needed.
STATE_DIR="${CLINE_RUNNER_STATE_DIR:-/tmp/cline-runner}"

# Context compaction mode: agentic|basic|off. Controls how Cline manages
# context window limits during long sessions.
COMPACTION="${COMPACTION:-basic}"

# Maximum consecutive mistakes (retries) before Cline halts a run.
RETRIES="${RETRIES:-6}"
# -----------------------------------------------------------------------------
# END CONFIGURATION
# -----------------------------------------------------------------------------

# Don't use set -e here because we handle errors explicitly in the loop.
# pipefail is useful for catching exit codes through pipes.
set -uo pipefail

# --- Resolve project root: the directory this script lives in ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SELF_PID=$$

# --- Load prompt from file (after SCRIPT_DIR is resolved) -------------------
# Cline can edit .cline/runner-prompt.txt to change its own prompt.
# Override with --prompt or the PROMPT env var.
if [[ -z "${PROMPT:-}" ]]; then
  _DEFAULT_PROMPT_FILE="$SCRIPT_DIR/.cline/runner-prompt.txt"
  if [[ -f "$_DEFAULT_PROMPT_FILE" ]]; then
    PROMPT="$(cat "$_DEFAULT_PROMPT_FILE")"
  else
    PROMPT="$DEFAULT_PROMPT_FALLBACK"
  fi
fi

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

# --- Locate the Cline binary ------------------------------------------------
find_cline_bin() {
  if [[ -n "${CLINE_BIN:-}" && -x "$CLINE_BIN" ]]; then
    return 0
  fi
  if command -v cline >/dev/null 2>&1; then
    CLINE_BIN="$(command -v cline)"
    return 0
  fi
  # Try common npm global locations
  local npm_bin
  npm_bin="$(npm config get prefix 2>/dev/null)/bin/cline"
  if [[ -x "$npm_bin" ]]; then
    CLINE_BIN="$npm_bin"
    return 0
  fi
  printf "${C_BOLD}${C_RED}error: could not find the 'cline' binary.${C_RESET}\n" >&2
  printf "       install it with: npm i -g cline\n" >&2
  printf "       or set CLINE_BIN=/path/to/cline\n" >&2
  return 1
}

# =============================================================================
# MODEL & PROVIDER REGISTRY
# =============================================================================
# Cline supports many providers, each with their own model IDs. OpenRouter
# uses "provider/model" format (e.g. "deepseek/deepseek-v4-flash"). Direct
# providers use bare model IDs (e.g. "claude-sonnet-4-6").
#
# The map_model function below resolves friendly aliases to canonical IDs.
# The map_provider function resolves provider aliases.
# Use --list-models to see all known models organized by provider.

map_provider() {
  local input="$1"
  [[ -z "$input" ]] && { echo ""; return; }
  case "$input" in
    anthropic|claude)                          echo "anthropic";          return ;;
    openai|openai-native|gpt)                  echo "openai-native";      return ;;
    openrouter|or)                             echo "openrouter";         return ;;
    gemini|google)                             echo "gemini";             return ;;
    deepseek|ds)                               echo "deepseek";           return ;;
    moonshot|kimi)                             echo "moonshot";           return ;;
    xai|grok)                                  echo "xai";                return ;;
    bedrock|aws)                               echo "bedrock";            return ;;
    cerebras)                                  echo "cerebras";           return ;;
    ollama|local)                              echo "ollama";             return ;;
    lmstudio)                                  echo "lmstudio";           return ;;
    sapaicore|sap)                             echo "sapaicore";          return ;;
    cline)                                     echo "cline";              return ;;
    cline-pass|clinepass|pass)                 echo "cline-pass";         return ;;
    *)                                         echo "$input";             return ;;
  esac
}

map_model() {
  local input="$1"
  local provider="${2:-}"
  [[ -z "$input" ]] && { echo ""; return; }

  # --- Provider-agnostic aliases (work with any provider) ---
  case "$input" in
    auto|adaptive)                             echo "";                   return ;;
  esac

  # --- ClinePass subscription models ($9.99/month, 2-5x usage) ---
  if [[ "$provider" == "cline-pass" ]]; then
    case "$input" in
      kimi-k3|k3|kimi)                          echo "cline-pass/kimi-k3";          return ;;
      kimi-k2.7-code|kimi-k2-7-code)            echo "cline-pass/kimi-k2.7-code";   return ;;
      kimi-k2.6|kimi-k2-6)                      echo "cline-pass/kimi-k2.6";        return ;;
      glm-5.3|glm-5-3)                          echo "cline-pass/glm-5.3";          return ;;
      glm-5.2|glm-5-2)                          echo "cline-pass/glm-5.2";          return ;;
      deepseek-v4-pro|ds-v4-pro)                echo "cline-pass/deepseek-v4-pro";  return ;;
      deepseek-v4-flash|ds-v4-flash)            echo "cline-pass/deepseek-v4-flash"; return ;;
      minimax-m3|minimax)                       echo "cline-pass/minimax-m3";       return ;;
      mimo-v2.5-pro|mimo-pro)                   echo "cline-pass/mimo-v2.5-pro";    return ;;
      mimo-v2.5|mimo)                           echo "cline-pass/mimo-v2.5";        return ;;
      qwen3.8-max|qwen-3.8-max)                 echo "cline-pass/qwen3.8-max";      return ;;
      qwen3.7-max|qwen-3.7-max)                 echo "cline-pass/qwen3.7-max";      return ;;
      qwen3.7-plus|qwen-3.7-plus)               echo "cline-pass/qwen3.7-plus";     return ;;
      # Pass through if already cline-pass/ prefixed
      cline-pass/*)                             echo "$input";                      return ;;
    esac
  fi

  # --- OpenRouter models (provider/model format) ---
  if [[ "$provider" == "openrouter" || ( -z "$provider" && "$input" == */* ) ]]; then
    case "$input" in
      # DeepSeek
      deepseek-v4|deepseek-v4-flash|ds-v4|ds-flash)
        echo "deepseek/deepseek-v4-flash"; return ;;
      deepseek-chat|ds-chat)
        echo "deepseek/deepseek-chat"; return ;;
      deepseek-r1|ds-r1)
        echo "deepseek/deepseek-r1"; return ;;
      # Anthropic via OpenRouter
      claude-sonnet-4.6|sonnet-4.6|sonnet-4-6)
        echo "anthropic/claude-sonnet-4-6"; return ;;
      claude-opus-4.6|opus-4.6|opus-4-6)
        echo "anthropic/claude-opus-4-6"; return ;;
      claude-haiku-4.5|haiku-4.5|haiku-4-5)
        echo "anthropic/claude-haiku-4-5"; return ;;
      claude-sonnet-4.5|sonnet-4.5|sonnet-4-5)
        echo "anthropic/claude-sonnet-4-5-20250929"; return ;;
      claude-3.7-sonnet|claude-3-7-sonnet)
        echo "anthropic/claude-3.7-sonnet"; return ;;
      # Google via OpenRouter
      gemini-2.5-pro|gemini-pro)
        echo "google/gemini-2.5-pro"; return ;;
      gemini-2.5-flash|gemini-flash)
        echo "google/gemini-2.5-flash"; return ;;
      gemini-3.7-flash|gemini-3-7-flash)
        echo "google/gemini-3.7-flash"; return ;;
      # OpenAI via OpenRouter
      gpt-4o|gpt-4o)
        echo "openai/gpt-4o"; return ;;
      gpt-5|gpt-5.5)
        echo "openai/gpt-5.5"; return ;;
      # xAI via OpenRouter
      grok-4|grok)
        echo "xai/grok-4"; return ;;
      # Moonshot via OpenRouter
      kimi-k3|k3|kimi)
        echo "moonshotai/kimi-k3"; return ;;
      kimi-k2.5)
        echo "moonshotai/kimi-k2.5"; return ;;
      # If it already has provider/ prefix, pass through
      */*)
        echo "$input"; return ;;
    esac
  fi

  # --- Direct Anthropic / Cline provider (same model IDs) ---
  if [[ "$provider" == "anthropic" || "$provider" == "cline" ]]; then
    case "$input" in
      sonnet-4.6|sonnet-4-6|claude-sonnet-4-6)       echo "claude-sonnet-4-6";           return ;;
      sonnet-4.6-1m|sonnet-4-6:1m)                   echo "claude-sonnet-4-6:1m";        return ;;
      opus-4.6|opus-4-6|claude-opus-4-6)             echo "claude-opus-4-6";             return ;;
      opus-4.6-1m|opus-4-6:1m)                       echo "claude-opus-4-6:1m";          return ;;
      haiku-4.5|haiku-4-5|claude-haiku-4-5)          echo "claude-haiku-4-5-20251001";   return ;;
      sonnet-4.5|sonnet-4-5|claude-sonnet-4-5)       echo "claude-sonnet-4-5-20250929";  return ;;
      sonnet-4|claude-sonnet-4)                      echo "claude-sonnet-4-20250514";    return ;;
      opus-4.5|opus-4-5|claude-opus-4-5)             echo "claude-opus-4-5-20251101";    return ;;
      opus-4.1|opus-4-1|claude-opus-4-1)             echo "claude-opus-4-1-20250805";    return ;;
      opus-4|claude-opus-4)                          echo "claude-opus-4-20250514";      return ;;
      claude-3.7-sonnet|claude-3-7-sonnet)           echo "claude-3-7-sonnet-20250219";  return ;;
      claude-3.5-sonnet|claude-3-5-sonnet)           echo "claude-3-5-sonnet-20241022";  return ;;
      claude-3.5-haiku|claude-3-5-haiku)             echo "claude-3-5-haiku-20241022";   return ;;
      claude-3-opus|claude-3-opus)                   echo "claude-3-opus-20240229";      return ;;
      claude-3-haiku|claude-3-haiku)                 echo "claude-3-haiku-20240307";     return ;;
    esac
  fi

  # --- Direct DeepSeek provider ---
  if [[ "$provider" == "deepseek" ]]; then
    case "$input" in
      v4|v4-flash|deepseek-v4|deepseek-v4-flash)     echo "deepseek-v4-flash";           return ;;
      chat|deepseek-chat)                            echo "deepseek-chat";               return ;;
      r1|deepseek-r1)                                echo "deepseek-r1";                 return ;;
    esac
  fi

  # --- Direct Gemini provider ---
  if [[ "$provider" == "gemini" ]]; then
    case "$input" in
      2.5-pro|gemini-2.5-pro)                        echo "gemini-2.5-pro";              return ;;
      2.5-flash|gemini-2.5-flash)                    echo "gemini-2.5-flash";            return ;;
      3.7-flash|gemini-3.7-flash)                    echo "gemini-3.7-flash";            return ;;
    esac
  fi

  # --- Direct OpenAI provider ---
  if [[ "$provider" == "openai-native" ]]; then
    case "$input" in
      gpt-4o)                                        echo "gpt-4o";                      return ;;
      gpt-4)                                         echo "gpt-4";                       return ;;
      gpt-5|gpt-5.5)                                 echo "gpt-5.5";                     return ;;
    esac
  fi

  # --- Direct Moonshot provider ---
  if [[ "$provider" == "moonshot" ]]; then
    case "$input" in
      k2.5|kimi-k2.5|kimi)                           echo "kimi-k2.5";                   return ;;
      k2|kimi-k2)                                    echo "kimi-k2";                     return ;;
    esac
  fi

  # --- Direct xAI provider ---
  if [[ "$provider" == "xai" ]]; then
    case "$input" in
      grok-4|grok)                                   echo "grok-4";                      return ;;
      grok-3|grok3)                                  echo "grok-3";                      return ;;
    esac
  fi

  # --- SAP AI Core ---
  if [[ "$provider" == "sapaicore" ]]; then
    case "$input" in
      gpt-5.5|gpt-5)                                 echo "gpt-5.5";                     return ;;
    esac
  fi

  # Pass through unrecognized model IDs as-is
  echo "$input"
}

# --- List all known models (--list-models) -----------------------------------
list_models() {
  printf "${C_BOLD}${C_MAGENTA}Available Models by Provider${C_RESET}\n"
  printf "${C_DIM}  Use --provider <id> --model <alias> or canonical ID.${C_RESET}\n\n"

  printf "${C_BOLD}${C_CYAN}cline-pass${C_RESET}  (\$9.99/month subscription, 2-5x usage on open models)\n"
  printf "  ${C_DIM}Aliases -> Canonical ID${C_RESET}\n"
  printf "    kimi-k3            -> cline-pass/kimi-k3\n"
  printf "    kimi-k2.7-code     -> cline-pass/kimi-k2.7-code\n"
  printf "    kimi-k2.6          -> cline-pass/kimi-k2.6\n"
  printf "    glm-5.3            -> cline-pass/glm-5.3\n"
  printf "    glm-5.2            -> cline-pass/glm-5.2\n"
  printf "    deepseek-v4-pro    -> cline-pass/deepseek-v4-pro\n"
  printf "    deepseek-v4-flash  -> cline-pass/deepseek-v4-flash\n"
  printf "    minimax-m3         -> cline-pass/minimax-m3\n"
  printf "    mimo-v2.5-pro      -> cline-pass/mimo-v2.5-pro\n"
  printf "    mimo-v2.5          -> cline-pass/mimo-v2.5\n"
  printf "    qwen3.8-max        -> cline-pass/qwen3.8-max\n"
  printf "    qwen3.7-max        -> cline-pass/qwen3.7-max\n"
  printf "    qwen3.7-plus       -> cline-pass/qwen3.7-plus\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}anthropic${C_RESET}  (direct Claude API)\n"
  printf "  ${C_DIM}Aliases -> Canonical ID${C_RESET}\n"
  printf "    sonnet-4.6         -> claude-sonnet-4-6\n"
  printf "    sonnet-4.6-1m      -> claude-sonnet-4-6:1m       (1M context)\n"
  printf "    opus-4.6           -> claude-opus-4-6\n"
  printf "    opus-4.6-1m        -> claude-opus-4-6:1m         (1M context)\n"
  printf "    haiku-4.5          -> claude-haiku-4-5-20251001\n"
  printf "    sonnet-4.5         -> claude-sonnet-4-5-20250929\n"
  printf "    sonnet-4           -> claude-sonnet-4-20250514\n"
  printf "    opus-4.5           -> claude-opus-4-5-20251101\n"
  printf "    opus-4.1           -> claude-opus-4-1-20250805\n"
  printf "    opus-4             -> claude-opus-4-20250514\n"
  printf "    claude-3.7-sonnet  -> claude-3-7-sonnet-20250219\n"
  printf "    claude-3.5-sonnet  -> claude-3-5-sonnet-20241022\n"
  printf "    claude-3.5-haiku   -> claude-3-5-haiku-20241022\n"
  printf "    claude-3-opus      -> claude-3-opus-20240229\n"
  printf "    claude-3-haiku     -> claude-3-haiku-20240307\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}openrouter${C_RESET}  (multi-provider, provider/model format)\n"
  printf "  ${C_DIM}Aliases -> Canonical ID${C_RESET}\n"
  printf "    deepseek-v4        -> deepseek/deepseek-v4-flash\n"
  printf "    deepseek-chat      -> deepseek/deepseek-chat\n"
  printf "    deepseek-r1        -> deepseek/deepseek-r1\n"
  printf "    sonnet-4.6         -> anthropic/claude-sonnet-4-6\n"
  printf "    opus-4.6           -> anthropic/claude-opus-4-6\n"
  printf "    haiku-4.5          -> anthropic/claude-haiku-4-5\n"
  printf "    gemini-2.5-pro     -> google/gemini-2.5-pro\n"
  printf "    gemini-2.5-flash   -> google/gemini-2.5-flash\n"
  printf "    gemini-3.7-flash   -> google/gemini-3.7-flash\n"
  printf "    gpt-4o             -> openai/gpt-4o\n"
  printf "    gpt-5.5            -> openai/gpt-5.5\n"
  printf "    grok-4             -> xai/grok-4\n"
  printf "    kimi-k3           -> moonshotai/kimi-k3\n"
  printf "    kimi-k2.5         -> moonshotai/kimi-k2.5\n"
  printf "    ${C_DIM}(or any provider/model ID from openrouter.ai/models)${C_RESET}\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}deepseek${C_RESET}  (direct DeepSeek API)\n"
  printf "    v4-flash           -> deepseek-v4-flash\n"
  printf "    chat               -> deepseek-chat\n"
  printf "    r1                 -> deepseek-r1\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}gemini${C_RESET}  (direct Google API)\n"
  printf "    2.5-pro            -> gemini-2.5-pro\n"
  printf "    2.5-flash          -> gemini-2.5-flash\n"
  printf "    3.7-flash          -> gemini-3.7-flash\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}openai-native${C_RESET}  (direct OpenAI API)\n"
  printf "    gpt-4o             -> gpt-4o\n"
  printf "    gpt-4              -> gpt-4\n"
  printf "    gpt-5.5            -> gpt-5.5\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}moonshot${C_RESET}  (direct Moonshot/Kimi API)\n"
  printf "    kimi-k2.5          -> kimi-k2.5\n"
  printf "    kimi-k2            -> kimi-k2\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}xai${C_RESET}  (direct xAI/Grok API)\n"
  printf "    grok-4             -> grok-4\n"
  printf "    grok-3             -> grok-3\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}sapaicore${C_RESET}  (SAP AI Core)\n"
  printf "    gpt-5.5            -> gpt-5.5\n"
  printf "\n"

  printf "${C_BOLD}${C_CYAN}ollama${C_RESET} / ${C_CYAN}lmstudio${C_RESET}  (local models)\n"
  printf "    ${C_DIM}Use any model ID supported by your local server.${C_RESET}\n"
  printf "\n"

  printf "${C_BOLD}${C_YELLOW}Other providers:${C_RESET} bedrock, cerebras, openai (OpenAI-compatible)\n"
  printf "${C_DIM}  See https://docs.cline.bot/cli/cli-reference for full provider list.${C_RESET}\n"
  exit 0
}

# --- Auth check --------------------------------------------------------------
check_auth() {
  local settings_dir
  settings_dir="$HOME/.cline/data/settings"
  local providers_file="$settings_dir/providers.json"
  if [[ ! -f "$providers_file" ]]; then
    printf "${C_BOLD}${C_YELLOW}Warning: No Cline providers config at %s${C_RESET}\n" "$providers_file" >&2
    printf "${C_DIM}  The runner will not work. To fix:${C_RESET}\n" >&2
    printf "    Run: cline auth -p <provider> -k <key> -m <model>${C_RESET}\n" >&2
    return 1
  fi

  # Parse providers.json and list configured providers (without exposing keys)
  local provider_list
  provider_list="$(python3 -c "
import json, sys
try:
    d = json.load(open('$providers_file'))
    for pid, info in d.get('providers', {}).items():
        s = info.get('settings', {})
        model = s.get('model', 'default')
        print(f'{pid}|{model}')
except Exception as e:
    sys.exit(1)
" 2>/dev/null)"

  if [[ -z "$provider_list" ]]; then
    printf "${C_BOLD}${C_YELLOW}Warning: No authenticated providers found.${C_RESET}\n" >&2
    printf "${C_DIM}  Run: cline auth -p <provider> -k <key> -m <model>${C_RESET}\n" >&2
    return 1
  fi

  printf "${C_DIM}  Authenticated providers:${C_RESET}\n"
  while IFS='|' read -r provider model; do
    printf "    ${C_GREEN}%s${C_RESET} ${C_DIM}(model: %s)${C_RESET}\n" "$provider" "$model"
  done <<< "$provider_list"
  return 0
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
  local short_model; short_model="${MODEL:-default}"
  {
    printf 'pid=%s\n'           "$SELF_PID"
    printf 'project=%s\n'       "$PROJECT_ROOT"
    printf 'started=%s\n'       "$started"
    printf 'interval=%s\n'      "$INTERVAL"
    printf 'continuation=%s\n'  "$SESSION_CONTINUATION"
    printf 'model=%s\n'         "$short_model"
    printf 'provider=%s\n'      "${PROVIDER:-default}"
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
  local f pid proj started intvl cont mdl prov pr
  for f in "$STATE_DIR"/*.state; do
    [[ -e "$f" ]] || continue
    pid=""; proj=""; started=""; intvl=""; cont=""; mdl=""; prov=""; pr=""
    while IFS='=' read -r k v; do
      case "$k" in
        pid) pid="$v" ;;  project) proj="$v" ;;
        started) started="$v" ;;  interval) intvl="$v" ;;
        continuation) cont="$v" ;;  model) mdl="$v" ;;
        provider) prov="$v" ;;  prompt) pr="$v" ;;
      esac
    done < "$f"
    [[ "$pid" == "$SELF_PID" ]] && continue
    if ! pid_alive "$pid"; then
      rm -f "$f"; continue
    fi
    others+=("$pid|$proj|$started|$intvl|$cont|$mdl|$prov|$pr")
  done

  OTHER_RUNNERS_COUNT=${#others[@]}
  if [[ "$OTHER_RUNNERS_COUNT" -eq 0 ]]; then
    printf "${C_DIM}Other cline-runner runners: none${C_RESET}\n"
    return 0
  fi
  printf "${C_BOLD}${C_YELLOW}Other cline-runner runners (%s):${C_RESET}\n" "$OTHER_RUNNERS_COUNT"
  local entry
  for entry in "${others[@]}"; do
    IFS='|' read -r pid proj started intvl cont mdl prov pr <<< "$entry"
    local proj_short; proj_short="$(basename "$proj")"
    printf "  %b-%b pid=%-7s  %-20s  %s/%s  every %ss  cont=%s  since %s\n" \
      "$C_CYAN" "$C_RESET" "$pid" "$proj_short" "${prov:-?}" "${mdl:-?}" "$intvl" "${cont:-1}" "$started"
    printf "    %bproject :%b %s\n" "$C_DIM" "$C_RESET" "$proj"
    printf "    %bprompt  :%b %s\n" "$C_DIM" "$C_RESET" "$pr"
  done
}

# --- Show all runners (--status) --------------------------------------------
show_status() {
  [[ -d "$STATE_DIR" ]] || { echo "No runners registered."; exit 0; }
  local found=0
  local f pid proj started intvl cont mdl prov pr
  for f in "$STATE_DIR"/*.state; do
    [[ -e "$f" ]] || continue
    pid=""; proj=""; started=""; intvl=""; cont=""; mdl=""; prov=""; pr=""
    while IFS='=' read -r k v; do
      case "$k" in
        pid) pid="$v" ;;  project) proj="$v" ;;
        started) started="$v" ;;  interval) intvl="$v" ;;
        continuation) cont="$v" ;;  model) mdl="$v" ;;
        provider) prov="$v" ;;  prompt) pr="$v" ;;
      esac
    done < "$f"
    if ! pid_alive "$pid"; then
      rm -f "$f"; continue
    fi
    found=$((found + 1))
    if [[ "$found" -eq 1 ]]; then
      printf "%b${C_BOLD}${C_MAGENTA}Active cline-runner instances:${C_RESET}\n" ""
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
    printf "  %b#%b pid=%-7s  %-20s  %s/%s  every %ss  cont=%s  uptime %s\n" \
      "$C_CYAN" "$C_RESET" "$pid" "$proj_short" "${prov:-?}" "${mdl:-?}" "$intvl" "${cont:-1}" "$age_str"
    printf "    %bproject :%b %s\n" "$C_DIM" "$C_RESET" "$proj"
    printf "    %bprompt  :%b %s\n" "$C_DIM" "$C_RESET" "$pr"
  done
  [[ "$found" -eq 0 ]] && echo "No active cline-runner instances."
  exit 0
}

# --- Detect project context files -------------------------------------------
detect_context() {
  local ctx=""
  [[ -f "$PROJECT_ROOT/.clinerules" ]] && ctx="$ctx .clinerules"
  [[ -f "$PROJECT_ROOT/.cline/rules" ]] && ctx="$ctx .cline/rules"
  [[ -d "$PROJECT_ROOT/.cline" ]] && ctx="$ctx .cline/"
  [[ -f "$PROJECT_ROOT/AGENTS.md" ]] && ctx="$ctx AGENTS.md"
  [[ -f "$PROJECT_ROOT/CLAUDE.md" ]] && ctx="$ctx CLAUDE.md"
  [[ -f "$PROJECT_ROOT/.devin/rules" ]] && ctx="$ctx .devin/rules"
  [[ -d "$PROJECT_ROOT/.devin/workflows" ]] && ctx="$ctx .devin/workflows/"
  [[ -n "$ctx" ]] && printf "${C_DIM}  context :%s${C_RESET}\n" "$ctx"
}

# --- Build and run the Cline command ----------------------------------------
run_cline() {
  local session_id="${1:-}"
  local run_num="$2"

  # Resolve provider and model
  local resolved_provider resolved_model
  resolved_provider="$(map_provider "${PROVIDER:-}")"
  resolved_model="$(map_model "${MODEL:-}" "$resolved_provider")"

  # Build the cline command
  local -a cmd=()
  cmd+=("$CLINE_BIN")
  cmd+=(--cwd "$PROJECT_ROOT")
  cmd+=(--auto-approve "$AUTO_APPROVE")
  cmd+=(--compaction "$COMPACTION")
  cmd+=(--retries "$RETRIES")

  # Provider
  if [[ -n "$resolved_provider" ]]; then
    cmd+=(-P "$resolved_provider")
  fi

  # Model
  if [[ -n "$resolved_model" ]]; then
    cmd+=(-m "$resolved_model")
  fi

  # Thinking level
  if [[ -n "${THINKING:-}" ]]; then
    cmd+=(--thinking "$THINKING")
  fi

  # Timeout
  if [[ "$RUN_TIMEOUT" -gt 0 ]]; then
    cmd+=(-t "$RUN_TIMEOUT")
  fi

  # Plan mode
  if [[ "$PLAN_MODE" -eq 1 ]]; then
    cmd+=(-p)
  fi

  # Session continuation: resume existing session
  if [[ -n "$session_id" ]]; then
    cmd+=(--id "$session_id")
  fi

  # The prompt (positional argument)
  cmd+=("$PROMPT")

  # --- Print run header ---
  local ts; ts="$(date '+%H:%M:%S')"
  printf "\n${C_BOLD}${C_MAGENTA}━━━ Cline Run #%d ━━━ %s ${C_RESET}\n" "$run_num" "$ts"
  printf "${C_DIM}  provider: %s${C_RESET}\n" "${resolved_provider:-default}"
  printf "${C_DIM}  model   : %s${C_RESET}\n" "${resolved_model:-default}"
  printf "${C_DIM}  thinking: %s${C_RESET}\n" "${THINKING:-default}"
  printf "${C_DIM}  mode    : %s${C_RESET}\n" "$([[ "$PLAN_MODE" -eq 1 ]] && echo 'plan (read-only)' || echo 'act (autonomous)')"
  if [[ -n "$session_id" ]]; then
    printf "${C_DIM}  session : %s (continuing)${C_RESET}\n" "$session_id"
  else
    printf "${C_DIM}  session : new${C_RESET}\n"
  fi
  printf "${C_DIM}  prompt  : %s${C_RESET}\n" "$(printf '%s' "$PROMPT" | head -c 100)"
  detect_context
  printf "${C_BOLD}${C_MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}\n\n"

  # --- Dry run: print command and return ---
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf "${C_YELLOW}[dry-run] Command:${C_RESET}\n  %s\n" "${cmd[*]}"
    return 0
  fi

  # --- Run Cline ---
  # Cline renders its own output (markdown, thinking, tool calls) natively.
  # We just pass stdout/stderr through. If logging is enabled, tee to file.
  local exit_code
  if [[ -n "$LOG_FILE" ]]; then
    {
      printf '\n=== Run #%d at %s ===\n' "$run_num" "$(date '+%Y-%m-%d %H:%M:%S')"
      printf '=== Provider: %s | Model: %s | Session: %s ===\n\n' \
        "${resolved_provider:-default}" "${resolved_model:-default}" "${session_id:-new}"
    } >> "$LOG_FILE"
    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
  else
    "${cmd[@]}"
    exit_code=$?
  fi

  printf "\n${C_DIM}  exit code: %s${C_RESET}\n" "$exit_code"

  # --- Capture session ID for continuation ---
  if [[ "$SESSION_CONTINUATION" != "1" && "$exit_code" -eq 0 ]]; then
    LAST_SESSION_ID="$("$CLINE_BIN" history --json --limit 1 2>/dev/null \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['sessionId'] if d else '')" 2>/dev/null)"
    if [[ -n "$LAST_SESSION_ID" ]]; then
      printf "${C_DIM}  session : %s (saved for continuation)${C_RESET}\n" "$LAST_SESSION_ID"
    fi
  fi

  return $exit_code
}

# --- Cleanup on exit --------------------------------------------------------
cleanup() {
  unregister_self
  printf "\n${C_BOLD}${C_YELLOW}Stopping cline-runner (pid=%s).${C_RESET}\n" "$SELF_PID"
  exit 0
}
trap cleanup INT TERM

# --- Parse command-line arguments -------------------------------------------
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)              MAX_LOOPS=1; shift ;;
    --interval)          INTERVAL="$2"; shift 2 ;;
    --timeout)           RUN_TIMEOUT="$2"; shift 2 ;;
    --loops)             MAX_LOOPS="$2"; shift 2 ;;
    --session-continuation) SESSION_CONTINUATION="$2"; shift 2 ;;
    --prompt)            PROMPT="$2"; shift 2 ;;
    --model)             MODEL="$2"; shift 2 ;;
    --provider)          PROVIDER="$2"; shift 2 ;;
    --thinking)          THINKING="$2"; shift 2 ;;
    --plan)              PLAN_MODE=1; shift ;;
    --yolo)              AUTO_APPROVE="true"; shift ;;
    --no-auto-approve)   AUTO_APPROVE="false"; shift ;;
    --log)               LOG_FILE="$2"; shift 2 ;;
    --dry-run)           DRY_RUN=1; shift ;;
    --no-color)          COLOR="never"; USE_COLOR=0; shift ;;
    --color)             COLOR="$2"; shift 2 ;;
    --verbose|-v)        VERBOSE=1; shift ;;
    --check-auth)        find_cline_bin && check_auth; exit $? ;;
    --status)            show_status ;;
    --list-models)       list_models ;;
    --help|-h)
      sed -n '3,/^# ====/p' "$0" | sed 's/^# \?//' | head -60
      exit 0 ;;
    *)
      # If it's the first non-flag arg and no prompt set via --prompt, treat as prompt
      if [[ -z "${PROMPT_SET:-}" && "$1" != -* ]]; then
        PROMPT="$1"; PROMPT_SET=1; shift
      else
        printf "${C_RED}error: unknown option: %s${C_RESET}\n" "$1" >&2
        printf "  run --help for usage.\n" >&2
        exit 1
      fi
      ;;
  esac
done

# --- Startup banner ---------------------------------------------------------
printf "${C_BOLD}${C_MAGENTA}"
printf "╔══════════════════════════════════════════════════════════════╗\n"
printf "║                    C L I N E   R U N N E R                    ║\n"
printf "╚══════════════════════════════════════════════════════════════╝\n"
printf "${C_RESET}"
printf "${C_DIM}  project : %s${C_RESET}\n" "$(basename "$PROJECT_ROOT")"
printf "${C_DIM}  path    : %s${C_RESET}\n" "$PROJECT_ROOT"
printf "${C_DIM}  provider: %s${C_RESET}\n" "${PROVIDER:-default}"
printf "${C_DIM}  model   : %s${C_RESET}\n" "${MODEL:-default}"
printf "${C_DIM}  thinking: %s${C_RESET}\n" "${THINKING:-default}"
printf "${C_DIM}  interval: %ss${C_RESET}\n" "$INTERVAL"
printf "${C_DIM}  loops   : %s${C_RESET}\n" "$([[ "$MAX_LOOPS" -eq 0 ]] && echo 'infinite' || echo "$MAX_LOOPS")"
printf "${C_DIM}  timeout : %ss${C_RESET}\n" "$([[ "$RUN_TIMEOUT" -eq 0 ]] && echo 'none' || echo "$RUN_TIMEOUT")"
printf "${C_DIM}  cont.   : %s${C_RESET}\n" "$([[ "$SESSION_CONTINUATION" -eq 0 ]] && echo 'forever' || echo "$SESSION_CONTINUATION runs/session")"
detect_context

# --- Find cline binary ---
if ! find_cline_bin; then
  exit 1
fi
printf "${C_DIM}  binary  : %s${C_RESET}\n" "$CLINE_BIN"
printf "${C_DIM}  version : %s${C_RESET}\n" "$("$CLINE_BIN" version 2>/dev/null || "$CLINE_BIN" --version 2>/dev/null || echo 'unknown')"

# --- Check auth ---
check_auth || printf "${C_YELLOW}  (proceeding anyway — cline may prompt for auth)${C_RESET}\n"

# --- List other runners ---
list_other_runners

# --- Register self ---
register_self
printf "${C_DIM}  pid     : %s${C_RESET}\n" "$SELF_PID"

printf "\n${C_BOLD}${C_GREEN}Starting loop...${C_RESET}  ${C_DIM}(Ctrl+C to stop)${C_RESET}\n"

# --- Main loop --------------------------------------------------------------
LAST_SESSION_ID=""
run_count=0
session_run=0

while true; do
  run_count=$((run_count + 1))

  # Determine if we should start a new session or continue
  if [[ "$SESSION_CONTINUATION" -eq 0 ]]; then
    # Continue forever (use last session if available)
    session_to_use="$LAST_SESSION_ID"
  elif [[ "$SESSION_CONTINUATION" -eq 1 ]]; then
    # New session every run
    session_to_use=""
    session_run=0
  else
    # Continue for N runs, then start fresh
    if [[ "$session_run" -ge "$SESSION_CONTINUATION" ]]; then
      session_to_use=""
      session_run=0
    else
      session_to_use="$LAST_SESSION_ID"
    fi
  fi

  session_run=$((session_run + 1))

  # Run Cline
  run_cline "$session_to_use" "$run_count"
  local_exit=$?

  # Dry run: exit after one iteration
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf "\n${C_BOLD}${C_GREEN}[dry-run] Done. Exiting.${C_RESET}\n"
    break
  fi

  # Check if we should stop
  if [[ "$MAX_LOOPS" -gt 0 && "$run_count" -ge "$MAX_LOOPS" ]]; then
    printf "\n${C_BOLD}${C_GREEN}Completed %d run(s). Stopping.${C_RESET}\n" "$run_count"
    break
  fi

  # Sleep before next run
  printf "\n${C_DIM}  sleeping %ss before next run...${C_RESET}\n" "$INTERVAL"
  sleep "$INTERVAL"
done

# --- Cleanup ---
cleanup
