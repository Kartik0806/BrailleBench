#!/usr/bin/env bash
# Real-image-matched synthetic data: same buckets / text mix / shuffling as
# generate_balanced_shuffled.sh, but rendered with braille_gen_fineweb_v2.py in
# real-matching mode, so scale, tightness, cell/word spacing, dot size and
# lines-per-image follow the distributions measured on real braille photos
# (configs/realmatch_profile_v1.json, built by src/realstats/build_profile.py).
#
# Differences from generate_balanced_shuffled.sh (everything else is unchanged):
#   * GENERATOR   src/braille_gen_fineweb_v2.py  (original generator is untouched)
#   * --real_profile configs/realmatch_profile_v1.json
#   * random-resolution is off (scale is set per sample from the profile)
#   * SPLIT=val generates a real-matched, UNSHUFFLED validation set (the old script
#     symlinked data_balanced/val, which does not exist next to data_new_balanced)
#
# Useful variables (all optional):
#   SPLIT=train|val  TOTAL  START_INDEX  SEED_BASE  DATA_ROOT  OUTPUT_DIR  WORKERS  PYTHON
#   REAL_PROFILE        profile JSON (default configs/realmatch_profile_v1.json;
#                       configs/realmatch_profile_v1_noeval.json leaves the test pool out)
#   REAL_POOL_WEIGHTS   e.g. "eval_combined=0.4,new_data=0.4,book_pages=0.2"
#   REAL_OPTS           space-separated key=value overrides of REAL_DEFAULTS, e.g. "p_explore=0.2 max_lines=6"
#   G1_SHORT_MAX_CHARS / G1_MID_MAX_CHARS / G2_SHORT_MAX_CHARS / G2_LONG_MAX_CHARS
#                       character caps per bucket (defaults 120/180/120/450 = the original script).
#                       Real eval crops are short (1-3 word median 8 chars, 4+ words median 31);
#                       FineWeb windows are ~2.5x longer, which makes the images wider than real
#                       ones and so under-scaled after the model's resize. 16/48/16/120 brought the
#                       1-3 word model-input dot pitch to 21.5 px vs 23.0 for the eval (24/60/24/120: 18.2,
#                       original caps: 17.3); see src/realstats/README.md.
#   G1_SHORT_PCT / G1_MID_PCT / G2_SHORT_PCT / G2_LONG_PCT
#                       bucket shares in percent, must sum to 100 (defaults 40/10/20/30 = original)
#   G1_SHORT_WORDS / G1_MID_WORDS / G2_SHORT_WORDS / G2_LONG_WORDS
#                       word ranges "min-max" (defaults 1-2 / 6-10 / 1-3 / 6-50 = original; nothing
#                       produces 4-5 words with these)
#   DATASET_CONFIG      FineWeb config/subset (default sample-10BT = original), e.g. CC-MAIN-2013-20
#   STRIP_PUNCT_TOKENS=1  drop punctuation-only texts and leading/trailing punctuation-only tokens
#                       (e.g. "- and" -> "and"); default 0 = original texts
set -euo pipefail

SPLIT="${SPLIT:-train}"
if [[ "$SPLIT" != train && "$SPLIT" != val ]]; then
  echo "SPLIT must be train or val" >&2
  exit 2
fi

if [[ -z "${TOTAL:-}" ]]; then
  if [[ "$SPLIT" == train ]]; then TOTAL=100000; else TOTAL=2000; fi
fi
if ! [[ "$TOTAL" =~ ^[0-9]+$ ]] || (( TOTAL < 100 || TOTAL % 100 != 0 )); then
  echo "TOTAL must be a positive multiple of 100" >&2
  exit 2
fi

# START_INDEX: first sample index of this run. 0 = brand-new dataset; N > 0 appends
# to an existing OUTPUT_DIR that already holds exactly N samples (0..N-1).
START_INDEX="${START_INDEX:-0}"
if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "START_INDEX must be a non-negative integer" >&2
  exit 2
fi

# Seeds must differ from earlier runs or the same FineWeb text windows repeat,
# so the default is shifted by START_INDEX. (Different from the legacy defaults so a
# realmatch dataset never reuses the legacy text windows by accident.)
if [[ -z "${SEED_BASE:-}" ]]; then
  if [[ "$SPLIT" == train ]]; then SEED_BASE=$((2337 + START_INDEX)); else SEED_BASE=$((8337 + START_INDEX)); fi
fi

DATA_ROOT="${DATA_ROOT:-data_realmatch}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/$SPLIT}"
GENERATOR="${GENERATOR:-src/braille_gen_fineweb_v2.py}"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-32}"
BACKGROUND_DIR="${BACKGROUND_DIR:-background}"
REAL_PROFILE="${REAL_PROFILE:-configs/realmatch_profile_v1.json}"
REAL_POOL_WEIGHTS="${REAL_POOL_WEIGHTS:-}"
REAL_OPTS="${REAL_OPTS:-}"
if [[ "$SPLIT" == train ]]; then
  P_SHUFFLE_WORDS="${P_SHUFFLE_WORDS:-0.10}"
  P_SHUFFLE_CHARS="${P_SHUFFLE_CHARS:-0.15}"
else
  # validation stays clean (no shuffling), like the legacy script's val set
  P_SHUFFLE_WORDS="${P_SHUFFLE_WORDS:-0}"
  P_SHUFFLE_CHARS="${P_SHUFFLE_CHARS:-0}"
fi
SHUFFLE_CHAR_WORD_FRACTION="${SHUFFLE_CHAR_WORD_FRACTION:-0.25}"
G1_SHORT_MAX_CHARS="${G1_SHORT_MAX_CHARS:-120}"
G1_MID_MAX_CHARS="${G1_MID_MAX_CHARS:-180}"
G2_SHORT_MAX_CHARS="${G2_SHORT_MAX_CHARS:-120}"
G2_LONG_MAX_CHARS="${G2_LONG_MAX_CHARS:-450}"
G1_SHORT_PCT="${G1_SHORT_PCT:-40}"; G1_MID_PCT="${G1_MID_PCT:-10}"
G2_SHORT_PCT="${G2_SHORT_PCT:-20}"; G2_LONG_PCT="${G2_LONG_PCT:-30}"
G1_SHORT_WORDS="${G1_SHORT_WORDS:-1-2}"; G1_MID_WORDS="${G1_MID_WORDS:-6-10}"
G2_SHORT_WORDS="${G2_SHORT_WORDS:-1-3}"; G2_LONG_WORDS="${G2_LONG_WORDS:-6-50}"
STRIP_PUNCT_TOKENS="${STRIP_PUNCT_TOKENS:-0}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
if (( G1_SHORT_PCT + G1_MID_PCT + G2_SHORT_PCT + G2_LONG_PCT != 100 )); then
  echo "Bucket shares must sum to 100 (got $G1_SHORT_PCT+$G1_MID_PCT+$G2_SHORT_PCT+$G2_LONG_PCT)" >&2
  exit 2
fi
for r in "$G1_SHORT_WORDS" "$G1_MID_WORDS" "$G2_SHORT_WORDS" "$G2_LONG_WORDS"; do
  if ! [[ "$r" =~ ^[0-9]+-[0-9]+$ ]] || (( ${r%-*} < 1 || ${r%-*} > ${r#*-} )); then
    echo "Word range must look like MIN-MAX with 1 <= MIN <= MAX (got '$r')" >&2
    exit 2
  fi
done

if (( START_INDEX == 0 )); then
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists: $OUTPUT_DIR. Set a new DATA_ROOT/OUTPUT_DIR, or START_INDEX=<existing sample count> to append." >&2
    exit 1
  fi
else
  existing=0
  [[ -s "$OUTPUT_DIR/labels/labels.jsonl" ]] && existing=$(wc -l < "$OUTPUT_DIR/labels/labels.jsonl")
  if (( existing != START_INDEX )); then
    echo "START_INDEX=$START_INDEX but $OUTPUT_DIR has $existing labels; they must match to append contiguously." >&2
    exit 1
  fi
  echo "Appending to $OUTPUT_DIR: indices $START_INDEX..$((START_INDEX + TOTAL - 1)), SEED_BASE=$SEED_BASE"
fi
if [[ ! -f "$GENERATOR" ]]; then
  echo "Generator not found: $GENERATOR" >&2
  exit 1
fi
if [[ ! -s "$REAL_PROFILE" ]]; then
  echo "Real-style profile not found: $REAL_PROFILE (build it with src/realstats/build_profile.py)" >&2
  exit 1
fi

real_args=(--real_profile "$REAL_PROFILE")
[[ -n "$REAL_POOL_WEIGHTS" ]] && real_args+=(--real_pool_weights "$REAL_POOL_WEIGHTS")
for kv in $REAL_OPTS; do real_args+=(--real_opt "$kv"); done
[[ "$STRIP_PUNCT_TOKENS" == 1 ]] && real_args+=(--strip_punct_tokens)

base=$((TOTAL / 100))
grade1_short=$((G1_SHORT_PCT * base))
grade1_mid=$((G1_MID_PCT * base))
grade2_short=$((G2_SHORT_PCT * base))
grade2_long=$((G2_LONG_PCT * base))

generate_bucket() {
  local start="$1" count="$2" grade="$3" min_words="$4" max_words="$5" max_chars="$6" seed="$7"
  echo "Generating $count Grade $grade samples, $min_words-$max_words words, indices $start-$((start + count - 1))"
  "$PYTHON" "$GENERATOR" \
    --output_dir "$OUTPUT_DIR" \
    --background_dir "$BACKGROUND_DIR" \
    --dataset HuggingFaceFW/fineweb \
    --dataset_config "$DATASET_CONFIG" \
    --n_per_text 1 \
    --num_samples "$count" \
    --start_index "$start" \
    --seed "$seed" \
    --n_workers "$WORKERS" \
    --p_grade2 "$grade" \
    --min_text_words "$min_words" \
    --max_text_words "$max_words" \
    --min_text_chars 1 \
    --max_text_chars "$max_chars" \
    --bg_prob 0.30 \
    --p_perspective 0.40 \
    --p_page_curl 0.10 \
    --p_elastic 0.15 \
    --p_skew 0.20 \
    --max_skew_deg 8 \
    --p_vertical_flip 0 \
    --p_rotation 0 \
    --p_random_resolution 0 \
    --p_random_boundary 0.05 \
    --p_paper_texture 0.70 \
    --p_stain 0.10 \
    --p_lighting 0.60 \
    --p_low_light 0.15 \
    --p_shadow_band 0.15 \
    --p_noise 0.40 \
    --p_salt_pepper 0.10 \
    --p_shuffle_words "$P_SHUFFLE_WORDS" \
    --p_shuffle_chars "$P_SHUFFLE_CHARS" \
    --shuffle_char_word_fraction "$SHUFFLE_CHAR_WORD_FRACTION" \
    "${real_args[@]}"

  local expected=$((start + count)) actual
  actual=$(wc -l < "$OUTPUT_DIR/labels/labels.jsonl")
  if (( actual != expected )); then
    echo "Expected $expected labels after this bucket; found $actual. Stop and inspect generation errors." >&2
    exit 1
  fi
}

B="$START_INDEX"
generate_bucket "$B" "$grade1_short" 0 "${G1_SHORT_WORDS%-*}" "${G1_SHORT_WORDS#*-}" "$G1_SHORT_MAX_CHARS" "$SEED_BASE"
generate_bucket "$((B + grade1_short))" "$grade1_mid" 0 "${G1_MID_WORDS%-*}" "${G1_MID_WORDS#*-}" "$G1_MID_MAX_CHARS" "$((SEED_BASE + 1))"
generate_bucket "$((B + grade1_short + grade1_mid))" "$grade2_short" 1 "${G2_SHORT_WORDS%-*}" "${G2_SHORT_WORDS#*-}" "$G2_SHORT_MAX_CHARS" "$((SEED_BASE + 2))"
generate_bucket "$((B + grade1_short + grade1_mid + grade2_short))" "$grade2_long" 1 "${G2_LONG_WORDS%-*}" "${G2_LONG_WORDS#*-}" "$G2_LONG_MAX_CHARS" "$((SEED_BASE + 3))"

echo "Completed $TOTAL samples (indices $START_INDEX..$((START_INDEX + TOTAL - 1))) in $OUTPUT_DIR"
