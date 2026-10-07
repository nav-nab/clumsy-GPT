"""Settings that are not exposed in the GUI, plus GUI defaults and slider ranges."""

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Generation.
MAX_NEW_TOKENS = 150
SAMPLING_TEMPERATURE = 0.7

# A word-start token counts as a whole word if its Zipf frequency (from the
# wordfreq package) is at least MIN_ZIPF. Zipf 3 means about one in a million words.
# Short strings are more often fragments or abbreviations, so they need a higher
# frequency, and strings under three letters are never counted as words.
MIN_ZIPF = 2.5
MIN_ZIPF_SHORT = {3: 3.5, 4: 3.0}  # keyed by word length
MIN_WORD_LENGTH = 3

# GUI sliders: (default, minimum, maximum).
CLUMSINESS = (0.1, 0.0, 0.3)
LOOSENESS = (0.05, 0.01, 0.3)
TOP_K = (1000, 10, 5000)
SLIP_TYPE = (0.0, 0.0, 1.0)
