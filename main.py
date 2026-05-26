import json
from pathlib import Path

from tokenizer.utils import find_pairs, merge_pair

MIN_PAIR_FREQUENCY = 5
MAX_TOKEN_LENGTH = 20

# -----------------------------------------------------------------------------
# BPE training loop
# -----------------------------------------------------------------------------
corpus_path = Path("corpus-for-tokenizer-7KChar.txt")
corpus_string = corpus_path.read_text(encoding="utf-8")
tokens = list(corpus_string)


print(f"Tokens: {len(tokens):,}")
target_vocab_size = 3000
print(f"Target vocabulary size: {target_vocab_size}")
vocabulary: set[str] = set()

step = 0

# While have not reached the target vocabulary size
while len(vocabulary) < target_vocab_size:

    # Count all adjacent token pairs in the current corpus
    pairs = find_pairs(tokens)
    # Safety check
    if not pairs:
        print("\nNo more pairs found.")
        break

    # -------------------------------------------------------------------------
    # Find the best pair to merge
    #
    # We do NOT blindly take the most frequent pair.
    #
    # Why?
    #
    # Because the most frequent pair could create a giant token like:
    #
    # "is an unincorporated community and census-designated place"
    #
    # We therefore iterate through pairs ordered by frequency and pick the
    # first pair that satisfies our constraints.
    # -------------------------------------------------------------------------

    best_pair = None
    best_count = 0

    for pair, pair_count in pairs.most_common():

        # Example:
        #
        # pair = ("th", "e")
        #
        # merged_token = "the"

        merged_token = pair[0] + pair[1]

        # Stop searching if frequencies become too small.
        #
        # pairs.most_common() is sorted from highest frequency
        # to lowest frequency.
        #
        # Therefore if we reach a pair occurring less than
        # MIN_PAIR_FREQUENCY times, all remaining pairs will
        # also be below the threshold.

        if pair_count < MIN_PAIR_FREQUENCY:
            break

        # Skip tokens that would become too large.

        if len(merged_token) > MAX_TOKEN_LENGTH:
            continue

        # We found a valid candidate.

        best_pair = pair
        best_count = pair_count

        break

    # No valid pair found.
    #
    # Either:
    # - all remaining pairs are too rare
    # - all remaining pairs create tokens that are too long

    if best_pair is None:
        print("\nNo valid pair found.")
        break

    # -------------------------------------------------------------------------
    # Create the new token
    # -------------------------------------------------------------------------

    # Suming two
    merged_token = best_pair[0] + best_pair[1]

    # Add the new token to the vocabulary.
    #
    # Example:
    #
    # "th" + "e" -> "the"

    vocabulary.add(merged_token)

    # Replace every occurrence of the pair in the current corpus.
    #
    # Example:
    #
    # before:
    # ["th", "e", " ", "c", "a", "t"]
    #
    # after:
    # ["the", " ", "c", "a", "t"]

    tokens = merge_pair(tokens, best_pair)

    step += 1

    print(f"\rRound {step} - (max is {target_vocab_size})", end="", flush=True)


# -----------------------------------------------------------------------------
# End of training
# -----------------------------------------------------------------------------

if step > 0:
    print()

print("\nTraining finished")
print(f"Vocabulary size: {len(vocabulary):,}")

vocabulary_by_id = {
    token: token_id for token_id, token in enumerate(sorted(vocabulary))
}

with open("vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocabulary_by_id, f, ensure_ascii=False, indent=2)
