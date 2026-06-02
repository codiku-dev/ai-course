from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tokenizer.utils import find_pairs as count_pairs
from tokenizer.utils import merge_pair as apply_merge


class Tokenizer:
    def __init__(
        self,
        target_vocab_size: int = 3000,
        min_pair_frequency: int = 5,
        max_token_length: int = 20,
    ):
        self.target_vocab_size = target_vocab_size
        self.min_pair_frequency = min_pair_frequency
        self.max_token_length = max_token_length

        self.tokens: list[str] = []
        self.vocabulary: set[str] = set()
        self.merge_pairs: list[tuple[str, str]] = []

    def load_corpus(self, path: str | Path) -> None:
        corpus_path = Path(path)
        corpus_string = corpus_path.read_text(encoding="utf-8")
        self.tokens = list(corpus_string)

    def find_pairs(self) -> Counter:
        return count_pairs(self.tokens)

    def merge_pair(self, pair_to_merge: tuple[str, str]) -> list[str]:
        return apply_merge(self.tokens, pair_to_merge)

    def generate_vocabulary(self) -> dict[str, int]:
        step = 0

        while len(self.vocabulary) < self.target_vocab_size:
            pairs = self.find_pairs()

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

                if pair_count < self.min_pair_frequency:
                    break

                # Skip tokens that would become too large.

                if len(merged_token) > self.max_token_length:
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
            self.merge_pairs.append(best_pair)
            self.vocabulary.add(merged_token)

            # Replace every occurrence of the pair in the current corpus.
            #
            # Example:
            #
            # before:
            # ["th", "e", " ", "c", "a", "t"]
            #
            # after:
            # ["the", " ", "c", "a", "t"]

            self.tokens = self.merge_pair(best_pair)

            step += 1

            print(
                f"\rRound {step} - (max is {self.target_vocab_size})",
                end="",
                flush=True,
            )

        if step > 0:
            print()

        return {
            token: token_id for token_id, token in enumerate(sorted(self.vocabulary))
        }

    def save(
        self,
        vocabulary_path: str | Path = "vocabulary.json",
        merges_path: str | Path = "merges.json",
    ) -> None:
        vocabulary_by_id = {
            token: token_id for token_id, token in enumerate(sorted(self.vocabulary))
        }

        with open(vocabulary_path, "w", encoding="utf-8") as f:
            json.dump(vocabulary_by_id, f, ensure_ascii=False, indent=2)

        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(self.merge_pairs, f, ensure_ascii=False, indent=2)

    def reconstruct_tokens_from_merges(
        self,
        word: str,
        merge_ranks: dict[tuple[str, str], int],
    ) -> list[str]:

        # Start from characters
        #
        # "the"
        #
        # ->
        #
        # ["t", "h", "e"]

        tokens = list(word)

        while True:

            best_pair = None
            best_rank = float("inf")

            # ---------------------------------------------------------------------
            # Find best merge candidate
            # ---------------------------------------------------------------------

            for i in range(len(tokens) - 1):

                pair = (tokens[i], tokens[i + 1])

                if pair in merge_ranks:

                    rank = merge_ranks[pair]

                    # Smaller rank = earlier merge learned during training
                    #
                    # Earlier merges have higher priority.

                    if rank < best_rank:

                        best_rank = rank
                        best_pair = pair

            # No more merges possible

            if best_pair is None:
                break

            # ---------------------------------------------------------------------
            # Apply merge
            # ---------------------------------------------------------------------

            new_tokens = []

            i = 0

            while i < len(tokens):

                # Merge the selected pair
                #
                # Example:
                #
                # ["t", "h", "e"]
                #
                # best_pair = ("t", "h")
                #
                # ->
                #
                # ["th", "e"]

                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:

                    merged_token = tokens[i] + tokens[i + 1]

                    new_tokens.append(merged_token)

                    i += 2

                else:

                    new_tokens.append(tokens[i])

                    i += 1

            tokens = new_tokens

        return tokens
