from collections import Counter


def find_pairs(tokens: list[str]) -> Counter:

    pairs = Counter()
    for i in range(len(tokens) - 1):

        # We create a pair [ token1, token2 ]
        pair = (tokens[i], tokens[i + 1])
        # We increment the count for this pair
        ## Contratrly to js where the index would have to be a string here
        # it can be an object ( in this case a tuple)
        pairs[pair] += 1

    return pairs


def merge_pair(tokens: list[str], pair_to_merge: tuple[str, str]) -> list[str]:
    new_tokens = []
    i = 0

    while i < len(tokens):
        # Check if the current token and the next token
        if (
            i < len(tokens) - 1
            and tokens[i] == pair_to_merge[0]
            and tokens[i + 1] == pair_to_merge[1]
        ):
            # Merge the two tokens
            new_tokens.append(pair_to_merge[0] + pair_to_merge[1])

            i += 2

        else:
            # Keep the token unchanged
            new_tokens.append(tokens[i])
            # Move to the next token
            i += 1
    return new_tokens
