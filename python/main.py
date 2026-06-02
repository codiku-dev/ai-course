from pathlib import Path

from tokenizer import Tokenizer

MIN_PAIR_FREQUENCY = 5
MAX_TOKEN_LENGTH = 20

corpus_path = Path("corpus-for-tokenizer-7KChar.txt")
target_vocab_size = 3000

bpe = Tokenizer(
    target_vocab_size=target_vocab_size,
    min_pair_frequency=MIN_PAIR_FREQUENCY,
    max_token_length=MAX_TOKEN_LENGTH,
)

bpe.load_corpus(corpus_path)

print(f"Tokens: {len(bpe.tokens):,}")
print(f"Target vocabulary size: {target_vocab_size}")

vocabulary = bpe.generate_vocabulary()

print("\nTraining finished")
print(f"Vocabulary size: {len(vocabulary):,}")

bpe.save()
exampleTokenReconstructed = bpe.reconstruct_tokens_from_merges(
    "theold", bpe.merge_pairs
)
print(exampleTokenReconstructed)
