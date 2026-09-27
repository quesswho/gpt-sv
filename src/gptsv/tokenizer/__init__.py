"""Swedish tokenizer training and evaluation."""

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

# ChatML turn delimiters. `gptsv.tokenizer.reserve` adds these, followed by
# `<|reserved_N|>` placeholders, in the IDs of the tokenizer's rarest merges.
CHAT_TOKENS = ["<|im_start|>", "<|im_end|>"]
