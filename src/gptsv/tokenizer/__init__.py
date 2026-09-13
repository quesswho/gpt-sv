"""Swedish-first tokenizer training and evaluation."""

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

# ChatML turn delimiters. Tokenizers derived with `gptsv.tokenizer.reserve`
# carry these, then `<|reserved_N|>` placeholders, in the IDs of the base
# tokenizer's rarest merges.
CHAT_TOKENS = ["<|im_start|>", "<|im_end|>"]
