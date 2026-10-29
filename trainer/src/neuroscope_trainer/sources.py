"""Where the data and tokenizer come from, pinned to exact Hugging Face commits so every learner trains on the
same bytes."""

HF = "https://huggingface.co"

# Pretraining text: the "wiki" slice of OLMo-mix-1124, English Wikipedia and Wikibooks (ODC-BY; the articles are CC BY-SA).
WIKI_REPO = "allenai/olmo-mix-1124"
WIKI_REVISION = "99ee6aaace88779d1ef099d36251b91101c1679b"
WIKI_FILE = "data/wiki/wiki-0000.json.gz"
WIKI_URL = f"{HF}/datasets/{WIKI_REPO}/resolve/{WIKI_REVISION}/{WIKI_FILE}"

# Instruction tuning: a small, filtered subset of the Tulu 3 SFT mixture (ODC-BY; each source keeps its license).
TULU_REPO = "allenai/tulu-3-sft-mixture"
TULU_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
TULU_FILES = [f"data/train-{i:05d}-of-00006.parquet" for i in range(6)]

# Tokenizer: OLMo 2's (the base model's tokenizer.json; the Instruct model's tokenizer_config.json carries the chat
# template). Both Apache 2.0. The Instruct revision is the one the app downloads.
TOKENIZER_REPO = "allenai/OLMo-2-0425-1B"
TOKENIZER_REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
INSTRUCT_REPO = "allenai/OLMo-2-0425-1B-Instruct"
INSTRUCT_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"


def model_file_url(repo: str, revision: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{revision}/{path}"
