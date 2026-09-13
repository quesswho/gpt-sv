#!/usr/bin/env bash
# Pull the sv64k-v2 shards onto a training host. Run after the code is synced.
#
# Kept out of any provider's machine-boot hook on purpose: those run before this
# repo exists on the host, and the download needs HF_TOKEN for a private repo.
#
#   HF_TOKEN=hf_... HF_DATASET=<user>/gptsv-sv64k-v2 scripts/fetch_shards_hf.sh
#
# Downloads land where configs/phase1_430m.toml expects them
# (data/shards/sv64k-v2/{train,val}). Re-running resumes a partial download.
#
set -euo pipefail
cd "$(dirname "$0")/.."
HF=${HF:-$([ -x .venv/bin/hf ] && echo .venv/bin/hf || echo hf)}
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
SHARDS=${SHARDS:-data/shards/sv64k-v2}

if [ -z "${HF_DATASET:-}" ]; then
    echo "set HF_DATASET=<hf-user>/<repo-name>" >&2
    exit 1
fi

mkdir -p "$SHARDS"
"$HF" download "$HF_DATASET" --repo-type dataset --local-dir "$SHARDS"

# The loader trusts meta.json's n_tokens; a truncated download would otherwise
# surface as a confusing memmap error thousands of steps in.
"$PYTHON" - "$SHARDS" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
for split in ("train", "val"):
    meta = json.loads((root / split / "meta.json").read_text())
    have = sum(f.stat().st_size for f in (root / split).glob("shard_*.bin")) // 2
    want = meta["n_tokens"]
    status = "ok" if have == want else "MISMATCH"
    print(f"{split}: {have:,} / {want:,} tokens  {status}")
    if have != want:
        sys.exit(1)
PY
