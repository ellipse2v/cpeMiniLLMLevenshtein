# CPE Matcher — MiniLLM + Levenshtein

Match software names and versions to NVD CPE identifiers using a hybrid semantic + lexical scoring pipeline.

## Features

- **Product-level index**: the NVD dictionary has ~1.8 M CPE names but only ~150 k distinct `part:vendor:product` triples. One embedding is computed per product (10× fewer vectors, ~220 MB instead of ~2.2 GB for MiniLM)
- **Version substitution**: once the product is identified, the requested version is looked up in the dictionary. If NVD does not list it, a valid CPE is built from the product with the version substituted (e.g. Internet Explorer `2.0` → `cpe:2.3:a:microsoft:internet_explorer:2.0:*:*:*:*:*:*:*`). Results say whether the version was found (`version_in_dictionary`) and which dictionary entry was used (`reference_cpe`)
- **Hybrid scoring**: semantic cosine similarity (sentence-transformers) + Levenshtein vendor/product matching (company suffixes such as *Inc.*, *Corporation* ignored, vendor name stripped from the product: *Microsoft SQL Server* → *sql server*) + numeric version comparison
- **Batch search**: Excel rows are encoded and compared in batches (one matrix product per chunk, FAISS when installed) instead of one query at a time
- **Incremental embeddings**: after an NVD update only the new products are encoded
- **Automatic refresh**: when `official-cpe-dictionary_v2.3.xml` (produced by `generate_cpe_dictionary.py`) is newer than `cpe_data.json.gz`, the cache is rebuilt from it; deprecated CPEs are skipped
- **Secure JSON cache**: `cpe_data.json.gz` + SHA-256 integrity file
- **Score breakdown**: every result shows `semantic`, `vendor`, `product`, `version` sub-scores
- **Before/after benchmark**: `benchmark/` holds a labelled test set and a runner to measure accuracy and speed

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install sentence-transformers faiss-cpu scikit-learn numpy pandas tqdm lxml openpyxl python-Levenshtein
```

## Project Structure

```
cpeMiniLLMLevenshtein/
├── src/
│   ├── cpe_matcher/
│   │   ├── cpe_matcher.py          # Main matcher (CPEMatcher, ProductIndex)
│   │   ├── config.ini              # Optional, overrides the built-in defaults
│   │   └── test_cpe_matcher.py     # Unit tests
│   └── generate_cpe_dictionary/
│       ├── generate_cpe_dictionary.py
│       └── test_generate_cpe_dictionary.py
├── benchmark/
│   ├── cpe_benchmark_cases.csv     # Labelled queries (exact, substituted, variants, typos, negatives)
│   └── run_benchmark.py            # Accuracy + timing, before/after comparison
├── cpe_data.json.gz                # CPE metadata cache (gzip + JSON), versioned — see "CPE data"
├── cpe_data.json.gz.sha256         # Integrity check for the cache
├── cpe_product_embeddings_<model>.npy        # One embedding per product (generated)
├── cpe_product_embeddings_<model>.texts.json.gz  # Texts of those embeddings (incremental reuse)
├── official-cpe-dictionary_v2.3.xml          # Produced by generate_cpe_dictionary.py
├── requirements.txt
└── README.md
```

## Cache Migration (old pickle to JSON)

If you have an existing `cpe_data.pkl`, convert it once with Python:

```bash
python - <<'MIGRATE'
import json, gzip, hashlib, importlib
pkl = importlib.import_module("pickle")
with open("cpe_data.pkl", "rb") as fh:
    data = pkl.load(fh)
payload = json.dumps(data).encode("utf-8")
with gzip.open("cpe_data.json.gz", "wb") as fh:
    fh.write(payload)
digest = hashlib.sha256(payload).hexdigest()
with open("cpe_data.json.gz.sha256", "w") as fh:
    fh.write(digest)
print("Migration complete:", len(data.get("cpe_items", [])), "CPEs")
MIGRATE
```

After migration the `.pkl` file is no longer needed.

## Usage

### Interactive mode

```bash
cd src/cpe_matcher
python cpe_matcher.py
```

Enter queries like `microsoft windows_11 22000`.

### Batch Excel processing

```bash
python cpe_matcher.py --input path/to/software_list.xlsx --output results.xlsx
```

### Force regeneration of the embeddings

```bash
python cpe_matcher.py --force-regenerate
```

### Update the NVD CPE dictionary

```bash
# 1. Fetch all CPEs from the NVD API (needs an API key in src/generate_cpe_dictionary/config.ini)
python src/generate_cpe_dictionary/generate_cpe_dictionary.py

# 2. Run the matcher: the XML being newer than cpe_data.json.gz, the cache is
#    rebuilt from it and only the new products are embedded
python src/cpe_matcher/cpe_matcher.py
```

`--limit 5000` fetches only the first N CPEs (for testing).

Commit `cpe_data.json.gz` and `cpe_data.json.gz.sha256` together (the hash must match or the cache is rejected). Never commit `src/generate_cpe_dictionary/config.ini` once it holds your API key.

### CPE data

`cpe_data.json.gz` is versioned so that a fresh clone works without the XML (~900 MB) or an NVD API key; the product embeddings are computed on first run (~30 s on GPU).

| | |
|---|---|
| Snapshot | NVD API, 2026-09-27 |
| CPEs fetched | 1,846,965 |
| Deprecated (excluded) | 102,108 |
| Active CPEs in cache | 1,744,857 |
| Distinct products | 150,645 |

## Configuration (`config.ini`)

`src/cpe_matcher/config.ini` is optional: every key has a built-in default.

```ini
[Models]
DEFAULT_MODEL  = sentence-transformers/all-MiniLM-L6-v2
FALLBACK_MODEL = sentence-transformers/all-mpnet-base-v2

[Paths]
DEFAULT_MODEL_PATH     = models/all-MiniLM-L6-v2
FALLBACK_MODEL_PATH    = models/all-mpnet-base-v2
CPE_DATA_JSON          = cpe_data.json.gz
CPE_DICTIONARY_XML     = official-cpe-dictionary_v2.3.xml
EMBEDDINGS_DIR         = .       # where cpe_product_embeddings_<model>.npy is stored

[Settings]
BATCH_SIZE             = 256     # Embedding batch size
NUM_RESULTS            = 5       # Products returned per query
FORCE_REGENERATE       = false
SEMANTIC_SCORE_WEIGHT  = 0.5
VENDOR_SCORE_WEIGHT    = 0.2
PRODUCT_SCORE_WEIGHT   = 0.2
VERSION_SCORE_WEIGHT   = 0.1
MIN_SCORE_THRESHOLD    = 0.6     # Excel mode: discard results below this score
SEMANTIC_TOP_K         = 50      # Products kept from the semantic search before re-ranking
```

Keys of older config files (`CPE_EMBEDDINGS_NUMPY`, `MAX_WORKERS`, `FAISS_MIN_ROWS`) are ignored.

### Key parameters

| Parameter | Effect |
|---|---|
| `MIN_SCORE_THRESHOLD` | Filter low-confidence matches (0.0–1.0) |
| `SEMANTIC_TOP_K` | More candidates = better recall, slightly slower |
| `SEMANTIC_SCORE_WEIGHT` | Weight of embedding cosine similarity in final score |
| `VERSION_SCORE_WEIGHT` | 1.0 when the version is in NVD, 0.5–1.0 by closeness to the nearest known version otherwise |

## How It Works

1. **Load** `cpe_data.json.gz` (or parse the XML dictionary, skipping deprecated entries)
2. **Group** the ~1.7 M active CPE names by `part:vendor:product` (~150 k products); each product gets a version-less text (`vendor product title`) and a map of its known versions
3. **Encode** one embedding per product — saved once, only new products are encoded after an update
4. **At query time** — encode `vendor product` (batched in Excel mode), take the `SEMANTIC_TOP_K` closest products plus the products whose name matches exactly (modulo case/separators, with and without the vendor prefix)
5. **Re-rank** candidates with Levenshtein vendor/product similarity; exact product-name matches first
6. **Resolve the version**: dictionary CPE when the version is known (`4` and `4.0` are equivalent; the least specific variant is preferred), otherwise the product CPE with the requested version substituted

### Score breakdown example (illustrative values)

```
cpe:2.3:a:microsoft:internet_explorer:2.0:*:*:*:*:*:*:*  [version substituted]
  Score: 0.93  (semantic=0.95, vendor=1.000, product=1.000, version=0.800)
  Based on: cpe:2.3:a:microsoft:internet_explorer:3.0:*:*:*:*:*:*:*
```

## Model Comparison

| Model | Embedding dim | `.npy` size (products) | Quality |
|---|---|---|---|
| all-MiniLM-L6-v2 (default) | 384 | ~220 MB | Good |
| all-mpnet-base-v2 (fallback) | 768 | ~440 MB | Better |

Switch model via `DEFAULT_MODEL` in `config.ini` (or `--use-mini-llm` to prefer the fallback). Each model has its own embeddings file, generated on first use.

## Benchmark (before / after)

`benchmark/cpe_benchmark_cases.csv` contains ~110 labelled queries checked against the NVD data:
`exact` (version listed in NVD), `substituted` (version not listed), `variant` (inventory-style names such as *Mozilla Foundation / Mozilla Firefox*), `variant_substituted`, `typo`, and `negative` (invented software that must not match).

A case is `correct` when the CPE the Excel mode would write (top result above `MIN_SCORE_THRESHOLD`, version applied) has the expected `part:vendor:product` and the requested version.

```bash
# Old version (c8b795b = last commit before the product-level matcher)
git stash -u                       # if you have local changes
git fetch origin main              # make sure c8b795b is available locally (git fetch --unshallow for shallow clones)
git checkout c8b795b -- src/cpe_matcher/cpe_matcher.py
python benchmark/run_benchmark.py --label before

# New version
git checkout HEAD -- src/cpe_matcher/cpe_matcher.py
python benchmark/run_benchmark.py --label after

python benchmark/run_benchmark.py --compare benchmark/results_before.json benchmark/results_after.json
```

### Results (2026-09-27, all-MiniLM-L6-v2 on CUDA, same NVD dictionary of 1,736,681 CPEs for both versions)

| Category | Correct before → after |
|---|---|
| exact | 100.0 → 100.0 |
| substituted | 100.0 → 100.0 |
| variant | 88.0 → 92.0 |
| variant_substituted | 60.0 → 86.7 |
| typo | 100.0 → 100.0 |
| negative | 100.0 → 100.0 |
| **All (113)** | **92.0 → 96.5** |

- Right product ranked first: 96.4 % → 98.2 %; in top 5: 98.2 % → 100 %. 5 cases fixed, 0 regressed (threshold 0.7).
- Batch: 8.4 → 225 rows/s (~27×); single query ~1.8 s → ~40 ms.
- `MIN_SCORE_THRESHOLD` lowered to 0.6: invented software scores ≤ 0.56, correct products ≥ 0.64.

The comparison prints accuracy per category, timings (data load, mean query latency, batch throughput over `--repeat` × the cases) and the list of cases fixed/regressed. `--fake-model` runs the new version offline with a hashing encoder (sanity check only).

## Tests

```bash
# Fast unit tests (no model, no data files required)
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestModuleImport -v
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestJSONCache -v
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestVersionSimilarity -v
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestProductIndex -v
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestCPEMatcherOffline -v   # fake encoder, tiny dictionary

# Full end-to-end test (requires model + data files)
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestCPEMatcher -v

# All tests
python -m pytest src/cpe_matcher/test_cpe_matcher.py -v

# Dictionary generator tests
python -m pytest src/generate_cpe_dictionary/test_generate_cpe_dictionary.py -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
