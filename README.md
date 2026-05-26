# CPE Matcher — MiniLLM + Levenshtein

Match software names and versions to NVD CPE identifiers using a hybrid semantic + lexical scoring pipeline.

## Features

- **Hybrid scoring**: semantic cosine similarity (sentence-transformers) + Levenshtein vendor/product matching + numeric version comparison
- **Auto-FAISS ANN search**: builds a FAISS index automatically when processing ≥ `FAISS_MIN_ROWS` rows; falls back to brute-force cosine for small inputs
- **Memory-mapped embeddings**: `mmap_mode='r'` loads the `.npy` file on demand — avoids loading 2–4 GB into RAM at startup
- **Secure JSON cache**: `cpe_data.json.gz` + SHA-256 integrity file replaces the old pickle cache
- **Score breakdown**: every result shows `semantic`, `vendor`, `product`, `version` sub-scores
- **Configurable threshold**: `MIN_SCORE_THRESHOLD` filters low-confidence matches
- **Parallel Excel processing**: `MAX_WORKERS` threads for batch mode
- **Benchmark timers**: model load, JSON load, FAISS build, per-query embedding and scoring times

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
│   │   ├── cpe_matcher.py          # Main matcher (CPEMatcher class)
│   │   ├── config.ini              # All tunable parameters
│   │   └── test_cpe_matcher.py     # Unit tests
│   └── generate_cpe_dictionary/
│       ├── generate_cpe_dictionary.py
│       └── test_generate_cpe_dictionary.py
├── cpe_data.json.gz                # CPE metadata cache (gzip + JSON)
├── cpe_data.json.gz.sha256         # Integrity check for the cache
├── cpe_embeddings_minilm.npy       # Sentence-transformer embeddings
├── official-cpe-dictionary_v2.3.xml
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

### Force regeneration of cache and embeddings

```bash
python cpe_matcher.py --force-regenerate
```

### Generate / update the CPE dictionary XML

```bash
# Fetch all CPEs from NVD (default)
python src/generate_cpe_dictionary/generate_cpe_dictionary.py

# Fetch only the first N CPEs (for testing)
python src/generate_cpe_dictionary/generate_cpe_dictionary.py --limit 5000
```

## Configuration (`config.ini`)

```ini
[Models]
FALLBACK_MODEL = sentence-transformers/all-mpnet-base-v2
DEFAULT_MODEL  = sentence-transformers/all-MiniLM-L6-v2

[Paths]
FALLBACK_MODEL_PATH    = models/all-mpnet-base-v2
DEFAULT_MODEL_PATH     = models/all-MiniLM-L6-v2
CPE_DATA_JSON          = cpe_data.json.gz
CPE_EMBEDDINGS_NUMPY   = cpe_embeddings_minilm.npy
CPE_DICTIONARY_XML     = official-cpe-dictionary_v2.3.xml

[Settings]
BATCH_SIZE             = 128     # Embedding batch size
NUM_RESULTS            = 5       # Candidates returned per query
FORCE_REGENERATE       = false
SEMANTIC_SCORE_WEIGHT  = 0.5
VENDOR_SCORE_WEIGHT    = 0.2
PRODUCT_SCORE_WEIGHT   = 0.2
VERSION_SCORE_WEIGHT   = 0.1
MIN_SCORE_THRESHOLD    = 0.7     # Discard results below this score
MAX_WORKERS            = 1       # Parallel threads for Excel batch
FAISS_MIN_ROWS         = 100     # Build FAISS index above this row count
```

### Key parameters

| Parameter | Effect |
|---|---|
| `MIN_SCORE_THRESHOLD` | Filter low-confidence matches (0.0–1.0) |
| `FAISS_MIN_ROWS` | Row count above which FAISS is built; brute-force below |
| `MAX_WORKERS` | Threads for batch Excel mode; keep low on limited RAM |
| `SEMANTIC_SCORE_WEIGHT` | Weight of embedding cosine similarity in final score |

## How It Works

1. **Parse CPE XML** — extract 1.5 M+ CPE URIs and human-readable titles
2. **Encode titles** with sentence-transformers — `.npy` embeddings saved once
3. **At query time** — encode the query string; cosine similarity against all embeddings (or FAISS ANN when `FAISS_MIN_ROWS` is reached)
4. **Re-rank** top candidates using Levenshtein vendor/product similarity and numeric version comparison
5. **Return** results above `MIN_SCORE_THRESHOLD` with full score breakdown

### Score breakdown example

```
cpe:2.3:o:microsoft:windows_11:22000:...   score=0.912
  semantic=0.87  vendor=1.00  product=1.00  version=0.95
```

## Model Comparison

| Model | Embedding dim | `.npy` size | Startup (mmap) | Quality |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (default) | 384 | ~2.2 GB | ~11 s | Good |
| all-mpnet-base-v2 (fallback) | 768 | ~4.4 GB | ~20 s | Better |

Switch model via `DEFAULT_MODEL` and `CPE_EMBEDDINGS_NUMPY` in `config.ini`. Run with `--force-regenerate` once after switching.

## Tests

```bash
# Fast unit tests (no model, no data files required)
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestModuleImport -v
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestJSONCache -v
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestVersionSimilarity -v
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestFAISSSearch -v

# Full end-to-end test (requires model + data files)
python -m pytest src/cpe_matcher/test_cpe_matcher.py::TestCPEMatcher -v

# All tests
python -m pytest src/cpe_matcher/test_cpe_matcher.py -v

# Dictionary generator tests
python -m pytest src/generate_cpe_dictionary/test_generate_cpe_dictionary.py -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
