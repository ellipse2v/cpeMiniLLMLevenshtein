# Technical Documentation

This document provides a technical overview of the CPE MiniLM Levenshtein project.

## CPE Matcher (`cpe_matcher.py`)

The `cpe_matcher.py` script is the core component of this project. It is designed to find the most relevant Common Platform Enumeration (CPE) names for a given software product, vendor, and version.

### Features

-   **Sentence Transformer Model**: Utilizes a pre-trained sentence transformer model (e.g., `all-MiniLM-L6-v2` or `all-mpnet-base-v2`) to generate semantic embeddings for CPE data and user queries.
-   **Levenshtein Distance**: Combines semantic similarity with Levenshtein distance calculations on vendor, product, and version strings for more accurate matching.
-   **Weighted Scoring**: A combined score is calculated using configurable weights for semantic similarity and individual Levenshtein scores.
-   **Product-level index**: CPE names are grouped by `part:vendor:product` (~145 k products for ~1.5 M names). One embedding is computed per product; versions are resolved afterwards.
-   **Version substitution**: a version that NVD does not list is substituted into the product's CPE, which keeps the CPE valid (e.g. Internet Explorer 2.0).
-   **Data Caching**: the CPE list is cached in `cpe_data.json.gz` (+ SHA-256); product embeddings in `cpe_product_embeddings_<model>.npy` together with their texts, so that only new products are encoded after an NVD update. A newer `official-cpe-dictionary_v2.3.xml` triggers a rebuild of the cache.
-   **Force Regeneration**: The `--force-regenerate` command-line argument recomputes all product embeddings.
-   **Modes of Operation**:
    -   **Interactive Mode**: Allows users to enter vendor, product, and version information interactively.
    -   **Excel Mode**: Processes an Excel file containing lists of software to find matching CPEs.

### How it Works

1.  **Configuration**: Loads settings from `config.ini`, including model names, file paths, and scoring weights.
2.  **Model Loading**: Loads the specified sentence transformer model, with a fallback mechanism to try different models or download them if they are not available locally.
3.  **Data Preparation**:
    -   If cached data exists and `--force-regenerate` is not specified, it loads the pre-processed CPE items and embeddings from disk.
    -   Otherwise, it parses the CPE dictionary XML file (`official-cpe-dictionary_v2.3.xml`).
    -   For each CPE entry, it creates a descriptive text string.
    -   It then uses the sentence transformer model to generate a high-dimensional vector embedding for each CPE's descriptive text.
    -   The CPE items and their embeddings are saved to disk for future use.
4.  **Matching**:
    -   The user provides a vendor, product, and version (either interactively or from an Excel file; Excel rows are processed in batches).
    -   The `vendor product` query is embedded and compared with every product embedding; the `SEMANTIC_TOP_K` closest products are kept, plus products whose name matches exactly (ignoring case and separators, with or without the vendor name in front of the product).
    -   Candidates are scored with a weighted sum of semantic similarity and Levenshtein vendor/product similarity; exact product-name matches are ranked first.
    -   For the best products the version is resolved: the dictionary CPE if the version exists (`4` = `4.0`), otherwise the product's CPE with the requested version substituted. The version sub-score is 1.0 for a known version and 0.5–1.0 (closeness to the nearest known version) otherwise.
5.  **Output**: The script displays the top matching CPEs with their scores and details.

### Usage

**Interactive Mode:**

```bash
python3 src/cpe_matcher/cpe_matcher.py
```

**Excel Processing Mode:**

```bash
python3 src/cpe_matcher/cpe_matcher.py -data /path/to/your/file.xlsx
```

**Force Data Regeneration:**

```bash
python3 src/cpe_matcher/cpe_matcher.py --force-regenerate
```
