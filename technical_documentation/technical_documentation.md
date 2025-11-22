# Technical Documentation

This document provides a technical overview of the CPE MiniLM Levenshtein project.

## CPE Matcher (`cpe_matcher.py`)

The `cpe_matcher.py` script is the core component of this project. It is designed to find the most relevant Common Platform Enumeration (CPE) names for a given software product, vendor, and version.

### Features

-   **Sentence Transformer Model**: Utilizes a pre-trained sentence transformer model (e.g., `all-MiniLM-L6-v2` or `all-mpnet-base-v2`) to generate semantic embeddings for CPE data and user queries.
-   **Levenshtein Distance**: Combines semantic similarity with Levenshtein distance calculations on vendor, product, and version strings for more accurate matching.
-   **Weighted Scoring**: A combined score is calculated using configurable weights for semantic similarity and individual Levenshtein scores.
-   **Data Caching**: To improve performance, the script pre-processes the official CPE dictionary XML file, generates embeddings, and saves them to local files (`cpe_data.pkl` and `cpe_embeddings.npy`).
-   **Force Regeneration**: The `--force-regenerate` command-line argument allows the user to force the script to delete and regenerate the cached data and embeddings.
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
    -   The user provides a vendor, product, and version (either interactively or from an Excel file).
    -   The script generates an embedding for the user's query.
    -   It calculates the cosine similarity between the query embedding and all CPE embeddings to find the most semantically similar candidates.
    -   The results are then sorted to prioritize exact matches on the product name, with the weighted score serving as a secondary sorting criterion.
    -   It then refines these results by calculating the Levenshtein similarity for the vendor, product, and version fields.
    -   A final, weighted score is computed to rank the results.
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
