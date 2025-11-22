# cpeMiniLLMLevenshtein
Search CPE codes within the NVD dictionary.

# CPE Matcher with Sentence-Transformers and Levenshtein

This tool helps identify CPE (Common Platform Enumeration) codes for software products based on vendor, product, and version information. It uses semantic similarity via Sentence-Transformers embeddings combined with Levenshtein distance to find the most relevant CPE matches.

## Overview

CPE Matcher uses a robust approach to search through the official CPE dictionary. Key features include:

- **Sentence-Transformers Embeddings**: Uses the `all-mpnet-base-v2` transformer model by default to generate 768-dimensional embeddings, offering a strong balance between performance and accuracy. The model is configurable via `config.ini`.
- **Levenshtein Distance**: Incorporates string similarity metrics to improve matching accuracy.
- **GPU Acceleration**: Takes advantage of CUDA when available.
- **Efficient Processing**: Processes the CPE dictionary in batches to manage memory usage.
- **Multiple Operation Modes**: Interactive command-line interface and batch processing via Excel.
- **Configurable**: Key parameters like model choice, scoring weights, and file paths are now configurable via `config.ini` files for each script.

## Requirements

- Python 3.6+
- PyTorch
- sentence-transformers
- lxml
- numpy
- scikit-learn
- python-Levenshtein
- tqdm
- pandas (for Excel processing)
- requests (for generating the CPE dictionary)

## Installation

```bash
pip install torch sentence-transformers lxml numpy scikit-learn python-Levenshtein tqdm pandas requests
```

## Project Structure

The project now follows a more organized structure:

```
.
├── src/
│   ├── cpe_matcher/
│   │   ├── cpe_matcher.py
│   │   └── config.ini
│   └── generate_cpe_dictionary/
│       ├── generate_cpe_dictionary.py
│       └── config.ini
├── models/
│   └── all-MiniLM-L6-v2/  (or all-mpnet-base-v2, depending on download)
├── cpe_dictionary.csv
├── official-cpe-dictionary_v2.3.xml
├── LICENSE
├── README.md
└── template.xlsx
```

## Generating the CPE Dictionary

This script fetches all CPEs from the NVD API and generates an XML dictionary file.

1.  **Get an NVD API Key**:
    *   Visit the [NVD API key request page](https://nvd.nist.gov/developers/request-an-api-key) and follow the instructions to get your free API key.

2.  **Configure the API Key**:
    *   Open the file: `src/generate_cpe_dictionary/config.ini`
    *   Replace `YOUR_API_KEY_HERE` with the key you obtained.
    ```ini
    [NVD]
    API_KEY = YOUR_ACTUAL_API_KEY
    ```
    *   You can also configure proxy settings and output filenames in this `config.ini`.

3.  **Run the Generation Script**:
    *   Execute the following command in your terminal from the project root:
    ```bash
    python3 src/generate_cpe_dictionary/generate_cpe_dictionary.py
    ```
    *   To also generate a CSV file with vendor, product, version, and CPE data, use the `--output-csv` flag:
    ```bash
    python3 src/generate_cpe_dictionary/generate_cpe_dictionary.py --output-csv
    ```
    *   The script will fetch all CPE entries from the NVD API and create the `official-cpe-dictionary_v2.3.xml` file (and `cpe_dictionary.csv` if `--output-csv` is used) in the project root. This may take several minutes as it needs to download a large amount of data while respecting API rate limits.

## Usage

After generating the CPE dictionary, you can use the `cpe_matcher.py` script.

1.  **Run the CPE Matcher**:
    *   Open your terminal in the project root and run:
    ```bash
    python3 src/cpe_matcher/cpe_matcher.py
    ```

    **First Run Considerations:**
    *   **Model Download:** On the very first run, the script will attempt to download the `all-mpnet-base-v2` model (approx. 420MB) from Hugging Face and save it locally in the `models/all-mpnet-base-v2` directory. This requires an active internet connection.
        *   The script includes an **internet connectivity check** before attempting to download. If no internet is detected, it will inform you and exit.
    *   **Embedding Generation:** After the model is loaded, the script will parse the CPE dictionary and generate embeddings for all CPE entries. This is a one-time process and can take several minutes, depending on your hardware. The generated embeddings and parsed CPE data will be saved as `.pkl` and `.npy` files (e.g., `cpe_data.pkl`, `cpe_embeddings.npy`) in the project root for faster loading in future runs.

    **Subsequent Runs:**
    *   The script will load the locally saved model and embeddings, significantly reducing startup time.

2.  **Interactive Mode**:
    *   After initialization, the tool will prompt you to enter vendor, product, and version information interactively.

3.  **Excel Batch Processing Mode**:
    *   To process multiple entries at once using an Excel file, use the `-data` argument:
    ```bash
    python3 src/cpe_matcher/cpe_matcher.py -data your_data.xlsx
    ```
    *   Ensure your Excel file (`your_data.xlsx`) has the following columns: `Vendor`, `Product`, `Version`, `CPE`, `Levenshtein score`. The tool will fill in the `CPE` and `Levenshtein score` columns for matches with a combined score > 0.7. A new Excel file with "_updated" appended to the filename will be created.

## Configuration

You can customize the behavior of `cpe_matcher.py` by editing `src/cpe_matcher/config.ini`:

```ini
[Models]
DEFAULT_MODEL = sentence-transformers/all-mpnet-base-v2
FALLBACK_MODEL = sentence-transformers/all-MiniLM-L6-v2

[Paths]
DEFAULT_MODEL_PATH = models/all-mpnet-base-v2
FALLBACK_MODEL_PATH = models/all-MiniLM-L6-v2
CPE_DATA_PICKLE = cpe_data.pkl
CPE_EMBEDDINGS_NUMPY = cpe_embeddings.npy
CPE_DICTIONARY_XML = official-cpe-dictionary_v2.3.xml

[Settings]
BATCH_SIZE = 128
NUM_RESULTS = 5
FORCE_REGENERATE = false
SEMANTIC_SCORE_WEIGHT = 0.5
VENDOR_SCORE_WEIGHT = 0.2
PRODUCT_SCORE_WEIGHT = 0.2
VERSION_SCORE_WEIGHT = 0.1
```
*   **`DEFAULT_MODEL`**: The primary model to use.
*   **`FALLBACK_MODEL`**: Used if the default model cannot be loaded or downloaded.
*   **`*_PATH`**: Local paths where models are stored.
*   **`CPE_DATA_PICKLE`**, **`CPE_EMBEDDINGS_NUMPY`**, **`CPE_DICTIONARY_XML`**: File paths for cached data.
*   **`BATCH_SIZE`**: Number of items processed at once for embeddings.
*   **`NUM_RESULTS`**: Number of top CPE matches to return.
*   **`FORCE_REGENERATE`**: Set to `true` to force re-generation of embeddings even if cached files exist.
*   **`*_SCORE_WEIGHT`**: Adjust these values to change the influence of each scoring component on the combined score.

## How It Works

1.  **Data Loading**: The tool first loads the official CPE dictionary XML file and extracts all CPE entries along with their titles.
2.  **Text Processing**: For each CPE entry, it extracts the vendor, product, and version components, normalizes them, and combines them with the title to create a text representation.
3.  **Embedding Generation**: The tool uses the configured Sentence-Transformer model (e.g., `all-mpnet-base-v2`) to generate semantic embeddings for all CPE entries.
4.  **Search Process**:
    *   When searching for a product, the tool creates an embedding for the search query.
    *   It computes the cosine similarity between the query embedding and all CPE embeddings.
    *   It computes Levenshtein similarity for the vendor, product, and version components.
    *   A combined score is calculated using weighted averages of these similarity metrics (weights are configurable in `config.ini`).
    *   Results are sorted and the top matches are returned.

## Performance Improvements

Compared to the original MiniLM-based implementation, this version offers:

1.  **Improved Accuracy**: `all-mpnet-base-v2` generally provides better semantic understanding than `all-MiniLM-L6-v2`.
2.  **Configurability**: Easier to switch models and fine-tune scoring weights.
3.  **Robustness**: Internet connection check before model download.

## Future Improvements

- Add support for more advanced filtering options
- Implement export functionality for various formats
- Create a web interface
- Add multi-threading support for batch processing

## License
Licensed under the Apache License, Version 2.0 (the "License")

This project is provided for educational and research purposes.

## Acknowledgments

- NIST for providing the CPE dictionary
- The Sentence-Transformers team for the `all-mpnet-base-v2` and `all-MiniLM-L6-v2` models