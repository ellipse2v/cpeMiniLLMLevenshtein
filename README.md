# cpeMiniLLMLevenshtein
search CPE code into the nvd dictionary

# CPE Matcher with MiniLM and Levenshtein

This tool helps identify CPE (Common Platform Enumeration) codes for software products based on vendor, product, and version information. It uses semantic similarity via MiniLM embeddings combined with Levenshtein distance to find the most relevant CPE matches.

## Overview

CPE Matcher uses a lightweight approach to search through the official CPE dictionary. Key features include:

- **MiniLM Embeddings**: Uses the smaller `all-MiniLM-L6-v2` transformer model to generate 384-dimensional embeddings, offering a good balance between performance and resource usage
- **Levenshtein Distance**: Incorporates string similarity metrics to improve matching accuracy
- **GPU Acceleration**: Takes advantage of CUDA when available
- **Efficient Processing**: Processes the CPE dictionary in batches to manage memory usage

## Requirements

- Python 3.6+
- PyTorch
- sentence-transformers
- lxml
- numpy
- scikit-learn
- python-Levenshtein
- tqdm

## Installation

```bash
pip install torch sentence-transformers lxml numpy scikit-learn python-Levenshtein tqdm
```

## Usage

1. Download the official CPE dictionary XML file from NIST
2. Place the XML file in the same directory as the script (or update the file path in the Config class)
3. Run the script:

```bash
python cpe_matcher.py
```

On first run, the tool will:
1. Parse the CPE dictionary
2. Generate embeddings for all CPE entries
3. Save the data for faster loading in future runs

After this preparation step, you can search for CPE codes by entering vendor, product, and version information.

## How It Works

1. **Data Loading**: The tool first loads the official CPE dictionary XML file and extracts all CPE entries along with their titles.

2. **Text Processing**: For each CPE entry, it extracts the vendor, product, and version components, normalizes them, and combines them with the title to create a text representation.

3. **Embedding Generation**: The tool uses MiniLM to generate semantic embeddings for all CPE entries.

4. **Search Process**:
   - When a user searches for a product, the tool creates an embedding for the search query
   - It computes the cosine similarity between the query embedding and all CPE embeddings
   - It computes Levenshtein similarity for the vendor, product, and version components
   - A combined score is calculated using weighted averages of these similarity metrics
   - Results are sorted and the top matches are returned

5. **Score Weighting**:
   - Semantic similarity (40%)
   - Product name similarity (30%)
   - Vendor name similarity (20%)
   - Version similarity (10%)

## Performance Improvements

Compared to the original BERT-based implementation, this version offers:

1. **Reduced Memory Usage**: MiniLM (384 dimensions) vs BERT (768 dimensions)
2. **Faster Processing**: Smaller model with more efficient tokenization
3. **Improved String Matching**: Direct Levenshtein distance calculation for better text similarity

## Future Improvements

- Add support for more advanced filtering options
- Implement export functionality for results
- Create a web interface

## License
Licensed under the Apache License, Version 2.0 (the "License")

This project is provided for educational and research purposes.

## Acknowledgments

- NIST for providing the CPE dictionary
- The Sentence-Transformers team for the MiniLM model
