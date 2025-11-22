# Copyright 2025 ellipse2v
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import lxml.etree as ET
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import Levenshtein
import time
import argparse
import configparser
import pandas as pd
import socket

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check for internet connection by trying to connect to a well-known host.
    Host: Google's primary DNS server.
    Port: DNS port.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(f"Internet connection check failed: {ex}")
        return False

def load_config():
    """Loads configuration from config.ini and resolves paths."""
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config.ini')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config.read(config_path)
    
    # Resolve paths relative to the project root (which is two levels up from the script)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    cfg = {
        'default_model': config.get('Models', 'DEFAULT_MODEL'),
        'fallback_model': config.get('Models', 'FALLBACK_MODEL'),
        'default_model_path': os.path.join(project_root, config.get('Paths', 'DEFAULT_MODEL_PATH')),
        'fallback_model_path': os.path.join(project_root, config.get('Paths', 'FALLBACK_MODEL_PATH')),
        'pickle_filepath': os.path.join(project_root, config.get('Paths', 'CPE_DATA_PICKLE')),
        'embeddings_filepath': os.path.join(project_root, config.get('Paths', 'CPE_EMBEDDINGS_NUMPY')),
        'xml_filepath': os.path.join(project_root, config.get('Paths', 'CPE_DICTIONARY_XML')),
        'batch_size': config.getint('Settings', 'BATCH_SIZE'),
        'num_results': config.getint('Settings', 'NUM_RESULTS'),
        'force_regenerate': config.getboolean('Settings', 'FORCE_REGENERATE'),
        'semantic_weight': config.getfloat('Settings', 'SEMANTIC_SCORE_WEIGHT'),
        'vendor_weight': config.getfloat('Settings', 'VENDOR_SCORE_WEIGHT'),
        'product_weight': config.getfloat('Settings', 'PRODUCT_SCORE_WEIGHT'),
        'version_weight': config.getfloat('Settings', 'VERSION_SCORE_WEIGHT'),
    }
    return cfg

config = load_config()

def load_model_with_fallback(device):
    """
    Loads a sentence transformer model with a fallback mechanism.
    1. Tries to load the default model from the local path.
    2. If not found, tries to download the default model from Hugging Face.
    3. If download fails or no internet, tries to load the fallback model from the local path.
    4. If all else fails, exits the program.
    """
    default_path = config['default_model_path']
    default_name = config['default_model']
    fallback_path = config['fallback_model_path']
    fallback_name = config['fallback_model']

    # 1. Try to load the default model locally
    if os.path.exists(default_path):
        print(f"Loading default model from local path: {default_path}")
        try:
            return SentenceTransformer(default_path, device=device)
        except Exception as e:
            print(f"Error loading default model from {default_path}: {e}")
            # Proceed to download or fallback

    # 2. If not local, try to download it
    print(f"Default model not found at {default_path}. Attempting to download...")
    if check_internet_connection():
        try:
            print(f"Downloading default model: {default_name}...")
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            model = SentenceTransformer(default_name, device=device)
            print(f"Saving model to {default_path} for future offline use...")
            model.save(default_path)
            print(f"Model saved successfully to {default_path}")
            return model
        except Exception as e:
            print(f"Failed to download default model '{default_name}': {e}")
    else:
        print("No internet connection. Cannot download default model.")

    # 3. If download fails or no internet, try to load fallback model locally
    print(f"Trying to load fallback model '{fallback_name}' from local path: {fallback_path}")
    if os.path.exists(fallback_path):
        try:
            return SentenceTransformer(fallback_path, device=device)
        except Exception as e:
            print(f"Error loading fallback model from {fallback_path}: {e}")
    
    # 4. If all else fails, error out
    print("\nFATAL: Could not load any model.")
    print("Neither the default model nor the fallback model could be loaded.")
    print("Please connect to the internet to allow the script to download a model,")
    print(f"or manually place a model in '{default_path}' or '{fallback_path}'.")
    exit(1)


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to check GPU usage
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        return True
    return False

# Load the model using the new robust logic
model = load_model_with_fallback(device)
print(f"Model loaded successfully and moved to {device}")
check_gpu_usage()

def parse_cpe_name(cpe_string):
    """
    Extract vendor, product, and version from a CPE string.
    CPE 2.3 format: cpe:2.3:part:vendor:product:version:update:edition:language:sw_edition:target_sw:target_hw:other
    """
    parts = cpe_string.split(':')
    if len(parts) >= 5:
        vendor = parts[3] if parts[3] != '*' else ""
        product = parts[4] if parts[4] != '*' else ""
        version = parts[5] if len(parts) > 5 and parts[5] != '*' else ""
        return vendor, product, version
    return "", "", ""

def load_cpe_xml(filepath):
    """Load the CPE XML file and return the root of the XML tree."""
    abs_path = os.path.abspath(filepath)
    print(f"Attempting to load CPE dictionary from: {abs_path}")
    if not os.path.exists(abs_path):
        print(f"Error: File does not exist at the specified path: {abs_path}")
        return None
    try:
        tree = ET.parse(filepath)
        return tree.getroot()
    except Exception as e:
        print(f"Error loading or parsing XML file: {e}")
        return None

def extract_cpe_items(root):
    """Extract CPE items from the XML root."""
    namespace = {
        'cpe': 'http://cpe.mitre.org/dictionary/2.0',
        'cpe-23': 'http://scap.nist.gov/schema/cpe-extension/2.3'
    }
    
    cpe_items = []
    titles = []
    
    for cpe_item in root.findall(".//cpe:cpe-item", namespaces=namespace):
        cpe23_item = cpe_item.find(".//cpe-23:cpe23-item", namespaces=namespace)
        if cpe23_item is not None and cpe23_item.get("name"):
            cpe_name = cpe23_item.get("name")
            
            # Extract title if it exists
            title_element = cpe_item.find(".//cpe:title", namespaces=namespace)
            title = title_element.text if title_element is not None else ""
            
            cpe_items.append(cpe_name)
            titles.append(title)
    
    return cpe_items, titles

def clean_text(text):
    """Clean text for processing."""
    if not text:
        return ""
    # Replace special characters with spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    return text.lower()

def create_embeddings(texts, model, device, batch_size=128):
    """Create embeddings for a list of texts."""
    embedding_dim = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
    
    # Check GPU usage before processing
    print("GPU status before embedding generation:")
    check_gpu_usage()
    
    # Process in batches to save memory
    with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Force synchronization to free GPU memory
            if i > 0 and i % (batch_size * 10) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Progress: {i}/{len(texts)} texts processed")
                check_gpu_usage()
            
            try:
                # Generate embeddings with SentenceTransformer
                batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
                
                # Store in the main numpy array
                end_idx = min(i + batch_size, len(texts))
                embeddings[i:end_idx] = batch_embeddings
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Continue with the next batch
            
            # Update progress bar
            pbar.update(len(batch_texts))
    
    # Check GPU usage after processing
    print("GPU status after embedding generation:")
    check_gpu_usage()
    
    return embeddings

def levenshtein_similarity(str1, str2):
    """Calculate similarity based on Levenshtein distance."""
    if not str1 or not str2:
        return 0.0
    
    distance = Levenshtein.distance(str1.lower(), str2.lower())
    max_len = max(len(str1), len(str2))
    
    if max_len == 0:
        return 1.0  # Two empty strings are identical
    
    # Convert distance to similarity (0 to 1 where 1 is identical)
    return 1.0 - (distance / max_len)

def prepare_cpe_data(xml_filepath, model, device):
    """Prepare CPE data, generate embeddings, and create a product map."""
    root = load_cpe_xml(xml_filepath)
    if root is None:
        return None, None, None, None

    print("Extracting CPE items...")
    cpe_items, titles = extract_cpe_items(root)
    print(f"Number of extracted CPE items: {len(cpe_items)}")

    # Create a map from normalized product name to list of indices
    product_map = {}
    print("Creating product to index map...")
    for i, cpe_string in enumerate(tqdm(cpe_items, desc="Mapping products")):
        _, product, _ = parse_cpe_name(cpe_string)
        if product:
            normalized_product = product.lower().replace('_', ' ')
            if normalized_product not in product_map:
                product_map[normalized_product] = []
            product_map[normalized_product].append(i)

    # Prepare texts for embeddings
    print("Preparing texts for embeddings...")
    texts = []
    for cpe, title in zip(cpe_items, titles):
        vendor, product, version = parse_cpe_name(cpe)
        vendor_clean = clean_text(vendor.replace('_', ' '))
        product_clean = clean_text(product.replace('_', ' '))
        version_clean = clean_text(version.replace('_', ' '))
        
        text = f"{vendor_clean} {product_clean}"
        if version_clean:
            text += f" {version_clean}"
        if title:
            text += f" {clean_text(title)}"
        texts.append(text.strip())

    print("Generating embeddings...")
    embeddings = create_embeddings(texts, model, device, batch_size=config['batch_size'])
    
    return cpe_items, titles, embeddings, product_map

def save_cpe_data(cpe_items, titles, embeddings, product_map, pickle_filepath, embeddings_filepath):
    """Save CPE data, embeddings, and product map."""
    try:
        print(f"Saving CPE data to {pickle_filepath}...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump({'cpe_items': cpe_items, 'titles': titles, 'product_map': product_map}, f)
        
        print(f"Saving embeddings to {embeddings_filepath}...")
        np.save(embeddings_filepath, embeddings)
        
        print("Save completed successfully.")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_cpe_data(pickle_filepath, embeddings_filepath):
    """Load CPE data, embeddings, and product map."""
    try:
        print(f"Loading data from {pickle_filepath}...")
        with open(pickle_filepath, 'rb') as f:
            data = pickle.load(f)
        
        cpe_items = data.get('cpe_items', [])
        titles = data.get('titles', [])
        product_map = data.get('product_map', {})
        
        print(f"Loading embeddings from {embeddings_filepath}...")
        embeddings = np.load(embeddings_filepath)
        
        print(f"CPE data loaded: {len(cpe_items)} items")
        print(f"Embeddings loaded: {embeddings.shape}")
        if not product_map:
            print("Warning: Product map not found or is empty in pickle file.")

        return cpe_items, titles, embeddings, product_map
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def find_closest_cpes(vendor, product, version, cpe_items, titles, embeddings, model, device, product_map, num_results=5):
    """Find the closest CPE codes using a hybrid search approach."""
    # Build the query
    query = ""
    if vendor:
        query += clean_text(vendor) + " "
    if product:
        query += clean_text(product) + " "
    if version:
        query += clean_text(version)
    query = query.strip()
    
    if not query:
        return []

    print("Generating embedding for the query...")
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    print("Computing similarities...")
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # --- Hybrid Candidate Selection ---
    # 1. Get indices of exact product matches from the pre-computed map
    exact_match_indices = set()
    if product:
        normalized_query_product = product.lower().replace('_', ' ')
        exact_match_indices = set(product_map.get(normalized_query_product, []))

    # 2. Get top semantic matches
    top_semantic_indices = set(similarities.argsort()[-num_results*20:][::-1])

    # 3. Combine them to form the candidate pool
    combined_indices = exact_match_indices.union(top_semantic_indices)
    
    print(f"Preparing results from a pool of {len(combined_indices)} candidates...")
    results = []
    for idx in combined_indices:
        cpe = cpe_items[idx]
        title = titles[idx] if idx < len(titles) else ""
        cpe_vendor, cpe_product, cpe_version = parse_cpe_name(cpe)

        is_exact_product_match = product.lower().replace('_', ' ') == cpe_product.lower().replace('_', ' ') if product and cpe_product else False
        
        vendor_score = levenshtein_similarity(vendor, cpe_vendor.replace('_', ' ')) if vendor and cpe_vendor else 0.5
        product_score = levenshtein_similarity(product, cpe_product.replace('_', ' ')) if product and cpe_product else 0.0
        version_score = levenshtein_similarity(version, cpe_version.replace('_', ' ')) if version and cpe_version else 0.5
        
        semantic_score = similarities[idx]
        combined_score = (semantic_score * config['semantic_weight']) + \
                         (vendor_score * config['vendor_weight']) + \
                         (product_score * config['product_weight']) + \
                         (version_score * config['version_weight'])
        
        results.append((combined_score, cpe, title, is_exact_product_match))
    
    results.sort(key=lambda x: (x[3], x[0]), reverse=True)
    
    # Deduplicate results to show only one architecture per CPE
    final_results = deduplicate_results(results)
    
    return [(score, cpe, title) for score, cpe, title, _ in final_results[:num_results]]

def get_canonical_cpe(cpe_string):
    """
    Creates a canonical representation of a CPE string by taking only the first 10 parts
    (up to sw_edition), effectively ignoring target_sw, target_hw, and other.
    """
    parts = cpe_string.split(':')
    # Take up to the 10th part (index 9), which is sw_edition
    canonical_parts = parts[:10]
    return ':'.join(canonical_parts)

def deduplicate_results(results):
    """
    Deduplicates a list of CPE results, keeping only the highest-scoring one
    for each unique CPE (ignoring architecture).
    """
    deduplicated = []
    seen_canonical_cpes = set()
    
    for result in results:
        _, cpe, _, _ = result
        canonical_cpe = get_canonical_cpe(cpe)
        
        if canonical_cpe not in seen_canonical_cpes:
            deduplicated.append(result)
            seen_canonical_cpes.add(canonical_cpe)
            
    return deduplicated

def adjust_cpe_version(cpe_string, source_version):
    """
    Adjusts the version of a CPE string based on the source version from the input file.
    - If source_version is provided, it replaces the version in the CPE string.
    - If source_version is empty, the version in the CPE string is replaced with a wildcard '*'.
    """
    parts = cpe_string.split(':')
    
    if len(parts) > 5:
        if source_version and str(source_version).strip():
            parts[5] = str(source_version).strip().lower().replace(' ', '_')
        else:
            parts[5] = '*'
            
    return ':'.join(parts)

def process_excel_file(excel_path, cpe_items, titles, embeddings, model, device, product_map):
    """Process an Excel file with vendor, product, and version data to find CPE codes."""
    try:
        print(f"Loading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        required_columns = ['Vendor', 'Product', 'Version', 'CPE', 'Levenshtein score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            return False
        
        total_rows = len(df)
        print(f"Processing {total_rows} entries...")
        
        for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing Excel data"):
            vendor = str(row['Vendor']) if not pd.isna(row['Vendor']) else ""
            product = str(row['Product']) if not pd.isna(row['Product']) else ""
            source_version = str(row['Version']) if not pd.isna(row['Version']) else ""
            
            if not product:
                continue
            
            results = find_closest_cpes(vendor, product, source_version, cpe_items, titles, embeddings, model, device, product_map, num_results=1)
            
            if results:
                score, original_cpe, _ = results[0]
                if score > 0.7:
                    adjusted_cpe = adjust_cpe_version(original_cpe, source_version)
                    df['CPE'] = df['CPE'].astype(str)
                    df.at[index, 'CPE'] = adjusted_cpe
                    df.at[index, 'Levenshtein score'] = score
        
        output_path = excel_path.replace('.xlsx', '_updated.xlsx')
        if excel_path.endswith('.xls'):
            output_path = excel_path.replace('.xls', '_updated.xlsx')
        
        print(f"Saving updated Excel file to: {output_path}")
        df.to_excel(output_path, index=False)
        
        return True
    
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return False

def interactive_mode(cpe_items, titles, embeddings, model, device, product_map):
    """Run the program in interactive mode."""
    while True:
        print("\n=== Search for CPE codes ===")
        vendor = input("Enter vendor name (leave empty if unknown, 'q' to quit): ")
        if vendor.lower() == 'q': break
            
        product = input("Enter product name ('q' to quit): ")
        if product.lower() == 'q': break
        if not product:
            print("Product name is required for search.")
            continue
            
        version = input("Enter product version: ")
        
        print("\nSearching...")
        search_start = time.time()
        results = find_closest_cpes(vendor, product, version, cpe_items, titles, embeddings, model, device, product_map)
        search_time = time.time() - search_start
        
        if results:
            print(f"\nCPE codes found for '{vendor} {product} {version}' (in {search_time:.2f} seconds):")
            for score, cpe, title in results:
                vendor_part, product_part, version_part = parse_cpe_name(cpe)
                print(f"- {cpe}")
                print(f"  Score: {score:.4f}")
                print(f"  Vendor: {vendor_part.replace('_', ' ')}")
                print(f"  Product: {product_part.replace('_', ' ')}")
                print(f"  Version: {version_part.replace('_', ' ')}")
                if title:
                    print(f"  Description: {title}")
                print()
        else:
            print(f"No CPE codes found for '{vendor} {product} {version}'.")

def main(model):
    parser = argparse.ArgumentParser(description="CPE Matcher with MiniLM and Levenshtein")
    parser.add_argument("-data", help="Path to Excel file with vendor, product, and version data", type=str, default=None)
    parser.add_argument("--force-regenerate", help="Force regeneration of CPE data and embeddings", action="store_true")
    args = parser.parse_args()

    print("\n=== CPE Matcher with MiniLM and Levenshtein (Auto-download version) ===")
    
    start_time = time.time()
    
    cpe_items, titles, embeddings, product_map = None, None, None, None
    
    # Decide whether to load or regenerate data
    should_regenerate = args.force_regenerate or not os.path.exists(config['pickle_filepath']) or not os.path.exists(config['embeddings_filepath'])

    if not should_regenerate:
        print("Loading existing data...")
        cpe_items, titles, embeddings, product_map = load_cpe_data(config['pickle_filepath'], config['embeddings_filepath'])
        # If product_map is missing, we must regenerate
        if not product_map:
            print("Product map is missing from cached data. Forcing regeneration.")
            should_regenerate = True

    if should_regenerate:
        if args.force_regenerate:
            print("Forcing regeneration of CPE data and embeddings...")
        else:
            print("No complete existing data found. Preparing new CPE data...")
        cpe_items, titles, embeddings, product_map = prepare_cpe_data(config['xml_filepath'], model, device)
        if cpe_items is not None:
            save_cpe_data(cpe_items, titles, embeddings, product_map, config['pickle_filepath'], config['embeddings_filepath'])

    processing_time = time.time() - start_time
    print(f"Data preparation/loading time: {processing_time:.2f} seconds")
    
    if not cpe_items or embeddings is None or product_map is None:
        print("Cannot continue without valid CPE data and product map.")
        return
    
    if args.data:
        excel_path = args.data
        if not os.path.exists(excel_path):
            print(f"Error: The specified Excel file does not exist: {excel_path}")
            return
        
        print(f"Running in Excel processing mode with file: {excel_path}")
        process_excel_file(excel_path, cpe_items, titles, embeddings, model, device, product_map)
    else:
        print("Running in interactive mode.")
        interactive_mode(cpe_items, titles, embeddings, model, device, product_map)

if __name__ == "__main__":
    main(model)