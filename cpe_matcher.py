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
import pandas as pd

# Configuration
class Config:
    xml_filepath = "official-cpe-dictionary_v2.3.xml"  # Path to the CPE dictionary XML file
    pickle_filepath = "cpe_data.pkl"                   # Path to save CPE data
    embeddings_filepath = "cpe_embeddings_minilm.npy"  # Path to save embeddings
    batch_size = 128                                   # Batch size for processing (increased because MiniLM is lighter)
    model_path = "models/all-MiniLM-L6-v2"             # Path to locally saved model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Original model name (for download if needed)

config = Config()

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

# Load or download MiniLM model
print("Loading MiniLM model...")
if os.path.exists(config.model_path):
    print(f"Loading model from local path: {config.model_path}")
    model = SentenceTransformer(config.model_path, device=device)
else:
    print(f"Local model not found. Downloading model: {config.model_name}")
    print("This requires internet connection.")
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
        
        # Download the model
        model = SentenceTransformer(config.model_name, device=device)
        
        # Save the model locally for future offline use
        print(f"Saving model to {config.model_path} for future offline use...")
        model.save(config.model_path)
        print(f"Model saved successfully to {config.model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please ensure you have internet connection for the first run.")
        exit(1)

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
    try:
        tree = ET.parse(filepath)
        return tree.getroot()
    except Exception as e:
        print(f"Error loading XML file: {e}")
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
    """Create embeddings for a list of texts using MiniLM."""
    # Initialize empty numpy array to store all embeddings
    embeddings = np.zeros((len(texts), 384), dtype=np.float32)  # MiniLM produces 384-dimensional embeddings
    
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
    """Prepare CPE data and generate embeddings."""
    root = load_cpe_xml(xml_filepath)
    if root is None:
        return None, None, None
    
    print("Extracting CPE items...")
    cpe_items, titles = extract_cpe_items(root)
    
    print(f"Number of extracted CPE items: {len(cpe_items)}")
    
    # Prepare texts for embeddings
    print("Preparing texts for embeddings...")
    texts = []
    for cpe, title in zip(cpe_items, titles):
        vendor, product, version = parse_cpe_name(cpe)
        
        vendor_clean = clean_text(vendor.replace('_', ' '))
        product_clean = clean_text(product.replace('_', ' '))
        version_clean = clean_text(version.replace('_', ' '))
        
        # Build textual representation of the CPE
        text = f"{vendor_clean} {product_clean}"
        if version_clean:
            text += f" {version_clean}"
        if title:
            text += f" {clean_text(title)}"
        
        texts.append(text.strip())
    
    print("Generating embeddings...")
    embeddings = create_embeddings(texts, model, device, batch_size=config.batch_size)
    
    return cpe_items, titles, embeddings

def save_cpe_data(cpe_items, titles, embeddings, pickle_filepath, embeddings_filepath):
    """Save CPE data and embeddings."""
    try:
        print(f"Saving CPE data to {pickle_filepath}...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump({'cpe_items': cpe_items, 'titles': titles}, f)
        
        print(f"Saving embeddings to {embeddings_filepath}...")
        np.save(embeddings_filepath, embeddings)
        
        print("Save completed successfully.")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_cpe_data(pickle_filepath, embeddings_filepath):
    """Load CPE data and embeddings."""
    try:
        print(f"Loading data from {pickle_filepath}...")
        with open(pickle_filepath, 'rb') as f:
            data = pickle.load(f)
        
        cpe_items = data.get('cpe_items', [])
        titles = data.get('titles', [])
        
        print(f"Loading embeddings from {embeddings_filepath}...")
        embeddings = np.load(embeddings_filepath)
        
        print(f"CPE data loaded: {len(cpe_items)} items")
        print(f"Embeddings loaded: {embeddings.shape}")
        
        return cpe_items, titles, embeddings
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def find_closest_cpes(vendor, product, version, cpe_items, titles, embeddings, model, device, num_results=5):
    """Find the closest CPE codes using MiniLM and Levenshtein distance."""
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
    
    # Encode the query with SentenceTransformer
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    print("Computing similarities...")
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of sorted results
    top_indices = similarities.argsort()[-num_results*2:][::-1]
    
    print("Preparing results...")
    results = []
    for idx in top_indices:
        cpe = cpe_items[idx]
        title = titles[idx] if idx < len(titles) else ""
        cpe_vendor, cpe_product, cpe_version = parse_cpe_name(cpe)
        
        # Calculate scores with Levenshtein distance
        vendor_score = levenshtein_similarity(vendor, cpe_vendor.replace('_', ' ')) if vendor and cpe_vendor else 0.5
        product_score = levenshtein_similarity(product, cpe_product.replace('_', ' ')) if product and cpe_product else 0.0
        version_score = levenshtein_similarity(version, cpe_version.replace('_', ' ')) if version and cpe_version else 0.5
        
        # Weight the scores
        semantic_score = similarities[idx]
        combined_score = (semantic_score * 0.4) + (vendor_score * 0.2) + (product_score * 0.3) + (version_score * 0.1)
        
        results.append((combined_score, cpe, title))
    
    # Sort results by combined score
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Return only the top N results
    return results[:num_results]

def process_excel_file(excel_path, cpe_items, titles, embeddings, model, device):
    """Process an Excel file with vendor, product, and version data to find CPE codes."""
    try:
        print(f"Loading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # Verify the Excel structure
        required_columns = ['Vendor', 'Product', 'Version', 'CPE', 'Levenshtein score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            print("Required columns: A (Vendor), B (Product), C (Version), D (CPE), E (Levenshtein score)")
            return False
        
        total_rows = len(df)
        processed_count = 0
        updated_count = 0
        
        print(f"Processing {total_rows} entries...")
        
        # Process each row
        for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing Excel data"):
            vendor = str(row['Vendor']) if not pd.isna(row['Vendor']) else ""
            product = str(row['Product']) if not pd.isna(row['Product']) else ""
            version = str(row['Version']) if not pd.isna(row['Version']) else ""
            
            if not product:  # Product is required
                continue
            
            # Find closest CPE codes
            results = find_closest_cpes(vendor, product, version, cpe_items, titles, embeddings, model, device, num_results=1)
            
            processed_count += 1
            
            if results:
                score, cpe, _ = results[0]
                
                # Only update if the Levenshtein score is above 0.7
                if score > 0.7:
                    df['CPE'] = df['CPE'].astype(str)
                    df.at[index, 'CPE'] = cpe
                    df.at[index, 'Levenshtein score'] = score
                    updated_count += 1
        
        # Save the updated Excel file
        output_path = excel_path.replace('.xlsx', '_updated.xlsx')
        if excel_path.endswith('.xls'):
            output_path = excel_path.replace('.xls', '_updated.xlsx')
        
        print(f"Saving updated Excel file to: {output_path}")
        df.to_excel(output_path, index=False)
        
        print(f"Processing complete. Processed {processed_count} entries, updated {updated_count} entries.")
        return True
    
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return False

def interactive_mode(cpe_items, titles, embeddings, model, device):
    """Run the program in interactive mode."""
    while True:
        print("\n=== Search for CPE codes ===")
        vendor = input("Enter vendor name (leave empty if unknown, 'q' to quit): ")
        
        if vendor.lower() == 'q':
            break
            
        product = input("Enter product name ('q' to quit): ")
        
        if product.lower() == 'q':
            break
            
        if not product:
            print("Product name is required for search.")
            continue
            
        version = input("Enter product version: ")
        
        print("\nSearching...")
        search_start = time.time()
        results = find_closest_cpes(vendor, product, version, cpe_items, titles, embeddings, model, device)
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CPE Matcher with MiniLM and Levenshtein")
    parser.add_argument("-data", help="Path to Excel file with vendor, product, and version data", type=str, default=None)
    args = parser.parse_args()

    print("\n=== CPE Matcher with MiniLM and Levenshtein (Auto-download version) ===")
    print(f"Using PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        n_gpu = torch.cuda.device_count()
        print(f"Available GPUs: {n_gpu}")
    
    check_gpu_usage()
    
    start_time = time.time()
    
    # Load or prepare CPE data
    if os.path.exists(config.pickle_filepath) and os.path.exists(config.embeddings_filepath):
        print("Loading existing data...")
        cpe_items, titles, embeddings = load_cpe_data(config.pickle_filepath, config.embeddings_filepath)
        if cpe_items is None:
            print("Error loading data. Preparing new data...")
            cpe_items, titles, embeddings = prepare_cpe_data(config.xml_filepath, model, device)
            if cpe_items:
                save_cpe_data(cpe_items, titles, embeddings, config.pickle_filepath, config.embeddings_filepath)
    else:
        print("Preparing CPE data...")
        cpe_items, titles, embeddings = prepare_cpe_data(config.xml_filepath, model, device)
        if cpe_items:
            save_cpe_data(cpe_items, titles, embeddings, config.pickle_filepath, config.embeddings_filepath)
    
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.2f} seconds")
    
    if not cpe_items or embeddings is None:
        print("Cannot continue without valid CPE data.")
        return
    
    # Run in Excel mode or interactive mode based on arguments
    if args.data:
        excel_path = args.data
        if not os.path.exists(excel_path):
            print(f"Error: The specified Excel file does not exist: {excel_path}")
            return
        
        print(f"Running in Excel processing mode with file: {excel_path}")
        process_excel_file(excel_path, cpe_items, titles, embeddings, model, device)
    else:
        print("Running in interactive mode.")
        interactive_mode(cpe_items, titles, embeddings, model, device)

if __name__ == "__main__":
    main()