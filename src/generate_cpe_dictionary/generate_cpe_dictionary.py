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

import configparser
import requests
import time
import sys
import argparse
import csv
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent, register_namespace
import os
from datetime import datetime, timezone

def get_api_key():
    """Reads the API key from the config.ini file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    if not config.read(config_path) or 'NVD' not in config or 'API_KEY' not in config['NVD']:
        print("Error: 'config.ini' not found or 'API_KEY' is missing.", file=sys.stderr)
        print("Please create a 'config.ini' file with your NVD API key:", file=sys.stderr)
        print("[NVD]\nAPI_KEY = YOUR_API_KEY_HERE", file=sys.stderr)
        sys.exit(1)
    
    api_key = config['NVD']['API_KEY']
    if 'YOUR_API_KEY_HERE' in api_key:
        print("Error: Please replace 'YOUR_API_KEY_HERE' with your actual NVD API key in 'config.ini'.", file=sys.stderr)
        sys.exit(1)
        
    return api_key

def get_proxy_settings():
    """Reads proxy settings from the config.ini file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)
    
    proxy_settings = {
        "enabled": False,
        "http": None,
        "https": None
    }

    if 'Proxy' in config and config.getboolean('Proxy', 'enabled', fallback=False):
        host = config.get('Proxy', 'host', fallback='')
        port = config.get('Proxy', 'port', fallback='')
        username = config.get('Proxy', 'username', fallback='')
        password = config.get('Proxy', 'password', fallback='')

        if not host or not port:
            print("Warning: Proxy enabled but host or port is missing in config.ini. Ignoring proxy settings.", file=sys.stderr)
            return proxy_settings

        proxy_url = f"{host}:{port}"
        if username and password:
            proxy_url = f"{username}:{password}@{proxy_url}"
        
        proxy_settings["enabled"] = True
        proxy_settings["http"] = f"http://{proxy_url}"
        proxy_settings["https"] = f"https://{proxy_url}"
        print(f"Using proxy: {proxy_settings['http']}")
    
    return proxy_settings

def get_path_settings():
    """Reads the path settings from the config.ini file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)
    
    xml_output_filename = config.get('Paths', 'XML_OUTPUT_FILENAME', fallback='official-cpe-dictionary_v2.3.xml')
    csv_output_filename = config.get('Paths', 'CSV_OUTPUT_FILENAME', fallback='cpe_dictionary.csv')
    
    return xml_output_filename, csv_output_filename

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

def generate_cpe_dictionary():
    """
    Fetches all CPEs from the NVD API and generates an XML dictionary file
    that mimics the structure of the old official-cpe-dictionary_v2.3.xml.
    """
    parser = argparse.ArgumentParser(description="Generate CPE dictionary from NVD API.")
    parser.add_argument("--output-csv", action="store_true", help="Also generate a CSV file with CPE data.")
    parser.add_argument("--fetch-all", action="store_true", help="Fetch all CPEs from the NVD API.")
    args = parser.parse_args()

    api_key = get_api_key()
    proxy_settings = get_proxy_settings()
    xml_output_filename, csv_output_filename = get_path_settings()
    
    base_url = "https://services.nvd.nist.gov/rest/json/cpes/2.0"
    
    headers = {"apiKey": api_key}
    params = {
        "resultsPerPage": 2000,
        "startIndex": 0
    }

    proxies = None
    if proxy_settings["enabled"]:
        proxies = {
            "http": proxy_settings["http"],
            "https": proxy_settings["https"]
        }

    # Register namespaces
    register_namespace('cpe-23', 'http://scap.nist.gov/schema/cpe-extension/2.3')

    # Create the root of the XML tree
    root = Element("cpe-list")
    
    # Set root attributes
    root.set("xmlns", "http://cpe.mitre.org/dictionary/2.0")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xmlns:scap-core", "http://scap.nist.gov/schema/scap-core/0.3")
    root.set("xmlns:cpe-23", "http://scap.nist.gov/schema/cpe-extension/2.3")
    root.set("xmlns:meta", "http://scap.nist.gov/schema/cpe-dictionary-metadata/0.2")
    root.set("xsi:schemaLocation", 
             "http://cpe.mitre.org/dictionary/2.0 https://scap.nist.gov/schema/cpe/2.3/cpe-dictionary_2.3.xsd "
             "http://scap.nist.gov/schema/cpe-extension/2.3 https://scap.nist.gov/schema/cpe/2.3/cpe-dictionary-extension_2.3.xsd "
             "http://scap.nist.gov/schema/cpe-dictionary-metadata/0.2 https://scap.nist.gov/schema/cpe/2.1/cpe-dictionary-metadata_0.2.xsd "
             "http://scap.nist.gov/schema/scap-core/0.3 https://scap.nist.gov/schema/nvd/scap-core_0.3.xsd")

    # Add generator block
    generator = SubElement(root, "generator")
    SubElement(generator, "product_name").text = "cpeMiniLLMLevenshtein"
    SubElement(generator, "product_version").text = "1.0"
    SubElement(generator, "schema_version").text = "2.3"
    SubElement(generator, "timestamp").text = datetime.now(timezone.utc).isoformat()

    all_cpe_data_for_csv = [] # To store data for CSV output

    total_results = 0
    processed_count = 0
    
    if args.fetch_all:
        # Fetch all results
        try:
            response = requests.get(base_url, headers=headers, params={"resultsPerPage": 1, "startIndex": 0}, proxies=proxies, timeout=30)
            response.raise_for_status()
            total_results = response.json().get("totalResults", 0)
            print(f"Total CPEs to fetch: {total_results}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to get total results: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Limit to 2 for testing
        total_results = 2
        print("Fetching only 2 CPEs for testing. Use --fetch-all to get all CPEs.")


    print("Starting CPE data fetch from NVD API...")

    while processed_count < total_results:
        try:
            response = requests.get(base_url, headers=headers, params=params, proxies=proxies, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx) 
            
            data = response.json()
            products = data.get("products", [])

            if not products:
                print("No more products found. Exiting.")
                break

            for product in products:
                if processed_count >= total_results:
                    break
                
                cpe_data = product.get("cpe", {})
                cpe_uri = cpe_data.get("cpeName")
                
                if not cpe_uri:
                    continue

                # For XML output
                cpe_item = SubElement(root, "cpe-item", name=cpe_uri)
                
                # Add titles
                for title_data in cpe_data.get("titles", []):
                    title_text = title_data.get("title")
                    lang = title_data.get("lang")
                    if title_text and lang:
                        title_elem = SubElement(cpe_item, "title")
                        title_elem.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
                        title_elem.text = title_text
                
                # Add references
                refs = cpe_data.get("refs", [])
                if refs:
                    references_elem = SubElement(cpe_item, "references")
                    for ref_data in refs:
                        ref_elem = SubElement(references_elem, "reference", href=ref_data.get("ref"))
                        ref_elem.text = ref_data.get("type")

                # Add CPE 2.3 item
                cpe23_item = SubElement(cpe_item, "cpe-23:cpe23-item", name=cpe_uri)

                # For CSV output
                if args.output_csv:
                    vendor, product_name, version = parse_cpe_name(cpe_uri)
                    all_cpe_data_for_csv.append({
                        "vendor": vendor,
                        "product": product_name,
                        "version": version,
                        "cpe": cpe_uri
                    })

                processed_count += 1
            
            params["startIndex"] += len(products)
            
            print(f"Fetched {processed_count} of {total_results} CPEs...")

            # NVD API rate limiting
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"\nAn API request error occurred: {e}", file=sys.stderr)
            print("Please check your network connection, API key, and proxy settings.", file=sys.stderr)
            break # Exit loop on error
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
            break # Exit loop on error

    print(f"\nSuccessfully fetched {processed_count} CPEs.")
    
    # Write XML file
    print(f"Generating XML file: {xml_output_filename}...")
    tree = ElementTree(root)
    indent(tree, space="  ", level=0)
    tree.write(xml_output_filename, encoding="UTF-8", xml_declaration=True)
    print("XML generation done.")

    # Write CSV file if requested
    if args.output_csv:
        print(f"Generating CSV file: {csv_output_filename}...")
        if all_cpe_data_for_csv:
            keys = all_cpe_data_for_csv[0].keys()
            with open(csv_output_filename, 'w', newline='', encoding='utf-8') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_cpe_data_for_csv)
            print("CSV generation done.")
        else:
            print("No CPE data to write to CSV.")

    print("All operations completed.")

if __name__ == "__main__":
    generate_cpe_dictionary()
