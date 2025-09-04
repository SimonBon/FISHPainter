import os
import requests
import argparse
import sys
from pathlib import Path
from tqdm import tqdm  # Import tqdm for the progress bar

def download_from_zenodo(doi, output_directory):
    output_directory = Path(output_directory)
    # Fetch the Zenodo deposition metadata using the DOI
    response = requests.get(f"https://zenodo.org/api/records/{doi}")
    if response.status_code != 200:
        print("Error fetching Zenodo record:", response.status_code)
        return

    data = response.json()
    files = data.get("files", [])
    
    print(f"Downloading {len(files)} files!")

    if not files:
        print("No files found in the Zenodo record.")
        return
    
    if output_directory.exists():
        print(f"{output_directory} already exists. Saving to {output_directory}")
    else:
        output_directory.mkdir()
    
    for file in files:
        file_name = file["key"]
        download_url = file["links"]["self"]
        local_file_path = output_directory.joinpath(file_name)
        
        if not local_file_path.parent.exists():
            local_file_path.parent.mkdir()

        # Stream the download and update the progress bar
        print(f"Downloading '{file_name}' from Zenodo...")
        response = requests.get(download_url, stream=True)  # Stream the download
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_file_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print(f"\nError downloading '{file_name}' from Zenodo.")
        else:
            print(f"'{file_name}' has been downloaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from Zenodo using a DOI.")
    parser.add_argument("-id", "--doi", type=str, help="The DOI of the Zenodo record.")
    parser.add_argument("-o", "--output_directory", type=str, help="The directory to save the downloaded files.")
    args = parser.parse_args()
    download_from_zenodo(args.doi, args.output_directory)
