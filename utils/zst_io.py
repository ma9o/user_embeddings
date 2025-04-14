import dask.bag as db
import dask.dataframe as dd
import zstandard as zstd
import json
import os
import glob
from typing import List, Dict, Any, Optional, Iterable
import pandas as pd
import io # Import io module

def read_single_zst_ndjson(filepath: str) -> Iterable[Dict[str, Any]]:
    """
    Reads a single Zstandard compressed NDJSON file line by line.

    Args:
        filepath: Path to the .zst file.

    Yields:
        Dictionaries representing the JSON objects parsed from each line.
    """
    try:
        with open(filepath, 'rb') as fh:
            # Increase max_window_size further to handle potentially large frames
            dctx = zstd.ZstdDecompressor(max_window_size=2**31) # 2 GiB
            # Use stream_reader for potentially large files
            with dctx.stream_reader(fh) as reader:
                # Wrap the binary reader in a TextIOWrapper for line-based iteration
                text_reader = io.TextIOWrapper(reader, encoding='utf-8')
                line_num = 0
                for line in text_reader:
                    line_num += 1
                    try:
                        # Line is already decoded by TextIOWrapper
                        if line.strip(): # Avoid empty lines
                            yield json.loads(line)
                    except json.JSONDecodeError as json_err:
                        print(f"Warning: Skipping line due to JSON decode error in {os.path.basename(filepath)}: {json_err} - Line: {line[:100]}...")
                    except UnicodeDecodeError as unicode_err:
                        print(f"Warning: Skipping line due to Unicode decode error in {os.path.basename(filepath)}: {unicode_err} - Line: {line[:100]}...")
                    except Exception as e:
                        # Print more specific error info
                        print(f"Warning: Skipping line due to unexpected error ({type(e).__name__}) in {os.path.basename(filepath)}: {e} - Line: {line[:100]}...")
                        
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}, skipping.")
    except zstd.ZstdError as zstd_err:
        print(f"Warning: Zstandard decompression error in {filepath}: {zstd_err}, skipping file.")
    except Exception as e:
        # Print more specific error info
        print(f"Warning: Failed to read {filepath} due to unexpected error ({type(e).__name__}): {e}, skipping file.")

def read_zst_ndjson_files(
    directory_path: str, 
    file_pattern: str = "*.zst", 
    columns: Optional[List[str]] = None,
    meta: Optional[pd.DataFrame | Dict[str, Any]] = None
) -> dd.DataFrame:
    """
    Reads all Zstandard compressed NDJSON files matching a pattern within a directory 
    into a Dask DataFrame.

    Args:
        directory_path: Path to the directory containing .zst files.
        file_pattern: Glob pattern for the files to read (default: "*.zst").
        columns: Optional list of columns to keep in the resulting DataFrame. 
                 If `meta` is provided, `columns` is ignored as `meta` defines the schema.
        meta: Optional DataFrame or dict mapping column names to dtypes,
              used to specify the schema. Recommended for performance and correctness.

    Returns:
        A Dask DataFrame containing the data from the files.
    """
    filepaths = glob.glob(os.path.join(directory_path, file_pattern))
    if not filepaths:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in directory: {directory_path}")
        
    print(f"Found {len(filepaths)} files matching '{file_pattern}' in {directory_path}")

    # Create a Dask Bag by reading each file
    # Each partition in the bag will likely correspond to one file or part of a file
    bag = db.from_sequence(filepaths).map_partitions(
        lambda paths: [record for path in paths for record in read_single_zst_ndjson(path)]
    )

    # Convert the bag of dictionaries to a Dask DataFrame
    if meta is None and columns is not None:
        # If no meta, but columns are given, construct basic meta
        # This assumes 'object' dtype, which might not be optimal but avoids inference error
        print("Warning: No meta provided, constructing basic meta from columns with object dtype. Performance might suffer.")
        meta = pd.DataFrame({col: pd.Series(dtype='object') for col in columns})
        
    # Pass the provided or constructed meta to to_dataframe
    # Dask handles dict conversion internally if meta is a dict
    ddf = bag.to_dataframe(meta=meta)

    # If meta was provided, columns argument is less relevant as meta defines the schema.
    # If meta was *not* provided but columns *were*, we constructed basic meta above.
    # If columns were provided *without* meta, and inference *did* work (unlikely now),
    # we might still want to select. However, prioritizing meta is safer.
    # The selection based on `columns` can happen AFTER meta-driven creation if needed,
    # but usually meta defines the exact desired columns.
    if columns and meta is None: 
         # This case is less likely now meta is constructed if columns are present
         # but kept for theoretical completeness, though redundant.
         available_columns = ddf.columns
         missing_cols = [col for col in columns if col not in available_columns]
         if missing_cols:
              print(f"Warning: Requested columns not found after DataFrame creation: {missing_cols}")
              columns_to_select = [col for col in columns if col in available_columns]
              if not columns_to_select:
                   raise ValueError("None of the requested columns were found in the data.")
              ddf = ddf[columns_to_select]
         else:
              ddf = ddf[columns]
        
    return ddf 