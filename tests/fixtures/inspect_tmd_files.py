"""
Utility to inspect TMD files for debugging test failures.
Run this on failing test output files to understand what's wrong.
"""
import os
import sys
import struct
import argparse
from typing import List, Dict, Any, Optional, Tuple

def hexdump(data: bytes, start: int = 0, length: Optional[int] = None, width: int = 16, show_ascii: bool = True) -> str:
    """
    Generate a hexdump of the binary data.
    
    Args:
        data: Binary data to dump
        start: Starting offset
        length: Maximum length to show (or None for all)
        width: Number of bytes per line
        show_ascii: Whether to include ASCII representation
        
    Returns:
        Formatted hexdump string
    """
    if length is not None:
        data = data[start:start+length]
    else:
        data = data[start:]
        
    result = []
    for i in range(0, len(data), width):
        chunk = data[i:i+width]
        hex_repr = ' '.join(f"{b:02x}" for b in chunk)
        
        # Pad with spaces to align ASCII part
        hex_repr = hex_repr.ljust(width * 3 - 1)
        
        if show_ascii:
            ascii_repr = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
            line = f"{i+start:08x}:  {hex_repr}  |{ascii_repr}|"
        else:
            line = f"{i+start:08x}:  {hex_repr}"
            
        result.append(line)
        
    return '\n'.join(result)

def inspect_tmd_file(file_path: str) -> Dict[str, Any]:
    """
    Inspect a TMD file to diagnose issues.
    
    Args:
        file_path: Path to TMD file
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    results = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path)
    }
    
    try:
        with open(file_path, 'rb') as f:
            # Read first 128 bytes for header analysis
            header_data = f.read(128)
            results["header_hex"] = hexdump(header_data)
            
            # Try to decode ASCII strings
            try:
                header_ascii = header_data.decode('ascii', errors='replace')
                results["header_ascii"] = header_ascii.strip()
            except:
                results["header_ascii"] = None
            
            # Look for potential dimensions (int32 pairs)
            f.seek(0)
            file_data = f.read()
            
            # Look for potential dimensions at various offsets
            dimension_candidates = []
            for offset in range(0, min(len(file_data) - 8, 128), 4):
                try:
                    f.seek(offset)
                    # Try little endian
                    width_le = struct.unpack('<I', f.read(4))[0]
                    height_le = struct.unpack('<I', f.read(4))[0]
                    
                    # Check if dimensions seem reasonable
                    if 1 <= width_le <= 10000 and 1 <= height_le <= 10000:
                        # Check if expected data size matches file size
                        expected_data_size = width_le * height_le * 4  # 4 bytes per float32
                        if expected_data_size + offset + 24 <= results["file_size"]:
                            dimension_candidates.append({
                                "offset": offset,
                                "endian": "little",
                                "width": width_le,
                                "height": height_le,
                                "data_size": expected_data_size
                            })
                    
                    # Try big endian
                    f.seek(offset)
                    width_be = struct.unpack('>I', f.read(4))[0]
                    height_be = struct.unpack('>I', f.read(4))[0]
                    
                    # Only add if different from little endian
                    if width_be != width_le or height_be != height_le:
                        if 1 <= width_be <= 10000 and 1 <= height_be <= 10000:
                            expected_data_size = width_be * height_be * 4
                            if expected_data_size + offset + 24 <= results["file_size"]:
                                dimension_candidates.append({
                                    "offset": offset,
                                    "endian": "big",
                                    "width": width_be,
                                    "height": height_be,
                                    "data_size": expected_data_size
                                })
                except:
                    pass
            
            results["dimension_candidates"] = dimension_candidates
            
        return results
    
    except Exception as e:
        return {
            **results,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Inspect TMD file for debugging")
    parser.add_argument("file", help="Path to TMD file to inspect")
    args = parser.parse_args()
    
    results = inspect_tmd_file(args.file)
    
    print(f"File: {results['file_path']}")
    print(f"Size: {results['file_size']} bytes")
    print("\nHeader:")
    print(results.get("header_hex", ""))
    
    print("\nPossible ASCII header:")
    print(results.get("header_ascii", "None"))
    
    print("\nPotential dimensions:")
    for candidate in results.get("dimension_candidates", []):
        print(f"  At offset {candidate['offset']} ({candidate['endian']} endian): "
              f"{candidate['width']}x{candidate['height']} "
              f"(data size: {candidate['data_size']} bytes)")
    
    if "error" in results:
        print(f"\nError: {results['error']}")

if __name__ == "__main__":
    main()
