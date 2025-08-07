#!/usr/bin/env python3
"""
Script to clean up old files from src/attacks/ directory
Run this to remove files that have been consolidated/moved
"""

import os
import shutil
from pathlib import Path

# Files and directories to remove from src/attacks/
files_to_remove = [
    "api_attacks.log",
    "api_classifier_wrapper.py", 
    "google_nlp_rest.py",
    "mlass_api_clients.py",
    "model_wrapper.py",
    "quick_api_test.py",
    "results/",  # entire directory
    "run_api_attacks_limited.py",
    "run_api_blackbox.py", 
    "run_blackbox.py",
    "run_blackbox_ibm.py",
    "run_locals.py",
    "run_locals.sh",
    "test_api_attack.py",
    "test_api_clients.py", 
    "utils.py"
]

def cleanup_attacks_directory():
    """Remove old files from src/attacks/"""
    attacks_dir = Path("src/attacks")
    
    if not attacks_dir.exists():
        print("❌ src/attacks/ directory not found")
        return
    
    print("🧹 Cleaning up src/attacks/ directory...")
    print("Removing old files that have been consolidated/moved:")
    
    removed_count = 0
    
    for file_name in files_to_remove:
        file_path = attacks_dir / file_name
        
        try:
            if file_path.exists():
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"  🗑️  Removed directory: {file_name}")
                else:
                    file_path.unlink()
                    print(f"  🗑️  Removed file: {file_name}")
                removed_count += 1
            else:
                print(f"  ⚪ Already removed: {file_name}")
        except Exception as e:
            print(f"  ❌ Error removing {file_name}: {e}")
    
    print(f"\n✅ Cleanup complete! Removed {removed_count} items.")
    
    # Show what's left
    print(f"\n📁 Files remaining in src/attacks/:")
    remaining_files = sorted(attacks_dir.glob("*"))
    for file_path in remaining_files:
        if file_path.name != "__pycache__":
            print(f"  ✅ {file_path.name}")

def main():
    print("TextBugger Project Cleanup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the textbugger project root directory")
        return
    
    response = input("🧹 Clean up old files from src/attacks/? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_attacks_directory()
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()