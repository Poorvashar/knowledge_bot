# scripts/test_drive.py

from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_sources.google_drive_fetch import list_files_in_folder

# REPLACE THIS with your actual Folder ID
FOLDER_ID = "1WhWso5U_WLSyMFU8z0g0ZqpoHWWj1LjT"

print("Testing Google Drive connection...\n")
files = list_files_in_folder(FOLDER_ID)

if files:
    print("\n Success! Google Drive connection is working.")
else:
    print("\n No files found. Check:")
    print("  1. Is the Folder ID correct?")
    print("  2. Did you share the folder with the service account?")
    print("  3. Are there files in the folder?")