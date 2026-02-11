from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os


def get_drive_service():
    """Set up connection to Google Drive API"""
    credentials = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_CREDENTIALS_PATH"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=credentials)

def list_files_in_folder(folder_id):
    """List all files in the specified folder"""
    service = get_drive_service()

    results = service.files().list(
        q=f"'{folder_id}' in parents",     # Files inside this folder
        fields="files(id, name, mimeType)", # Get file ID, name, and type
    ).execute()

    files = results.get("files", [])
    print(f"📁 Found {len(files)} files")
    for f in files:
        print(f"  - {f['name']} ({f['mimeType']})")
    return files

def download_file_content(file_id, mime_type):
    """Download file content and return as text"""
    service = get_drive_service()

    # Google Docs need to be exported (PDFs and txt can be downloaded directly)
    if mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="text/plain"  # Export as plain text
        )
    else:
        request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    return buffer.getvalue().decode("utf-8", errors="ignore")