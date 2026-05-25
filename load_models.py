# load_models.py
import os
import requests

MODEL_DIR = 'models'

# Replace these with your actual Google Drive file IDs
FILES = {
    'model_efficientnet.keras': 'YOUR_EFFICIENTNET_FILE_ID',
    'model_mobilenet.keras':    'YOUR_MOBILENET_FILE_ID',
    'class_labels.json':        'YOUR_LABELS_FILE_ID',
}

def download_file(file_id, dest_path):
    """Download file from Google Drive."""
    URL = 'https://drive.google.com/drive/folders/1-2oIqakuihqFZSfGky0OLD-MzINfSleJ?usp=sharing'
    session = requests.Session()

    # First request to get confirmation token
    response = session.get(URL, params={'id': file_id}, stream=True)
    token    = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # Second request with token for large files
    if token:
        response = session.get(
            URL,
            params={'id': file_id, 'confirm': token},
            stream=True
        )

    # Write file in chunks
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f'✅ Downloaded: {dest_path}')


def ensure_models():
    """Download models if not already present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, file_id in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            print(f'Downloading {filename}...')
            try:
                download_file(file_id, dest)
            except Exception as e:
                print(f'❌ Failed to download {filename}: {e}')
        else:
            print(f'✅ Already exists: {filename}')