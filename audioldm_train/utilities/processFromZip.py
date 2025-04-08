import os
import json
import pandas as pd
import shutil
from tqdm import tqdm
import zipfile

def unzipDataFile(zipPath):
    extractPath = './cache/extract/'
    with zipfile.ZipFile(zipPath,"r") as zip_ref:
        zip_ref.extractall(extractPath)
        
        
    return extractPath

def findCSV(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv") and not "__MACOSX" in root:
                return os.path.join(root, file)
    return None

# Structure the data as the audioldm yaml expects it (like audioset)
# Adapted from:
# https://github.com/haoheliu/AudioLDM-training-finetuning/issues/41
def structureData(csvPath, train_split_proportion=0.6):
    # Load the CSV file
    data = pd.read_csv(csvPath)
    
    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Define paths
    root_dir = './data'
    csv_dir = os.path.dirname(csvPath)
    audioset_dir = os.path.join(root_dir, 'dataset/audioset')
    metadata_dir = os.path.join(root_dir, 'dataset/metadata')
    datafiles_dir = os.path.join(metadata_dir, 'datafiles')
    testset_subset_dir = os.path.join(metadata_dir, 'testset_subset')
    valset_subset_dir = os.path.join(metadata_dir, 'valset_subset')

    # Create directories if they don't exist
    os.makedirs(audioset_dir, exist_ok=True)
    os.makedirs(datafiles_dir, exist_ok=True)
    os.makedirs(testset_subset_dir, exist_ok=True)
    os.makedirs(valset_subset_dir, exist_ok=True)

    # Copy audio files to the audioset directory
    for audio_file in tqdm(data['audio']):
        file_name = os.path.basename(audio_file)
        base_audio_path = os.path.join(csv_dir, audio_file)
        new_path = os.path.join(audioset_dir, file_name)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        try:
            shutil.copy(base_audio_path, new_path)
        except Exception as e:
            print(f"Error copying {audio_file}: {e}")

    # Create metadata JSON files
    train_data = []
    test_data = []
    val_data = []
    
    # Calculate split indices
    data_len = len(data)
    train_end = int(data_len * train_split_proportion)
    val_end = train_end + int((data_len - train_end) / 2)

    for i, row in data.iterrows():
        datapoint = {
            'wav': os.path.basename(row['audio']),
            'caption': row['caption']
        }
        if i < train_end:
            train_data.append(datapoint)
        elif i < val_end:
            val_data.append(datapoint)
        else:
            test_data.append(datapoint)

    # Save the train metadata
    train_metadata = {'data': train_data}
    with open(os.path.join(datafiles_dir, 'audiocaps_train_label.json'), 'w') as f:
        json.dump(train_metadata, f, indent=4)

    # Save the test metadata
    test_metadata = {'data': test_data}
    with open(os.path.join(testset_subset_dir, 'audiocaps_test_nonrepeat_subset_0.json'), 'w') as f:
        json.dump(test_metadata, f, indent=4)

    # Save the validation metadata
    val_metadata = {'data': val_data}
    with open(os.path.join(valset_subset_dir, 'audiocaps_val_label.json'), 'w') as f:
        json.dump(val_metadata, f, indent=4)

    # Save the dataset root metadata
    dataset_root_metadata = {
        'audiocaps': 'data/dataset/audioset',
        'metadata': {
            'path': {
                'audiocaps': {
                    'train': 'data/dataset/metadata/datafiles/audiocaps_train_label.json',
                    'test': 'data/dataset/metadata/testset_subset/audiocaps_test_nonrepeat_subset_0.json',
                    'val': 'data/dataset/metadata/valset_subset/audiocaps_val_label.json'
                }
            }
        }
    }
    with open(os.path.join(metadata_dir, 'dataset_root.json'), 'w') as f:
        json.dump(dataset_root_metadata, f, indent=4)

    print("Dataset structured successfully!")
    
    return(metadata_dir)

## Parses data from zip file at zipPath
### 1. unzips .zip file
### 2. looks for .yaml file of captions
### 3. processes it into the "audiocaps" format that AudioLDM expects

def process(zipPath, train_split_proportion=0.6):
    extractPath = unzipDataFile(zipPath)
    csvPath = findCSV(extractPath)
    structureData(csvPath, train_split_proportion)
    print("Dataset extracted and processed")
    return "Successful Dataset Processing"