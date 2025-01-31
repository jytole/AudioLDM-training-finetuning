import os
import json
import pandas as pd
import shutil
from tqdm import tqdm

# Load the CSV file
csv_path = './data/labels.csv' #### Change to your own CSV file!
data = pd.read_csv(csv_path)

# Define paths
root_dir = './data'
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
    new_path = os.path.join(audioset_dir, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    try:
        shutil.copy(audio_file, new_path)
    except Exception as e:
        print(f"Error copying {audio_file}: {e}")

# Create metadata JSON files
train_data = []
test_data = []
val_data = []

for i, row in data.iterrows():
    datapoint = {
        'wav': os.path.basename(row['audio']),
        'caption': row['caption']
    }
    # You can define your own condition to split between train, test, and val
    if i % 5 == 0:  # Example condition for test
        test_data.append(datapoint)
    elif i % 5 == 1:  # Example condition for validation
        val_data.append(datapoint)
    else:
        train_data.append(datapoint)

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