import os
import pandas as pd
import shutil

EXPORT_FOLDER_NAME = "exported-dataset"

# filter the dataset
os.makedirs("filtered", exist_ok=True)

for filename in os.listdir("data"):
    df = pd.read_parquet(os.path.join("data", filename))
    filtered_df = df[df["caption"].str.contains("Dog") |
                     df["caption"].str.contains("dog") |
                     df["caption"].str.contains("Cat") |
                     df["caption"].str.contains("cat")]
    filtered_df.to_parquet(os.path.join("filtered", filename))

# Combine all filtered files, 5 at a time
file_count = len(os.listdir("filtered"))

while(file_count > 5):
    file_list = os.listdir("filtered")  
    groups_count = int(file_count / 5)
    for i in range(groups_count):
        dfs = []
        for j in range(5):
            # filename = "filtered/{:05d}-of-{:05d}".format(i*5+j,int(groups_count))
            filename = os.path.join("filtered", file_list[i*5 + j])
            dfs.append(pd.read_parquet(filename))
            
        fiveRows = pd.concat(dfs).reset_index(drop=True)
        fiveRows.to_parquet(os.path.join("filtered", "combined-{:05d}-of-{:05d}.parquet".format(i, groups_count)))
        
        for j in range(5):
            os.remove(os.path.join("filtered", file_list[i*5 + j]))
        
    file_count = len(os.listdir("filtered"))

dfs = []
file_list = os.listdir("filtered")
for i in range(len(file_list)):
    filename = os.path.join("filtered", file_list[i])
    dfs.append(pd.read_parquet(filename))
    
finalCombination_df = pd.concat(dfs).reset_index(drop=True)
finalCombination_df.to_parquet("filtered/combined.parquet")

for i in range(len(file_list)):
    os.remove(os.path.join("filtered", file_list[i]))

## Format, export, and zip dataset
os.makedirs(os.path.join(EXPORT_FOLDER_NAME, "data"), exist_ok=True)

# Format dataset
formatted_df = pd.DataFrame({
    'audio': './data/' + finalCombination_df['audio.path'],
    'caption': finalCombination_df['caption']
})

# Export as metadata.csv
formatted_df.to_csv(os.path.join(EXPORT_FOLDER_NAME, "metadata.csv"), index=False)

# Export .wavs
for index, row in finalCombination_df.iterrows():
    with open(os.path.join(EXPORT_FOLDER_NAME, "data", row['audio.path']), 'wb') as f:
        f.write(row['audio.bytes'])

# Zip
shutil.make_archive(EXPORT_FOLDER_NAME, "zip", EXPORT_FOLDER_NAME)