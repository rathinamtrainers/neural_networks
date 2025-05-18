import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Step 1: Define our custom dataset
class MyCSVData(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):                                                  # idx is the index of the row
        x = torch.tensor([self.data.iloc[idx, 0]], dtype=torch.float32)     # Access first column
        y = torch.tensor([self.data.iloc[idx, 1]], dtype=torch.float32)     # Access second column
        return x, y

if __name__ == "__main__":
    # Step-2: Save sample CSV file before running further.
    import csv
    with open("122-sample_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for i in range(100):
            x = i * 0.1
            y = 2 * x + 1
            writer.writerow([x, y])

    # Step-3: Load using custom dataset
    dataset = MyCSVData("122-sample_data.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # Step-4: Loop through DataLoader
    for batch in dataloader:
        inputs, targets = batch
        print(f"Inputs: {inputs.shape}, Targets: {targets.shape}")
        break




