import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SpaceCollisionDataset(Dataset):
    def __init__(self, csv_file, selected_columns=None, is_train=True):
        if selected_columns is None:
            # Load all columns if selected_columns is not provided
            self.data = pd.read_csv(csv_file)
        else:
            # Load specific columns
            selected_columns.extend(["risk"])
            self.data = pd.read_csv(csv_file, usecols=selected_columns)

        print("Columns in the dataset:", self.data.columns)
        self.is_train = is_train

        # Data Preprocessing
        # All columns are numerical except for c_object_type: object type which is at collision risk with satellite
        if "c_object_type" in self.data.columns:
            self.label_encoder = LabelEncoder()
            self.data["c_object_type"] = self.label_encoder.fit_transform(
                self.data["c_object_type"]
            )

        # Handle Missing Values (replace with appropriate strategy if needed)
        self.data = self.data.dropna()

        # Split into features and targets for training
        self.features = self.data.drop(["risk"], axis=1)
        self.target = self.data["risk"]

        if self.is_train:
            # Split into train and validation sets
            (
                self.features_train,
                self.features_val,
                self.target_train,
                self.target_val,
            ) = train_test_split(
                self.features, self.target, test_size=0.2, random_state=42
            )

            # Standardize numerical features
            self.scaler = StandardScaler()
            self.features_train_scaled = self.scaler.fit_transform(self.features_train)
            self.features_val_scaled = self.scaler.transform(self.features_val)

    def __len__(self):
        if self.is_train:
            return len(self.features_train)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx, :].values, dtype=torch.float32)
        label = torch.tensor(self.target.iloc[idx], dtype=torch.float32)
        return features, label


if __name__ == "__main__":
    # To load the whole dataset set columns_to_process to None
    columns_to_process = ["mission_id", "time_to_tca"]
    train_dataset = SpaceCollisionDataset(
        "../data/training/train_data.csv",
        selected_columns=columns_to_process,
        is_train=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=2
    )

    # Print a few samples from the dataset
    for i in range(5):
        sample_features, sample_label = train_dataset[i]
        print(f"Sample {i + 1} - Features: {sample_features}, Risk: {sample_label}")
