import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SpaceCollisionDataset(Dataset):
    def __init__(
        self,
        csv_file,
        selected_columns=None,
        is_train=True,
        split_ratio=0.2,
        filter=False,
    ):
        if selected_columns is None:
            # Load all columns if selected_columns is not provided
            self.data = pd.read_csv(csv_file)
        else:
            # Load specific columns
            selected_columns.extend(["risk"])
            self.data = pd.read_csv(csv_file, usecols=selected_columns)

        print("Columns in the dataset:", self.data.columns)

        # Handle Missing Values
        self.data = self.data.dropna(axis=0, how="any")

        self.is_train = is_train
        if filter:
            # Apply constraints for test set eligibility
            self.data = self.filter_test_set_events()
        # Data Preprocessing
        if "c_object_type" in self.data.columns:
            self.label_encoder = LabelEncoder()
            self.data["c_object_type"] = self.label_encoder.fit_transform(
                self.data["c_object_type"]
            )
        # Normalize the 'risk' variable
        # if "risk" in self.data.columns:
        #     self.scaler_risk = StandardScaler()
        #     self.data["risk"] = self.scaler_risk.fit_transform(
        #         self.data[["risk"]]
        #     ).squeeze()

        # Split into features and targets for training
        self.features = self.data.drop(["risk"], axis=1)
        self.target = self.data["risk"]

        # Split into features and targets for training and validation
        (
            self.features_train,
            self.features_val,
            self.target_train,
            self.target_val,
        ) = train_test_split(
            self.features, self.target, test_size=split_ratio, random_state=42
        )

        print(f"Length of features_train: {len(self.features_train)}")
        print(f"Length of features_val: {len(self.features_val)}")

    def __len__(self):
        if self.is_train:
            return len(self.features_train)
        else:
            return len(self.features_val)

    def __getitem__(self, idx):
        if self.is_train:
            features = torch.tensor(
                self.features_train.iloc[idx, :].values, dtype=torch.float32
            )
            label = torch.tensor(self.target_train.iloc[idx], dtype=torch.float32)
        else:
            features = torch.tensor(
                self.features_val.iloc[idx, :].values, dtype=torch.float32
            )
            label = torch.tensor(self.target_val.iloc[idx], dtype=torch.float32)

        return features, label

    def filter_test_set_events(self):
        """
        Filters the dataset to include only events that meet specific criteria:

        1. The event must contain at least two CDMs, one to infer from and one to use as the target.
        2. The last CDM released for the event must be within a day (timetotca < 1) of the TCA.
        3. The first CDM released for the event must be at least two days before the TCA (time to tca ⩾ 2),
           and all the CDMs that were within two days from the TCA (time to tca < 2) are removed.

        Returns:
            pandas.DataFrame: The filtered dataset containing events that meet the specified criteria.
        """
        # Group by event_id and count the number of CDMs for each event
        event_counts = (
            self.data.groupby("event_id").size().reset_index(name="cdm_count")
        )

        # Keep events with at least two CDMs
        valid_events = event_counts[event_counts["cdm_count"] >= 2]["event_id"]

        # Filter the data based on valid event IDs
        self.data = self.data[self.data["event_id"].isin(valid_events)]

        # Keep the last CDM within a day of TCA
        self.data = self.data.loc[self.data.groupby("event_id")["time_to_tca"].idxmin()]

        # Keep events with the first CDM at least two days before TCA (remove redundant line)
        self.data = self.data[self.data["time_to_tca"] >= 2]

        # Filter out events with risk values below the threshold (e.g., -6)
        # self.data = self.data[self.data['risk'] >= -6]

        # Reset index to avoid ambiguity
        # self.data.reset_index(drop=True, inplace=True)

        return self.data


def create_dataloader(
    csv_file, selected_columns, is_train, batch_size, filter, split_ratio=0.2
):
    dataset = SpaceCollisionDataset(
        csv_file=csv_file,
        selected_columns=selected_columns,
        is_train=is_train,
        split_ratio=split_ratio,
        filter=filter,
    )

    if batch_size is None:
        # Use the entire dataset without batching
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=is_train)
    else:
        # Use DataLoader with the specified batch size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return dataloader
