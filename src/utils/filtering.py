import pandas as pd
from pandas.core.frame import DataFrame


def filter_data(data: DataFrame) -> DataFrame:
    """
    Filters the dataset to include only events that meet specific criteria:

    1. The event must contain at least two CDMs, one to infer from and one to use as the target.
    2. The last CDM released for the event must be within a day (timetotca < 1) of the TCA.
    3. The first CDM released for the event must be at least two days before the TCA (time to tca ⩾ 2),
       and all the CDMs that were within two days from the TCA (time to tca < 2) are removed.

    Args:
        data (DataFrame): The original dataset.

    Returns:
        DataFrame: The filtered dataset containing events that meet the specified criteria.
    """
    # Group by event_id and count the number of CDMs for each event
    event_counts = data.groupby("event_id").size().reset_index(name="cdm_count")

    # Keep events with at least two CDMs
    valid_events = event_counts[event_counts["cdm_count"] >= 2]["event_id"]

    # Filter the data based on valid event IDs
    filtered_data = data[data["event_id"].isin(valid_events)]

    # Keep the last CDM within a day of TCA
    filtered_data = filtered_data.loc[
        filtered_data.groupby("event_id")["time_to_tca"].idxmin()
    ]

    # Keep events with the first CDM at least two days before TCA
    filtered_data = filtered_data[filtered_data["time_to_tca"] >= 2]

    # Reset index to avoid ambiguity
    filtered_data.reset_index(drop=True, inplace=True)

    print("Filtered data")
    filtered_data.info()

    return filtered_data
