import pandas as pd
from pandas.core.frame import DataFrame


def filter_data(data: DataFrame) -> DataFrame:
    """
    Filters the dataset to include only events that meet specific criteria:

    1. The event must contain at least two CDMs, one to infer from and one to use as the target.
    2. The last CDM released for the event must be within a day (time_to_tca < 1) of the TCA.
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

# This Python function, named filter_data, takes a DataFrame (data) as input and filters it based on specific criteria, returning a new DataFrame with events that meet those criteria. Here's a breakdown of the function:
#
# Grouping by Event ID and Counting CDMs:
# The function starts by grouping the input DataFrame data by the "event_id" column. For each unique event, it counts the number of Conjunction Data Messages (CDMs) associated with that event. This count is stored in a new DataFrame called event_counts.
# Filtering Events with at Least Two CDMs:
# The next step involves identifying events that have at least two CDMs. The valid_events DataFrame is created by selecting only those rows in event_counts where the "cdm_count" is greater than or equal to 2. These events are considered valid for further processing.
# Filtering the Original Data Based on Valid Events:
# The function then filters the original DataFrame data to include only the rows corresponding to the valid events identified in the previous step. This is done using the isin method to filter rows where the "event_id" is in the list of valid events.
# Keeping the Last CDM Within a Day of TCA:
# For each valid event, the function keeps only the last CDM that is within a day of the Time to Closest Approach (TCA). The groupby and idxmin operations help identify the index of the row with the minimum "time_to_tca" value for each group (each event). This ensures that the last CDM before TCA is selected.
# Keeping Events with the First CDM at Least Two Days Before TCA:
# The function further filters the data to keep only those events where the first CDM occurred at least two days before the TCA. This ensures that there is sufficient time for analysis and decision-making before the closest approach.
# Resetting Index:
# Finally, the index of the resulting filtered DataFrame (filtered_data) is reset to avoid ambiguity, and the filtered data is printed out for inspection.
# Return:
# The function returns the filtered DataFrame containing events that meet the specified criteria.
