import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_distribution(y_array, threshold=None, classification=True):
    """
    Visualizes the distribution of binary risk classes and prints relevant statistics if classification is True.

    Parameters:
    - y_array (numpy.ndarray): Array containing risk values or risk classes.
    - threshold (float or None): Threshold value to classify high and low risk. Default is None.
    - classification (bool): Specifies whether risk classification is performed.

    Returns:
    - dict or None: Dictionary containing statistics if classification is True, otherwise None.
    """

    # Classify risk based on threshold if classification is True
    if classification:
        # If threshold is provided, use it to classify risk
        if threshold is not None:
            y_risk_class = np.where(y_array >= threshold, 1, 0)
        else:
            # If threshold is not provided, assume risk classes are already provided
            y_risk_class = y_array

        # Plot distribution
        sns.histplot(y_risk_class, bins=30, kde=True, color="blue")
        # Set the background color to white
        # plt.gca().set_facecolor('white')
        #
        # # Save the plot as a PNG file with a white background
        # plt.savefig("histogram.png", pad_inches=0)

        # Calculate statistics
        y_high_risk = np.sum(y_risk_class == 1)
        y_low_risk = np.sum(y_risk_class == 0)

        # Check if the denominator is not zero before calculating the percentage
        if y_low_risk + y_high_risk != 0:
            y_risk_percent = 100 * y_high_risk / (y_low_risk + y_high_risk)
        else:
            y_risk_percent = (
                np.nan
            )  # Set y_risk_percent to NaN if the denominator is zero

        # Print statistics
        print("Risk Distribution Statistics:")
        print(f"High Risk: {y_high_risk}")
        print(f"Low Risk: {y_low_risk}")
        print(f"Percent High Risk: {y_risk_percent}")

        # Return statistics as dictionary
        return {
            "High Risk": y_high_risk,
            "Low Risk": y_low_risk,
            "Percent High Risk": y_risk_percent,
        }

    else:
        return sns.histplot(y_array, bins=30, kde=True, color="blue")
