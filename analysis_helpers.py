# Serves for enriching metadata

import pandas as pd

OUTLET_METADATA = {
    "CNN": {"country": "USA", "region": "North America", "leaning": "Center-Left"},
    "Fox News": {"country": "USA", "region": "North America", "leaning": "Right"},
    "BBC News": {"country": "UK", "region": "Europe", "leaning": "Center"},
    "Al Jazeera English": {"country": "Qatar", "region": "Middle East", "leaning": "Center"},
}

def add_outlet_metadata(df):
    df = df.copy()

    def lookup(source):
        if source in OUTLET_METADATA:
            return pd.Series(OUTLET_METADATA[source])
        return pd.Series({
            "country_label": "Unknown",
            "region_label": "Unknown",
            "leaning_label": "Unknown"
        })

    meta = df["source"].apply(lookup)
    meta.columns = ["country_label", "region_label", "leaning_label"]
    return pd.concat([df, meta], axis=1)