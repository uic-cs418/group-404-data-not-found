# Serves for enriching metadata

import pandas as pd

OUTLET_METADATA = {
    "CNN": {
        "country_label": "USA",
        "region_label": "North America",
        "leaning_label": "Center-Left"
    },
    "Fox News": {
        "country_label": "USA",
        "region_label": "North America",
        "leaning_label": "Right"
    },
    "BBC News": {
        "country_label": "UK",
        "region_label": "Europe",
        "leaning_label": "Center"
    },
    "Al Jazeera English": {
        "country_label": "Qatar",
        "region_label": "Middle East",
        "leaning_label": "Center"
    },
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
    return pd.concat([df, meta], axis=1)