import pandas as pd
import numpy as np

def load_yelp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_b = pd.read_csv("data/df_b_processed.csv")
    df_r = pd.read_csv("data/df_r_processed.csv")

    df_b["categories"] = df_b["categories"].str.split(", ")
    df_b["postal_code"] = df_b["postal_code"].astype("Int64")
    df_b["is_open"] = df_b["is_open"] > 0

    def parse_hours(hours: str | float) -> dict[str, tuple[np.datetime64, np.datetime64]] | float:
        if isinstance(hours, float):
            return hours
        hours = eval(hours)
        def parse_hours_time(interval: str) -> tuple[np.datetime64, np.datetime64]:
            a, b = interval.split("-")
            a_h, a_m = a.split(":")
            b_h, b_m = b.split(":")
            return int(a_h) + int(a_m) / 60, int(b_h) + int(b_m) / 60
        return {day: parse_hours_time(h) for day, h in hours.items()}

    df_b["hours"] = df_b["hours"].apply(parse_hours)

    df_b["price_range"] = df_b["attributes"].apply(lambda a: (eval(a)["RestaurantsPriceRange2"] if isinstance(a, str) and "RestaurantsPriceRange2" in eval(a) and eval(a)["RestaurantsPriceRange2"] in "1234" else pd.NA)).astype("Int64")

    df_r["date"] = pd.to_datetime(df_r["date"])

    # set review latitude/longitude to business lat/long by looking up business_id
    df_b_by_id = df_b.set_index("business_id")
    df_r["latitude"] = df_r["business_id"].apply(df_b_by_id["latitude"].get)
    df_r["longitude"] = df_r["business_id"].apply(df_b_by_id["longitude"].get)

    return df_b, df_r

def load_crime_data() -> pd.DataFrame:
    df_cx = [pd.read_csv(f"data/Crime_Incidents/incidents_{y}.csv") for y in range(2006, 2023)]
    df_c = pd.concat(df_cx)
    df_c["text_general_code"] = pd.Categorical(df_c["text_general_code"])
    df_c["dispatch_date_time"] = pd.to_datetime(df_c["dispatch_date_time"])
    df_c["dispatch_date_time"] = df_c["dispatch_date_time"].dt.tz_localize(None)

    return df_c.dropna(subset=["lat", "lng"], ignore_index=True).rename(columns={"lat": "latitude", "lng": "longitude"})
