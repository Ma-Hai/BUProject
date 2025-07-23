import pandas as pd
import numpy as np
# import tkinter
# import tkintermapview
from tqdm import tqdm

CITY_CENTER = 39.9526, -75.1652

def dist(
    lat1: float | np.ndarray[np.float32],
    lon1: float | np.ndarray[np.float32],
    lat2: float | np.ndarray[np.float32],
    lon2: float | np.ndarray[np.float32]
) -> float | np.ndarray[np.float32]:
    """
    Calculate the great-circle distance between points on the earth,
    specified in decimal degrees.

    This implementation uses the haversine formula and supports inputs as
    single numbers (float) or as NumPy arrays for vectorized calculations.
    It correctly broadcasts between single points and arrays of points.

    Args:
        lat1: Latitude of the first point(s).
        lon1: Longitude of the first point(s).
        lat2: Latitude of the second point(s).
        lon2: Longitude of the second point(s).

    Returns:
        The great-circle distance(s) in miles. Returns a float for single
        point calculations and a NumPy array for vectorized calculations.

    Example:
        >>> # 1. Single point to single point (New York to London)
        >>> haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        3456.49...

        >>> # 2. Broadcasting a single point to an array of points
        >>> ny_lat, ny_lon = 40.7128, -74.0060
        >>> other_lats = np.array([51.5074, 48.8566]) # London, Paris
        >>> other_lons = np.array([-0.1278, 2.3522])
        >>> haversine_distance(ny_lat, ny_lon, other_lats, other_lons)
        array([3456.49..., 3624.99...])
    """
    # Earth radius in miles
    R = 3958.8

    # --- Haversine Calculation ---
    # Convert all inputs to radians for the trigonometric functions
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate differences. NumPy handles broadcasting automatically.
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Apply the haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    # If the result is a single-element array, return it as a float
    return distance.item() if isinstance(distance, np.ndarray) and distance.size == 1 else distance

STATIONS = pd.read_csv("https://docs.google.com/spreadsheets/d/1_v8HYe_p2xumjlXTIIhmSyWE32Bm6LgGjhwmbDOi2M0/export?format=csv", index_col="Name")

def select_businesses(name: str, df_b: pd.DataFrame) -> pd.DataFrame:
    """Select businesses near a station by name"""
    sel = dist(
        df_b["latitude"],
        df_b["longitude"],
        STATIONS["Lat"][name],
        STATIONS["Long"][name],
    ) < STATIONS["Radius"][name]
    return df_b[sel]

def select_reviews(name: str, df_r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select reviews near a station by name, split into pre-construction and post-construction"""
    sel_pos = dist(
        df_r["latitude"],
        df_r["longitude"],
        STATIONS["Lat"][name],
        STATIONS["Long"][name],
    ) < STATIONS["Radius"][name]
    sel_pre = df_r["date"] < pd.to_datetime(STATIONS["Start Date"][name])
    sel_post = df_r["date"] < pd.to_datetime(STATIONS["End Date"][name])
    return df_r[sel_pos & sel_pre], df_r[sel_pos & sel_post]
    


# # def draw_transit_lines(
#     map_widget: tkintermapview.TkinterMapView, 
#     routes: pd.DataFrame, 
#     stops: pd.DataFrame, 
#     trips: pd.DataFrame, 
#     shapes: pd.DataFrame,
# ) -> None:
#     for row in routes.itertuples():
#         route_id = row.route_id
#         color = row.route_color
#         shape_ids = set(trips[trips["route_id"] == route_id]["shape_id"])
#         for shape_id in shape_ids:
#             shape = shapes[shapes["shape_id"] == shape_id].sort_values("shape_pt_sequence")
#             map_widget.set_path(list(zip(shape["shape_pt_lat"], shape["shape_pt_lon"])), color="#" + color)

# if __name__ == "__main__":
#     import os
#     # from src import load_data
#     # df_b, df_r = load_data.load_yelp_data()


#     # create tkinter window
#     root_tk = tkinter.Tk()
#     root_tk.geometry(f"{1000}x{700}")
#     root_tk.title("BU Project")

#     # create map widget

#     # path for the database to use
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     database_path = os.path.join(script_directory, "offline_tiles.db")

#     # create map widget and only use the tiles from the database, not the online server (use_database_only=True)
#     map_widget = tkintermapview.TkinterMapView(root_tk, width=1000, height=700, corner_radius=0, use_database_only=True,
#                                 max_zoom=17, database_path=database_path)
#     map_widget.pack(fill="both", expand=True)
#     # map_widget = tkintermapview.TkinterMapView(root_tk, width=800, height=600, corner_radius=0)

#     map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")  # OpenStreetMap (default)

#     map_widget.set_overlay_tile_server("http://a.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png")  # railway infrastructure


#     map_widget.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
#     map_widget.set_position(*CITY_CENTER)
#     map_widget.set_zoom(10)
#     print("EEE")

#     # df_rt = pd.read_csv("data/septa_gtfs_public/google_rail/routes.txt")
#     # df_st = pd.read_csv("data/septa_gtfs_public/google_rail/stops.txt")
#     # df_tp = pd.read_csv("data/septa_gtfs_public/google_rail/trips.txt")
#     # df_sh = pd.read_csv("data/septa_gtfs_public/google_rail/shapes.txt")
#     # draw_transit_lines(map_widget, df_rt, df_st, df_tp, df_sh)



#     root_tk.mainloop()
