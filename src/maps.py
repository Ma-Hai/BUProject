import pandas as pd
import numpy as np

import rasterio
from rasterstats import zonal_stats
import geopandas as gpd
from shapely.geometry import Polygon

from tqdm import tqdm


CITY_CENTER = 39.9526, -75.1652
SQUARE = 0.01448, 0.01888 # 1-mile square in lat/long

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

def map_bc(df_b: pd.DataFrame, df_r: pd.DataFrame, df_c: pd.DataFrame, grid_size: float = 1.0) -> pd.DataFrame:
    min_lat = min(df_b["latitude"].min(), df_c["latitude"].min())
    max_lat = max(df_b["latitude"].max(), df_c["latitude"].max())
    min_lon = min(df_b["longitude"].min(), df_c["longitude"].min())
    max_lon = max(df_b["longitude"].max(), df_c["longitude"].max())

    lat_step, lon_step = grid_size * SQUARE[0], grid_size * SQUARE[1]

    lats = np.arange(min_lat, max_lat, lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)
    lats_bc, lons_bc = lats[:, None] + 0 * lons[None, :], lons[None, :] + 0 * lats[:, None]


    df_b["grid_id"] = ((df_b["latitude"] - min_lat) // lat_step).astype(int) * len(lons) + ((df_b["longitude"] - min_lon) // lon_step).astype(int)
    df_r["grid_id"] = ((df_r["latitude"] - min_lat) // lat_step).astype(int) * len(lons) + ((df_r["longitude"] - min_lon) // lon_step).astype(int)
    df_c["grid_id"] = ((df_c["latitude"] - min_lat) // lat_step).astype(int) * len(lons) + ((df_c["longitude"] - min_lon) // lon_step).astype(int)


    grid_ids = np.arange(len(lats) * len(lons))
    df_g = pd.DataFrame(
        {
            "grid_id": grid_ids,
            "latitude": lats_bc.flatten(),
            "longitude": lons_bc.flatten(),
            "business_ids": [list(df_b.loc[df_b["grid_id"] == grid_id, "business_id"]) for grid_id in grid_ids],
            "review_ids": [list(df_r.loc[df_r["grid_id"] == grid_id, "review_id"]) for grid_id in grid_ids],
            "crime_ids": [list(df_c.loc[df_c["grid_id"] == grid_id, "objectid"]) for grid_id in grid_ids],
        }
    )

    df_g["n_businesses"] = df_g["business_ids"].apply(len)
    df_g["n_reviews"] = df_g["review_ids"].apply(len)
    df_g["n_crimes"] = df_g["crime_ids"].apply(len)

    raster_path = "data/usa_ppp_2020_constrained.tif"
    
    population_raster = rasterio.open(raster_path)
    
    # --- Step 3: Prepare Your Grid for Analysis ---

    polygons = []

    for index, row in df_g.iterrows():
        sw_corner_lat = row["latitude"]
        sw_corner_lon = row["longitude"]
        ne_corner_lat = sw_corner_lat + lat_step
        ne_corner_lon = sw_corner_lon + lon_step
        polygons.append(Polygon([
            (sw_corner_lon, sw_corner_lat), (ne_corner_lon, sw_corner_lat),
            (ne_corner_lon, ne_corner_lat), (sw_corner_lon, ne_corner_lat)
        ]))

    grid_gdf = gpd.GeoDataFrame(df_g, geometry=polygons, crs="EPSG:4326")


    # --- Step 4: Perform Zonal Statistics ---

    # This is the core operation. It calculates statistics of the raster pixel values
    # that fall within each polygon of your grid.

    # Ensure your grid is in the same Coordinate Reference System (CRS) as the raster.
    if grid_gdf.crs != population_raster.crs:
        grid_gdf = grid_gdf.to_crs(population_raster.crs)

    # Use zonal_stats to calculate the sum of population pixels within each grid square.
    # 'stats="sum"' tells the function to add up all the population values.
    # The result is a list of dictionaries, one for each grid square.
    stats = zonal_stats(
        grid_gdf.geometry,
        raster_path,
        stats="sum",
        all_touched=True # Includes cells that are only touched by a polygon's border
    )

    df_g["population"] = [d["sum"] for d in stats]
    df_g["population"] = df_g["population"].fillna(0).round().astype(int)

    df_g["crime_rate"] = df_g["n_crimes"] / df_g["population"]
    df_g.loc[df_g["population"] == 0, "crime_rate"] = pd.NA
    
    return df_g


def match(index: pd.Series, target: pd.Series) -> pd.Series:
    """
    given an index series of lists of indices into the target series,
    return a series where each index is replaced by a list of the values in the target series
    corresponding to the indices in the index series.
    """

    return index.apply(lambda ix: list(target.get(ix, pd.NA)))


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

#     # create map widget and only use the tiles from the database, not the online server (use_database_only=True)
#     # map_widget = tkintermapview.TkinterMapView(root_tk, width=1000, height=700, corner_radius=0,
#                                 # max_zoom=17)
    
#     map_widget = tkintermapview.TkinterMapView(root_tk, width=1000, height=700, corner_radius=0)
#     map_widget.pack(fill="both", expand=True)

#     # map_widget.se t_tile_server("https://tiles.wmflabs.org/osm-no-labels/{z}/{x}/{y}.png")  # OpenStreetMap (default)

#     map_widget.set_overlay_tile_server("http://a.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png")  # railway infrastructure


#     map_widget.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
#     map_widget.set_position(*CITY_CENTER)
#     map_widget.set_zoom(12)



#     root_tk.mainloop()
