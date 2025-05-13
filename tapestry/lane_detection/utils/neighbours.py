import pandas as pd
import geopandas as gpd


def _compute_neighbours_local(gdf: gpd.GeoDataFrame, buffer_meters: float = 25):
    """
    Computes neighbors in a GeoDataFrame already projected in local CRS.
    """
    sindex = gdf.sindex
    records = []

    for idx, row in gdf.iterrows():
        link_id = row["link_segment_id"]
        geom = row["geom"]
        buf = geom.buffer(buffer_meters)

        neighbors_idx = list(sindex.intersection(buf.bounds))
        neighbors = gdf.iloc[neighbors_idx]

        for _, n in neighbors.iterrows():
            if n["link_segment_id"] != link_id and n["geom"].intersects(buf):
                records.append({
                    "link_segment_id": link_id,
                    "neighbor_segment_id": n["link_segment_id"]
                })

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(gdf)} segments...")

    return pd.DataFrame(records)


def compute_neighbours(
    all_link_segments: pd.DataFrame,
    crs_lookup: dict[str, int],
    buffer_meters: float = 50
) -> pd.DataFrame:
    """
    Computes lateral neighbor relationships for all link segments, grouped by base network.

    Args:
        all_link_segments: DataFrame with 'geom' (EPSG:3857) and 'base_network_id'
        crs_lookup: Dict of base_network_id → EPSG code
        buffer_meters: Buffer radius in meters (in projected CRS)

    Returns:
        pd.DataFrame with columns: link_segment_id, neighbor_segment_id
    """
    all_neighbours = []

    for base_network_id, epsg in crs_lookup.items():
        subset = all_link_segments[all_link_segments["base_network_id"] == base_network_id]
        if subset.empty:
            continue

        print(f"🔍 Processing base network {base_network_id} in EPSG:{epsg}...")

        gdf = gpd.GeoDataFrame(subset, geometry="geom", crs="EPSG:3857")
        gdf_proj = gdf.to_crs(f"EPSG:{epsg}")
        gdf_proj["geom"] = gdf_proj.geometry  # overwrite original column name

        neighbours = _compute_neighbours_local(gdf_proj, buffer_meters=buffer_meters)
        all_neighbours.append(neighbours)

    if all_neighbours:
        return pd.concat(all_neighbours, ignore_index=True)
    else:
        print("⚠️ No neighbors computed. Empty input?")
        return pd.DataFrame(columns=["link_segment_id", "neighbor_segment_id"])
