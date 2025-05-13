import pandas as pd
import geopandas as gpd


def compute_neighbours(gdf: gpd.GeoDataFrame, buffer_meters: float=25):

    # Create spatial index
    sindex = gdf.sindex

    # Build neighbor relationships
    records = []

    for idx, row in gdf.iterrows():
        link_id = row["link_segment_id"]
        geom = row["geom"]
        buf = geom.buffer(buffer_meters)

        # Find nearby segments (excluding self)
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

    # Save
    out = pd.DataFrame(records)
    return out
