import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def get_camera_point_offset(segs, crs_lookup):
    segs_with_offsets = []

    for base_network_id, epsg in crs_lookup.items():
        subset = segs[segs["base_network_id"] == base_network_id].copy()
        if subset.empty:
            continue

        subset['geom_proj'] = subset['geom'].to_crs(f"EPSG:{epsg}")
        subset['camera_geom_proj'] = subset['camera_geom'].to_crs(f"EPSG:{epsg}")

        subset['dist_camera_ls_start'] = subset['camera_geom_proj'].distance(
            subset['geom_proj'].apply(lambda line: Point(line.coords[0]))
        )
        subset['dist_camera_ls_end'] = subset['camera_geom_proj'].distance(
            subset['geom_proj'].apply(lambda line: Point(line.coords[-1]))
        )

        subset['camera_point_offset'] = (
            subset['dist_camera_ls_start'] - (subset['geom_proj'].length / 2)
        )

        # Adjust for points located beyond the end (extrapolated)
        condition = (
            (subset['dist_camera_ls_start'] < subset['dist_camera_ls_end']) &
            (subset['dist_camera_ls_end'] > subset['geom_proj'].length)
        )
        subset.loc[condition, 'camera_point_offset'] = (
            -subset['dist_camera_ls_start'] - (subset['geom_proj'].length / 2)
        )

        subset = gpd.GeoDataFrame(
            subset.drop(columns=[
                'camera_geom',
                'geom_proj',
                'camera_geom_proj',
                'dist_camera_ls_start',
                'dist_camera_ls_end',
            ], errors="ignore"),
            geometry='geom',
            crs=segs.crs
        )

        segs_with_offsets.append(subset)

    # Combine results and clean up
    segs_with_offsets = gpd.GeoDataFrame(
        pd.concat(segs_with_offsets),
        geometry='geom',
        crs=segs.crs
    )

    return segs_with_offsets
