import os
import psycopg2
import pandas as pd
import geopandas as gpd
from shapely import wkb
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("TOPANIMEX_DB_URL")

# ───── COLUMN SETS ─────

CAMERA_POINT_COLUMNS = [
    "camera_point_id",
    "base_network_id",
    "bearing",
    "length",
    "extracted",
    "sampled_for_annotation",
]

NODE_COLUMNS = [
    "node_id",
    "base_network_id",
    "n_links",
    "camera_point_id",
    "annotated",
    "annotator",
    "annotation_date",
    "n_locks",
]

LINK_SEGMENT_COLUMNS = [
    "link_segment_id",
    "link_id",
    "segment_ix_uv",
    "segment_ix_vu",
    "camera_point_id",
    "annotated",
    "annotator",
    "annotation_date",
    "n_locks",
]

SECTION_COLUMNS = [
    "id",
    "link_segment_id",
    "component"
]
TURN_COLUMNS = [
    "id",
    "node_id",
    "component"
]

# ───── BASE NETWORKS ─────

def get_base_networks() -> pd.DataFrame:
    query = """
        SELECT *
        FROM basenetwork_basenetwork;
    """
    return pd.read_sql(query, con=DB_URL)

# ───── CAMERA POINTS ─────

def get_camera_points_by_ids(
        camera_point_ids: list[str],
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if exclude_geom:
        columns = ", ".join(CAMERA_POINT_COLUMNS)
    else:
        columns = "*"

    placeholders = ", ".join(["%s"] * len(camera_point_ids))
    query = f"""
        SELECT {columns}
        FROM basenetwork_camerapoint
        WHERE camera_point_id IN ({placeholders})
    """

    params = tuple(camera_point_ids)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom", params=params).to_crs(3857)
    return gdf

# ───── NODES ─────

def get_nodes_by_base_network(
        base_network_ids: list[str],
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    placeholders = ",".join("%s" for _ in base_network_ids)
    if ids_only:
        columns = "node_id"
    elif exclude_geom:
        columns = ", ".join(NODE_COLUMNS)
    else:
        columns = "*"

    query = f"""
        SELECT {columns}
        FROM basenetwork_node
        WHERE base_network_id IN ({placeholders});
    """

    params = tuple(base_network_ids)
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL, params=params)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857", params=params)
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf


def get_nodes_by_annotation_area(
        buffer_meters: float = 50.0,
        area_names: list[str] | None = None,
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    name_filter = ""
    if area_names:
        placeholders = ",".join("%s" for _ in area_names)
        name_filter = f"WHERE aa.name IN ({placeholders})"

    if ids_only:
        columns = "n.node_id"
    elif exclude_geom:
        columns = ", ".join([f"n.{col}" for col in NODE_COLUMNS])
    else:
        columns = "n.*"

    query = f"""
        WITH buffered_areas AS (
            SELECT id, ST_Buffer(ST_Transform(geom, 3857), %s) AS geom_3857
            FROM topologyannotator_annotationarea aa
            {name_filter}
        )
        SELECT {columns}
        FROM basenetwork_node n
        JOIN buffered_areas b ON ST_Intersects(ST_Transform(n.geom, 3857), b.geom_3857);
    """
    params = (buffer_meters,) + tuple(area_names or [])
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL, params=params)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857", params=params)
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf


def get_annotated_nodes(
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if ids_only:
        columns = "node_id"
    elif exclude_geom:
        columns = ", ".join(NODE_COLUMNS)
    else:
        columns = "*"

    query = f"""
        SELECT {columns}
        FROM basenetwork_node
        WHERE annotated = 'Y';
    """
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857")
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf

def get_all_nodes(
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if ids_only:
        columns = "node_id"
    elif exclude_geom:
        columns = ", ".join(NODE_COLUMNS)
    else:
        columns = "*"

    query = f"""
        SELECT {columns}
        FROM basenetwork_node;
    """
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857")
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf

# ───── LINK SEGMENTS ─────

def get_link_segments_by_base_network(
        base_network_ids: list[str],
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    placeholders = ",".join("%s" for _ in base_network_ids)
    if ids_only:
        columns = "ls.link_segment_id"
    elif exclude_geom:
        columns = ", ".join([f"ls.{col}" for col in LINK_SEGMENT_COLUMNS])
    else:
        columns = "ls.*"

    query = f"""
        SELECT {columns}
        FROM basenetwork_linksegment ls
        JOIN basenetwork_camerapoint cp ON cp.camera_point_id = ls.camera_point_id
        WHERE cp.base_network_id IN ({placeholders});
    """

    params = tuple(base_network_ids)
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL, params=params)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857", params=params)
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf


def get_link_segments_by_annotation_area(
        buffer_meters: float = 25.0,
        area_names: list[str] | None = None,
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    name_filter = ""
    if area_names:
        placeholders = ",".join("%s" for _ in area_names)
        name_filter = f"WHERE aa.name IN ({placeholders})"

    if ids_only:
        columns = "ls.link_segment_id"
    elif exclude_geom:
        columns = ", ".join([f"ls.{col}" for col in LINK_SEGMENT_COLUMNS])
    else:
        columns = "ls.*"

    query = f"""
        WITH buffered_areas AS (
            SELECT id, ST_Buffer(ST_Transform(geom, 3857), %s) AS geom_3857
            FROM topologyannotator_annotationarea aa
            {name_filter}
        )
        SELECT {columns}
        FROM basenetwork_linksegment ls
        WHERE ST_Intersects(ls.geom_3857, (SELECT ST_Union(geom_3857) FROM buffered_areas));
    """
    params = (buffer_meters,) + tuple(area_names or [])
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL, params=params)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857", params=params)
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf


def get_annotated_link_segments(
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if ids_only:
        columns = "link_segment_id"
    elif exclude_geom:
        columns = ", ".join(LINK_SEGMENT_COLUMNS)
    else:
        columns = "*"

    query = f"""
        SELECT {columns}
        FROM basenetwork_linksegment
        WHERE annotated = 'Y';
    """
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857")
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf


def get_link_segments_for_annotated_nodes(
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if ids_only:
        columns = "ls.link_segment_id"
    elif exclude_geom:
        columns = ", ".join([f"ls.{col}" for col in LINK_SEGMENT_COLUMNS])
    else:
        columns = "ls.*"

    query = f"""
        SELECT {columns}
        FROM basenetwork_linksegment ls
        JOIN basenetwork_link l ON l.link_id = ls.link_id
        JOIN basenetwork_node n1 ON n1.node_id = l.u_id
        JOIN basenetwork_node n2 ON n2.node_id = l.v_id
        WHERE n1.annotated = 'Y' OR n2.annotated = 'Y';
    """
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857")
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf

def get_all_link_segments(
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if ids_only:
        columns = "link_segment_id"
    elif exclude_geom:
        columns = ", ".join(LINK_SEGMENT_COLUMNS)
    else:
        columns = "*"

    query = f"""
        SELECT {columns}
        FROM basenetwork_linksegment;
    """
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL)
    gdf = gpd.read_postgis(query, con=DB_URL, geom_col="geom_3857")
    gdf = gdf.drop(columns=["geom"], errors="ignore").rename(columns={"geom_3857": "geom"}).set_geometry("geom")
    return gdf

# ───── SECTIONS & TURNS ─────

def get_sections_by_link_segment_ids(
        link_segment_ids: list[str],
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if not link_segment_ids:
        raise ValueError("No link_segment_ids provided.")

    placeholders = ",".join("%s" for _ in link_segment_ids)

    if ids_only:
        columns = "s.id"
    elif exclude_geom:
        columns = ", ".join([f"s.{col}" for col in SECTION_COLUMNS]) + ", cp.base_network_id::text"
    else:
        columns = "s.*, cp.base_network_id::text"

    query = f"""
        SELECT {columns}
        FROM topologyannotator_section s
        JOIN basenetwork_linksegment ls ON ls.link_segment_id = s.link_segment_id
        JOIN basenetwork_camerapoint cp ON cp.camera_point_id = ls.camera_point_id
        WHERE s.link_segment_id IN ({placeholders});
    """

    params = tuple(link_segment_ids)
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL, params=params)
    return gpd.read_postgis(query, con=DB_URL, geom_col="geom", params=params).to_crs(3857)


def get_turns_by_node_ids(
        node_ids: list[str],
        ids_only: bool = False,
        exclude_geom: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:

    if not node_ids:
        raise ValueError("No node_ids provided.")

    placeholders = ",".join("%s" for _ in node_ids)

    if ids_only:
        columns = "t.id"
    elif exclude_geom:
        columns = ", ".join([f"t.{col}" for col in TURN_COLUMNS]) + ", n.base_network_id::text"
    else:
        columns = "t.*, n.base_network_id::text"

    query = f"""
        SELECT {columns}
        FROM topologyannotator_turn t
        JOIN basenetwork_node n ON n.node_id = t.node_id
        WHERE t.node_id IN ({placeholders});
    """

    params = tuple(node_ids)
    if ids_only or exclude_geom:
        return pd.read_sql(query, con=DB_URL, params=params)
    return gpd.read_postgis(query, con=DB_URL, geom_col="geom", params=params).to_crs(3857)
