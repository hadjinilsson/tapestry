import os
import psycopg2
import pandas as pd
import geopandas as gpd
from shapely import wkb
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("TOPANIMEX_DB_URL")


def get_base_network_crs() -> dict[str, int]:
    """
    Fetch mapping of base_network_id → EPSG CRS code.

    Returns:
        Dict[str, int]: e.g., {"dalby25": 3006, "cph25": 25832, ...}
    """
    query = """
        SELECT base_network_id, crs
        FROM basenetwork_basenetwork;
    """

    with psycopg2.connect(DB_URL, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return {row["base_network_id"]: row["crs"] for row in rows}


def get_camera_point_ids_for_base_network(base_network_id: str) -> list[str]:
    query = """
        SELECT camera_point_id
        FROM basenetwork_camerapoint
        WHERE base_network_id = %s
          AND extracted = 'Y'
        ORDER BY camera_point_id;
    """

    with psycopg2.connect(DB_URL, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (base_network_id,))
            rows = cur.fetchall()
            return [row["camera_point_id"] for row in rows]


def get_camera_point_ids_for_annotated_link_segments() -> list[str]:
    query = """
        SELECT DISTINCT camera_point_id
        FROM basenetwork_linksegment
        WHERE annotated = 'Y'
          AND camera_point_id IS NOT NULL
    """

    with psycopg2.connect(DB_URL, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return [row["camera_point_id"] for row in rows]


def get_link_segments_near_annotation_areas(
    buffer_meters: float = 25.0,
    annotation_area_names: list[str] | None = None,
    get_camera_point_geoms: bool = True,
) -> gpd.GeoDataFrame:
    """
    Returns all LinkSegments that intersect buffered AnnotationAreas,
    and whose camera point has been extracted.

    Args:
        buffer_meters: Buffer distance in meters (applied to AnnotationArea geom_3857).
        annotation_area_names: List of AnnotationArea IDs, or None for all.
        get_camera_point_geoms: Optional flag to get camera point geometries.

    Returns:
        GeoDataFrame of link segments intersecting buffered annotation areas.
    """
    if annotation_area_names:
        name_list = ",".join(f"'{i}'" for i in annotation_area_names)
        area_filter = f"WHERE aa.name IN ({name_list})"
    else:
        area_filter = ""  # Use all areas

    columns = """
        ls.link_segment_id,
        cp.base_network_id::text,
        ls.link_id,
        ls.segment_ix_uv,
        ls.segment_ix_vu,
        ls.annotated,
        ls.camera_point_id,
        ls.geom_3857 AS geom
    """

    if get_camera_point_geoms:
        columns += ", ST_Transform(cp.geom, 3857) as camera_geom"

    query = f"""
        WITH selected_areas AS (
            SELECT id, ST_Buffer(ST_Transform(geom, 3857), {buffer_meters}) AS geom_3857
            FROM topologyannotator_annotationarea aa
            {area_filter}
        )
        SELECT
            {columns}
        FROM basenetwork_linksegment ls
        JOIN basenetwork_camerapoint cp ON cp.camera_point_id = ls.camera_point_id
        JOIN selected_areas aa ON ST_Intersects(ls.geom_3857, aa.geom_3857)
        WHERE cp.extracted = 'Y';
    """

    df = pd.read_sql(query, con=DB_URL)
    df["geom"] = df["geom"].apply(wkb.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geom", crs="EPSG:3857")

    if get_camera_point_geoms:
        gdf["camera_geom"] = gdf["camera_geom"].apply(wkb.loads)
        gdf["camera_geom"] = gpd.GeoSeries(gdf["camera_geom"], crs="EPSG:3857")

    return gdf


def get_sections_by_link_segment_ids(link_segment_ids: list[str]) -> gpd.GeoDataFrame:
    """
    Fetches all Sections tied to the given link_segment_ids, including base_network_id.

    Args:
        link_segment_ids: List of link_segment_id strings.

    Returns:
        GeoDataFrame of sections, with base_network_id column.
    """
    if not link_segment_ids:
        raise ValueError("No link_segment_ids provided.")

    formatted_ids = ",".join(f"'{sid}'" for sid in link_segment_ids)

    query = f"""
        SELECT s.id AS section_id,
               s.link_segment_id,
               s.component,
               cp.base_network_id::text,
               s.geom
        FROM topologyannotator_section s
        JOIN basenetwork_linksegment ls ON ls.link_segment_id = s.link_segment_id
        JOIN basenetwork_camerapoint cp ON cp.camera_point_id = ls.camera_point_id
        WHERE s.link_segment_id IN ({formatted_ids});
    """

    return gpd.read_postgis(query, con=DB_URL, geom_col="geom")