import os
import psycopg2
import geopandas as gpd
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("TOPANIMEX_DB_URL")


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
          AND camera_point_id IS NOT NULL;
    """

    with psycopg2.connect(DB_URL, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return [row["camera_point_id"] for row in rows]


def get_link_segments_near_annotation_areas(buffer_meters: float = 25.0, annotation_area_ids: list[str] | None = None) -> gpd.GeoDataFrame:
    """
    Returns all LinkSegments that intersect buffered AnnotationAreas.

    Args:
        buffer_meters: Buffer distance in meters (applied to AnnotationArea geom_3857).
        annotation_area_ids: List of AnnotationArea IDs, or None for all.

    Returns:
        GeoDataFrame of link segments intersecting buffered annotation areas.
    """
    area_filter = ""
    if annotation_area_ids:
        id_list = ",".join(str(i) for i in annotation_area_ids)
        area_filter = f"WHERE aa.id IN ({id_list})"

    query = f"""
        WITH selected_areas AS (
            SELECT id, ST_Transform(ST_Buffer(ST_Transform(geom, 3857), {buffer_meters}), 4326) AS geom
            FROM topologyannotator_annotationarea aa
            {area_filter}
        )
        SELECT ls.link_segment_id,
               ls.base_network_id::text,
               ls.annotated,
               ls.camera_point_id,
               ls.geom_3857 AS geom
        FROM basenetwork_linksegment ls
        JOIN selected_areas aa
          ON ST_Intersects(ls.geom_3857, aa.geom)
        LIMIT 100;
    """

    return gpd.read_postgis(query, con=DB_URL, geom_col="geom")


def get_sections_by_link_segment_ids(link_segment_ids: list[str]) -> gpd.GeoDataFrame:
    """
    Fetches all Sections tied to the given link_segment_ids.

    Args:
        link_segment_ids: List of link_segment_id strings.

    Returns:
        GeoDataFrame of sections.
    """
    if not link_segment_ids:
        raise ValueError("No link_segment_ids provided.")

    formatted_ids = ",".join(f"'{sid}'" for sid in link_segment_ids)

    query = f"""
        SELECT section_id,
               link_segment_id,
               class,
               subtype,
               direction,
               base_network_id,
               geom
        FROM topologyannotator_section
        WHERE link_segment_id IN ({formatted_ids});
    """

    return gpd.read_postgis(query, con=DB_URL, geom_col="geom")