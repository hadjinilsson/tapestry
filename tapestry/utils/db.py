import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("TOPANIMEX_DB_URL")


def get_camera_point_ids(base_network_id: str) -> list[str]:
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


def get_camera_points_for_annotated_link_segments() -> list[str]:
    query = """
        SELECT DISTINCT camera_point_id
        FROM basenetwork_linksegment
        WHERE annotated = 'Y'
          AND camera_point_id IS NOT NULL
        LIMIT 200;
    """

    with psycopg2.connect(DB_URL, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return [row["camera_point_id"] for row in rows]
