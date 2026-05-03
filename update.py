#!/usr/bin/env python3
"""
update.py

Sync mine metadata and NDVI/NBR images to a Neon PostgreSQL database.
Creates tables on first run (idempotent).  Re-running updates any changed rows.

Usage:
    export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
    python update.py                    # all mines with output images
    python update.py --mine hobet-mine  # one mine only
    python update.py --schema-only      # create tables, skip image upload
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from PIL import Image
from tqdm import tqdm

HERE         = Path(__file__).parent
METADATA_CSV = HERE / "mine-metadata.csv"
OUTPUT_DIR   = HERE / "outputs"

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mines (
    id                SERIAL PRIMARY KEY,
    slug              TEXT             NOT NULL UNIQUE,
    mine_name         TEXT             NOT NULL,
    county            TEXT,
    state             CHAR(2),
    lat               DOUBLE PRECISION,
    lon               DOUBLE PRECISION,
    bounds_west       DOUBLE PRECISION,
    bounds_south      DOUBLE PRECISION,
    bounds_east       DOUBLE PRECISION,
    bounds_north      DOUBLE PRECISION,
    operational_start INT,
    operational_end   INT,
    status            TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS mine_images (
    id           SERIAL PRIMARY KEY,
    mine_id      INT              NOT NULL REFERENCES mines(id) ON DELETE CASCADE,
    year         INT              NOT NULL,
    index_type   TEXT             NOT NULL CHECK (index_type IN ('ndvi', 'nbr')),
    image_data   BYTEA            NOT NULL,
    width_px     INT              NOT NULL,
    height_px    INT              NOT NULL,
    updated_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    UNIQUE (mine_id, year, index_type)
);

CREATE TABLE IF NOT EXISTS mine_timeseries (
    id                    SERIAL PRIMARY KEY,
    mine_id               INT              NOT NULL REFERENCES mines(id) ON DELETE CASCADE,
    year                  INT              NOT NULL,
    ndvi_mean             DOUBLE PRECISION,
    ndvi_median           DOUBLE PRECISION,
    ndvi_std              DOUBLE PRECISION,
    ndvi_p25              DOUBLE PRECISION,
    ndvi_p75              DOUBLE PRECISION,
    ndvi_valid_px         INT,
    ndvi_pct_above_threshold DOUBLE PRECISION,
    nbr_mean              DOUBLE PRECISION,
    nbr_median            DOUBLE PRECISION,
    nbr_std               DOUBLE PRECISION,
    nbr_p25               DOUBLE PRECISION,
    nbr_p75               DOUBLE PRECISION,
    nbr_valid_px          INT,
    nbr_pct_above_threshold  DOUBLE PRECISION,
    updated_at            TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    UNIQUE (mine_id, year)
);

-- Flat view: one row per image with all mine metadata alongside it.
-- Excludes the image_data blob so this is safe to SELECT * from.
CREATE OR REPLACE VIEW mine_data AS
SELECT
    m.id         AS mine_id,
    m.slug,
    m.mine_name,
    m.county,
    m.state,
    m.lat,
    m.lon,
    m.bounds_west,
    m.bounds_south,
    m.bounds_east,
    m.bounds_north,
    m.operational_start,
    m.operational_end,
    m.status,
    m.notes,
    i.id         AS image_id,
    i.year,
    i.index_type,
    i.width_px,
    i.height_px,
    i.updated_at
FROM mines m
JOIN mine_images i ON i.mine_id = m.id
ORDER BY m.slug, i.index_type, i.year;
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def mine_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _int_or_none(val: str):
    return int(val) if val.strip() else None


def _float_or_none(val: str):
    return float(val) if val.strip() else None


def load_mines_csv() -> dict:
    mines = {}
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slug = mine_slug(row["mine_name"])
            mines[slug] = row
    return mines


# ── Database operations ───────────────────────────────────────────────────────

def apply_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    print("Schema applied.")


def upsert_mines(conn, mines: dict) -> dict:
    """Insert or update all mine rows.  Returns {slug: mine_id}."""
    slug_to_id = {}
    with conn.cursor() as cur:
        for slug, row in mines.items():
            cur.execute(
                """
                INSERT INTO mines (
                    slug, mine_name, county, state, lat, lon,
                    bounds_west, bounds_south, bounds_east, bounds_north,
                    operational_start, operational_end, status, notes
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (slug) DO UPDATE SET
                    mine_name         = EXCLUDED.mine_name,
                    county            = EXCLUDED.county,
                    state             = EXCLUDED.state,
                    lat               = EXCLUDED.lat,
                    lon               = EXCLUDED.lon,
                    bounds_west       = EXCLUDED.bounds_west,
                    bounds_south      = EXCLUDED.bounds_south,
                    bounds_east       = EXCLUDED.bounds_east,
                    bounds_north      = EXCLUDED.bounds_north,
                    operational_start = EXCLUDED.operational_start,
                    operational_end   = EXCLUDED.operational_end,
                    status            = EXCLUDED.status,
                    notes             = EXCLUDED.notes
                RETURNING id
                """,
                (
                    slug,
                    row["mine_name"],
                    row["county"] or None,
                    row["state"] or None,
                    _float_or_none(row["lat"]),
                    _float_or_none(row["lon"]),
                    _float_or_none(row["bounds_west"]),
                    _float_or_none(row["bounds_south"]),
                    _float_or_none(row["bounds_east"]),
                    _float_or_none(row["bounds_north"]),
                    _int_or_none(row["operational_start"]),
                    _int_or_none(row["operational_end"]),
                    row["status"] or None,
                    row["notes"] or None,
                ),
            )
            slug_to_id[slug] = cur.fetchone()[0]
    conn.commit()
    print(f"Upserted {len(mines)} mine(s).")
    return slug_to_id


def upload_images(conn, slug_to_id: dict, selected_slug: str | None = None) -> None:
    slugs = [selected_slug] if selected_slug else list(slug_to_id.keys())

    for slug in slugs:
        mine_id = slug_to_id.get(slug)
        if mine_id is None:
            print(f"  {slug}: not in mines table, skipping")
            continue

        mine_dir = OUTPUT_DIR / slug
        if not mine_dir.exists():
            print(f"  {slug}: no outputs directory, skipping")
            continue

        # Collect (index_type, year, path) for all available PNGs
        images: list[tuple[str, int, Path]] = []
        for index_type in ("ndvi", "nbr"):
            img_dir = mine_dir / index_type
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.glob(f"{index_type}_*.png")):
                try:
                    year = int(p.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                images.append((index_type, year, p))

        if not images:
            print(f"  {slug}: no images found")
            continue

        print(f"\n  {slug} ({mine_id}): {len(images)} images")

        for index_type, year, img_path in tqdm(images, desc=f"  {slug}"):
            raw = img_path.read_bytes()
            with Image.open(img_path) as im:
                w, h = im.size

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mine_images
                        (mine_id, year, index_type, image_data, width_px, height_px, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (mine_id, year, index_type) DO UPDATE SET
                        image_data = EXCLUDED.image_data,
                        width_px   = EXCLUDED.width_px,
                        height_px  = EXCLUDED.height_px,
                        updated_at = NOW()
                    """,
                    (mine_id, year, index_type, psycopg2.Binary(raw), w, h),
                )
            conn.commit()


def upload_timeseries(conn, slug_to_id: dict, selected_slug: str | None = None) -> None:
    slugs = [selected_slug] if selected_slug else list(slug_to_id.keys())

    for slug in slugs:
        mine_id = slug_to_id.get(slug)
        if mine_id is None:
            continue

        ts_path = OUTPUT_DIR / slug / "timeseries.json"
        if not ts_path.exists():
            print(f"  {slug}: no timeseries.json, skipping")
            continue

        with open(ts_path, encoding="utf-8") as f:
            data = json.load(f)

        series = data.get("series", [])
        if not series:
            print(f"  {slug}: timeseries.json has no series data")
            continue

        print(f"  {slug}: uploading {len(series)} timeseries rows")
        with conn.cursor() as cur:
            for row in series:
                cur.execute(
                    """
                    INSERT INTO mine_timeseries (
                        mine_id, year,
                        ndvi_mean, ndvi_median, ndvi_std, ndvi_p25, ndvi_p75,
                        ndvi_valid_px, ndvi_pct_above_threshold,
                        nbr_mean, nbr_median, nbr_std, nbr_p25, nbr_p75,
                        nbr_valid_px, nbr_pct_above_threshold,
                        updated_at
                    ) VALUES (
                        %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s,
                        NOW()
                    )
                    ON CONFLICT (mine_id, year) DO UPDATE SET
                        ndvi_mean                = EXCLUDED.ndvi_mean,
                        ndvi_median              = EXCLUDED.ndvi_median,
                        ndvi_std                 = EXCLUDED.ndvi_std,
                        ndvi_p25                 = EXCLUDED.ndvi_p25,
                        ndvi_p75                 = EXCLUDED.ndvi_p75,
                        ndvi_valid_px            = EXCLUDED.ndvi_valid_px,
                        ndvi_pct_above_threshold = EXCLUDED.ndvi_pct_above_threshold,
                        nbr_mean                 = EXCLUDED.nbr_mean,
                        nbr_median               = EXCLUDED.nbr_median,
                        nbr_std                  = EXCLUDED.nbr_std,
                        nbr_p25                  = EXCLUDED.nbr_p25,
                        nbr_p75                  = EXCLUDED.nbr_p75,
                        nbr_valid_px             = EXCLUDED.nbr_valid_px,
                        nbr_pct_above_threshold  = EXCLUDED.nbr_pct_above_threshold,
                        updated_at               = NOW()
                    """,
                    (
                        mine_id, row["year"],
                        row.get("ndvi_mean"),    row.get("ndvi_median"),
                        row.get("ndvi_std"),     row.get("ndvi_p25"),
                        row.get("ndvi_p75"),     row.get("ndvi_valid_px"),
                        row.get("ndvi_pct_above_threshold"),
                        row.get("nbr_mean"),     row.get("nbr_median"),
                        row.get("nbr_std"),      row.get("nbr_p25"),
                        row.get("nbr_p75"),      row.get("nbr_valid_px"),
                        row.get("nbr_pct_above_threshold"),
                    ),
                )
        conn.commit()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload mine metadata and images to Neon PostgreSQL."
    )
    parser.add_argument(
        "--db", metavar="URL",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres connection string (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--mine", metavar="SLUG",
        help="Upload images for this mine slug only (metadata always synced for all mines)",
    )
    parser.add_argument(
        "--schema-only", action="store_true",
        help="Create/verify tables then exit without uploading",
    )
    args = parser.parse_args()

    if not args.db:
        sys.exit(
            "ERROR: supply --db <url> or set the DATABASE_URL environment variable.\n"
            "Example: export DATABASE_URL='postgresql://user:pass@host/db?sslmode=require'"
        )

    print("Connecting to database…")
    try:
        conn = psycopg2.connect(args.db)
    except Exception as exc:
        sys.exit(f"Connection failed: {exc}")

    try:
        apply_schema(conn)

        if args.schema_only:
            return

        mines = load_mines_csv()
        slug_to_id = upsert_mines(conn, mines)

        upload_images(conn, slug_to_id, selected_slug=args.mine)
        upload_timeseries(conn, slug_to_id, selected_slug=args.mine)
        print("\nDone.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
