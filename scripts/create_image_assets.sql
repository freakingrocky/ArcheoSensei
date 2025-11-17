-- Schema for storing AI-generated image annotations and references
-- This script creates a dedicated table for lecture images with
-- structured metadata, detailed notes, and highlight areas.

CREATE TABLE IF NOT EXISTS lecture_image_assets (
    id              BIGSERIAL PRIMARY KEY,
    img_url         TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    description     TEXT,
    notes           TEXT,
    lecture_key     TEXT,
    area_description JSONB      NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS lecture_image_assets_lecture_idx
    ON lecture_image_assets (lecture_key);

CREATE INDEX IF NOT EXISTS lecture_image_assets_url_idx
    ON lecture_image_assets (img_url);

CREATE OR REPLACE FUNCTION set_lecture_image_assets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_lecture_image_assets_updated_at ON lecture_image_assets;
CREATE TRIGGER trg_lecture_image_assets_updated_at
BEFORE UPDATE ON lecture_image_assets
FOR EACH ROW EXECUTE FUNCTION set_lecture_image_assets_updated_at();
