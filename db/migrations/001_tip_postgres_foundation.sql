CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL DEFAULT 'club',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT organizations_type_check CHECK (type IN ('club', 'consultancy', 'demo', 'internal'))
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS organization_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT organization_users_role_check CHECK (role IN ('owner', 'admin', 'analyst', 'viewer')),
    CONSTRAINT organization_users_unique_membership UNIQUE (organization_id, user_id)
);

CREATE TABLE IF NOT EXISTS clubs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    country TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS organization_clubs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    club_id UUID NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT organization_clubs_relationship_type_check CHECK (
        relationship_type IN ('owner', 'client', 'opponent', 'scouting_target', 'partner')
    ),
    CONSTRAINT organization_clubs_unique_link UNIQUE (organization_id, club_id, relationship_type)
);

CREATE TABLE IF NOT EXISTS teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    club_id UUID NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    category TEXT,
    gender TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT teams_gender_check CHECK (gender IN ('male', 'female', 'mixed') OR gender IS NULL),
    CONSTRAINT teams_unique_name_per_club UNIQUE (club_id, name, category, gender)
);

CREATE TABLE IF NOT EXISTS competitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    country TEXT,
    gender TEXT,
    level TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT competitions_gender_check CHECK (gender IN ('male', 'female', 'mixed') OR gender IS NULL)
);

CREATE TABLE IF NOT EXISTS seasons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competition_id UUID NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT seasons_unique_name_per_competition UNIQUE (competition_id, name)
);

CREATE TABLE IF NOT EXISTS team_seasons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    season_id UUID NOT NULL REFERENCES seasons(id) ON DELETE CASCADE,
    external_label TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT team_seasons_unique_link UNIQUE (team_id, season_id)
);

CREATE TABLE IF NOT EXISTS matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_match_id TEXT NOT NULL UNIQUE,
    competition_id UUID REFERENCES competitions(id) ON DELETE SET NULL,
    season_id UUID REFERENCES seasons(id) ON DELETE SET NULL,
    match_date TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT matches_status_check CHECK (
        status IN ('scheduled', 'available', 'processing', 'completed', 'failed', 'archived')
    )
);

CREATE TABLE IF NOT EXISTS organization_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    visibility TEXT NOT NULL DEFAULT 'owned',
    source TEXT NOT NULL DEFAULT 'local_demo',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT organization_matches_visibility_check CHECK (
        visibility IN ('owned', 'shared', 'readonly')
    ),
    CONSTRAINT organization_matches_unique_link UNIQUE (organization_id, match_id)
);

CREATE TABLE IF NOT EXISTS match_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_id UUID NOT NULL REFERENCES teams(id) ON DELETE RESTRICT,
    side TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT match_participants_side_check CHECK (side IN ('home', 'away')),
    CONSTRAINT match_participants_unique_side UNIQUE (match_id, side)
);

CREATE TABLE IF NOT EXISTS match_provider_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    provider_match_id TEXT NOT NULL,
    provider_competition_id TEXT,
    provider_season_id TEXT,
    adapter_version TEXT NOT NULL DEFAULT 'v1',
    source_priority INTEGER,
    data_quality TEXT,
    provider_payload_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT match_provider_links_unique_match_provider UNIQUE (match_id, provider),
    CONSTRAINT match_provider_links_unique_provider_match UNIQUE (provider, provider_match_id)
);

CREATE TABLE IF NOT EXISTS provider_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_link_id UUID NOT NULL REFERENCES match_provider_links(id) ON DELETE CASCADE,
    snapshot_type TEXT NOT NULL,
    snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS storage_objects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    storage_provider TEXT NOT NULL,
    bucket TEXT NOT NULL,
    object_key TEXT NOT NULL,
    file_name TEXT,
    mime_type TEXT,
    size_bytes BIGINT,
    checksum TEXT,
    asset_version INTEGER NOT NULL DEFAULT 1,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT storage_objects_provider_check CHECK (storage_provider IN ('local', 'r2')),
    CONSTRAINT storage_objects_size_check CHECK (size_bytes IS NULL OR size_bytes >= 0),
    CONSTRAINT storage_objects_unique_location UNIQUE (storage_provider, bucket, object_key, asset_version)
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    match_id UUID REFERENCES matches(id) ON DELETE SET NULL,
    requested_by UUID REFERENCES users(id) ON DELETE SET NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    pipeline_version TEXT NOT NULL DEFAULT 'v1',
    error_message TEXT,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT processing_jobs_status_check CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
    )
);

CREATE TABLE IF NOT EXISTS match_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    storage_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    produced_by_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS event_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    provider_link_id UUID REFERENCES match_provider_links(id) ON DELETE SET NULL,
    dataset_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    adapter_version TEXT,
    storage_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    produced_by_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT event_datasets_unique_version UNIQUE (
        match_id,
        provider_link_id,
        dataset_type,
        schema_version,
        adapter_version
    )
);

CREATE TABLE IF NOT EXISTS metric_sets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    metric_family TEXT NOT NULL,
    metric_version TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_ref TEXT,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    storage_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    produced_by_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS quality_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    pipeline TEXT NOT NULL,
    pipeline_version TEXT NOT NULL DEFAULT 'v1',
    status TEXT NOT NULL,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    storage_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    produced_by_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT quality_summaries_status_check CHECK (
        status IN ('pending', 'ready', 'warning', 'failed')
    )
);

CREATE TABLE IF NOT EXISTS ai_coach_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    match_id UUID REFERENCES matches(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_coach_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES ai_coach_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    match_context_storage_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_coach_messages_role_check CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE TABLE IF NOT EXISTS organization_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT organization_settings_unique_key UNIQUE (organization_id, key)
);

CREATE INDEX IF NOT EXISTS idx_clubs_name ON clubs (name);
CREATE INDEX IF NOT EXISTS idx_teams_club_id ON teams (club_id);
CREATE INDEX IF NOT EXISTS idx_matches_match_date ON matches (match_date DESC);
CREATE INDEX IF NOT EXISTS idx_organization_matches_match_id ON organization_matches (match_id);
CREATE INDEX IF NOT EXISTS idx_match_provider_links_match_id ON match_provider_links (match_id);
CREATE INDEX IF NOT EXISTS idx_storage_objects_object_key ON storage_objects (object_key);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_match_id ON processing_jobs (match_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_organization_id_status ON processing_jobs (organization_id, status);
CREATE INDEX IF NOT EXISTS idx_event_datasets_match_id ON event_datasets (match_id);
CREATE INDEX IF NOT EXISTS idx_metric_sets_match_id ON metric_sets (match_id);
CREATE INDEX IF NOT EXISTS idx_quality_summaries_match_id ON quality_summaries (match_id);
CREATE INDEX IF NOT EXISTS idx_ai_coach_sessions_match_id ON ai_coach_sessions (match_id);
CREATE INDEX IF NOT EXISTS idx_ai_coach_messages_session_id_created_at ON ai_coach_messages (session_id, created_at);

DROP TRIGGER IF EXISTS trg_organizations_set_updated_at ON organizations;
CREATE TRIGGER trg_organizations_set_updated_at
BEFORE UPDATE ON organizations
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_users_set_updated_at ON users;
CREATE TRIGGER trg_users_set_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_organization_users_set_updated_at ON organization_users;
CREATE TRIGGER trg_organization_users_set_updated_at
BEFORE UPDATE ON organization_users
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_clubs_set_updated_at ON clubs;
CREATE TRIGGER trg_clubs_set_updated_at
BEFORE UPDATE ON clubs
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_organization_clubs_set_updated_at ON organization_clubs;
CREATE TRIGGER trg_organization_clubs_set_updated_at
BEFORE UPDATE ON organization_clubs
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_teams_set_updated_at ON teams;
CREATE TRIGGER trg_teams_set_updated_at
BEFORE UPDATE ON teams
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_competitions_set_updated_at ON competitions;
CREATE TRIGGER trg_competitions_set_updated_at
BEFORE UPDATE ON competitions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_seasons_set_updated_at ON seasons;
CREATE TRIGGER trg_seasons_set_updated_at
BEFORE UPDATE ON seasons
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_team_seasons_set_updated_at ON team_seasons;
CREATE TRIGGER trg_team_seasons_set_updated_at
BEFORE UPDATE ON team_seasons
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_matches_set_updated_at ON matches;
CREATE TRIGGER trg_matches_set_updated_at
BEFORE UPDATE ON matches
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_organization_matches_set_updated_at ON organization_matches;
CREATE TRIGGER trg_organization_matches_set_updated_at
BEFORE UPDATE ON organization_matches
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_match_participants_set_updated_at ON match_participants;
CREATE TRIGGER trg_match_participants_set_updated_at
BEFORE UPDATE ON match_participants
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_match_provider_links_set_updated_at ON match_provider_links;
CREATE TRIGGER trg_match_provider_links_set_updated_at
BEFORE UPDATE ON match_provider_links
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_provider_snapshots_set_updated_at ON provider_snapshots;
CREATE TRIGGER trg_provider_snapshots_set_updated_at
BEFORE UPDATE ON provider_snapshots
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_storage_objects_set_updated_at ON storage_objects;
CREATE TRIGGER trg_storage_objects_set_updated_at
BEFORE UPDATE ON storage_objects
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_processing_jobs_set_updated_at ON processing_jobs;
CREATE TRIGGER trg_processing_jobs_set_updated_at
BEFORE UPDATE ON processing_jobs
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_match_assets_set_updated_at ON match_assets;
CREATE TRIGGER trg_match_assets_set_updated_at
BEFORE UPDATE ON match_assets
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_event_datasets_set_updated_at ON event_datasets;
CREATE TRIGGER trg_event_datasets_set_updated_at
BEFORE UPDATE ON event_datasets
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_metric_sets_set_updated_at ON metric_sets;
CREATE TRIGGER trg_metric_sets_set_updated_at
BEFORE UPDATE ON metric_sets
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_quality_summaries_set_updated_at ON quality_summaries;
CREATE TRIGGER trg_quality_summaries_set_updated_at
BEFORE UPDATE ON quality_summaries
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_ai_coach_sessions_set_updated_at ON ai_coach_sessions;
CREATE TRIGGER trg_ai_coach_sessions_set_updated_at
BEFORE UPDATE ON ai_coach_sessions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_ai_coach_messages_set_updated_at ON ai_coach_messages;
CREATE TRIGGER trg_ai_coach_messages_set_updated_at
BEFORE UPDATE ON ai_coach_messages
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_organization_settings_set_updated_at ON organization_settings;
CREATE TRIGGER trg_organization_settings_set_updated_at
BEFORE UPDATE ON organization_settings
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
