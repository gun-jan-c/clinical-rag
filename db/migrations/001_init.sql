create extension if not exists vector;

-- Structured registry facts: filtered with SQL, never embedded
create table trials (
  nct_id          text primary key,
  title           text,
  phase           text[],
  status          text,                        -- as of ingestion date
  sponsor         text,
  enrollment      int,
  first_posted    date,
  start_date      text,
  completion_date text,
  interventions   text[],
  conditions      text[],
  has_results     boolean,
  last_updated    text
);

-- Every source document
create table documents (
  doc_id         text primary key,             -- 'ctgov:NCT…' | 'pubmed:<pmid>' | 'epmc:<pmcid>' | 'label:<set_id>'
  source         text not null check (source in ('clinicaltrials.gov','pubmed','europepmc','dailymed')),
  url            text not null,
  title          text,
  published_date date,
  license        text,
  sections       jsonb not null,               -- {"results": "...", "methods": "..."}
  metadata       jsonb not null default '{}',  -- drugs, year, nct_ids, linked_trials, label effective date, ...
  content_hash   text not null,                -- skip re-chunk/re-embed when unchanged
  ingested_at    timestamptz not null default now()
);

create table chunks (
  chunk_id    text primary key,                -- '<doc_id>:<strategy>:<n>'
  doc_id      text not null references documents(doc_id) on delete cascade,
  strategy    text not null check (strategy in ('fixed','section')),
  section     text,
  content     text not null,                   -- what gets embedded and searched
  context     text,                            -- what the LLM receives (parent section / full table / caption + description)
  token_count int,
  metadata    jsonb not null default '{}',     -- content_type (text|table|figure), drugs, year, source, license, label
  fts         tsvector generated always as (to_tsvector('english', content)) stored
);
create index chunks_strategy_idx on chunks (strategy);
create index chunks_fts_idx      on chunks using gin (fts);
create index chunks_meta_idx     on chunks using gin (metadata jsonb_path_ops);

-- One row per (chunk, embedding model) so models can be compared side by side
create table chunk_embeddings (
  chunk_id  text not null references chunks(chunk_id) on delete cascade,
  model     text not null check (model in ('oai-3-small','oai-3-large')),
  embedding vector(1024) not null,
  primary key (chunk_id, model)
);
create index emb_small_hnsw  on chunk_embeddings using hnsw (embedding vector_cosine_ops) where model = 'oai-3-small';
create index emb_large_hnsw  on chunk_embeddings using hnsw (embedding vector_cosine_ops) where model = 'oai-3-large';

create table drug_synonyms (
  alias   text primary key,                    -- lowercase
  generic text not null
);
-- Seed (architect to verify and extend):
insert into drug_synonyms values
  ('ly3298176','tirzepatide'),
  ('ly3502970','orforglipron'),
  ('ly3437943','retatrutide'),
  ('bi 456906','survodutide');

create table briefs (
  brief_id     uuid primary key default gen_random_uuid(),
  requested_by text not null,
  request      jsonb not null,
  status       text not null default 'generating'
               check (status in ('generating','draft','in_review','approved','failed')),
  model_id     text not null,
  config       jsonb not null,                 -- strategy, embedding model, k, rerank model, data as-of date
  total_input_tokens  int default 0,
  total_output_tokens int default 0,
  total_cost_usd      numeric(10,4) default 0,
  latency_ms   int,
  created_at   timestamptz not null default now()
);

create table brief_sections (
  brief_id       uuid not null references briefs(brief_id) on delete cascade,
  section_key    text not null,
  draft          jsonb not null,
  retrieved_ids  text[] not null default '{}',
  review_status  text not null default 'pending'
                 check (review_status in ('pending','approved','edited','rejected')),
  reviewer_text  text,
  reviewed_by    text,
  reviewed_at    timestamptz,
  primary key (brief_id, section_key)
);

create table audit_log (
  id         bigserial primary key,
  brief_id   uuid references briefs(brief_id) on delete cascade,
  actor      text not null,
  event      text not null,                    -- generated | claim_flagged | approved | edited | rejected
  payload    jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create table usage_daily (
  day    date primary key,
  briefs int not null default 0,
  asks   int not null default 0
);

create table eval_runs (
  run_id     uuid primary key default gen_random_uuid(),
  config     jsonb not null,
  metrics    jsonb not null,
  notes      text,
  created_at timestamptz not null default now()
);

create table eval_results (
  run_id            uuid not null references eval_runs(run_id) on delete cascade,
  question_id       text not null,
  retrieved_doc_ids text[] not null,
  recall_at_5       real,
  recall_at_10      real,
  reciprocal_rank   real,
  answer            jsonb,
  citation_valid    real,
  numbers_grounded  real,
  not_found_correct boolean,
  primary key (run_id, question_id)
);

-- Security: RLS on, no public policies. The app connects server-side as the DB owner.
do $$ declare t text; begin
  foreach t in array array['trials','documents','chunks','chunk_embeddings','drug_synonyms','briefs',
                           'brief_sections','audit_log','usage_daily','eval_runs','eval_results']
  loop execute format('alter table %I enable row level security', t); end loop;
end $$;
