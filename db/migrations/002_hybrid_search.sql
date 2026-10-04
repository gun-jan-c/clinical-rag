create or replace function hybrid_search(
  query_text      text,
  query_embedding vector(1024),
  embedding_model text  default 'oai-3-small',
  chunk_strategy  text  default 'section',
  metadata_filter jsonb default '{}'::jsonb,   -- e.g. {"drugs":["tirzepatide"],"content_type":"table"}
  match_count     int   default 20,
  candidate_pool  int   default 50,
  rrf_k           int   default 60
)
returns table (
  chunk_id text, doc_id text, section text, content text, context text, metadata jsonb,
  dense_rank bigint, keyword_rank bigint, rrf_score double precision
)
language sql stable
as $$
with q as (
  -- websearch_to_tsquery ANDs every term, too strict for questions; switch to OR
  select replace(websearch_to_tsquery('english', query_text)::text, '&', '|')::tsquery as tsq
),
dense as (
  select c.chunk_id,
         row_number() over (order by e.embedding <=> query_embedding) as rnk
  from chunks c
  join chunk_embeddings e on e.chunk_id = c.chunk_id and e.model = embedding_model
  where c.strategy = chunk_strategy
    and c.metadata @> metadata_filter
  order by e.embedding <=> query_embedding
  limit candidate_pool
),
keyword as (
  select c.chunk_id,
         row_number() over (order by ts_rank_cd(c.fts, q.tsq) desc) as rnk
  from chunks c, q
  where c.strategy = chunk_strategy
    and c.metadata @> metadata_filter
    and c.fts @@ q.tsq
  order by ts_rank_cd(c.fts, q.tsq) desc
  limit candidate_pool
),
fused as (
  select coalesce(d.chunk_id, k.chunk_id) as chunk_id,
         d.rnk as dense_rank,
         k.rnk as keyword_rank,
         coalesce(1.0 / (rrf_k + d.rnk), 0) + coalesce(1.0 / (rrf_k + k.rnk), 0) as rrf_score
  from dense d
  full outer join keyword k on d.chunk_id = k.chunk_id
)
select c.chunk_id, c.doc_id, c.section, c.content, c.context, c.metadata,
       f.dense_rank, f.keyword_rank, f.rrf_score::double precision
from fused f
join chunks c on c.chunk_id = f.chunk_id
order by f.rrf_score desc
limit match_count;
$$;
