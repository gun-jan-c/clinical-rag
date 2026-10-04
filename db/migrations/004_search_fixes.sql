-- Keyword search also indexes the document's own ID (NCT05872620, a PMID, PMC7318331), weighted 'A' so a
-- trial's own chunks outrank papers that only mention its ID. Content keeps the default weight 'D'.
alter table chunks alter column fts set expression as (
  setweight(to_tsvector('simple', split_part(doc_id, ':', 2)), 'A') || to_tsvector('english', content)
);

-- Filtered vector search: by default the HNSW index returns its 40 nearest candidates and the filter
-- (strategy, drug, content type) then removes some, so far fewer than candidate_pool rows came back.
-- Iterative scans keep searching the index until enough rows pass the filter (pgvector 0.8+).
alter function hybrid_search(text, vector, text, text, jsonb, int, int, int)
  set hnsw.iterative_scan = 'relaxed_order';
