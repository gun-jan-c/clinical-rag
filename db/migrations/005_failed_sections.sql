-- A section whose model reply failed the format twice is saved with review_status 'failed'
-- (schemas.SectionDraft allows it; 001 did not).
alter table brief_sections drop constraint brief_sections_review_status_check;
alter table brief_sections add constraint brief_sections_review_status_check
  check (review_status in ('pending','approved','edited','rejected','failed'));
