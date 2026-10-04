-- US brand names -> generic names, so trials and searches that use a brand name are tagged correctly.
insert into drug_synonyms values
  ('wegovy',   'semaglutide'),
  ('ozempic',  'semaglutide'),
  ('rybelsus', 'semaglutide'),
  ('zepbound', 'tirzepatide'),
  ('mounjaro', 'tirzepatide'),
  ('saxenda',  'liraglutide'),
  ('victoza',  'liraglutide')
on conflict (alias) do nothing;
