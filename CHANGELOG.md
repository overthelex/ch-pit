# Changelog

## v2026.09 (2026-09-04) -- first public build

- 15,000 items (5,000 per language, de/fr/it) from Fedlex Akoma Ntoso XML
  editions of federal acts; the 500-per-language `core` subset every
  published baseline runs on.
- Scorer, templates, report and core split moved here from
  `services/ch-pipeline/chpipe/bench` (SecondLayer monorepo); this
  repository is now the source of truth for them.
- Oracle 1.000 in all three languages, 0 errors.
- Baselines on `core` via OpenRouter (recite, closed, current, pit, agentic)
  for Claude Haiku 4.5, Claude Sonnet 5, GPT-5.6-terra, DeepSeek V4 Pro;
  `recite_label` / `recite_as_of` stamped on every published item.
