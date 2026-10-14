# Immutable campaign manifests

`planned.json` is generated from a frozen experiment configuration and
the *executing* commit SHA. An individual run is identified by dataset,
variant, noise, data seed, optimizer seed, split hash, config hash, stage and
code SHA. Results use atomically replaced, SHA-256 checked JSON files.

The `resume_plan` summarizer counts all planned identities and treats
missing/corrupt outputs as **incomplete**. Conflicting duplicate results
are rejected. A crash of an Actions VM before artifact upload can still
lose its local files; it must be explicitly rerun, never called complete.
