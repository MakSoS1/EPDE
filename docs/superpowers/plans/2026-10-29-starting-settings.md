# Implementation and verification

1. Build pure metadata-only recommendation rules and wizard state transitions;
   test hosted limits, valid differentiation, no reference leakage, heavy inputs,
   manual edit preservation and approval invalidation.
2. Integrate all four existing settings editors; retain method/protocol meaning,
   rename the old budget to Default, introduce Suggested and the sparkle action.
3. Build a compact reproducible historical metadata index from committed baseline
   runs; preserve old implementation and environment provenance.
4. Audit real comparison record semantics, add explained metric leaders and
   actionable follow-up checks, and test all four method records and missing data.
5. Verify real Streamlit transitions, scientific regressions and four-method
   Compare in Chromium. Include Python 3.13 in exact Cloud dependency CI.
6. Review the combined diff, publish with an expected-parent lease, inspect CI and
   verify the live deployed revision and export after the app-code update.

Root owns the pure recommendation/state helpers and Lab page; independent agents
own the four editors, historical index, and Compare implementation. A final
independent review checks correctness and scientific interpretation.
