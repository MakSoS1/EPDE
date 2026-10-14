# S0 launch gate

Only a new or modified `*.yaml` in this directory triggers the heavy
research workflow. Source-code commits only trigger the inexpensive smoke
checks; failed or timed-out task identities remain visible in the ledger.

GitHub Actions does not support `workflow_dispatch` on a workflow defined
only in this non-default branch, so a launch file is the explicit dispatch.

Only stage S0 (frozen-candidate tests) is enabled initially. Stage S1 and
later require integration and measured feasibility before the launch gate
is relaxed. Absence of S1/S2/S3 results must be reported as NOT RUN.
