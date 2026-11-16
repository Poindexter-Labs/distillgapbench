# The judge

In coding and mathematics, the v1 leaderboard applies one more check to every frontier gap. A reviewer model (GPT-6
Astra, reasoning effort high) reads both tasks of the pair. If it marks either task as ambiguous, the pair is vetoed.
The reviewer is Poindexter's internal light-path auto-review, so its prompts are not part of this release.

Without them, `scripts/run_grid.py --judge auto` (the default) runs with `--judge none` and reports the unjudged
score. In v1 the ambiguity veto removed 7 frontier gaps across the whole study, 0 to 3 per researcher, and changed no
rank. The leaderboard's "no judge" view shows those scores.

To use your own reviewer, add `distillgap/vendor/light_path.py` defining:

- `LIGHT_REVIEW_SCHEMA`: a JSON schema whose output includes the boolean `"ambiguity verdict"` and the string
  `"final verdict"`;
- `get_light_review(problem, solution)` (mathematics) and `get_light_review_generic(problem, solution)` (coding),
  each returning `(system_parts, user_parts)` as lists of strings.

Then run with `--judge light_path` and an Azure key. `distillgap/judge.py` sends the request and applies the veto rule.
Report results from a different reviewer as a separate judge rule, not as v1 scores.
