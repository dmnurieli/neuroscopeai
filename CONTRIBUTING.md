# Contributing to NeuroScope AI

Thanks for helping. A few things keep pull requests easy to review and merge.

- **Start with an issue** for anything bigger than a small fix, so we can shape it together. That goes double for
  anything marked open in [Upcoming additions](README.md#upcoming-additions): say you're on it, and we'll agree the
  scope before you build.
- **Leave `README.md` out of your pull request.** The README is curated by hand and updated as features land. If your
  change adds something it should mention, say so in the pull request description and it will be added when your
  change merges. Pull requests that edit the README need the maintainer's review before they can merge. Spotted a
  typo or a broken link in it? Open an issue.
- **Read [docs/DEVELOPING.md](docs/DEVELOPING.md) first.** One rule matters above the others: every number a lesson
  shows is computed, never typed.
- **Keep each pull request to one change**, and run the tests for the parts you touched:
  - `cd app && swift test --filter AirlineKitTests` for the airline course
  - `cd engine && cargo test --release` for the engine
  - `cd trainer && uv run pytest` for the trainer

By contributing, you agree that your work is licensed under the project's [Apache License 2.0](LICENSE).
