# Developing NeuroScope

Last updated: 2026-10-29 13:05 EDT

How the code is laid out, how to build and test it, and the few rules that keep the lessons honest.

## Layout

| Folder | Job |
|---|---|
| `engine/` | Rust: [candle](https://github.com/huggingface/candle) on Metal. Runs OLMo 2 (and Qwen3, kept but unused) with activation capture and in-pass edits; builds a static library. `src/types.rs` holds the shared types |
| `engine/include/neuroscope.h` | the C ABI: the contract between Rust and Swift |
| `engine/examples/lesson_numbers.rs` | measures every number the "Inside a real AI" lessons state (see below) |
| `app/` | SwiftUI + Metal app (SwiftPM). `Model/` state and data, `Render/` the Metal neuron field, `UI/` panels, `Bridge/` the C ABI wrapper |
| `app/Sources/AirlineKit/` | the airline course's tiny network and its seeded synthetic flight ledger. Pure Swift, no Rust, no UI. `Contract.swift` is the public API the app codes against |
| `app/Sources/NeuroScope/Airline/` | the airline lessons' UI: the map, the flight card, the lesson shell, every lesson |
| `app/Tests/AirlineKitTests/` | determinism, a gradient check, the lesson premises, the storm hub, the sealed exam split |
| `trainer/` | Python + MLX: trains a small OLMo 2 shaped model on this Mac ("Train your own AI"): `bench`, `prepare`, `pretrain`, `sft`, `export`, `compare`. Runs as its own process; talks JSON lines (`docs/trainer-protocol.md`) |
| `engine/examples/logits.rs` | dumps the engine's next-token logits for a prompt and a few greedy steps (the trainer's `compare` checks them) |
| `scripts/reference/` | Python reference runs (Hugging Face transformers) the engine is checked against |
| `scripts/stages/` | records the Graduation lesson's model outputs from published OLMo 2 training stages |
| `docs/airline-course.md` | the lesson-by-lesson spec of the airline course |
| `docs/GUIDE.md` | the in-app guide for "Inside a real AI" |
| `docs/trainer-protocol.md` | the trainer's event protocol: the contract the "Train your own AI" lesson codes against |
| `docs/plans/` | designs for lessons not built yet (`train-your-own.md`) |

## Build, run, test

Requirements: macOS 15+, Xcode (full Xcode, not only the Command Line Tools), and a Rust toolchain.

```sh
scripts/build.sh --run                           # engine + app → build/NeuroScope.app, then open it
cd app && swift test --filter AirlineKitTests    # the airline course's engine tests
cd engine && cargo test --release                # engine unit tests
```

If `xcode-select` points at the Command Line Tools, set
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` for `swift` commands (`scripts/build.sh` does this
for you). Metal shaders compile at runtime from source, so no offline `metal` compiler is needed.

Model weights never live in the repo. The app downloads OLMo 2 1B Instruct (~3 GB, a pinned revision in
`app/Sources/NeuroScope/Model/ModelStore.swift`) into
`~/Library/Application Support/NeuroScope/models/` the first time a lesson needs it. The airline course itself
needs no download.

### Checking the engine against a reference

```sh
python3 -m venv scripts/reference/.venv
scripts/reference/.venv/bin/pip install torch transformers
scripts/reference/.venv/bin/python scripts/reference/ref_olmo.py      # writes olmo_fixtures.json + logits
cd engine && cargo test --release --test reference_olmo -- --ignored
```

`NS_OLMO_DIR` and `NS_OLMO_FIXTURES` point the test at another model folder or fixture prefix; `NS_DTYPE=f32`
runs the engine in full precision.

### The trainer

Needs [uv](https://docs.astral.sh/uv/); it creates `trainer/.venv` from the pinned `trainer/uv.lock` (Python 3.13
fetched by uv, never the system Python). Data, runs and exports go to
`~/Library/Application Support/NeuroScope/training` (`--workdir` or `NEUROSCOPE_TRAIN_DIR` to change).

```sh
cd trainer
uv run pytest                                           # unit tests (seconds)
uv run neuroscope-train bench --time 30m                # measure this Mac, print the plan's derivation
uv run neuroscope-train prepare --time 30m --sft        # Wikipedia + instruction-tuning set
uv run neuroscope-train pretrain --run my-model --time 30m
uv run neuroscope-train sft --run my-model --time 30m
uv run neuroscope-train export --run my-model-sft      # → .../training/exports/my-model-sft
```

To check an export against the app's engine (bf16 as the app runs it, and f32 for a tight bound):

```sh
cd engine
cargo run --release --example logits -- --model DIR --out /tmp/eng.json "The capital of France is"
NS_DTYPE=f32 cargo run --release --example logits -- --model DIR --out /tmp/eng32.json "The capital of France is"
cd ../trainer
uv run neuroscope-train compare --export DIR --engine /tmp/eng.json
uv run neuroscope-train compare --export DIR --engine /tmp/eng32.json
```

## Changing the C ABI

Edit these three together: `engine/include/neuroscope.h`, `engine/src/ffi.rs` and
`app/Sources/NeuroScope/Bridge/NeuroEngine.swift`.

## Rules for the airline course

- **No typed numbers.** Every number a lesson shows is computed live from AirlineKit on
  `Architecture.lessonSeed` (42). If copy needs a number, compute it.
- **The world must stay non-additive.** The ledger's delays come from AND-combinations (storm = rain AND wind,
  holiday rush = holiday AND a full plane). An additive world makes direct routes as good as hubs and breaks
  lessons 3–6.
- **The exam is sealed.** Years 8–9 are never read before Exam day. Hub names, comparisons and rulebook scores
  use the check year (7).
- **Motion shows real values.** Animations trace variables the engine computed (passengers, blame, route
  changes); a key press skips to the end state and Reduce Motion cuts straight to it.

## Rules for "Inside a real AI" (OLMo 2 1B)

- Counts in copy come from the engine's geometry (`Geometry.fill`: `{floors}`, `{desks}`, `{allDesks}`,
  `{memo}`); never type them.
- Measured numbers in `Model/Lessons.swift` (desk 11:2962 "French", "La", +1.5, ink 2.8 after floor 12, …) come
  from `cd engine && cargo run --release --example lesson_numbers -- <answer|follow L I|desks2|ink SITE|derive|add|trace|memo>`.
  Re-run it after any engine or model change.
- OLMo 2 is post-norm: desks read the memo raw, and each floor tidies what it *writes* (`FollowPanel`,
  `SlowMotionBar`).
- OLMo 2 has no thinking mode: `ModelSpec.hasThinking` hides every thinking control and pane.
- Remove-concept tryouts span the upper half of the floors (`removeDepthCap` 1.0): French only goes away from
  floor 12 up.
- A finished run rests on its last real word (`AppState.restingIndex`): the closing `<|endoftext|>` frame is
  nearly grey.
- Ready-made changes are only the ones verified to keep the reply coherent on OLMo 2 1B: Remove French and
  Remove brevity. Add (steering) garbled replies on this model, so it lives only in Make your own, behind a
  one-line warning.

## Graduation recordings

Pages 1–5 of the Graduation lesson show recorded outputs, not live ones, so the app ships text instead of nine
model checkpoints. `scripts/stages/record.sh` downloads one published stage at a time (pinned commit), records
the fixed prompts with the app's own engine, then deletes it; `scripts/stages/to_swift.py` turns
`recordings.jsonl` into `app/Sources/AirlineKit/GraduationRecordings.swift`. Phrase counts in the training text
come from the public [infini-gram](https://infini-gram.io) index (`scripts/stages/trainingdata.sh`).
