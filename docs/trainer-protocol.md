# Trainer protocol

Last updated: 2026-10-29 13:05 EDT

The contract between the trainer (`trainer/`, Python + MLX, run as its own process) and the app's "Train your own
AI" lesson. The app launches one subcommand at a time, reads JSON lines from its stdout, and stops it with
SIGTERM. `trainer/src/neuroscope_trainer/events.py` holds the same schema in code and refuses to emit an event that
does not match it; `trainer/tests/test_events.py` fails if this page leaves out an event or a field.

## Launching

```sh
cd trainer
uv run neuroscope-train [--workdir DIR] <subcommand> [options]
```

| Subcommand | What it does | Typical options |
|---|---|---|
| `bench` | measures this Mac's training speed for each size (about 20 s), saves `bench.json`, and derives a plan for each `--time` | `--time 30m --time 2h --time overnight`; `--plan-only` re-derives from the saved `bench.json` |
| `prepare` | downloads and tokenizes the Wikipedia text the plan needs (resumable), and with `--sft` the instruction-tuning conversations | `--time 30m --sft`, or `--tokens 50e6`, or `--sft-only` |
| `pretrain` | trains a new model from scratch | `--run NAME --time 30m`, or `--preset tiny --tokens 50e6` |
| `sft` | instruction-tunes a finished pretraining run into `NAME-sft` ("flight school") | `--run NAME --time 30m`, or `--steps N` |
| `export` | writes a run as a Hugging Face OLMo 2 folder the neuron viewer opens | `--run NAME-sft [--step N] [--out DIR]` |
| `compare` | checks the app engine's logits against MLX on an exported model (a developer check) | `--export DIR --engine out.json` |

`--workdir` defaults to `~/Library/Application Support/NeuroScope/training` (or `$NEUROSCOPE_TRAIN_DIR`). Inside it:

```
bench.json                  last measurement
tokenizer/                  OLMo 2 tokenizer files
data/wiki/                  manifest.json, val-00000.u32, train-NNNNN.u32 (uint32 token ids)
data/sft/                   manifest.json, train/val shards + .mask files
runs/<name>/                run.json, events.jsonl (a copy of every event), milestones.json,
                            milestones/step-NNNNNNN.safetensors (bf16 weights at each milestone),
                            state/ (latest resumable state), final.safetensors (when finished)
exports/<name>/             the Hugging Face folder
```

A typical lesson runs `bench --time <choice>`, then `prepare --time <choice> --sft`, `pretrain --run R --time
<choice>`, `sft --run R --time <choice>` and `export --run R-sft`. Rerunning `prepare`, `pretrain` or `sft` with
the same arguments continues where it stopped. `pretrain` refuses to reuse a run name with different settings.

## Lines

- stdout carries only protocol events: one JSON object per line, UTF-8, flushed per line. stderr carries
  human-readable logs (show them in a details pane at most; never parse them).
- Every event has `v` (protocol version, currently 1), `event` (its type), `t` (unix time in seconds, float) and
  `cmd` (the subcommand). Fields below are required unless marked optional. Numbers are finite JSON numbers;
  "int" fields are integers.
- Ignore event types and optional fields you do not know; a new required field or a changed meaning bumps `v`.
- The first line is always `hello` and the last is always `done`.

## Stopping and exit codes

SIGTERM (or SIGINT) asks for a clean stop: the trainer finishes the current step, saves a resumable state, emits a
`checkpoint` with `kind` `"resume"`, then `done` with `status` `"stopped"`. A second signal stops at once. Exit
codes: 0 completed, 3 stopped, 1 failed, 2 bad arguments. A failed run emits `error` before `done`.

## Events

### `hello`
First line of every subcommand. `version` (str, trainer version), `protocol` (int, = `v`), `pid` (int),
`args` (object: the parsed command-line options).

### `status`
A phase change worth a sentence on screen. `phase` (str: `"bench"`, `"prepare"`), `message` (str, plain
English), optional `detail` (object).

### `bench`
The measurement. `device` (str, e.g. "Apple M5 Ultra"), `memory_gb` (number: the GPU's recommended working set),
`presets` (array, one object per size: `preset`, `label`, `non_embedding_params`, `total_params`,
`tokens_per_step`, `micro_batch`, `seq_len`, `step_s`, `tok_per_s`, `peak_memory_gb`, `steps_measured`).

### `plan`
What a chosen time turns into, derived from the measured speed. `time_budget_s` (number), `preset` (str: `tiny`,
`small` or `medium`), `tokens` (int: pretraining tokens), `sft_steps` (int), `derivation` (array of str: the
reasoning, one sentence per line, ready to show), `candidates` (array: per size `preset`, `tok_per_s`, `tokens`,
`tokens_per_param`, `fits`). Optional: `time_label` (str), `seq_len`, `tokens_per_step`, `pretrain_steps` (int).

The rule (in `bench.py`): pretraining gets 85% of the time, instruction tuning 7%, samples and checkpoints the
rest; the chosen size is the largest that still sees at least 20 training tokens per non-embedding parameter.
On the machine this was built on, 30 minutes picks `tiny`, 2 hours `small` and overnight (8 hours) `medium`.

### `prepare`
Download and tokenizing progress, about every 2 s. `split` (str: `"wiki"` or `"sft"`), `tokens` (int: tokens
written so far), `target_tokens` (int; 0 while the SFT set is still being read), `docs` (int: documents or
conversations read), `bytes_downloaded` (int), `tok_per_s` (number), `eta_s` (number, seconds; 0 when unknown).

### `progress`
One training step's summary, about once a second. `phase` (str: `"pretrain"` or `"sft"`), `step` (int, steps
done), `total_steps` (int), `tokens` (int, tokens seen), `total_tokens` (int), `loss` (number: mean training loss
over the steps since the last `progress`, in nats per token), `lr` (number), `tok_per_s` (number, smoothed),
`elapsed_s` (number: training time, summed across resumes), `eta_s` (number: remaining training time at the
current speed, milestones not included). Optional `grad_norm` (number).

A fresh model starts at a loss of about ln(100,352) = 11.5 (a uniform guess over the vocabulary).

### `eval`
Loss on held-out text the model never trains on (the first ~1M Wikipedia tokens, or 200 held-out conversations),
at every milestone. `phase`, `step`, `tokens` (as in `progress`), `val_loss` (number).

### `sample`
What the model writes at a milestone, one event per prompt. `phase`, `step`, `tokens`, `prompt_id` (int, stable
across milestones so the app can line samples up), `prompt` (str: the text shown to the learner; for chat this is
the question, without the template), `text` (str: the continuation; it can contain `<|endoftext|>` only if the
model wrote it mid-text, otherwise generation stops there). Optional `kind` (str: `"plain"` for pretraining
continuations, `"chat"` for answers after the chat template).

Milestones fall at 0, 0.5%, 1%, 2%, 5%, 10%, 20%, 35%, 50%, 75% and 100% of the run's steps: dense early, where
the text changes fastest from gibberish to words. Step 0 is the untrained (or, for `sft`, the pretrained-only)
model. Samples use temperature 0.8, top-k 40 and a fixed seed per prompt, so differences between milestones come
from the weights, not the dice.

### `checkpoint`
Something was saved. `phase`, `step`, `tokens`, `path` (str, absolute), `kind` (str): `"milestone"` (bf16 weights at
a milestone, in `milestones/`; a later lesson can open these in the viewer to watch the model grow) or `"resume"`
(the full resumable state, saved at milestones, every 10 minutes and on stop).

### `export`
`path` (str: the exported folder), `files` (array of str), `params` (int: total parameters).

### `compare`
One position of an engine-vs-MLX check. `prompt` (str), `max_abs_diff` (number), `cosine` (number),
`top1_match` (bool), `dtype` (str), `ok` (bool).

### `error`
`code` (str, one of `disk_full`, `network`, `no_data`, `no_checkpoint`, `bad_args`, `nan_loss`, `out_of_memory`,
`internal`), `message` (str, one plain sentence), optional `detail` (str: a traceback for bug reports).

### `done`
Last line. `status` (str: `"completed"`, `"stopped"` or `"failed"`), optional `summary` (object).

## Example

```json
{"v":1,"event":"hello","t":1793291854.469,"cmd":"pretrain","version":"0.1.0","protocol":1,"pid":17307,"args":{"cmd":"pretrain","run":"wiki30","time":"30m","seed":0}}
{"v":1,"event":"progress","t":1793291900.1,"cmd":"pretrain","phase":"pretrain","step":1200,"total_steps":22578,"tokens":9830400,"total_tokens":184958976,"loss":5.91,"lr":0.003,"tok_per_s":118000.0,"elapsed_s":83.2,"eta_s":1484.0,"grad_norm":0.41}
{"v":1,"event":"sample","t":1793291901.0,"cmd":"pretrain","phase":"pretrain","step":1129,"tokens":9248768,"prompt_id":0,"prompt":"The capital of France is","text":" the city of the city of ...","kind":"plain"}
{"v":1,"event":"done","t":1793293500.0,"cmd":"pretrain","status":"completed"}
```
