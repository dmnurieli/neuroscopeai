# Train your own AI — lesson design (phase B)

Last updated: 2026-10-29 13:05 EDT

Phase A built the engine: `trainer/` measures the Mac, downloads Wikipedia, trains a small OLMo 2 shaped model
with MLX, tunes it to answer questions, and exports it in the layout the neuron viewer already reads. Its
contract with the app is `docs/trainer-protocol.md`. This page is the plan for phase B, the lesson a learner
walks through in the app. Nothing here is built yet.

## Who it is for

Someone who finished the airline course. They know a network learns by guessing, being told how wrong it was,
and nudging its routes. They watched Graduation, where a real model went from nonsense to sentences across
Ai2's published training snapshots. This lesson lets them do that themselves, on their own Mac, and keep what
they made.

## The idea in one sentence

Graduation showed someone else's model growing up; here you grow your own, then open it and look inside.

## What the learner sees and does

### 1. Pick a time

One question: "How long can your Mac work on this?" Three buttons: **About 30 minutes**, **About 2 hours**,
**Overnight**. No sizes, no parameter counts, no batch sizes. A small line underneath says the Mac should stay
plugged in and awake (the app holds a power assertion while training).

Why a time and not a size: a learner cannot judge "50 million parameters", but they know whether they have half an
hour. The trainer turns the time into a size for them, and shows its reasoning on the next page.

### 2. The Mac measures itself

The app runs `bench` (about 20 seconds). A short animation fills while it measures: "Timing how fast this Mac can
read." Then one card with the result, written from the `plan` event's `derivation` lines, shortened:

> This Mac reads about 120,000 word pieces a second while learning. In 30 minutes that is about 185 million word
> pieces, enough to train a small model well (36 word pieces for each of its 5 million connections). A bigger
> model would only see 9 per connection, too few to learn much.

The numbers come straight from the event, never typed into the copy (the airline course's rule). A "why?" link
expands the full derivation for curious learners: the 20-tokens-per-parameter rule, named once in plain words as
"roughly how much reading a model of this size needs".

### 3. Fetching the books

`prepare --time <choice> --sft` streams Wikipedia (and Wikibooks) from Ai2's OLMo-mix and turns it into word pieces.
On a good connection this takes under a minute for the 30-minute plan; overnight needs a few gigabytes. Show a
progress bar from `prepare` events, and one honest sentence about where the text comes from: "Articles written by
Wikipedia's volunteers, packaged for training by the Allen Institute for AI." A disk-space error (`disk_full`)
gets its own page that says how much room is needed.

The airline bridge: in the airline course the clipboard held a flight's facts. Here the clipboard is a window of
1,024 word pieces of an article, and the question at every position is "what comes next?"

### 4. Watch it learn (the main event)

`pretrain`. The screen has two halves.

- **The loss line.** A live chart from `progress` events. The y-axis is labeled "minutes off" for one beat,
  then "how surprised it is by the next word piece" (the airline course called loss "minutes off"; here it is
  surprise). A dashed line at 11.5 marks "a blind guess among 100,352 word pieces". The `eval` points sit on the
  same chart as dots: "on articles it has never seen". When the two lines stay together, a one-liner says it is
  learning, not memorizing (the cramming lesson, revisited).
- **The notebook.** At each milestone (`sample` events) a new page appears with the model's continuation of five
  fixed openings ("The capital of France is", "An airplane is", …). Pages stack, newest on top, and the learner
  can flip back. The first real run (the 30-minute plan on a Mac Studio: tiny size, 185M word pieces, 26
  minutes) shows the arc this lesson is about, in its own words, for "The capital of France is":
  - before training: `_ALL compliantanteed.getDeclared) … potato autor marin_regs jumps-bootstrap` (random word
    pieces)
  - after 1M word pieces: `,,. on  was the. (, the and the and are..` (the most common words, no order)
  - after 9M: `known as the "R" is known as "C" or "B".` (phrases, borrowed shapes)
  - after 37M: `often the second most popular city. The French-speaking German and the Dutch are also the largest
    cities in Asia` (sentences, invented facts)
  - after 185M: `Béla. It was the capital of France. In the mid-20th century, the army's army and the army was
    stationed.` (fluent, confidently wrong)

  Loss on unseen articles fell from 11.59 to 7.91 (1M), 5.80 (9M), 4.84 (37M) and 4.28 (185M).

  Because each prompt keeps the same random seed at every milestone, the only thing that changes between pages is
  the model. Say so once.
- Time remaining from `eta_s`, as "about 12 minutes left", never as seconds.
- **Pause** sends SIGTERM; the trainer saves and stops cleanly, and the lesson can resume later (rerunning the
  same command continues). Closing the app does the same.

Milestones are dense early (0.5%, 1%, 2%, 5% of the run) because that is when the text changes fastest. The 30
minute plan produces its first readable phrases within the first two minutes, which keeps attention.

### 5. Flight school

`sft`. The same model is now shown conversations: a question, then a good answer. The airline course's
Graduation already used "flight school" for instruction tuning; reuse the name. Before tuning (step 0), the model
answers "How do airplanes fly?" by continuing the text like an encyclopedia page; after, it tries to answer. The
notebook shows those two side by side. Copy should be honest: a small model will answer confidently and often
wrongly. That is the point to make: answering is a learned habit, separate from knowing. In the first run
(921 steps, 80 seconds), "What is the capital of France?" went from `how, if any, that is? (F. 2006, "What is
the capital?` to `The capital of France is the capital of French-Hungary`, and "Write one sentence about the
ocean." from a row of `#` signs to `The ocean is composed of three different types of ocean:`.

The tuning set is small (about 12,000 English conversations from Tulu 3's OpenAssistant and FLAN sources) and
the step takes about 7% of the chosen time.

### 6. Export

`export` writes the model as a Hugging Face OLMo 2 folder with an `ATTRIBUTION.md` and a `training.json`. The app
copies it into its models folder under a name the learner types ("My first model") and shows the attribution in
plain words (see "Responsible notes").

### 7. Open your own model in the neuron viewer

The payoff. The same viewer from "Inside a real AI" opens the learner's model: fewer floors (6 for tiny, 8 for
small, 16 for medium) and fewer desks per floor, but the same layout. Ask it something and watch the desks
light up. Two things to try, both prepared by the lesson:

- **Find a desk that fires on a word you choose**, as in the OLMo 2 lessons. On a small model the concepts are
  cruder; that is interesting in itself.
- **Compare with your model's earlier self.** Milestone weights are saved at every milestone; exporting one with
  `export --step N` gives the viewer a younger version of the same model. Seeing the same desk before it learned
  anything is a strong picture of what training does.

## How it bridges from the airline course

| Airline course | Train your own |
|---|---|
| the clipboard (inputs) | a window of 1,024 word pieces of an article |
| one gate: how late? | 100,352 word-piece destinations: what comes next? |
| minutes off (loss) | surprise at the next word piece |
| the blame walk and the foggy mountain | the same, millions of times; the learner only watches the line fall |
| cramming (overfitting) | training loss vs. loss on unseen articles |
| Graduation's snapshots | your own snapshots, every few minutes |
| flight school | flight school, on your model |
| naming a hub | finding a desk in your own model |

## What the app needs to build

- A trainer runner: launch `uv run neuroscope-train …` (or a bundled environment, below) as a child process, read
  stdout lines, decode the events in `docs/trainer-protocol.md`, send SIGTERM on pause or quit, hold a sleep
  assertion while it runs.
- A Python environment. Phase A uses `uv` with pinned versions (`trainer/pyproject.toml`, `uv.lock`). For release,
  either ship `uv` inside the app and create the environment on first use (the environment is about 420 MB on disk, most of it MLX and pyarrow),
  or bundle a relocatable Python with the wheels. The first is simpler; decide when packaging.
- "Open a model folder" in the model store, so the viewer can load an export (the engine already loads any OLMo 2
  folder, tied embeddings included; the store currently knows one pinned model).
- Lesson pages in the airline lesson shell, numbers from events only.

## Responsible notes on data and licenses

Say these plainly in the lesson, once, and keep them in `ATTRIBUTION.md` next to the weights:

- The pretraining text is Wikipedia and Wikibooks, as packaged in Ai2's OLMo-mix-1124 (ODC-By). The articles are
  CC BY-SA: free to reuse with credit, and adaptations must be shared alike.
- Whether a model trained on share-alike text must itself be shared alike is an open legal question. We do not claim either
  way. If the learner shares their model, the safe course is to credit the sources, keep the attribution file with
  it, and share it openly.
- The instruction tuning uses two sources from Ai2's Tulu 3 mixture (ODC-By): OpenAssistant (Apache 2.0, written
  by volunteers) and FLAN v2 (Apache 2.0). We left out sources with non-commercial licenses, replies written by
  another company's model, and Tulu's built-in "I am Tulu" identity answers, so the learner's model does not
  inherit someone else's name or terms.
- A model this small makes things up. The lesson should show that happening, not hide it.
- Everything runs on the learner's Mac. Nothing they type, and nothing the model writes, leaves it; the only
  network traffic is downloading the public data and tokenizer from Hugging Face.

## Open questions

- Laptops: the trainer sizes micro-batches from the GPU's memory, but an 8 GB machine has not been tested. The
  30-minute plan on a base laptop will pick the tiny size with fewer tokens; check that its samples still reach
  phrases.
- Thermals: a fanless laptop slows down when hot, so the measured speed overestimates. Re-plan after the first
  few minutes if `tok_per_s` drops well below the bench figure.
- Overnight plans need about 1.3 billion word pieces, about 5 GB of shards plus about 2.2 GB downloaded. Show this before
  starting.
