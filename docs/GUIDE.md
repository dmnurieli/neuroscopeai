# NeuroScope Guide

Last updated: 2026-10-28 13:50 EDT

NeuroScope lets you watch a small AI model at work, change one idea inside it, and see what that does to its answer. You need no background in AI. Follow the four steps in the bar at the top of the app: Ask, Watch, Change, Compare. The "Going deeper" sections after that are optional. They explain how it all works, still in plain words.

## Three scary words, and what they really mean

You will meet three words that sound hard. They are not.

- **Layer**: one floor of a building. The model is a stack of 16 of them.
- **Direction**: one idea, like "French", running through the model like a single highlighter color.
- **Abliteration**: a ten-dollar word (ablation + obliteration) for a simple act: erase one highlighter color, everywhere.

Everything else in this guide grows out of those three.

## The picture to keep in your head

The model is a **16-floor office building**. It reads your question and writes its reply one word at a time.

- Each **layer** is a floor, numbered 0 at the bottom to 15 at the top.
- Each **neuron** is a desk on a floor. Every floor has 8,192 desks, so the building has 131,072 desks in all.
- The **residual stream** is a memo that rides the elevator up all 16 floors. Each floor reads the memo and adds its own notes. The top floor uses the finished memo to pick the next word.

Every idea in this guide is told with this one building. Hold on to it.

---

## Step 1 - Ask

Type a question, or click one of the example questions, then press Ask. While the building is working, the button turns into Stop.

The model is OLMo 2 1B Instruct, made by the Allen Institute for AI (Ai2). It is fully open: its weights, its training data and its training code are all published. It is a small model that runs entirely on your Mac (the download is about 3.0 GB). Nothing is sent anywhere. Because it is small, it makes mistakes. That is fine for exploring, and small mistakes are often the most interesting thing to watch.

You do not need to choose any settings. The app picks them for you:

- It allows long answers, so a reply is not cut off halfway.
- It makes the building choose the same words every time, so asking twice gives the same answer. Without that, you could not tell whether a later change came from you or from chance.
- It picks sensible display settings.

All of these are in the Advanced panel if you ever want to set them yourself.

---

## Step 2 - Watch

The field of dots is the building at work. Each dot is one desk. For each word, the busiest desks on every floor turn colorful:

- **Amber** means the worker is pushing for "more of this".
- **Blue** means the worker is pushing for "less of this".
- **Grey** means the desk is quieter than those right now.

The stronger the color, the harder that worker is pushing right now. Only a few desks per floor light up at once, so you can see which ones are doing the work for this word.

The building writes in **tokens**: word-pieces, not always whole words. For example, it writes "volcanoes" as "volcan" + "oes". Each time it writes one piece, the dots update, so you watch the activity move as the building answers.

- Press the play controls or drag the timeline to go back to any moment. The answer follows along, piece by piece.
- The list of busiest desks shows which workers are working hardest at that moment.
- Click any dot to see what that desk does, in plain words.

After the first answer, the app reseats the desks. Coworkers who get busy at the same moments are moved to sit together, in "neighborhoods". This is the **cluster layout**. Why does the app do this? In the real building, desk numbers are arbitrary: desk 17 and desk 18 have nothing to do with each other. Left in number order, the dots would look like random static. Reseated, you can see teams. Just remember the app drew this seating plan; the building has none (see "Are there brain-like regions?" below).

---

## Slow motion: how one desk triggers the next

In Watch, press "Slow motion" to stretch one word's trip up the 16 floors until you can follow it. The floors become columns, and arrows are drawn from desks on one floor to desks on the floor above.

Desks never call each other. A desk writes a note into the memo, and a later desk reads the memo. So an arrow is not a wire. It is the **direct path**: how much the source desk's note pushed (amber) or held back (blue) the target desk's trigger on this word. Thicker means stronger.

What the arrows do not show: notes the target desk borrowed from earlier words through **attention**, and the part that comes from the word itself. Only the direct path from desks on the floor below is drawn, and only the strongest arrows, so the picture is a sample, not every connection.

Each floor shows an explained percentage: the share of its busiest desks' input that the arrows account for. The rest comes from attention and the word itself. A low number does not mean something is wrong; it means most of that floor's input arrived another way.

Click any desk to ask why it got busy. You will see notes from every floor below it that pushed it.

In real time the whole trip takes about 8 thousandths of a second per word. The slider shows how many times slower than real you are watching.

---

## Follow one desk: its story in seven steps

Slow motion shows many desks at once. Following shows one desk's whole job on one word, one step at a time. Click a desk and press "Follow this desk" in its card (or, in slow motion, "Follow the focused desk"; or Run > Follow Selected Desk, Option-Command-F). Move with Back and Next or the left and right arrow keys. Every step gives a plain sentence, then the formula, then this run's real numbers, so you can redo the arithmetic yourself.

The example below is real, measured on this app: desk #2962 on floor 11, at the word "La" in the model's French answer to "Answer in French, in one short sentence: Why is the sky blue?" ("La raison derrière le bleu ciel est que...").

### 1. Reads the memo

The desk reads the **memo** (the running note that rides the elevator, 2,048 numbers) exactly as it arrives. In this model the floor does no tidying before its desks read the memo. The tidying happens later, on the way out (step 5).

```
input = x        (the memo as it arrives, untouched)
```

- x: the memo arriving at the floor. |x| is its length.
- This run: |x| = 14.19.

### 2. Asks its question

The desk asks: how much of my pattern is in this memo? Its pattern, the **gate row**, is 2,048 fixed numbers set in training. The answer is a dot product: multiply the two lists number by number and add up. Big and positive means "yes, my pattern is here".

```
gate = W_gate[j] · x
```

- This run: gate = +1.52, a clear yes.
- What wakes it up: the app compares the pattern with every word the building knows. The best matches are " French", "French", " french", " France", "France", " Paris", "_vectors", " Jean", " france", "Jean". Nobody labelled this desk; a French desk was found automatically, by comparing how busy it gets at French words (French, France, Paris, 法国, 巴黎, Französisch, Marseille) against the same sentences with another country's word. Floor 11 has by far the biggest gap of any floor.
- Who raised the question: earlier desks whose notes pushed its answer. Floor 7 desk #1910 added +1.30; floor 5 desk #4127 added +0.55; floor 1 desk #5638 added +0.46. These are only the biggest few, and the listed notes together add up to +2.46, which is more than the desk's whole answer of +1.52. That is fine: other inputs pull it back down. The rest of its input comes from earlier words (**attention**, notes borrowed from words already read), from the word itself, and from desks not listed.

### 3. Decides whether to speak

A soft on/off switch, **σ** (sigma), turns the answer into a number between 0 (off) and 1 (on). What passes is **silu**. Why a switch? A yes passes nearly whole and a no is squashed toward zero, so a desk stays quiet on words that are not its business.

```
silu(gate) = gate × σ(gate)        σ(z) = 1 / (1 + e^−z)
```

- This run: σ(1.52) = 0.820, so silu = 1.52 × 0.820 = +1.24.

### 4. How loudly and which way

A second fixed pattern, **up**, asks: how much, and which way? Multiplying gives **h**, the desk's busyness on this word. Its size is how loud the desk speaks; its sign says whether the note is written as is (+) or backwards (−). h is what the dot's color saturation shows. If you edited this desk (off, turned up, held), h is the value after your edit.

```
up = W_up[j] · x        h = silu(gate) × up
```

- This run: up = −0.84, so h = 1.24 × −0.84 = −1.05 (the app multiplies the unrounded numbers). Loud-ish, and written backwards.

### 5. Writes its note

The desk's raw note is its **column** (its fixed message, 2,048 numbers) times h. But the floor does not add raw notes to the memo. It adds up all its desks' notes, rescales that total to a standard size, and multiplies it by the floor's own fixed weights. Why? It keeps each floor's voice at a steady volume, so no floor can shout over the memo. Attention output is tidied the same way.

```
memo += w_out ⊙ m / rms(m)        m = sum over desks of h × W_down[:, j]
one desk's note as it lands = h × c        c = (w_out ⊙ W_down[:, j]) / rms(m)
```

- m: the floor's desk notes added together. rms(m): its typical size, taken at this word.
- w_out: the floor's fixed output weights. ⊙ multiplies number by number.
- c: the desk's column after the tidy-up.
- This run: the tidied column is 1.55 long, so the note is 1.05 × 1.55 ≈ 1.62 long. The memo is 14.19 long, so this one desk out of 8,192 wrote a note about 11% of the memo's length: a loud voice.

What decides where it goes: the message, the column, is the same on every word. Only its loudness h changes from word to word. Who it reaches is fixed by the wiring of later desks' questions.

### 6. Who hears it

Desks never call each other. Later desks read the memo, and a later desk hears this note exactly as much as its own question lines up with the note as it landed.

```
push = h × (W_gate′[k] · c)
```

- k: a later desk. ′ means its floor's numbers.
- This run: floor 13 desk #2479 was pushed +0.70, floor 12 desk #8043 +0.56, floor 12 desk #5694 −0.41 and floor 13 desk #1276 +0.36. Desk #1276 writes French words itself: it promotes " de", " d", " des", " du" and " qu".
- The path: follow the loudest listener, then its loudest listener. Here: floor 11 #2962 → floor 13 #2479 → floor 14 #6083 → floor 15 #920. Click any desk on the path to follow it at the same word.

### 7. Effect on the final word

At the top, the memo becomes a score (a **logit**) for every possible next word. Straight through its own note, how much did this desk raise or lower the chosen word's score?

```
effect = h × (unembed[word] · (w_final ⊙ c)) / rms_final
```

- unembed[word]: the word's score pattern at the top. w_final and rms_final: the last tidy-up before scoring.
- This run: the next word was " raison" (making "La raison"), and the desk raised its score by +0.20. Small, because most of its effect travels through the later desks it reached in step 6, not straight to the top. The panel also shows the other words it pushed toward (amber): " sky" +0.24, " lum" +0.20 and " reason" +0.14, and the words it pushed away from (blue), such as " couleur" at −0.14.

---

## Step 3 - Change

Here you take one idea out of the building and see what happens. There are two ready-made cards, Remove French and Remove brevity, plus "Make your own".

- **Remove** puts a filter in the elevator that erases one highlighter color on every floor, so no floor can see that idea. Remove French, and a building that was asked to answer in French answers in English. Remove brevity, and a building that was asked for "at most five words" answers in a full sentence.
- **Add** is the opposite move: someone on one floor inks that highlighter onto every memo passing by, so the floors above read it as if you had asked for it. On this small model adding mostly garbled the reply (see "What we measured" below), so there is no ready-made Add card. You can still try it under "Make your own".

### How the app finds the highlighter

An idea like "French" is not stored at one desk. It is a **direction**: one highlighter color running through the memo. To find it, the app sends matched pairs of questions through the building, the same question twice, once "...in French" and once "...in English". Then it looks at what is consistently different in the memos. That difference is the French highlighter.

While the app does this, it tells you what it is doing in plain words. When it is done, the field shows a concept map: the desks that care most about the idea light up. A "your change" row of chips spells out exactly what will be applied.

### Why the app tests floors for you

The ink shows up on many floors, but erasing or adding it only works on some. On this model, the French highlighter removes French only when it is taken from after floor 12 or higher. Taken from after floor 11 or lower, it leaves the reply in French. Add the ink too strongly high up and the reply breaks into repeated words. So the app holds tryouts: it tries the likely floors on fresh questions it has never used before, and keeps the floor that removes the idea without breaking the reply. You can still pick the floor yourself in Advanced.

### A worked run: Remove French

- The question: "Answer in French, in one short sentence: What do dogs eat?"
- Before the change, the building answers: "Les chiots mangent de la nourriture fourragère."
- The app tried floors on fresh questions. It took about 10 seconds.
- After the change, the building answers: "Dogs eat mainly meat, with fruits, vegetables, and bones as occasional treats."

### Turning off one desk

You can also click a single dot and use Turn off or Turn up on its card. Expect small effects. Desks share jobs, so one desk is rarely the home of a whole idea. Most single-desk changes do nothing you can see (see "Why one desk rarely matters" below).

When you are ready, press Try it.

---

## Step 4 - Compare

The app asks your question again with your change in place, and shows the two answers side by side, Before and After.

- A sentence tells you what happened, for example "The answer changed starting at word 12", followed by the first difference. If the answers match, it says so and suggests a stronger change or a different question.
- The field is colored by what changed. **Red** means a desk worked harder after your change. **Blue** means it worked less.
- Try another change takes you back to Step 3 with the same question. Start over clears everything.

### Why an answer sometimes does not change

There are three common reasons.

- The change was too weak to matter.
- The question never used the idea. Removing French from a question that was never going to be answered in French changes nothing.
- The building was not leaning on that idea in the first place.

A different question that really uses the idea usually shows the effect.

### Side effects

Erasing one color can smudge others. Read the After answer for broken grammar, repeated words or an empty reply. If you see those, the change was too strong. Then try a question that has nothing to do with the idea. A good change leaves that one alone.

---

## The math, step by step

The four steps tell you what a change does. This chapter shows how the numbers inside the building actually change, so you can check the app's arithmetic yourself and then build edits nobody has made a button for.

Every change in the app has a "Show the math" panel. It always goes in the same order: one plain sentence, then the formula, then the real numbers from the run you just did. This chapter teaches the same things in the same order. You need only multiplying, adding and one square root.

## The memo is a list of 2,048 numbers

The memo riding the elevator is not words. It is a **vector**: a list of numbers, here exactly 2,048 of them. There is one memo per word-piece, and you can read it at each of the 17 sites (site 0 below floor 0, then one after every floor). We call it x.

```
x = (x1, x2, x3, ..., x2048)
```

- x: the memo at one site, for one word-piece.
- x1, x2, ...: its numbers, called entries. Entry 1 is the first number, entry 2,048 the last.

A real memo, at site 13 (after floor 12) on the reply word "La" of the French sky answer, starts like this (measured):

```
x = (0.025, 0.186, 0.077, -0.342, -0.551, -0.157, ...)   2,042 more numbers follow
```

For the worked examples we use a toy memo with just three numbers, so you can do every step in your head: x = (2, 5, 1). Everything works the same way with 2,048 numbers; there is just more adding.

## Dot product: how much two lists agree

The **dot product** of two lists multiplies them number by number and adds up the results. It is written with a dot, a·b.

```
a·b = a1×b1 + a2×b2 + ... + a2048×b2048
```

- a, b: two lists of the same length.
- ×: ordinary multiplication; the result is one single number, not a list.

Big and positive means the two lists point the same way. Near zero means they have nothing to do with each other. Negative means they point opposite ways.

## Length: how big a list is

The **length** of a list (also called its **norm**, written |x|) is the dot product of the list with itself, then a square root. It is Pythagoras with more sides.

```
|x| = √(x·x) = √(x1² + x2² + ... + x2048²)
```

- |x|: one number, always zero or more.
- x1²: x1 times itself.

Toy: |(2, 5, 1)| = √(4 + 25 + 1) = √30 ≈ 5.48. The real memo above has |x| = 19.74 (measured).

## The highlighter, d

The highlighter is a list of 2,048 numbers too, called d. The app builds it from your two sets of matched questions, A (with the idea, "...in French") and B (without it, "...in English"). This is the **difference of means** from Step 3.

```
d = mean(memo for A prompts) - mean(memo for B prompts)
```

- mean(...): add the memos entry by entry and divide by how many there are. The result is one "average memo".
- memo for A prompts: the memo at one chosen site, read at the last word-piece of each A question (the moment just before the building starts its reply).
- d: what the A memos have, on average, that the B memos do not. That leftover is the French highlighter.

It is measured separately at each site, so there is one candidate d per site.

## Making it a pure direction, d̂

The size of d depends on how loud the idea is at that site. To separate "which way" from "how much", divide d by its own length. The result, d̂ (say "d-hat"), has length exactly 1. It is a **unit vector**: a pure direction.

```
|d| = √(d·d)
d̂   = d / |d|        (divide every entry of d by |d|)
|d̂| = 1
```

- |d|: the highlighter's natural strength at that site. For French at site 13, |d| = 12.67 (measured).
- d̂: the same direction as d, scaled to length 1.

The first six entries of the real French d̂ at site 13 are (measured):

```
d̂ = (-0.006, -0.004, -0.007, -0.002, -0.025, -0.023, ...)
```

The toy highlighter is d̂ = (0, 0.6, 0.8). Check its length: √(0 + 0.36 + 0.64) = √1 = 1.

## Ink: how much highlighter a memo carries

**Ink** is how much of the highlighter color a memo carries. It is the dot product of the memo with d̂.

```
ink = x·d̂
```

- ink: one number. Positive means the memo leans toward A (French). Negative means it leans toward B (English). Zero means it carries none of this color.

Toy, step by step:

```
x·d̂ = 2×0 + 5×0.6 + 1×0.8
    = 0   + 3.0   + 0.8
    = 3.8
```

Real: the memo for "La" at site 13 has ink x·d̂ = 4.31 (measured). The first six entries alone give 0.025×(-0.006) + 0.186×(-0.004) + 0.077×(-0.007) + (-0.342)×(-0.002) + (-0.551)×(-0.025) + (-0.157)×(-0.023) ≈ +0.02; the other 2,042 products bring it to 4.31. A single word's ink can be noisy. The per-floor average over the whole reply, which the app's ink chart shows, is the steadier signal.

Notice how the ink compares with the memo's length: 4.31 out of 19.74. French is a clear slice of what the memo carries, but most of the memo is other things.

## Ink builds up floor by floor

On a normal run (French vs English, the sky question, ink measured along the site-13 highlighter and averaged over every word of the reply, measured):

- Up to after floor 10: around zero (between -0.5 and +0.03)
- After floor 11: 0.40
- After floor 12 (site 13): 2.80
- After floor 13: 3.90
- After floor 14: 4.20
- At the top (site 16): 3.92

So on this model the building does not write French in gradually. It paints most of it on by floor 12: the ink sits near zero on the low floors, barely stirs after floor 11, then jumps to 2.80 after floor 12 and stays roughly there to the top.

## Erase: Remove, or abliteration

Remove takes out exactly the ink, and nothing else.

```
x_new = x - (x·d̂) d̂
```

- (x·d̂): the ink, one number.
- (x·d̂) d̂: d̂ with every entry multiplied by the ink. This is the part of the memo that points along the highlighter.
- x_new: the memo with that part subtracted, entry by entry.

Toy, step by step, with x = (2, 5, 1) and d̂ = (0, 0.6, 0.8):

```
1. ink           x·d̂ = 3.8
2. the color part     3.8 × (0, 0.6, 0.8) = (0, 2.28, 3.04)
3. subtract           (2, 5, 1) - (0, 2.28, 3.04) = (2, 2.72, -2.04)
4. check              x_new·d̂ = 2×0 + 2.72×0.6 + (-2.04)×0.8
                             = 0 + 1.632 - 1.632 = 0
```

Two things to notice. The new ink is exactly 0, so the color is gone. And the first number, 2, did not change, because that entry never pointed along d̂ (its d̂ entry is 0). In general, every part of x that is not along d̂ stays exactly as it was. That is what makes erasing a careful edit, not a blunt one.

Real: in the erased run, the memo at site 13 for the first reply word "The" arrives with ink 2.77, so the filter subtracts 2.77 × d̂. Its first six entries become:

```
(-0.101, 0.137, -0.363, -0.035, -0.871, -0.050)
  - 2.77 × (-0.006, -0.004, -0.007, -0.002, -0.025, -0.023)
= (-0.083, 0.147, -0.344, -0.030, -0.805, 0.014)
```

The ink after the filter is 0.002, not exactly 0 (see bf16 below). The length barely moves, from 18.43 to √(18.43² - 2.77²) ≈ 18.22, because only the French slice was taken away.

## Why the filter runs on every floor

The filter sits at all 17 sites: entering floor 0, and after every floor. Why not just once? Because the floors keep writing French back in. On the same French vs English test (the sky question, measured):

- Normal run: ink after floor 12 (site 13) is 2.80, and at the top 3.92.
- With the highlighter erased everywhere: ink arriving at site 13, before that site's filter, is still 2.69. Floor 12 rebuilt that much from scratch, in a single floor. At site 14 it arrives at 0.95, and at the top at 1.01.
- After every filter: 0.001 to 0.003.

Erase only at the bottom and the floors above would write French back in, and it would pile up again toward 3.92. Erasing at every site keeps the pile from ever forming.

Why 0.001 to 0.003 and not exactly 0? The app stores numbers in **bf16**, a short number format that keeps only about 3 significant digits to save memory and time. Each step rounds a little, so "after the filter" reads about 0.001 to 0.003, not a perfect 0. The toy example came out at exactly 0 only because we did it by hand.

## Add: steering

Add inks the color on at one floor.

```
x_new = x + α d̂
```

- α (say "alpha"): how much ink to add, one number.
- α d̂: d̂ with every entry multiplied by α.

Because d̂ has length 1, the ink goes up by exactly α: x_new·d̂ = x·d̂ + α×(d̂·d̂) = x·d̂ + α.

Toy, with α = 2: x_new = (2, 5, 1) + 2 × (0, 0.6, 0.8) = (2, 6.2, 2.6). Ink: 0 + 3.72 + 2.08 = 5.8, which is 3.8 + 2. Right on.

How big is α? By default it is 1.5 × |d|: one and a half times the highlighter's own natural strength. The app tries 1.0, 1.5 and 2.0 times |d|. With |d| = 12.67 at site 13, that is α = 12.7, 19.0 and 25.3. Too little and nothing happens; too much and the reply turns into nonsense. A negative α pushes toward B instead.

## Desk edits: one desk's number

A desk (neuron) on a floor reads the memo and produces one number, its output h.

```
h = silu(gate·x) × (up·x)
```

- gate, up: two fixed lists of 2,048 numbers that belong to this desk (its wiring). gate·x and up·x are dot products.
- silu(z): a smooth on-off switch, z / (1 + e^(-z)). Big positive z passes through almost unchanged; big negative z gives almost 0.
- h: how busy the desk is on this word. This is what colors its dot.

The desk edits replace h:

```
turn off:  h' = 0
turn up:   h' = 3h
hold:      h' = v      (v is a value you pick)
```

The desk then writes h times its own column of W_down (the **down projection**) into the memo. That column is a fixed list of 2,048 numbers; its length is called out_norm. So changing h by Δh (say "delta h", the change) changes the desk's raw note by:

```
Δh = h' - h
change to raw note = Δh × (this desk's column of W_down)
size of change = |Δh| × out_norm
```

The floor then rescales all its desks' notes together on the way out (see "Follow one desk", step 5), so out_norm is a good guide to how big an edit is, not an exact figure.

A made-up example: a desk with h = 0.8 and out_norm = 0.5. Turn it off, so Δh = -0.8, and the raw note moves by 0.8 × 0.5 = 0.4. Next to a memo about 14 long (the size entering floor 11, measured at the word "La"), that is a small nudge, about 3%. This is the arithmetic behind "one desk rarely matters". To see what a desk edit does to an idea, take the dot product of its column with d̂: the ink changes by Δh × (column·d̂), before the floor's rescaling.

## Slow motion arrows

The arrow from a source desk on one floor to a target desk on a later floor is the **direct path**: how much the source's note pushes the target's trigger (its gate·x).

```
arrow = a_src × (W_gate row of target · c_src)
c_src = (w_out ⊙ W_down column of source) / rms(m)
```

- a_src: the source desk's output h on this word.
- c_src: the source's column after its floor's tidy-up on the way out, the same c as in "Follow one desk", step 5.
- W_gate row of target: the target's gate list. The target reads the memo as it arrives, so there is no tidy-up on the way in.
- w_out: the source floor's fixed output weights. ⊙: multiply number by number but do not add up, so the result is still a list.
- rms(m): the typical size of the source floor's summed desk notes at this word. Dividing by it is the "rescale to a standard size" step.

Read it as: what the source wrote (as it landed in the memo), times how much the target's gate cares about exactly that.

Each floor's explained percentage adds up the arrows from the busiest 12 source desks and compares that with the target's whole input. Measured on this model at the word "La": about 4-10% on most floors, 16.7% on floor 2, 16.9% on floor 12 and 20.9% on floor 13. On most floors, the other 90% or more arrives through attention and earlier floors. So the arrows are a true but small part of the story.

## A few facts that shape every experiment

- On this model, removal works only when the highlighter is taken from after floor 12 or higher. Highlighters taken from after floor 11 or lower leave the reply in French, even though they separate the two question sets well. That is why the app holds tryouts on fresh questions instead of trusting the cleanest floor.

---

## Make your own edit

Now you can build an edit nobody made a card for. In Step 3, choose the "Make your own" card. It walks you through the same recipe the built-in cards use.

- **Write two phrases.** One has the idea (for example "Answer in Spanish"), one is the same request without it ("Answer in English"). Add a test question to try the edit on.
- **Expect a partial result.** On this small model, home-made changes often work only partly. Removing Spanish, for example, turned "¿Cómo eruptan las volcánicas?" into "How do volcanoes erupt?": English, but the building only repeated the question. Add often garbles the reply. Read the whole After reply.
- **The app makes matched pairs.** It puts each phrase in front of the same set of practice questions: phrase A + question 1 against phrase B + question 1, and so on. Only the phrase differs, so the difference in the memos is the idea.
- **The app finds the highlighter.** At each site, d = mean(A memos) - mean(B memos), measured at the last word of each prompt, then d̂ = d / |d|.
- **Find the floor, or pick it.** Let the app hold tryouts on fresh questions and keep the floor that works without breaking the reply, or pick a site yourself. Picking skips the tryouts.
- **Choose erase or add.** Erase applies x - (x·d̂) d̂ at all 17 sites. Add applies x + α d̂ at your one site; start with α = 1.5 × |d| and go up or down from there.
- **Read the ink chart.** Compare shows ink per floor for Before and After. After an erase, the After line should sit near 0 everywhere (about 0.001 to 0.003, because of bf16). After an add, it should jump by α at your floor and carry on above it. If the ink changed but the answer did not, the building was not leaning on that idea for this question; try one that uses it.

Open "Show the math" at any point to see these formulas filled in with your own run's numbers.

---

## Going deeper

The rest of this guide explains how it works. You can skip it and come back later.

## Are there brain-like regions?

Short answer: no, not the way a brain has them. The "regions" you may have seen in other tools are drawn by a computer, not found inside the model.

**No seating plan.** In a real brain, place matters. Neighboring cells in the part of the brain that handles sight respond to neighboring spots in what you see, and sight lives in one physical place. The building has nothing like that. Desk 17 and desk 18 on a floor have no relationship at all. You could shuffle all 8,192 desks on a floor, as long as you move each desk's three sets of wiring (called `gate`, `up` and `down_proj`) along with it, and the building would compute exactly the same thing. Because desk numbers mean nothing, a desk's spot in a plain grid tells you nothing.

**Height is the real organization.** What does matter is the floor. Roughly:

- Low floors deal with the word-piece itself and the nearby grammar.
- Middle floors build bigger ideas: topic, meaning, what you are asking for.
- High floors turn all of that into a guess at the next word.

Two well-known pieces of evidence back this up. **Probing studies**, which test what each floor "knows", found that the floors of a model called BERT handle classic language tasks in rough order: parts of speech first, meaning later (Tenney et al. 2019). And the **logit lens**, which peeks at each floor's best guess of the next word, showed that in GPT-2 those guesses get closer to the final answer as you go up (nostalgebraist 2020; later refined by the Tuned Lens paper). NeuroScope's logit lens shows the same thing for this model. These are tendencies, not hard walls between floors.

**The neighborhoods are computed.** Teams of desks that get busy together (for example on code, or on French) really do exist, but their desk numbers are scattered all over the floor. So the app measures which desks get busy at the same moments and seats them together. The teams are real. The seating is a picture we drew, not anatomy.

**A seating plan only appears if you train for it.** TopoLM (Rathi et al. 2024, ICLR 2025) gave each desk a fixed spot on a flat sheet and, during training, rewarded neighbors for behaving alike. Only then did brain-like, place-based language regions appear. Ordinary models, this one included, are not trained that way.

---

## Why one desk rarely matters

If ideas do not live in places, where do they live? The best current answer: in directions in the memo. Each word-piece carries a memo of 2,048 numbers up the elevator, and every floor reads it and adds to it. A direction is one pattern across those numbers: one highlighter color.

**Desks share jobs.** The building needs to track far more ideas than it has desks. So it cheats, cleverly. One desk helps with several unrelated topics, and one topic is spread over many desks. This is called **superposition**. Elhage et al. 2022 (Toy Models of Superposition) showed that small networks pack many ideas into fewer slots by giving each idea its own direction, with the directions overlapping a little. The cost is a bit of crosstalk. The gain is far more room.

**One desk, many topics.** Because of this, a single desk usually reacts to several unrelated things. This is called **polysemanticity** ("many meanings"). The desk is not the unit of meaning. The idea is, and each idea is spread over many desks.

**Untangling the jobs.** To pull ideas apart again, researchers train a **sparse autoencoder**: a helper that rewrites each memo as a mix of just a few entries from a big "dictionary" of directions, each ideally meaning one thing. Bricken et al. 2023 (Towards Monosemanticity) turned one floor of 512 desks into thousands of mostly single-meaning ideas. Templeton et al. 2024 (Scaling Monosemanticity) scaled this to millions of ideas in a large commercial model.

**So why does unplugging one desk change so little?** An idea is spread across many desks, and part of it also lives in the building's other machinery (the **attention** parts, which let each word-piece look back at earlier ones). Unplugging one desk removes only a small slice of the idea, and other desks often make up for it. That is why Step 3 works with highlighter colors, not single desks.

---

## How abliteration is actually done

**Abliteration** does not delete desks. It removes one highlighter color from the memo, so the building can no longer use that idea. The method comes from Arditi et al. (2024), who showed that one behavior of chat models (refusing requests) is controlled by a single direction. The recipe works for any idea you can contrast, and NeuroScope runs it on harmless ones.

Here is the recipe, step by step.

- **Send matched pairs.** Run two matched sets of questions through the building. Set A shows the behavior ("in French") and set B does not ("in English"). Read the memo at the last word-piece of the question, the moment right before the building starts answering, at every checkpoint between floors.
- **Average and subtract.** At each checkpoint, average the memos from set A, average the memos from set B, and subtract one from the other. This is called the **difference of means**. It gives one candidate highlighter per floor. This is what Compute direction does in Advanced.
- **Choose the floor.** A score called **Cohen's d** measures how cleanly the two sets separate on each floor. Higher means cleaner. But the cleanest floor is not always where an edit works best. Treat the top of the chart as a starting point, and try nearby floors on questions that were not in your lists.
- **Erase it everywhere.** Take that color out of the memo at every checkpoint, from the very bottom (right after the question is turned into numbers) to the top floor. In math, x ← x − (x·d̂)d̂: find how much of the color the memo has, and subtract exactly that much. Abliterate does this while the building runs. Why everywhere? Erase it on only one floor and the floors above write it back.
- **Bake it in (optional).** People who want the change to be permanent rewrite the building's wiring instead, so it can never write that color into the memo at all. They change every part that writes to the memo: the part that turns words into numbers, each attention output and each desk-floor output (called the **down projection**), using W ← W − d̂d̂ᵀW. The result behaves like the live filter but needs no filter at run time. NeuroScope keeps its edits live, so you can always undo them.
- **Check the result.** First, check that the behavior really changed on fresh questions. Then check unrelated questions for damage: broken grammar, repeated words, empty answers. A good highlighter changes one thing and leaves everything else alone.

**What we measured on OLMo 2 1B Instruct** (French vs English, 14 matched pairs per side):

- The two sets separated most cleanly after floor 3 (Cohen's d 21.7), and also after floor 7 (20.6) and floor 8 (20.1). After floor 12 the score is lower, 12.6.
- Erasing the color taken from after floor 11 or lower left the reply in French, on all 12 floors tried. Erasing the color taken from after floor 12 or higher removed French. The tryouts picked the one from after floor 12.
- On a held-out check of 16 French requests (14 that were never in any list, plus the 2 tryout questions), all 16 answered in French before the change. After erasing, 15 of 16 answered in English. 5 were clean English, 10 were English with a few French words slipping in (for example "Ice floats because its densité relative to water is less than one."), and 1 stayed mixed from the first word. Plain English questions stayed English and nearly unchanged.
- Adding French, the opposite move, was weak. Its best floor, after floor 11, turned English answers into broken French-English ("Les volcanes eruptent souvent, mais pas toujours, ..."). Adding at floor 12 or higher at 1.5 times the color's own strength broke the reply into repeated words.
- Adding pirate talk was weak and garbled ("Ah, young scarrle! Heer heer..."). Adding enthusiasm looked right on short replies ("And now for a totally wild and awesome fact about these leafy, totally epic..."), but on a full-length reply it ran away into one repeated letter ("BLOWZZZZZZ..."). Adding "brief" went from no effect straight to an empty reply. So the app has no ready-made Add cards on this model.
- Remove brevity works: taken from after floor 9, it turned the one-word replies to 12 of 13 fresh "at most five words" questions into sentences (the 13th, magnets, stayed one word: "Magnetism."), for example "Volcano" into "Volcanoes erupt by releasing molten rock, gases, and ash from the Earth's interior." In the app, "How does the internet work?" went from "How" to "The internet works by connecting computers and devices."

The lesson: the cleanest floor is not where the edit works. Here the two sets separate best after floor 3, yet removal only works from after floor 12 up. Try floors beyond the cleanest one, and do not trust the cleanest floor alone. That is exactly why the app holds tryouts for you.

## Steering: adding a color

**Steering** is the adding version: x ← x + α·d̂ at one floor. Someone on that floor inks the color onto every memo going by, which pushes the answer toward set A. A negative amount (α below zero) pushes toward set B instead. It was introduced as **activation addition** (Turner et al., 2023) and refined as **contrastive activation addition**, which averages over many matched pairs (Panickssery et al., 2023). Too much ink breaks the reply into nonsense, so start near the color's own natural strength.

---

## Limits

- OLMo 2 1B is small. Its ideas are cruder and more tangled than a big model's, so its highlighters are noisier and edits break fluency more often.
- The difference of means catches whatever differs between your two lists, including accidents of wording. Badly matched lists give a misleading highlighter.
- One highlighter is a simplification. Many behaviors use several colors at once, or change with context, and one edit can have side effects you only see on other questions.
- The cluster layout is a picture. Two desks seated together got busy together on the questions measured. That does not prove they share one purpose.

---

## The Advanced panel

Everything the four steps decide for you is still within reach. Turn on View > Show Advanced (option-command-A). The panel holds:

- Make your own: write two matched question lists (A has the idea, B is the same without it), compute the highlighter, and read the separation chart for each checkpoint (Cohen's d; higher is cleaner).
- Site, strength and mode: pick the floor to edit and how strongly, instead of the automatic choice, and steer toward A (negative values steer toward B).
- Generation: the longest allowed answer in tokens, same-words-every-time (greedy) or varied (sampled) choice of words, and a seed that makes varied runs repeatable.
- Display: color mode, layout, attention heads, intensity, fade, dot size, and the logit lens (each floor's best guess of the next word).
- Desk and head editing: turn off, scale or pin chosen desks; turn off or scale attention heads.

A **site** is a checkpoint where you can read the memo. Site 0 is right after the question is turned into numbers, before floor 0. Site l+1 is the memo as it leaves floor l. So there are sites 0 to 16.

---

## Words you just learned

- **Layer**: a floor of the building; 16 of them, numbered 0 to 15.
- **Direction**: one idea, like "French", running through the memo as a single highlighter color.
- **Abliteration**: erasing one highlighter color on every floor, so the building cannot use that idea.
- **16-floor office building**: the picture this guide uses for the model, which reads your question and writes its reply one word at a time.
- **Neuron**: a desk on a floor; 8,192 per floor, 131,072 in all.
- **Residual stream**: the memo that rides the elevator up all 16 floors, gaining notes from each one.
- **Amber**: in Watch, a desk pushing for "more of this".
- **Blue**: in Watch, a desk pushing for "less of this"; in Compare, a desk that worked less after your change.
- **Grey**: a quieter desk, not among the busiest on its floor for the current word.
- **Token**: a word-piece the building writes, like "volcan" + "oes".
- **Cluster layout**: desks reseated so coworkers who get busy together sit together; a picture, not a real seating plan.
- **Remove**: a filter in the elevator that erases one highlighter color on every floor.
- **Add**: someone on one floor inking a highlighter color onto every passing memo.
- **Red**: in Compare, a desk that worked harder after your change.
- **Probing studies**: tests of what each floor "knows".
- **Logit lens**: peeking at each floor's best guess of the next word.
- **Superposition**: desks sharing jobs; one desk helps with several topics, and one topic spreads over many desks.
- **Polysemanticity**: one desk reacting to several unrelated things.
- **Sparse autoencoder**: a helper that untangles the memo into a few single-meaning ideas.
- **Attention**: the parts of the building that let each word-piece look back at earlier ones.
- **Difference of means**: average the "with" memos, average the "without" memos, subtract; what is left is the highlighter.
- **Cohen's d**: a score for how cleanly two sets of memos separate on a floor; higher is cleaner.
- **Down projection**: the wiring that carries a floor's desk work back into the memo.
- **Steering**: adding a highlighter color at one floor to push the answer toward an idea.
- **Activation addition**: the first published name for steering.
- **Contrastive activation addition**: steering with a highlighter averaged over many matched pairs.
- **Arrow (direct path)**: how much one desk's note pushed or held back a later desk on this word.
- **Site**: a checkpoint where the memo can be read; site 0 is before floor 0, site 16 is after the top floor.
- **Vector**: a list of numbers; the memo is one with 2,048 numbers.
- **Dot product**: multiply two lists number by number and add up; one number that says how much they agree.
- **Length (norm)**: |x|, the square root of x·x; how big a list is.
- **Unit vector (d̂)**: the highlighter divided by its own length, so its length is 1; a pure direction.
- **Ink**: x·d̂, how much of a highlighter color one memo carries.
- **Erase formula**: x - (x·d̂) d̂; removes exactly the ink and leaves everything else unchanged.
- **α (alpha)**: how much ink Add puts on; 1.5 × |d| by default.
- **bf16**: the short number format the app stores numbers in, about 3 significant digits, which is why erased ink reads 0.001 to 0.003, not exactly 0.
- **W_down column (out_norm)**: the list a desk writes into the memo, and its length; a desk's change moves its raw note by |Δh| × out_norm, before the floor's rescaling.
- **silu**: the smooth on-off switch inside each desk.
- **Explained percentage**: the share of a floor's busiest desks' input that the slow-motion arrows account for; about 4-10% on most floors.

---

## References

- Tenney, Das, Pavlick. *BERT Rediscovers the Classical NLP Pipeline*. ACL 2019. arXiv:1905.05950.
- nostalgebraist. *interpreting GPT: the logit lens*. LessWrong, 2020-08-31.
- *Eliciting Latent Predictions from Transformers with the Tuned Lens*. arXiv:2303.08112, 2023.
- Rathi, Mehrer, et al. *TopoLM: brain-like spatio-functional organization in a topographic language model*. ICLR 2025. arXiv:2410.11516.
- Elhage et al. *Toy Models of Superposition*. Transformer Circuits Thread, 2022. arXiv:2209.10652.
- Bricken et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread, 2023-10-04.
- Templeton et al. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Transformer Circuits Thread, 2024-05-21.
- Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. *Refusal in Language Models Is Mediated by a Single Direction*. arXiv:2406.11717, 2024.
- Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid. *Steering Language Models With Activation Engineering*. arXiv:2308.10248, 2023.
- Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner. *Steering Llama 2 via Contrastive Activation Addition*. arXiv:2312.06681, 2023.
