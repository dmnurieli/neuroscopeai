// The course: short lessons, one idea per page. Content is data so it can be reworded without touching code.
// Counts ({floors}, {desks}, {allDesks}, {memo}) are filled from the engine's geometry when the page is shown.
// Every other number was measured on OLMo 2 1B Instruct (greedy, bf16 on Metal, the app's own request) with
// `cd engine && cargo run --release --example lesson_numbers -- <answer|follow 11 2962|ink 13|derive>`
// on 2026-10-28; keep them in step with that model.

import Foundation

enum Lessons {
    static let skyQuestion = "Answer in French, in one short sentence: Why is the sky blue?"

    static let all: [Lesson] = [meet, slowWord, oneDesk, changeMind, math, yourTurn]

    // 1. The building, the desks, and a first question.
    static let meet = Lesson(
        id: "meet", title: "Meet the model", subtitle: "A building that writes, one word at a time", minutes: 2,
        pages: [
            LessonPage(id: "l1p1",
                       text: "Picture the model as a building with {floors} **floors**. Together they write one word at a time.",
                       field: .overview),
            LessonPage(id: "l1p2",
                       text: "Each dot is a worker at a **desk**. One floor has {desks} of them.",
                       detail: "{allDesks} desks in all.",
                       field: .floors([7])),
            LessonPage(id: "l1p3",
                       text: "Let's ask it something.",
                       detail: "“\(skyQuestion)”",
                       action: .ask(skyQuestion), field: .overview, continueLabel: "Ask", waits: true),
            LessonPage(id: "l1p4",
                       text: "Watch the desks light up as it writes, one word at a time.",
                       action: .replay,
                       field: .overview),
            LessonPage(id: "l1p5",
                       text: "Colorful desks are busy. Grey desks are resting.",
                       field: .keep),
            LessonPage(id: "l1p6",
                       text: "Here is what it wrote. Every word took one trip up the building.",
                       field: .keep),
        ])

    // 2. One word's trip, bottom to top, slowed down.
    static let slowWord = Lesson(
        id: "slow-word", title: "One word, slowed down", subtitle: "Follow a single word up the floors", minutes: 2,
        pages: [
            LessonPage(id: "l2p1",
                       text: "Let's slow one word way down. We'll follow its first word, “La”.",
                       action: .slowMotion, field: .columns, waits: true),
            LessonPage(id: "l2p2",
                       text: "A **note** travels up the floors, from the bottom to the top.",
                       field: .keep),
            LessonPage(id: "l2p3",
                       text: "Each floor reads the note and adds a little to it.",
                       action: .slowMotionTo(5), field: .keep),
            LessonPage(id: "l2p4",
                       text: "By floor 12, the note has picked up a lot along the way.",
                       action: .slowMotionTo(12), field: .keep),
            LessonPage(id: "l2p5",
                       text: "At the top, the last floor reads the note and picks the next word.",
                       action: .slowMotionTo(16), field: .keep),
            LessonPage(id: "l2p6",
                       text: "For real, the whole trip takes about 8 thousandths of a second. Here we stretched it out.",
                       detail: "About 8 ms per word.",
                       field: .keep),
        ])

    // 3. One desk's story at the word “La” (desk 11:2962: found by its activation on French/France/Paris versus
    // other countries' words, `lesson_numbers desks2`; its strongest matches are " French", "French", " france",
    // " France", "France", " Paris"; at "La": gate +1.52, h −1.05, heard by 13:2479, 12:8043, 13:1276; +0.198 on
    // " raison", +0.236 on " sky").
    static let oneDesk = Lesson(
        id: "one-desk", title: "Meet one desk", subtitle: "What a single worker does", minutes: 2,
        pages: [
            LessonPage(id: "l3p1",
                       text: "This desk on floor 11 cares about one thing: French.",
                       detail: "Desk #2962, at the word “La”.",
                       action: .followDesk(layer: 11, index: 2962, step: 0), field: .columns),
            LessonPage(id: "l3p2",
                       text: "It wakes up for words like French, France and Paris.",
                       detail: "Its closest words: French, France, Paris, and even the name Jean.",
                       action: .followDesk(layer: 11, index: 2962, step: 1), field: .keep),
            LessonPage(id: "l3p3",
                       text: "It asks the note one question: how French is this? The answer is +1.5, a clear yes.",
                       action: .followDesk(layer: 11, index: 2962, step: 2), field: .keep),
            LessonPage(id: "l3p4",
                       text: "So it speaks up and adds its own message to the note.",
                       action: .followDesk(layer: 11, index: 2962, step: 3), field: .keep),
            LessonPage(id: "l3p5",
                       text: "Its message reaches desks on higher floors.",
                       detail: "Floor 13 #2479, floor 12 #8043, and floor 13 #1276, a desk that writes French words like “de” and “des”.",
                       action: .followDesk(layer: 11, index: 2962, step: 5), field: .keep),
            LessonPage(id: "l3p6",
                       text: "In the end, it nudges the next word toward “raison”, making “La raison”.",
                       detail: "+0.20 toward “raison”. It pushes “sky” a little too (+0.24): one desk never decides alone.",
                       action: .followDesk(layer: 11, index: 2962, step: 6), field: .keep),
            LessonPage(id: "l3p7",
                       text: "One desk, one small job. Thousands of them together write the answer.",
                       field: .keep),
        ])

    // 4. An idea is a color in the note; erase it and the answer changes.
    static let changeMind = Lesson(
        id: "change-mind", title: "Change its mind", subtitle: "Find an idea, then erase it", minutes: 2,
        pages: [
            LessonPage(id: "l4p1",
                       text: "An idea doesn't live in one desk. It's like a **color** running through the note.",
                       field: .overview),
            LessonPage(id: "l4p2",
                       text: "Let's find the French color.",
                       action: .findConcept("remove-french"), field: .concept, continueLabel: "Find it", waits: true),
            LessonPage(id: "l4p3",
                       text: "These desks light up for French.",
                       field: .keep),
            LessonPage(id: "l4p4",
                       text: "The color hardly shows on the low floors. Then floor 12 paints it on, and it stays to the top.",
                       detail: "About 0.03 after floor 10, 2.8 after floor 12, 3.9 at the top.",
                       field: .keep),
            LessonPage(id: "l4p5",
                       text: "Now erase that color on every floor, and ask again.",
                       detail: "“Answer in French, in one short sentence: What do dogs eat?”",
                       action: .tryChange, field: .compare, continueLabel: "Erase and ask again", waits: true),
            LessonPage(id: "l4p6",
                       text: "Before, it answered in French. After, in English.",
                       field: .keep),
            LessonPage(id: "l4p7",
                       text: "Nothing else was touched. Only the French color is gone.",
                       field: .keep),
        ])

    // 5. The same erase, on three numbers.
    static let math = Lesson(
        id: "math", title: "The math, gently", subtitle: "How erasing works, on three numbers", minutes: 2,
        pages: [
            LessonPage(id: "l5p1",
                       text: "A note is just a list of numbers. Here is a tiny one.",
                       detail: "The real note has {memo} numbers.",
                       action: .mathToy(0), field: .hidden),
            LessonPage(id: "l5p2",
                       text: "The French color is a list of numbers too.",
                       action: .mathToy(1), field: .hidden),
            LessonPage(id: "l5p3",
                       text: "Multiply them pair by pair, then add. That tells you how French the note is.",
                       detail: "0 + 3 + 0.8 = 3.8",
                       action: .mathToy(2), field: .hidden),
            LessonPage(id: "l5p4",
                       text: "To erase, take away exactly that much of the color.",
                       detail: "3.8 × (0, 0.6, 0.8) = (0, 2.28, 3.04)",
                       action: .mathToy(3), field: .hidden),
            LessonPage(id: "l5p5",
                       text: "Check again: now it's 0. No French is left.",
                       detail: "0 + 1.632 − 1.632 = 0",
                       action: .mathToy(4), field: .hidden),
            LessonPage(id: "l5p6",
                       text: "The model does this on every floor, for every word.",
                       action: .mathToy(4), field: .hidden),
        ])

    // 6. Send-off into Explore.
    static let yourTurn = Lesson(
        id: "your-turn", title: "Your turn", subtitle: "Ask your own questions", minutes: 1,
        pages: [
            LessonPage(id: "l6p1",
                       text: "You've met the building, followed one word, and met one desk.",
                       field: .overview),
            LessonPage(id: "l6p2",
                       text: "You also found an idea and erased it.",
                       field: .keep),
            LessonPage(id: "l6p3",
                       text: "Now ask anything you like, and watch which floors light up.",
                       field: .keep),
            LessonPage(id: "l6p4",
                       text: "Have fun.",
                       action: .explore, field: .keep, continueLabel: "Explore freely"),
        ])
}
