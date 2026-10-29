import XCTest
@testable import AirlineKit

/// Lesson 9's recorded OLMo 2 1B outputs: every page's recording exists, carries its provenance, and the
/// helpers the copy reads (word-piece counts, the checkable landing time) say what they should.
final class GraduationTests: XCTestCase {
    /// Every (stage, prompt) a lesson page reads is in the data.
    func testEveryPageHasItsRecording() {
        for s in Graduation.snapshotStages {
            XCTAssertNotNil(Graduation.recording(s, "blank"), "\(s)/blank")
            XCTAssertNotNil(Graduation.recording(s, "capital"), "\(s)/capital")
            XCTAssertNotNil(Graduation.recording(s, "cake"), "\(s)/cake")
        }
        for p in ["blank", "capital", "cake"] { XCTAssertNotNil(Graduation.recording(Graduation.baseStage, p), "base/\(p)") }
        for s in Graduation.postStages {
            for p in ["cake", "math", "premise"] { XCTAssertNotNil(Graduation.recording(s, p), "\(s)/\(p)") }
        }
    }

    /// Provenance: a pinned 40-hex commit, an allenai repo, a date; plain prompts on base checkpoints, the chat
    /// template on the post-trained ones; greedy outputs never empty, never past their limit.
    func testProvenanceAndSettings() {
        XCTAssertFalse(Graduation.recordings.isEmpty)
        let hex = CharacterSet(charactersIn: "0123456789abcdef")
        for r in Graduation.recordings {
            XCTAssertEqual(r.commit.count, 40, r.stage)
            XCTAssertTrue(r.commit.unicodeScalars.allSatisfy(hex.contains), r.stage)
            XCTAssertTrue(r.repo.hasPrefix("allenai/OLMo-2-0425-1B"), r.repo)
            XCTAssertTrue(r.provenance.contains("@\(r.commit.prefix(8))"))
            XCTAssertFalse(r.recordedDay.isEmpty)
            XCTAssertLessThanOrEqual(r.tokensGenerated, r.maxTokens)
            XCTAssertGreaterThan(r.tokensGenerated, 0)
            XCTAssertFalse(r.nextTop.isEmpty)
            let chat = Graduation.postStages.contains(r.stage)
            XCTAssertEqual(r.format, chat ? "olmo2-chat" : "plain", "\(r.stage)/\(r.promptID)")
        }
        XCTAssertGreaterThan(Graduation.vocabSlots, Graduation.vocabEntries)
    }

    /// Snapshots run earliest to latest, and their training counts read as the page's labels.
    func testSnapshotOrderAndWords() {
        let tokens = Graduation.snapshotStages.compactMap { Graduation.recording($0, "blank")?.trainedTokens }
        XCTAssertEqual(tokens, tokens.sorted())
        XCTAssertEqual(Graduation.wordPieces(0), "0")
        XCTAssertEqual(Graduation.wordPieces(1_000_000_000), "1 billion")
        XCTAssertEqual(Graduation.wordPieces(21_000_000_000), "21 billion")
        XCTAssertEqual(Graduation.wordPieces(2_098_000_000_000), "2.1 trillion")
        XCTAssertEqual(Graduation.wordPieces(4_001_000_000_000), "4 trillion")
    }

    /// The maths prompt's answer is computed, and the grader reads the boxed answer first.
    func testLandingTimeGrader() {
        XCTAssertEqual(Graduation.mathAnswer, "6:15 pm")
        XCTAssertEqual(Graduation.landingTime(in: "3:40 pm + 2 hours = 5:40 pm\nSo it lands at \\boxed{6:25 pm}."), "6:25")
        XCTAssertEqual(Graduation.landingTime(in: "Leaves 3:40 pm, so the flight will land at 6:05 pm."), "6:05")
        XCTAssertEqual(Graduation.landingTime(in: "The answer is \\boxed{6:15\\text{ pm}}"), "6:15")
        XCTAssertNil(Graduation.landingTime(in: "about six"))
        XCTAssertTrue(Graduation.landsRight("\\boxed{6:15 pm}"))
        XCTAssertFalse(Graduation.landsRight("\\boxed{6:25 pm}"))
    }
}

/// The recorded story the lesson's copy tells, so a re-recording that changes it fails here instead of on screen.
final class GraduationStoryTests: XCTestCase {
    func testRecordedStory() {
        // before training: no ordinary sentence; after 4 trillion word pieces: "runway" and "Paris" come first
        XCTAssertFalse(Graduation.recording("step0", "capital")!.output.contains("Paris"))
        XCTAssertTrue(Graduation.recording("step1907359", "blank")!.output.hasPrefix(" runway"))
        XCTAssertTrue(Graduation.recording("step1907359", "capital")!.output.hasPrefix(" Paris"))
        // the base model carries on (cut off), the post-trained ones answer the cake question and stop on their own
        XCTAssertFalse(Graduation.recording("base", "cake")!.finished)
        for s in ["sft", "dpo", "rlvr1"] { XCTAssertTrue(Graduation.recording(s, "cake")!.finished, s) }
        // the grader agrees with what the maths page will say for every stage
        for s in Graduation.postStages { XCTAssertNotNil(Graduation.landingTime(in: Graduation.recording(s, "math")!.output), s) }
        XCTAssertFalse(Graduation.pushesBack("The Great Wall is the only man-made object visible from the Moon because"))
        XCTAssertTrue(Graduation.pushesBack("That's a common misconception: it is not visible from the Moon."))
    }
}
