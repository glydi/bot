// ttsd — a persistent, warm macOS text-to-speech helper.
//
// Spawning `say(1)` costs ~950 ms per utterance, essentially all of it process
// startup. This helper keeps one AVSpeechSynthesizer alive for the life of the
// process, so time-to-first-audio drops to a few milliseconds and synthesis
// runs at roughly 80x realtime.
//
// It speaks nothing to the speakers: audio is synthesised with
// AVSpeechSynthesizer.write and streamed to stdout as raw PCM, resampled to the
// requested rate by AVAudioConverter. Playback is the caller's problem.
//
//   Usage: ttsd [-voice ID] [-rate R] [-sr HZ]
//          ttsd -list
//
// Commands (stdin, one UTF-8 line each; text may not contain a newline):
//   SAY <text>    synthesise <text>; always answered by exactly one END frame
//   CANCEL        abandon the utterance in flight; its END frame still arrives
//   VOICE <id>    change voice for subsequent utterances
//   RATE <float>  change rate for subsequent utterances (0.0-1.0, 0.5 = normal)
//   (EOF)         exit cleanly
//
// Frames (stdout): 4-byte ASCII tag + 4-byte big-endian payload length + payload
//   "RDY "  once at startup, after the engine is warm. Payload: voice identifier.
//   "RATE"  first frame of an utterance. Payload: uint32 big-endian sample rate.
//   "PCM "  payload: little-endian int16 mono samples.
//   "END "  empty; end of utterance. Exactly one per SAY.
//   "ERR "  payload: UTF-8 message. Non-fatal; an END still follows if in an
//           utterance.
//
// Build: swiftc -O ttsd.swift -o ttsd

import Foundation
import AVFoundation

// MARK: - Framed output

private let stdoutHandle = FileHandle.standardOutput
private let writeLock = NSLock()

func emit(_ tag: String, _ payload: Data = Data()) {
    var frame = Data(tag.utf8)
    while frame.count < 4 { frame.append(0x20) }
    frame = frame.prefix(4)
    var n = UInt32(payload.count).bigEndian
    frame.append(Data(bytes: &n, count: 4))
    if !payload.isEmpty { frame.append(payload) }
    writeLock.lock()
    defer { writeLock.unlock() }
    stdoutHandle.write(frame)
}

func emitError(_ message: String) {
    emit("ERR ", Data(message.utf8))
}

func die(_ message: String) -> Never {
    FileHandle.standardError.write(Data("ttsd: \(message)\n".utf8))
    exit(1)
}

// MARK: - Arguments

func qualityName(_ q: AVSpeechSynthesisVoiceQuality) -> String {
    switch q {
    case .premium:  return "premium"
    case .enhanced: return "enhanced"
    default:        return "compact"
    }
}

var voiceID: String? = nil
var speechRate: Float = AVSpeechUtteranceDefaultSpeechRate
var sampleRate: Double = 24000

do {
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    func value(_ flag: String) -> String {
        guard i + 1 < args.count else { die("\(flag) needs a value") }
        i += 1
        return args[i]
    }
    while i < args.count {
        switch args[i] {
        case "-voice": voiceID = value("-voice")
        case "-rate":
            let raw = value("-rate")
            guard let r = Float(raw) else { die("bad -rate \(raw)") }
            speechRate = r
        case "-sr":
            let raw = value("-sr")
            guard let r = Double(raw), r > 0 else { die("bad -sr \(raw)") }
            sampleRate = r
        case "-list":
            // identifier \t name \t language \t quality
            for v in AVSpeechSynthesisVoice.speechVoices() {
                print("\(v.identifier)\t\(v.name)\t\(v.language)\t\(qualityName(v.quality))")
            }
            exit(0)
        case "-h", "-help", "--help":
            print("usage: ttsd [-voice ID] [-rate R] [-sr HZ] | ttsd -list")
            exit(0)
        default:
            die("unknown argument \(args[i])")
        }
        i += 1
    }
}

// MARK: - Synthesiser state

/// Everything mutated from more than one thread lives behind `stateLock`.
private let stateLock = NSLock()
private var currentVoice: AVSpeechSynthesisVoice? = nil
private var cancelRequested = false

func resolveVoice(_ id: String) -> AVSpeechSynthesisVoice? {
    AVSpeechSynthesisVoice(identifier: id) ?? AVSpeechSynthesisVoice(language: id)
}

if let id = voiceID {
    guard let v = resolveVoice(id) else { die("no such voice: \(id) (try -list)") }
    currentVoice = v
}

let synth = AVSpeechSynthesizer()

let outputFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                 sampleRate: sampleRate,
                                 channels: 1,
                                 interleaved: true)!

/// Synthesise `text`, emitting RATE/PCM/END frames. Emits exactly one END
/// unless `quiet`. Returns after the utterance is fully drained.
func speak(_ text: String, quiet: Bool = false) {
    stateLock.lock()
    let voice = currentVoice
    let rate = speechRate
    let startCancelled = cancelRequested
    cancelRequested = false
    stateLock.unlock()

    if startCancelled {
        if !quiet { emit("END ") }
        return
    }

    let utterance = AVSpeechUtterance(string: text)
    if let voice = voice { utterance.voice = voice }
    utterance.rate = rate

    var converter: AVAudioConverter? = nil
    var announced = false
    var finished = false
    let done = DispatchSemaphore(value: 0)

    func isCancelled() -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return cancelRequested
    }

    synth.write(utterance) { (buffer: AVAudioBuffer) in
        guard let pcm = buffer as? AVAudioPCMBuffer else { return }

        // AVSpeechSynthesizer signals completion with a zero-length buffer, and
        // some voices deliver more than one of them per utterance.
        if pcm.frameLength == 0 {
            if finished { return }
            finished = true
            done.signal()
            return
        }
        if isCancelled() { return }

        if !announced {
            announced = true
            if !quiet {
                var sr = UInt32(sampleRate).bigEndian
                emit("RATE", Data(bytes: &sr, count: 4))
            }
        }
        if converter == nil {
            converter = AVAudioConverter(from: pcm.format, to: outputFormat)
            if converter == nil {
                if !quiet { emitError("cannot convert \(pcm.format) to \(outputFormat)") }
                return
            }
        }
        guard let conv = converter else { return }

        let ratio = sampleRate / pcm.format.sampleRate
        let capacity = AVAudioFrameCount(Double(pcm.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }

        var supplied = false
        var err: NSError? = nil
        conv.convert(to: out, error: &err) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true
            status.pointee = .haveData
            return pcm
        }
        if let err = err, !quiet {
            emitError("resample: \(err.localizedDescription)")
            return
        }
        guard let channel = out.int16ChannelData, out.frameLength > 0 else { return }
        if !quiet {
            emit("PCM ", Data(bytes: channel[0], count: Int(out.frameLength) * 2))
        }
    }

    // A voice that never delivers the empty terminator must not wedge the
    // helper; synthesis runs ~80x realtime so 30 s is enormously generous.
    if done.wait(timeout: .now() + 30) == .timedOut && !quiet {
        emitError("timed out waiting for synthesis to finish")
    }

    stateLock.lock()
    cancelRequested = false
    stateLock.unlock()

    if !quiet { emit("END ") }
}

// MARK: - Command loop
//
// AVSpeechSynthesizer.write delivers its buffers on the main run loop, so the
// main thread must sit in RunLoop.run() and everything else runs off it.
// Reading stdin and synthesising are separate queues so that CANCEL can be
// processed while an utterance is in flight.

let speakQueue = DispatchQueue(label: "com.glydi.ttsd.speak")

DispatchQueue.global(qos: .userInitiated).async {
    // Warm the engine: the first utterance otherwise pays voice-load cost.
    speakQueue.sync { speak("a", quiet: true) }
    stateLock.lock()
    let ready = currentVoice?.identifier ?? ""
    stateLock.unlock()
    emit("RDY ", Data(ready.utf8))

    while let line = readLine(strippingNewline: true) {
        if line.isEmpty { continue }
        let space = line.firstIndex(of: " ")
        let verb = String(space.map { line[line.startIndex..<$0] } ?? Substring(line))
        let arg = space.map { String(line[line.index(after: $0)...]) } ?? ""

        switch verb {
        case "SAY":
            let text = arg.trimmingCharacters(in: .whitespaces)
            if text.isEmpty { emit("END "); continue }
            speakQueue.async { speak(text) }

        case "CANCEL":
            stateLock.lock()
            cancelRequested = true
            stateLock.unlock()
            synth.stopSpeaking(at: .immediate)

        case "VOICE":
            if arg.isEmpty {
                stateLock.lock(); currentVoice = nil; stateLock.unlock()
            } else if let v = resolveVoice(arg) {
                stateLock.lock(); currentVoice = v; stateLock.unlock()
            } else {
                emitError("no such voice: \(arg)")
            }

        case "RATE":
            if let r = Float(arg) {
                stateLock.lock(); speechRate = r; stateLock.unlock()
            } else {
                emitError("bad rate: \(arg)")
            }

        default:
            emitError("unknown command: \(verb)")
        }
    }

    // stdin closed: let any in-flight utterance finish, then leave.
    speakQueue.sync { }
    exit(0)
}

RunLoop.main.run()
