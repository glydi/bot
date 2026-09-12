package audio

// Turn detection: deciding when someone has finished speaking.
//
// This is the single biggest contributor to whether a voice bot feels
// responsive. The naive approach -- wait for N milliseconds of silence -- puts
// that entire N on the front of every reply, and still cuts people off when
// they pause mid-thought.
//
// What is implemented here is energy VAD with hangover, which is the honest
// floor: it needs a real pause. The Python build used smart-turn v3, a model
// that reads the waveform and judges whether the speaker is actually finished,
// and it is worth roughly 250ms a turn. That model is ONNX and can be loaded
// from Go the same way the face models are; Detector is deliberately an
// interface so it can be swapped in without touching the conversation loop.

import "time"

// Detector decides when a turn starts and ends.
type Detector interface {
	// Push feeds one chunk and reports the current state.
	Push(samples []float32) State
	Reset()
}

type State int

const (
	Silent State = iota
	Speaking
	// Ended means: the user was speaking and has now finished. Emitted once.
	Ended
)

// EnergyVAD triggers on loudness with a hangover period.
type EnergyVAD struct {
	// Threshold is mean absolute amplitude. Speech at conversational distance
	// sits around 0.02-0.08; room tone is well under 0.005.
	Threshold float32
	// StartFrames is how many loud chunks are needed before we believe speech
	// began. Guards against a door closing or a cough starting a turn.
	StartFrames int
	// HangoverFrames is how much silence ends the turn. At 32ms per chunk, 20
	// frames is ~640ms -- enough to ride out the pause inside a sentence.
	HangoverFrames int
	// MinSpeechFrames rejects utterances too short to be worth transcribing.
	MinSpeechFrames int

	loud     int
	quiet    int
	speaking bool
	spoken   int
}

func NewEnergyVAD() *EnergyVAD {
	return &EnergyVAD{
		Threshold:       0.015,
		StartFrames:     3,
		HangoverFrames:  20,
		MinSpeechFrames: 8,
	}
}

func (v *EnergyVAD) Reset() {
	v.loud, v.quiet, v.spoken, v.speaking = 0, 0, 0, false
}

func (v *EnergyVAD) Push(samples []float32) State {
	if MeanAbs(samples) >= v.Threshold {
		v.loud++
		v.quiet = 0
		if !v.speaking && v.loud >= v.StartFrames {
			v.speaking = true
			v.spoken = v.loud
		} else if v.speaking {
			v.spoken++
		}
	} else {
		v.loud = 0
		if v.speaking {
			v.quiet++
			if v.quiet >= v.HangoverFrames {
				wasLongEnough := v.spoken >= v.MinSpeechFrames
				v.Reset()
				if wasLongEnough {
					return Ended
				}
				return Silent
			}
			// Still inside the hangover -- treat as ongoing speech.
			v.spoken++
		}
	}
	if v.speaking {
		return Speaking
	}
	return Silent
}

// HangoverDuration reports the dead air this detector adds to every turn, which
// is useful to log so the cost stays visible rather than being forgotten.
func (v *EnergyVAD) HangoverDuration(sampleRate int) time.Duration {
	return time.Duration(v.HangoverFrames*FramesPerBuffer) * time.Second /
		time.Duration(sampleRate)
}
