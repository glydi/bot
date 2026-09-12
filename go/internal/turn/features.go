package turn

import "math"

// Whisper-style log-mel feature extraction, a direct port of pipecat's
// vendored numpy implementation (_whisper_features.py), which in turn mirrors
// transformers.WhisperFeatureExtractor with chunk_length=8.
//
// Pipeline: pad/truncate to 128000 samples (8 s @ 16 kHz) -> zero-mean
// unit-variance waveform normalisation in float32 -> reflect-padded STFT
// (n_fft=400, hop=160, periodic Hann) -> power spectrogram -> Slaney mel
// filterbank (80 filters, 0-8000 Hz) -> log10 -> drop last frame -> clamp to
// (max - 8) -> (x + 4) / 4.

const (
	sampleRate  = 16000
	windowSecs  = 8
	numSamples  = sampleRate * windowSecs // 128000
	nFFT        = 400
	hopLength   = 160
	numMels     = 80
	numFreqBins = nFFT/2 + 1 // 201
	numFrames   = 800        // frames kept after dropping the trailing one
	melFloor    = 1e-10
	normVarEps  = 1e-7
)

func hertzToMelSlaney(f float64) float64 {
	const minLogHertz = 1000.0
	const minLogMel = 15.0
	logstep := 27.0 / math.Log(6.4)
	if f >= minLogHertz {
		return minLogMel + math.Log(f/minLogHertz)*logstep
	}
	return 3.0 * f / 200.0
}

func melToHertzSlaney(m float64) float64 {
	const minLogHertz = 1000.0
	const minLogMel = 15.0
	logstep := math.Log(6.4) / 27.0
	if m >= minLogMel {
		return minLogHertz * math.Exp(logstep*(m-minLogMel))
	}
	return 200.0 * m / 3.0
}

// linspace replicates numpy.linspace (including its exact endpoint fixup).
func linspace(start, stop float64, num int) []float64 {
	out := make([]float64, num)
	step := (stop - start) / float64(num-1)
	for i := 0; i < num; i++ {
		out[i] = float64(i)*step + start
	}
	out[num-1] = stop
	return out
}

// buildMelFilters returns the Slaney-normalised triangular filterbank as
// [numMels][numFreqBins] (transposed relative to the numpy version so the
// matmul is row-major friendly).
func buildMelFilters() [][]float64 {
	melMin := hertzToMelSlaney(0.0)
	melMax := hertzToMelSlaney(sampleRate / 2.0)
	melFreqs := linspace(melMin, melMax, numMels+2)
	filterFreqs := make([]float64, len(melFreqs))
	for i, m := range melFreqs {
		filterFreqs[i] = melToHertzSlaney(m)
	}
	fftFreqs := linspace(0, sampleRate/2, numFreqBins)

	filterDiff := make([]float64, len(filterFreqs)-1)
	for i := range filterDiff {
		filterDiff[i] = filterFreqs[i+1] - filterFreqs[i]
	}

	filters := make([][]float64, numMels)
	for i := 0; i < numMels; i++ {
		enorm := 2.0 / (filterFreqs[i+2] - filterFreqs[i])
		row := make([]float64, numFreqBins)
		for j := 0; j < numFreqBins; j++ {
			down := -(filterFreqs[i] - fftFreqs[j]) / filterDiff[i]
			up := (filterFreqs[i+2] - fftFreqs[j]) / filterDiff[i+1]
			v := math.Min(down, up)
			if v < 0 {
				v = 0
			}
			row[j] = v * enorm
		}
		filters[i] = row
	}
	return filters
}

// hannPeriodic matches np.hanning(n+1)[:-1] (== torch.hann_window(n)).
func hannPeriodic(n int) []float64 {
	// np.hanning(M)[i] = 0.5 - 0.5*cos(2*pi*i/(M-1)); here M = n+1.
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	}
	return w
}

// featureExtractor holds the precomputed tables needed to turn 8 s of audio
// into an (80, 800) log-mel matrix.
type featureExtractor struct {
	window  []float64
	filters [][]float64
	// Each mel filter is a narrow triangle, so only bins [lo, hi) are nonzero.
	lo, hi []int
	fft    *bluestein

	padded []float64 // numSamples + nFFT
	frameA []float64 // nFFT
	frameB []float64 // nFFT
	powerA []float64 // numFreqBins
	powerB []float64 // numFreqBins
	mel    []float64 // numMels * (numFrames+1), mel-major
}

func newFeatureExtractor() *featureExtractor {
	fe := &featureExtractor{
		window:  hannPeriodic(nFFT),
		filters: buildMelFilters(),
		fft:     newBluestein(nFFT),
		padded:  make([]float64, numSamples+nFFT),
		frameA:  make([]float64, nFFT),
		frameB:  make([]float64, nFFT),
		powerA:  make([]float64, numFreqBins),
		powerB:  make([]float64, numFreqBins),
		mel:     make([]float64, numMels*(numFrames+1)),
	}
	fe.lo = make([]int, numMels)
	fe.hi = make([]int, numMels)
	for m := 0; m < numMels; m++ {
		row := fe.filters[m]
		l, h := 0, numFreqBins
		for l < numFreqBins && row[l] == 0 {
			l++
		}
		for h > l && row[h-1] == 0 {
			h--
		}
		fe.lo[m], fe.hi[m] = l, h
	}
	return fe
}

// accumulateMel projects one power-spectrum frame onto the mel filterbank.
func (fe *featureExtractor) accumulateMel(power []float64, t, nf int) {
	for m := 0; m < numMels; m++ {
		row := fe.filters[m]
		var acc float64
		for j := fe.lo[m]; j < fe.hi[m]; j++ {
			acc += row[j] * power[j]
		}
		if acc < melFloor {
			acc = melFloor
		}
		fe.mel[m*nf+t] = math.Log10(acc)
	}
}

// prepare pads/truncates raw 16 kHz mono float32 audio to exactly 8 seconds,
// keeping the END of the utterance (zero-padding at the FRONT when short) —
// this is what base_smart_turn/local_smart_turn_v3 does before feature
// extraction.
func prepare(samples []float32, dst []float32) []float32 {
	if cap(dst) < numSamples {
		dst = make([]float32, numSamples)
	}
	dst = dst[:numSamples]
	if len(samples) >= numSamples {
		copy(dst, samples[len(samples)-numSamples:])
		return dst
	}
	pad := numSamples - len(samples)
	for i := 0; i < pad; i++ {
		dst[i] = 0
	}
	copy(dst[pad:], samples)
	return dst
}

// compute writes the (80, 800) log-mel features, mel-major, into out.
// x must be exactly numSamples long.
func (fe *featureExtractor) compute(x []float32, out []float32) {
	// --- waveform normalisation (float32, as in the reference) ---
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := float32(sum / float64(len(x)))
	var vsum float64
	for _, v := range x {
		d := float64(v - mean)
		vsum += d * d
	}
	variance := float32(vsum / float64(len(x)))
	std := float32(math.Sqrt(float64(variance + normVarEps)))

	// --- reflect padding by nFFT/2 on both sides ---
	const pad = nFFT / 2
	p := fe.padded
	for i := 0; i < numSamples; i++ {
		p[pad+i] = float64((x[i] - mean) / std)
	}
	for i := 0; i < pad; i++ {
		p[pad-1-i] = p[pad+1+i]                     // reflect front (no edge repeat)
		p[pad+numSamples+i] = p[pad+numSamples-2-i] // reflect back
	}

	// --- STFT power spectrogram -> mel ---
	// Frames are transformed in pairs (two real frames per complex FFT).
	nf := numFrames + 1 // 801 frames before the trailing one is dropped
	t := 0
	for ; t+1 < nf; t += 2 {
		baseA := t * hopLength
		baseB := (t + 1) * hopLength
		for i := 0; i < nFFT; i++ {
			w := fe.window[i]
			fe.frameA[i] = p[baseA+i] * w
			fe.frameB[i] = p[baseB+i] * w
		}
		fe.fft.realFFTPowerPair(fe.frameA, fe.frameB, fe.powerA, fe.powerB)
		fe.accumulateMel(fe.powerA, t, nf)
		fe.accumulateMel(fe.powerB, t+1, nf)
	}
	for ; t < nf; t++ {
		base := t * hopLength
		for i := 0; i < nFFT; i++ {
			fe.frameA[i] = p[base+i] * fe.window[i]
		}
		fe.fft.realFFTPower(fe.frameA, fe.powerA)
		fe.accumulateMel(fe.powerA, t, nf)
	}

	// --- drop trailing frame, clamp to (max - 8), rescale ---
	maxv := math.Inf(-1)
	for m := 0; m < numMels; m++ {
		row := fe.mel[m*nf : m*nf+numFrames]
		for _, v := range row {
			if v > maxv {
				maxv = v
			}
		}
	}
	floor := maxv - 8.0
	for m := 0; m < numMels; m++ {
		src := fe.mel[m*nf : m*nf+numFrames]
		dst := out[m*numFrames : (m+1)*numFrames]
		for i, v := range src {
			if v < floor {
				v = floor
			}
			dst[i] = float32((v + 4.0) / 4.0)
		}
	}
}
