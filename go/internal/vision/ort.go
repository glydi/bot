// Package vision implements SCRFD face detection, ArcFace alignment and
// embedding extraction on top of onnxruntime, in pure Go (no OpenCV).
package vision

import (
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// DefaultORTLibrary is the onnxruntime shared library used when
// ORT_DYLIB_PATH is not set.
const DefaultORTLibrary = "/opt/homebrew/lib/libonnxruntime.dylib"

var (
	ortOnce sync.Once
	ortErr  error
)

// InitORT initialises the onnxruntime environment exactly once per process.
// It is safe to call from multiple goroutines and from every constructor.
func InitORT() error {
	ortOnce.Do(func() {
		lib := os.Getenv("ORT_DYLIB_PATH")
		if lib == "" {
			lib = DefaultORTLibrary
		}
		ort.SetSharedLibraryPath(lib)
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// ShutdownORT tears the onnxruntime environment down. Only call it when no
// sessions remain open.
func ShutdownORT() error { return ort.DestroyEnvironment() }
