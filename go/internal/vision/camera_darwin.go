//go:build darwin && cgo

package vision

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -Wno-deprecated-declarations
#cgo LDFLAGS: -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework Foundation
#include <stdlib.h>
#include "camera_darwin.h"
*/
import "C"

import (
	"fmt"
	"image"
	"sync"
	"time"
	"unsafe"
)

// AuthStatus reports whether this process may use the camera.
type AuthStatus int

const (
	AuthNotDetermined AuthStatus = 0
	AuthRestricted    AuthStatus = 1
	AuthDenied        AuthStatus = 2
	AuthAuthorized    AuthStatus = 3
)

func (a AuthStatus) String() string {
	switch a {
	case AuthNotDetermined:
		return "not-determined"
	case AuthRestricted:
		return "restricted"
	case AuthDenied:
		return "denied"
	case AuthAuthorized:
		return "authorized"
	}
	return "unknown"
}

// CameraAuthStatus returns the current TCC camera authorization state.
func CameraAuthStatus() AuthStatus { return AuthStatus(C.glydi_cam_auth_status()) }

// RequestCameraAccess triggers the macOS permission prompt and blocks until
// the user answers (or returns immediately if already decided).
func RequestCameraAccess() AuthStatus { return AuthStatus(C.glydi_cam_request_access()) }

// CameraDevices lists the available video capture devices, in index order.
func CameraDevices() []string {
	n := int(C.glydi_cam_device_count())
	out := make([]string, 0, n)
	buf := make([]byte, 256)
	for i := 0; i < n; i++ {
		C.glydi_cam_device_name(C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
		out = append(out, C.GoString((*C.char)(unsafe.Pointer(&buf[0]))))
	}
	return out
}

// Camera is a live capture session backed by AVFoundation.
type Camera struct {
	mu     sync.Mutex
	handle *C.glydi_cam
	frame  *RGB
	// BlackFrames counts consecutive frames whose mean brightness was
	// essentially zero; on macOS that is the signature of a TCC denial
	// rather than a broken capture path.
	BlackFrames int
	TotalFrames int
}

// CameraOptions configures capture.
type CameraOptions struct {
	Width  int // requested, default 1280
	Height int // requested, default 720
}

// OpenCamera opens the video device at deviceIndex.
//
// On macOS a process without a bundle Info.plist that carries
// NSCameraUsageDescription cannot show the permission prompt; the capture
// session then starts happily but every frame is all black. Read's
// ErrBlackFrames surfaces that case rather than failing silently.
func OpenCamera(deviceIndex int, opts CameraOptions) (*Camera, error) {
	if opts.Width == 0 {
		opts.Width = 1280
	}
	if opts.Height == 0 {
		opts.Height = 720
	}
	if st := CameraAuthStatus(); st == AuthNotDetermined {
		st = RequestCameraAccess()
		if st != AuthAuthorized {
			return nil, fmt.Errorf("camera access %s (macOS TCC)", st)
		}
	} else if st != AuthAuthorized {
		return nil, fmt.Errorf("camera access %s (macOS TCC); grant it in "+
			"System Settings > Privacy & Security > Camera", st)
	}
	errBuf := make([]byte, 256)
	h := C.glydi_cam_open(C.int(deviceIndex), C.int(opts.Width), C.int(opts.Height),
		(*C.char)(unsafe.Pointer(&errBuf[0])), C.int(len(errBuf)))
	if h == nil {
		return nil, fmt.Errorf("open camera %d: %s", deviceIndex,
			C.GoString((*C.char)(unsafe.Pointer(&errBuf[0]))))
	}
	return &Camera{handle: h}, nil
}

// ErrTimeout is returned by Read when no new frame arrived in time.
var ErrTimeout = fmt.Errorf("camera: timed out waiting for a frame")

// ErrBlackFrames is returned once enough consecutive all-black frames have
// been seen to conclude the process was denied camera access.
var ErrBlackFrames = fmt.Errorf("camera: frames are all black - this is a " +
	"macOS TCC camera-permission problem (an unbundled binary has no " +
	"NSCameraUsageDescription), not a capture failure")

// Read blocks for the next frame and returns it as a tightly packed RGB
// buffer. The returned *RGB is reused across calls; copy it if you need to
// retain it.
func (c *Camera) Read(timeout time.Duration) (*RGB, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle == nil {
		return nil, fmt.Errorf("camera: closed")
	}
	// The negotiated size is only known once a frame has landed, so poll
	// with a small scratch buffer until the delegate reports dimensions.
	deadline := time.Now().Add(timeout)
	for {
		w := int(C.glydi_cam_width(c.handle))
		h := int(C.glydi_cam_height(c.handle))
		if w > 0 && h > 0 {
			if c.frame == nil || c.frame.W != w || c.frame.H != h {
				c.frame = NewRGB(w, h)
			}
			break
		}
		if time.Now().After(deadline) {
			return nil, ErrTimeout
		}
		time.Sleep(10 * time.Millisecond)
	}
	remaining := time.Until(deadline)
	if remaining <= 0 {
		remaining = 100 * time.Millisecond
	}
	rc := C.glydi_cam_read(c.handle, (*C.uint8_t)(unsafe.Pointer(&c.frame.Pix[0])),
		C.int(len(c.frame.Pix)), C.int(remaining.Milliseconds()))
	switch rc {
	case 0:
		return nil, ErrTimeout
	case -1:
		return nil, fmt.Errorf("camera: read failed")
	}
	c.TotalFrames++
	if c.frame.MeanBrightness() < 0.5 {
		c.BlackFrames++
		if c.BlackFrames >= 5 {
			return c.frame, ErrBlackFrames
		}
	} else {
		c.BlackFrames = 0
	}
	return c.frame, nil
}

// ReadImage is Read returning a standard image.Image view of the frame.
func (c *Camera) ReadImage(timeout time.Duration) (image.Image, error) {
	f, err := c.Read(timeout)
	if f == nil {
		return nil, err
	}
	return f.AsImage(), err
}

// Close stops the capture session.
func (c *Camera) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle != nil {
		C.glydi_cam_close(c.handle)
		c.handle = nil
	}
	return nil
}
