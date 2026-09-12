//go:build !darwin || !cgo

package vision

import (
	"fmt"
	"image"
	"time"
)

// AuthStatus mirrors the darwin type on platforms without a camera backend.
type AuthStatus int

const (
	AuthNotDetermined AuthStatus = 0
	AuthRestricted    AuthStatus = 1
	AuthDenied        AuthStatus = 2
	AuthAuthorized    AuthStatus = 3
)

func (a AuthStatus) String() string { return "unsupported" }

var errUnsupported = fmt.Errorf("vision: camera capture is only implemented for darwin with cgo")

// ErrTimeout is returned when no new frame arrived in time.
var ErrTimeout = fmt.Errorf("camera: timed out waiting for a frame")

// ErrBlackFrames is returned when frames are all black.
var ErrBlackFrames = fmt.Errorf("camera: frames are all black")

// CameraOptions configures capture.
type CameraOptions struct{ Width, Height int }

// Camera is a live capture session.
type Camera struct{ BlackFrames, TotalFrames int }

// CameraAuthStatus reports camera permission state.
func CameraAuthStatus() AuthStatus { return AuthRestricted }

// RequestCameraAccess triggers the platform permission prompt.
func RequestCameraAccess() AuthStatus { return AuthRestricted }

// CameraDevices lists available capture devices.
func CameraDevices() []string { return nil }

// OpenCamera opens the video device at deviceIndex.
func OpenCamera(deviceIndex int, opts CameraOptions) (*Camera, error) { return nil, errUnsupported }

// Read blocks for the next frame.
func (c *Camera) Read(timeout time.Duration) (*RGB, error) { return nil, errUnsupported }

// ReadImage is Read returning an image.Image.
func (c *Camera) ReadImage(timeout time.Duration) (image.Image, error) {
	return nil, errUnsupported
}

// Close stops the capture session.
func (c *Camera) Close() error { return nil }
