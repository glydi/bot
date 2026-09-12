#ifndef GLYDI_CAMERA_DARWIN_H
#define GLYDI_CAMERA_DARWIN_H

#include <stdint.h>

// Opaque capture handle.
typedef struct glydi_cam glydi_cam;

// Returns the number of video capture devices.
int glydi_cam_device_count(void);

// Copies the name of device `index` into `out` (NUL terminated).
void glydi_cam_device_name(int index, char *out, int out_len);

// Authorization status: 0 not determined, 1 restricted, 2 denied, 3 authorized.
int glydi_cam_auth_status(void);

// Blocks until the user answers the permission prompt (or returns immediately
// if already decided). Returns the resulting auth status.
int glydi_cam_request_access(void);

// Opens device `index` requesting roughly width x height. On failure returns
// NULL and writes a message into `err`.
glydi_cam *glydi_cam_open(int index, int width, int height, char *err, int err_len);

// Actual negotiated frame size, valid after the first frame arrives.
int glydi_cam_width(glydi_cam *c);
int glydi_cam_height(glydi_cam *c);

// Copies the most recent frame as tightly packed RGB into `dst`
// (dst_len must be >= width*height*3). Waits up to timeout_ms for a frame
// newer than the last one returned. Returns 1 on success, 0 on timeout,
// -1 on error.
int glydi_cam_read(glydi_cam *c, uint8_t *dst, int dst_len, int timeout_ms);

void glydi_cam_close(glydi_cam *c);

#endif
