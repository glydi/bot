//go:build darwin

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <string.h>

#include "camera_darwin.h"

@interface GlydiCamDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic, assign) uint8_t *buf;   // BGRA, tightly packed
@property(nonatomic, assign) int width;
@property(nonatomic, assign) int height;
@property(nonatomic, assign) uint64_t seq;
@property(nonatomic, strong) NSCondition *cond;
@end

@implementation GlydiCamDelegate

- (instancetype)init {
  self = [super init];
  if (self) {
    _cond = [[NSCondition alloc] init];
    _buf = NULL;
    _seq = 0;
  }
  return self;
}

- (void)dealloc {
  if (_buf) free(_buf);
}

- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
  CVImageBufferRef pixels = CMSampleBufferGetImageBuffer(sampleBuffer);
  if (!pixels) return;
  CVPixelBufferLockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);
  int w = (int)CVPixelBufferGetWidth(pixels);
  int h = (int)CVPixelBufferGetHeight(pixels);
  size_t stride = CVPixelBufferGetBytesPerRow(pixels);
  uint8_t *base = (uint8_t *)CVPixelBufferGetBaseAddress(pixels);
  if (base && w > 0 && h > 0) {
    [self.cond lock];
    if (self.buf == NULL || self.width != w || self.height != h) {
      if (self.buf) free(self.buf);
      self.buf = (uint8_t *)malloc((size_t)w * h * 4);
      self.width = w;
      self.height = h;
    }
    for (int y = 0; y < h; y++) {
      memcpy(self.buf + (size_t)y * w * 4, base + (size_t)y * stride, (size_t)w * 4);
    }
    self.seq++;
    [self.cond broadcast];
    [self.cond unlock];
  }
  CVPixelBufferUnlockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);
}

@end

struct glydi_cam {
  AVCaptureSession *session;
  GlydiCamDelegate *delegate;
  dispatch_queue_t queue;
  uint64_t last_seq;
};

static NSArray<AVCaptureDevice *> *glydi_devices(void) {
  AVCaptureDeviceDiscoverySession *disco = [AVCaptureDeviceDiscoverySession
      discoverySessionWithDeviceTypes:@[
        AVCaptureDeviceTypeBuiltInWideAngleCamera,
        AVCaptureDeviceTypeExternalUnknown,
      ]
                            mediaType:AVMediaTypeVideo
                             position:AVCaptureDevicePositionUnspecified];
  return disco.devices;
}

int glydi_cam_device_count(void) {
  @autoreleasepool {
    return (int)glydi_devices().count;
  }
}

void glydi_cam_device_name(int index, char *out, int out_len) {
  @autoreleasepool {
    out[0] = 0;
    NSArray<AVCaptureDevice *> *devs = glydi_devices();
    if (index < 0 || index >= (int)devs.count) return;
    const char *name = devs[index].localizedName.UTF8String;
    if (name) {
      strncpy(out, name, out_len - 1);
      out[out_len - 1] = 0;
    }
  }
}

int glydi_cam_auth_status(void) {
  @autoreleasepool {
    switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]) {
      case AVAuthorizationStatusNotDetermined: return 0;
      case AVAuthorizationStatusRestricted: return 1;
      case AVAuthorizationStatusDenied: return 2;
      case AVAuthorizationStatusAuthorized: return 3;
    }
    return 1;
  }
}

int glydi_cam_request_access(void) {
  @autoreleasepool {
    if (glydi_cam_auth_status() != 0) return glydi_cam_auth_status();
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                             completionHandler:^(BOOL granted) {
                               (void)granted;
                               dispatch_semaphore_signal(sem);
                             }];
    dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 30LL * NSEC_PER_SEC));
    return glydi_cam_auth_status();
  }
}

glydi_cam *glydi_cam_open(int index, int width, int height, char *err, int err_len) {
  @autoreleasepool {
    err[0] = 0;
    NSArray<AVCaptureDevice *> *devs = glydi_devices();
    if (index < 0 || index >= (int)devs.count) {
      snprintf(err, err_len, "no video device at index %d (%d present)", index,
               (int)devs.count);
      return NULL;
    }
    AVCaptureDevice *dev = devs[index];
    NSError *nserr = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:dev
                                                                       error:&nserr];
    if (!input) {
      snprintf(err, err_len, "device input: %s",
               nserr.localizedDescription.UTF8String ?: "unknown");
      return NULL;
    }

    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    if (![session canAddInput:input]) {
      snprintf(err, err_len, "cannot add capture input");
      return NULL;
    }
    [session addInput:input];

    AVCaptureVideoDataOutput *out = [[AVCaptureVideoDataOutput alloc] init];
    out.alwaysDiscardsLateVideoFrames = YES;
    out.videoSettings = @{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
      (id)kCVPixelBufferWidthKey : @(width),
      (id)kCVPixelBufferHeightKey : @(height),
    };
    if (![session canAddOutput:out]) {
      snprintf(err, err_len, "cannot add capture output");
      return NULL;
    }
    [session addOutput:out];

    glydi_cam *c = (glydi_cam *)calloc(1, sizeof(glydi_cam));
    c->delegate = [[GlydiCamDelegate alloc] init];
    c->queue = dispatch_queue_create("ai.glydi.camera", DISPATCH_QUEUE_SERIAL);
    [out setSampleBufferDelegate:c->delegate queue:c->queue];
    c->session = session;
    c->last_seq = 0;
    CFRetain((__bridge CFTypeRef)session);
    CFRetain((__bridge CFTypeRef)c->delegate);
    [session startRunning];
    return c;
  }
}

int glydi_cam_width(glydi_cam *c) { return c ? c->delegate.width : 0; }
int glydi_cam_height(glydi_cam *c) { return c ? c->delegate.height : 0; }

int glydi_cam_read(glydi_cam *c, uint8_t *dst, int dst_len, int timeout_ms) {
  if (!c) return -1;
  @autoreleasepool {
    GlydiCamDelegate *d = c->delegate;
    [d.cond lock];
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout_ms / 1000.0];
    while (d.seq <= c->last_seq) {
      if (![d.cond waitUntilDate:deadline]) {
        [d.cond unlock];
        return 0;
      }
    }
    int w = d.width, h = d.height;
    if (dst_len < w * h * 3) {
      [d.cond unlock];
      return -1;
    }
    // BGRA -> RGB
    const uint8_t *src = d.buf;
    for (int i = 0; i < w * h; i++) {
      dst[i * 3 + 0] = src[i * 4 + 2];
      dst[i * 3 + 1] = src[i * 4 + 1];
      dst[i * 3 + 2] = src[i * 4 + 0];
    }
    c->last_seq = d.seq;
    [d.cond unlock];
    return 1;
  }
}

void glydi_cam_close(glydi_cam *c) {
  if (!c) return;
  @autoreleasepool {
    [c->session stopRunning];
    CFRelease((__bridge CFTypeRef)c->session);
    CFRelease((__bridge CFTypeRef)c->delegate);
    c->session = nil;
    c->delegate = nil;
    free(c);
  }
}
