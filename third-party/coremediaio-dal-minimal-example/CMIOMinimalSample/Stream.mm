//
//  Stream.mm
//  CMIOMinimalSample
//
//  Created by John Boiles  on 4/10/20.
//
//  CMIOMinimalSample is free software, and use is bound by the terms
//  set out in the LICENSE file distributed with this project.

#import "Stream.h"

#import <AppKit/AppKit.h>
#import <mach/mach_time.h>
#include <CoreMediaIO/CMIOSampleBuffer.h>

#import "Logging.h"

@interface Stream () {
    CMSimpleQueueRef _queue;
    CFTypeRef _clock;
    NSImage *_testImage;
    dispatch_source_t _frameDispatchSource;
    uint64_t _firstFrameDeliveryTime;
}

@property CMIODeviceStreamQueueAlteredProc alteredProc;
@property void * alteredRefCon;
@property (readonly) CMSimpleQueueRef queue;
@property (readonly) CFTypeRef clock;
@property UInt64 sequenceNumber;
@property (readonly) NSImage *testImage;

@end

@implementation Stream

#define FPS 30.0

- (instancetype _Nonnull)init {
    self = [super init];
    if (self) {
        _firstFrameDeliveryTime = 0;
        _frameDispatchSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                                     dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));

        dispatch_time_t startTime = dispatch_time(DISPATCH_TIME_NOW, 0);
        uint64_t intervalTime = (int64_t)(NSEC_PER_SEC / FPS);
        dispatch_source_set_timer(_frameDispatchSource, startTime, intervalTime, 0);

        __weak typeof(self) wself = self;
        dispatch_source_set_event_handler(_frameDispatchSource, ^{
            [wself fillFrame];
        });
    }
    return self;
}

- (void)dealloc {
    DLog(@"Stream Dealloc");
    CMIOStreamClockInvalidate(_clock);
    CFRelease(_clock);
    _clock = NULL;
    CFRelease(_queue);
    _queue = NULL;
    dispatch_suspend(_frameDispatchSource);
}

- (void)startServingFrames {
    dispatch_resume(_frameDispatchSource);
}

- (void)stopServingFrames {
    dispatch_suspend(_frameDispatchSource);
    _firstFrameDeliveryTime = 0;
}

- (CMSimpleQueueRef)queue {
    if (_queue == NULL) {
        // Allocate a one-second long queue, which we can use our FPS constant for.
        OSStatus err = CMSimpleQueueCreate(kCFAllocatorDefault, FPS, &_queue);
        if (err != noErr) {
            DLog(@"Err %d in CMSimpleQueueCreate", err);
        }
    }
    return _queue;
}

- (CFTypeRef)clock {
    if (_clock == NULL) {
        OSStatus err = CMIOStreamClockCreate(kCFAllocatorDefault, CFSTR("CMIOMinimalSample::Stream::clock"), (__bridge void *)self,  CMTimeMake(1, 10), 100, 10, &_clock);
        if (err != noErr) {
            DLog(@"Error %d from CMIOStreamClockCreate", err);
        }
    }
    return _clock;
}

- (NSImage *)testImage {
    if (_testImage == nil) {
        NSBundle *bundle = [NSBundle bundleForClass:[self class]];
        _testImage = [bundle imageForResource:@"hi"];
    }
    return _testImage;
}

- (CMSimpleQueueRef)copyBufferQueueWithAlteredProc:(CMIODeviceStreamQueueAlteredProc)alteredProc alteredRefCon:(void *)alteredRefCon {
    self.alteredProc = alteredProc;
    self.alteredRefCon = alteredRefCon;

    // Retain this since it's a copy operation
    CFRetain(self.queue);

    return self.queue;
}


// https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both

typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // a fraction between 0 and 1
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h * 360;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}

- (CVPixelBufferRef)createPixelBufferWithTestAnimation {
    int width = 1280;
    int height = 720;

    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGImageCompatibilityKey,
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGBitmapContextCompatibilityKey, nil];
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, (__bridge CFDictionaryRef) options, &pxbuffer);

    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);

    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    NSParameterAssert(pxdata != NULL);

    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, width, height, 8, CVPixelBufferGetBytesPerRow(pxbuffer), rgbColorSpace, kCGImageAlphaPremultipliedFirst | kCGImageByteOrder32Big);
    NSParameterAssert(context);
    
    NSString *filePath = @"/tmp/camera.bmp";
    bool exists =[[NSFileManager defaultManager] fileExistsAtPath:filePath];
    
    if (exists) {
        CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)[NSData dataWithContentsOfFile:filePath]);
        static const size_t kComponentsPerPixel = 3;
        static const size_t kBitsPerComponent = sizeof(unsigned char) * 8;
        CGImageRef image = CGImageCreate(width, height, kBitsPerComponent,
                      kBitsPerComponent * kComponentsPerPixel,
                      kComponentsPerPixel * width,
                      rgbColorSpace,
                      kCGBitmapByteOrderDefault | kCGImageAlphaNone,
                      provider, NULL, false, kCGRenderingIntentDefault);
        
        CGRect imageRect = CGRectMake(0, 0, width, height);
        
        CGContextSaveGState(context);
        CGContextTranslateCTM(context, 0, height);
        CGContextScaleCTM(context, 1.0, -1.0);
        CGContextDrawImage(context, imageRect, image);
        CGContextRestoreGState(context);
        CGImageRelease(image);
        CGDataProviderRelease(provider);
    } else {
        CGColorRef backgroundColor = CGColorCreateGenericRGB(1, 1, 1, 1);
        CGContextSetFillColorWithColor(context, backgroundColor);
        CGContextFillRect(context, CGRectMake(0, 0, width, height));
        CGColorRelease(backgroundColor);
        
        double time = double(mach_absolute_time()) / NSEC_PER_SEC;
        double period = 4.0;
        double fraction = sin(2 * M_PI * time / period) / 2 + 0.5;
        hsv rawBoxColorHSV;
        rawBoxColorHSV.h = fraction;
        rawBoxColorHSV.s = 0.5;
        rawBoxColorHSV.v = 1.0;
        rgb rawBoxColorRGB = hsv2rgb(rawBoxColorHSV);
        CGColorRef boxColor = CGColorCreateGenericRGB(rawBoxColorRGB.r, rawBoxColorRGB.g, rawBoxColorRGB.b, 1);
        CGContextSetFillColorWithColor(context, boxColor);
        for (int i = 0; i < 3; i++) {
            CGContextSaveGState(context);
            double rotation_fraction = time / 3.0 + double(i) / 3.0;
            CGContextTranslateCTM(context, width / 2, height / 2);
            CGContextRotateCTM(context, 2 * M_PI * rotation_fraction);
            double x_center = width * 0.1;
            double y_center = 0.0;
            CGContextFillEllipseInRect(context, CGRectMake(CGFloat(x_center - 50), CGFloat(y_center - 50), 100, 100));
            CGContextRestoreGState(context);
        }
        CGColorRelease(boxColor);
    }
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);

    return pxbuffer;
}

- (void)fillFrame {
    if (CMSimpleQueueGetFullness(self.queue) >= 1.0) {
        DLog(@"Queue is full, bailing out");
        return;
    }

    CVPixelBufferRef pixelBuffer = [self createPixelBufferWithTestAnimation];

    // The timing here is quite important. For frames to be delivered correctly and successfully be recorded by apps
    // like QuickTime Player, we need to be accurate in both our timestamps _and_ have a sensible scale. Using large
    // scales like NSEC_PER_SEC combined with a mach_absolute_time() value will work for display, but will error out
    // when trying to record.
    //
    // Instead, we start our presentation times from zero (using the sequence number as a base), and use a scale that's
    // a multiple of our framerate. This has been observed in parts of AVFoundation and lets us be frame-accurate even
    // on non-round framerates (i.e., we can use a scale of 2997 for 29,97 fps content if we want to).
    //
    // It's also been observed that we do seem to need a mach_absolute_time()-based value for presentation times in
    // order to get reliable output and recording. Since we don't want to just call mach_absolute_time() on every
    // frame (otherwise recorded output from this plugin will have frame timing based on the scheduling of our timer,
    // which isn't guaranteed to be accurate), we record the system's absolute time on our first frame, then calculate
    // a delta from these for subsequent frames. This keeps presentation times accurate even if our timer isn't.
    if (_firstFrameDeliveryTime == 0) {
        _firstFrameDeliveryTime = mach_absolute_time();
    }

    CMTimeScale scale = FPS * 100;
    CMTime firstFrameTime = CMTimeMake((_firstFrameDeliveryTime / (CFTimeInterval)NSEC_PER_SEC) * scale, scale);
    CMTime frameDuration = CMTimeMake(scale / FPS, scale);
    CMTime framesSinceBeginning = CMTimeMake(frameDuration.value * self.sequenceNumber, scale);
    CMTime presentationTime = CMTimeAdd(firstFrameTime, framesSinceBeginning);

    CMSampleTimingInfo timing;
    timing.duration = frameDuration;
    timing.presentationTimeStamp = presentationTime;
    timing.decodeTimeStamp = presentationTime;
    OSStatus err = CMIOStreamClockPostTimingEvent(presentationTime, mach_absolute_time(), true, self.clock);
    if (err != noErr) {
        DLog(@"CMIOStreamClockPostTimingEvent err %d", err);
    }

    CMFormatDescriptionRef format;
    CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer, &format);

    self.sequenceNumber = CMIOGetNextSequenceNumber(self.sequenceNumber);

    CMSampleBufferRef buffer;
    err = CMIOSampleBufferCreateForImageBuffer(
        kCFAllocatorDefault,
        pixelBuffer,
        format,
        &timing,
        self.sequenceNumber,
        kCMIOSampleBufferNoDiscontinuities,
        &buffer
    );
    CFRelease(pixelBuffer);
    CFRelease(format);
    if (err != noErr) {
        DLog(@"CMIOSampleBufferCreateForImageBuffer err %d", err);
    }

    CMSimpleQueueEnqueue(self.queue, buffer);

    // Inform the clients that the queue has been altered
    if (self.alteredProc != NULL) {
        (self.alteredProc)(self.objectId, buffer, self.alteredRefCon);
    }
}

- (CMVideoFormatDescriptionRef)getFormatDescription {
    CMVideoFormatDescriptionRef formatDescription;
    OSStatus err = CMVideoFormatDescriptionCreate(kCFAllocatorDefault, kCMVideoCodecType_422YpCbCr8, 1280, 720, NULL, &formatDescription);
    if (err != noErr) {
        DLog(@"Error %d from CMVideoFormatDescriptionCreate", err);
    }
    return formatDescription;
}

#pragma mark - CMIOObject

- (UInt32)getPropertyDataSizeWithAddress:(CMIOObjectPropertyAddress)address qualifierDataSize:(UInt32)qualifierDataSize qualifierData:(nonnull const void *)qualifierData {
    switch (address.mSelector) {
        case kCMIOStreamPropertyInitialPresentationTimeStampForLinkedAndSyncedAudio:
            return sizeof(CMTime);
        case kCMIOStreamPropertyOutputBuffersNeededForThrottledPlayback:
            return sizeof(UInt32);
        case kCMIOObjectPropertyName:
            return sizeof(CFStringRef);
        case kCMIOObjectPropertyManufacturer:
            return sizeof(CFStringRef);
        case kCMIOObjectPropertyElementName:
            return sizeof(CFStringRef);
        case kCMIOObjectPropertyElementCategoryName:
            return sizeof(CFStringRef);
        case kCMIOObjectPropertyElementNumberName:
            return sizeof(CFStringRef);
        case kCMIOStreamPropertyDirection:
            return sizeof(UInt32);
        case kCMIOStreamPropertyTerminalType:
            return sizeof(UInt32);
        case kCMIOStreamPropertyStartingChannel:
            return sizeof(UInt32);
        case kCMIOStreamPropertyLatency:
            return sizeof(UInt32);
        case kCMIOStreamPropertyFormatDescriptions:
            return sizeof(CFArrayRef);
        case kCMIOStreamPropertyFormatDescription:
            return sizeof(CMFormatDescriptionRef);
        case kCMIOStreamPropertyFrameRateRanges:
            return sizeof(AudioValueRange);
        case kCMIOStreamPropertyFrameRate:
        case kCMIOStreamPropertyFrameRates:
            return sizeof(Float64);
        case kCMIOStreamPropertyMinimumFrameRate:
            return sizeof(Float64);
        case kCMIOStreamPropertyClock:
            return sizeof(CFTypeRef);
        default:
            DLog(@"Stream unhandled getPropertyDataSizeWithAddress for %@", [ObjectStore StringFromPropertySelector:address.mSelector]);
            return 0;
    };
}

- (void)getPropertyDataWithAddress:(CMIOObjectPropertyAddress)address qualifierDataSize:(UInt32)qualifierDataSize qualifierData:(nonnull const void *)qualifierData dataSize:(UInt32)dataSize dataUsed:(nonnull UInt32 *)dataUsed data:(nonnull void *)data {
    switch (address.mSelector) {
        case kCMIOObjectPropertyName:
            *static_cast<CFStringRef*>(data) = CFSTR("CMIOMinimalSample Stream");
            *dataUsed = sizeof(CFStringRef);
            break;
        case kCMIOObjectPropertyElementName:
            *static_cast<CFStringRef*>(data) = CFSTR("CMIOMinimalSample Stream Element");
            *dataUsed = sizeof(CFStringRef);
            break;
        case kCMIOObjectPropertyManufacturer:
        case kCMIOObjectPropertyElementCategoryName:
        case kCMIOObjectPropertyElementNumberName:
        case kCMIOStreamPropertyTerminalType:
        case kCMIOStreamPropertyStartingChannel:
        case kCMIOStreamPropertyLatency:
        case kCMIOStreamPropertyInitialPresentationTimeStampForLinkedAndSyncedAudio:
        case kCMIOStreamPropertyOutputBuffersNeededForThrottledPlayback:
            DLog(@"TODO: %@", [ObjectStore StringFromPropertySelector:address.mSelector]);
            break;
        case kCMIOStreamPropertyDirection:
            *static_cast<UInt32*>(data) = 1;
            *dataUsed = sizeof(UInt32);
            break;
        case kCMIOStreamPropertyFormatDescriptions:
            *static_cast<CFArrayRef*>(data) = (__bridge_retained CFArrayRef)[NSArray arrayWithObject:(__bridge_transfer NSObject *)[self getFormatDescription]];
            *dataUsed = sizeof(CFArrayRef);
            break;
        case kCMIOStreamPropertyFormatDescription:
            *static_cast<CMVideoFormatDescriptionRef*>(data) = [self getFormatDescription];
            *dataUsed = sizeof(CMVideoFormatDescriptionRef);
            break;
        case kCMIOStreamPropertyFrameRateRanges:
            AudioValueRange range;
            range.mMinimum = FPS;
            range.mMaximum = FPS;
            *static_cast<AudioValueRange*>(data) = range;
            *dataUsed = sizeof(AudioValueRange);
            break;
        case kCMIOStreamPropertyFrameRate:
        case kCMIOStreamPropertyFrameRates:
            *static_cast<Float64*>(data) = FPS;
            *dataUsed = sizeof(Float64);
            break;
        case kCMIOStreamPropertyMinimumFrameRate:
            *static_cast<Float64*>(data) = FPS;
            *dataUsed = sizeof(Float64);
            break;
        case kCMIOStreamPropertyClock:
            *static_cast<CFTypeRef*>(data) = self.clock;
            // This one was incredibly tricky and cost me many hours to find. It seems that DAL expects
            // the clock to be retained when returned. It's unclear why, and that seems inconsistent
            // with other properties that don't have the same behavior. But this is what Apple's sample
            // code does.
            // https://github.com/lvsti/CoreMediaIO-DAL-Example/blob/0392cb/Sources/Extras/CoreMediaIO/DeviceAbstractionLayer/Devices/DP/Properties/CMIO_DP_Property_Clock.cpp#L75
            CFRetain(*static_cast<CFTypeRef*>(data));
            *dataUsed = sizeof(CFTypeRef);
            break;
        default:
            DLog(@"Stream unhandled getPropertyDataWithAddress for %@", [ObjectStore StringFromPropertySelector:address.mSelector]);
            *dataUsed = 0;
    };
}

- (BOOL)hasPropertyWithAddress:(CMIOObjectPropertyAddress)address {
    switch (address.mSelector){
        case kCMIOObjectPropertyName:
        case kCMIOObjectPropertyElementName:
        case kCMIOStreamPropertyFormatDescriptions:
        case kCMIOStreamPropertyFormatDescription:
        case kCMIOStreamPropertyFrameRateRanges:
        case kCMIOStreamPropertyFrameRate:
        case kCMIOStreamPropertyFrameRates:
        case kCMIOStreamPropertyMinimumFrameRate:
        case kCMIOStreamPropertyClock:
            return true;
        case kCMIOObjectPropertyManufacturer:
        case kCMIOObjectPropertyElementCategoryName:
        case kCMIOObjectPropertyElementNumberName:
        case kCMIOStreamPropertyDirection:
        case kCMIOStreamPropertyTerminalType:
        case kCMIOStreamPropertyStartingChannel:
        case kCMIOStreamPropertyLatency:
        case kCMIOStreamPropertyInitialPresentationTimeStampForLinkedAndSyncedAudio:
        case kCMIOStreamPropertyOutputBuffersNeededForThrottledPlayback:
            DLog(@"TODO: %@", [ObjectStore StringFromPropertySelector:address.mSelector]);
            return false;
        default:
            DLog(@"Stream unhandled hasPropertyWithAddress for %@", [ObjectStore StringFromPropertySelector:address.mSelector]);
            return false;
    };
}

- (BOOL)isPropertySettableWithAddress:(CMIOObjectPropertyAddress)address {
    DLog(@"Stream unhandled isPropertySettableWithAddress for %@", [ObjectStore StringFromPropertySelector:address.mSelector]);
    return false;
}

- (void)setPropertyDataWithAddress:(CMIOObjectPropertyAddress)address qualifierDataSize:(UInt32)qualifierDataSize qualifierData:(nonnull const void *)qualifierData dataSize:(UInt32)dataSize data:(nonnull const void *)data {
    DLog(@"Stream unhandled setPropertyDataWithAddress for %@", [ObjectStore StringFromPropertySelector:address.mSelector]);
}

@end
