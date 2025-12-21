// Minimal stub for ESP_I2S.h to allow building on non-audio hosts.
// This provides a lightweight I2SClass with the methods used by the
// project's sketch. It does not implement real audio functionality.

#pragma once

#include <Arduino.h>

// Minimal mode/bit definitions used by the sketch
#define I2S_MODE_PDM_RX 1
#define I2S_MODE_STD 2
#define I2S_DATA_BIT_WIDTH_16BIT 16
#define I2S_DATA_BIT_WIDTH_32BIT 32
#define I2S_SLOT_MODE_MONO 1
#define I2S_SLOT_MODE_STEREO 2

class I2SClass {
public:
  I2SClass() {}

  // For PDM input
  void setPinsPdmRx(int clockPin, int dataPin) {
    (void)clockPin; (void)dataPin;
  }

  // For standard output pins
  void setPins(int bclk, int lrck, int din) {
    (void)bclk; (void)lrck; (void)din;
  }

  // Begin with a mode and parameters. Return true for success.
  bool begin(int mode, int sampleRate, int bitWidth, int slotMode) {
    (void)mode; (void)sampleRate; (void)bitWidth; (void)slotMode;
    return true;
  }

  // Read a sample for PDM input. Return -1 when no data is available.
  int read() {
    // Provide no data; the sketch handles -1 as 'no sample yet'.
    return -1;
  }

  // Write audio data for playback. Return number of bytes written.
  size_t write(const uint8_t* buf, size_t len) {
    (void)buf; return len;
  }
};

// Instances are defined in the sketch; provide extern declarations here
extern I2SClass i2sIn;
extern I2SClass i2sOut;
