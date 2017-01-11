// Arduino library to find beats in music input from the 7 frequency bands of an MSGEQ7 IC.
// Copyright Greg Friedland 2011 greg.friedland@gmail.com

#ifndef FINDBEATS_H
#define FINDBEATS_H

#include <fftw3.h>
#include <Arduino.h>
#include "CircularBuffer.h"

typedef int EnergyVal;

template <int NUM_BANDS>
class FindBeats {
 public:

  FindBeats(size_t stepSize, size_t bufferSize);
  virtual ~FindBeats();

  // how long to hold the beat (in ms)
  void setBeatHoldTime(uint16_t beatHoldTime) { _beatHoldTime = beatHoldTime; }
  uint16_t getBeatHoldTime() { return _beatHoldTime; }

  // how long to wait before another beat is allowed (in ms)
  void setBeatWaitTime(uint16_t beatWaitTime) { _beatWaitTime = beatWaitTime; }
  uint16_t getBeatWaitTime() { return _beatWaitTime; }

  // when did the start of the last beat occur
  uint32_t getLastBeatTime(byte band) { return _lastBeatTime[band]; }

  // is a beat happening now
  boolean isBeat(byte band) { return _beats[band]; }

  // how recent is the beat where 255 is when it starts and 0 is when the hold time is over
  byte getBeatRecentness(byte band);
  
  // how sharp (abrupt) was the beat
  byte getBeatSharpness(byte band) { return _beatIntensity[band]; }
  
  // the energy level of this frequency band
  byte getEnergy(byte band, int historyPeriod);
  
  // run the beat calculations
  void updateBeats();

 private:
  void calcBeats();
  size_t m_bufferSize, m_stepSize;
  std::vector<size_t> m_logSpacing;

  fftw_complex *m_fftIn, *m_fftOut;
  fftw_plan m_fftPlan;  
};
#endif
