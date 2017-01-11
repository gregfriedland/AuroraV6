// Arduino library to find beats in music input from the 7 frequency bands of an MSGEQ7 IC.
// Copyright Greg Friedland 2011 greg.friedland@gmail.com

// TODO:
// - define history size based on time
// - clean up circularbuffer api to get rid of getoffsetindex

#include "FindBeats.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))

#define PRINT_FPS 1
#define DEBUG 0
#define PLOT 0                  // send data to serial out for processing sketch to plot; DEBUG must be off
#define MOVING_AVG_WINDOW_SIZE 5  // how many samples to use in the moving average
#define MIN_CUTOFF_DIFF 50        // the min diff between the energy and the cutoff


// create a temporary array storing the # of times each value was found then find the middle value
// won't work for floats/doubles!
// e.g. 0:1 1:0 2:3 3:0 4:1 (median=2)
// note: is not correct if number of values is zero and averaging is necessary; but good enough for these purposes
static unsigned int fastMedian(unsigned int *a, unsigned int n, unsigned int maxVal) {
  //Serial.println("gothere1");

  // create the temp array
  unsigned int *counts = (unsigned int*) malloc(sizeof(unsigned int) * (maxVal+1));
  memset(counts, 0, sizeof(unsigned int) * (maxVal+1));
  //Serial.println("gothere2");
  
  for (int i=0; i<n; i++) {
    counts[a[i]]++;
  }
  
  // find the median
  //Serial.println("gothere3");
  unsigned int count = 0, currVal = 0, currValCount = 0;
  while (count < (n+1)/2) {
//    Serial.print("count="); Serial.print(count); Serial.print(" currVal="); Serial.print(currVal);
//    Serial.print(" currValCount="); Serial.print(currValCount); Serial.print(" counts[currVal]="); Serial.println(counts[currVal]);
    
    currValCount++;
    count++;
    if (currValCount > counts[currVal]) {
      // find the next nonzero count value
      currVal++;
      while(counts[currVal] == 0) currVal++;
      currValCount = 1;
//      Serial.println("Moving to the next val");
    }
  }
  
//  Serial.print("count="); Serial.print(count); Serial.print(" currVal="); Serial.print(currVal);
//  Serial.print(" currValCount="); Serial.print(currValCount); Serial.println();
  
  free(counts);
  //Serial.println("gothere4");

  return currVal;
}

static unsigned int fastMad(unsigned int *a, unsigned int n, unsigned int median) {
  //Serial.println("gothere2.1");
  unsigned int maxVal = 0;
  unsigned int *diff = (unsigned int*)malloc(sizeof(unsigned int)*n);
  for (int i=0; i<n; i++) {
    diff[i] = abs(a[i] - median);
    maxVal = MAX(maxVal, diff[i]);
  //Serial.println("gothere2.1");
  }
  
  unsigned int madVal = fastMedian(diff, n, maxVal);
  
  free(diff);
  return madVal;
  
}

template <int NUM_BANDS>
FindBeats::FindBeats(size_t stepSize, size_t bufferSize) {
: m_stepSize(stepSize), m_bufferSize(bufferSize) {
    size_t N = m_bufferSize;
    m_fftIn = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    m_fftOut = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    m_fftPlan = fftw_plan_dft_1d(N, m_fftIn, m_fftOut, FFTW_FORWARD, FFTW_ESTIMATE);

    m_logSpacing = logspace<size_t>(0, bufferSize, NUM_BANDS + 1);
    std::cout << "logspacing=";
    for (auto val: m_logSpacing) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
}

~FindBeats() {
  fftw_destroy_plan(m_fftPlan);
  fftw_free(m_fftIn); fftw_free(m_fftOut);  
}

// Find beats by looking for signals above a certain number of MADs from the median
template <int NUM_BANDS>
void FindBeats::calcBeats() {
  uint32_t currentTime = millis();
  
  for (byte b=0; b<NUM_BANDS; b++) {
    if (!_bandsEnabled[b]) continue;

    byte scale = 10 - _bandsSensitivity[b];

    // get the moving avg of the energy
    EnergyVal meanE = _energies[b].mean(_energies[b].getOffsetIndex(-MOVING_AVG_WINDOW_SIZE), _energies[b].getIndex());
    _meanEnergies[b].append(meanE);
    
    // use the median and the MAD (median absoulate deviation) to compute the cutoff
    // since these are robust metrics
    _medianEnergy[b] = _meanEnergies[b].median();
    EnergyVal madE = _meanEnergies[b].mad(_medianEnergy[b]);
    
    EnergyVal cutoff = _medianEnergy[b] + scale * madE;

    // the energy must be at least a small amount above the cutoff to be considered an onset
    boolean onset = meanE - cutoff >= MIN_CUTOFF_DIFF;
    
    if (currentTime - _lastBeatTime[b] <= _beatHoldTime) {
      // hold the beat
      _beats[b] = true;
    } else if (currentTime - _lastBeatTime[b] < _beatWaitTime) {
      // haven't waited long enough for the next beat
      _beats[b] = false;
    } else if (onset) {
      // beginning of a new beat
      _beats[b] = true;
      _lastBeatTime[b] = currentTime;
      _beatIntensity[b] = constrain(map(meanE - cutoff, MIN_CUTOFF_DIFF, 250, 0, 255), 0, 255);
    }
  }
}


// store beat information
template <int NUM_BANDS>
void FindBeats::updateBeats(float* data) {
  fftw_execute(p); /* repeat as needed */

  // sum fft output into logarithmic bins
  for (size_t i = 0; i < m_bufferSize) {
    
  }    
}

// get the intensity of the beat as it decays from the max value over the onset hold time
template <int NUM_BANDS>
byte FindBeats::getBeatRecentness(byte band) {
  uint32_t currentTime = millis();
  return constrain(map(currentTime - _lastBeatTime[band], 0, _beatHoldTime, 255, 0), 0, 255);
}

template <int NUM_BANDS>
byte FindBeats::getEnergy(byte band, int historyPeriod) {
  int windowSize = historyPeriod / _updatePeriod;
  
  int meanE = _energies[band].mean(_energies[band].getIndex() - windowSize, _energies[band].getIndex());
  return constrain(map(meanE, 0, 1023, 0, 255), 0, 255);
}
