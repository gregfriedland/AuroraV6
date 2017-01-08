#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: overflowcheck=False
#cython: always_allow_keywords=False
#cython: profile=True


TIMING = True
DEF FFT_TYPE = "double" #"numpy_inner" #"numpy", "double"

import numpy as np
cimport numpy as np
cimport cython
import scipy
from time import time
import sys

ctypedef np.int32_t int32
ctypedef np.int16_t int16
from libc.math cimport log10, log2, sqrt, sin
from libc.stdlib cimport malloc, free


IF FFT_TYPE == "double":
  # Call cfftpack directly
  cdef extern from "AuroraViz/lib/cfftpack.h":
    ctypedef double Treal
  
    void rfftf(int N, Treal data[], const Treal wrk[])
    void rffti(int N, Treal wrk[])
ELIF FFT_TYPE == "float":
  # Call cfftpack directly
  cdef extern from "AuroraViz/lib/cfftpack.h":
    ctypedef float Treal
  
    void rfftf(int N, Treal data[], const Treal wrk[])
    void rffti(int N, Treal wrk[])
ELIF FFT_TYPE == "numpy_inner":
  import numpy.fft.fftpack as np_fftpack


# map 0->maxx to 0->1->0
cdef inline float map_triangle(float x):
  if x <= 0.5:
    return x * 2
  else:
    return (1.0 - x) * 2


cdef inline float map_sin(float x):
  return sin(x * 3.14159)



cdef class FindBeatData:
  cdef double startTime
  cdef object params, type
  cdef int32 meanEsIndex, onsetsIndex
  cdef unsigned char [:,:] onsets
  cdef unsigned char [:] hasRecentOnsets, beats, newOnsets
  cdef int32 [:] freqs, bandCounts, bandMap
  cdef float [:,:] meanEs
  cdef float [:] fftData, energies, newMeanEs, fluxEs, lastStdFluxEs, stdFluxEs
  cdef float [:] fluxCutoffEs, movVar, movMean, lastBeatTimes
  IF FFT_TYPE == "double":
    cdef double *fftpack_wsave
    cdef complex *fftpack_ret
  ELIF FFT_TYPE == "float":
    cdef float *fftpack_wsave
    cdef float complex *fftpack_ret
  ELIF FFT_TYPE == "numpy_inner":
    cdef float *dataIn
    cdef float [:] dataIn


  def __init__(self, params, type):
    self.startTime = time()
    self.params = params
    self.type = type
    if type == "group":
      numBands = params.numGroupBands
    else:
      numBands = params.numCalcBands
    
    self.fftData = np.zeros(params.numFrames/2+1, np.float32)
    self.energies = np.zeros(numBands, np.float32)

    self.meanEs, self.meanEsIndex = np.zeros((params.avgWindow, numBands), np.float32), 0
    self.newMeanEs = np.zeros(numBands,np.float32)

    self.onsets, self.onsetsIndex = np.zeros((params.onsetWait, numBands), np.uint8), 0
    self.newOnsets = np.zeros(numBands, np.uint8)

    self.fluxEs = np.zeros(numBands,np.float32)
    self.lastStdFluxEs = np.zeros(numBands,np.float32)
    self.stdFluxEs = np.zeros(numBands,np.float32)
    self.fluxCutoffEs = np.zeros(numBands,np.float32)

    self.hasRecentOnsets = np.zeros(numBands, np.uint8)

    self.movVar = np.zeros(numBands,np.float32)
    self.movMean = np.zeros(numBands,np.float32)

    self.freqs = np.zeros(numBands,np.int32)
    self.beats = np.zeros(numBands,np.uint8)

    self.bandMap = np.zeros(numBands,np.int32)
    self.bandCounts = np.zeros(numBands,np.int32)
    
    self.lastBeatTimes = np.zeros(numBands, np.float32)
    cdef int b
    for b in range(numBands):
      self.lastBeatTimes[b] = -self.params.beatWaitTime*2

    # allocate memory with malloc
    IF FFT_TYPE == "double":
      self.fftpack_wsave = <double *>malloc((2*params.numFrames+15) * sizeof(double))
      self.fftpack_ret = <complex *>malloc(params.numFrames * sizeof(complex))
      if not self.fftpack_ret or not self.fftpack_wsave:
        raise MemoryError()

      # init FFTPACK wsave array
      rffti(params.numFrames, self.fftpack_wsave)

    ELIF FFT_TYPE == "float":
      self.fftpack_wsave = <float *>malloc((2*params.numFrames+15) * sizeof(float))
      self.fftpack_ret = <float complex *>malloc(params.numFrames * sizeof(float complex))
      if not self.fftpack_ret or not self.fftpack_wsave:
        raise MemoryError()
    
      # init FFTPACK wsave array
      rffti(params.numFrames, self.fftpack_wsave)


  cpdef getOnsets(self):
    return self.newOnsets


  cpdef getBeats(self):
    return self.beats


  cpdef getBeatIntensity(self, type="triangle"):
    cdef float currTime = self.getCurrentTime()
    cdef int i, n=self.lastBeatTimes.shape[0]
    cdef float [:] result = np.zeros(n, np.float32)

    cdef float (*map_func)(float)
    if type == "triangle":
      map_func = map_triangle
    else:
      map_func = map_sin

    cdef float maxx = self.params.beatHoldTime
    for i in range(n):
      result[i] = (currTime - self.lastBeatTimes[i]) / maxx
      if result[i] > 1:
        result[i] = 0
      else:
        result[i] = map_func(result[i])
    
    return result


  cdef inline float getCurrentTime(self):
    return time() - self.startTime


  IF FFT_TYPE == "double" or FFT_TYPE == "float":
    def __dealloc__(self):
      free(self.fftpack_wsave)
      free(self.fftpack_ret)


  def __getitem__(self, attr):
    if attr == "energy":
      return np.array(self.energies)
    elif attr == "fft":
      return np.array(self.fftData)
    elif attr == "meanE":
      return np.array(self.newMeanEs)
    elif attr == "onset":
      return np.array(self.newOnsets)
    elif attr in ["derivE", "fluxE"]:
      return np.array(self.fluxEs)
    elif attr in ["stdDerivE", "stdFluxE"]:
      return np.array(self.lastStdFluxEs)
    elif attr in ["derivCutoffE", "fluxCutoffE"]:
      return np.array(self.fluxCutoffEs)
    elif attr in ["freq"]:
      return np.array(self.freqs)
    elif attr in ["bandMap"]:
      return np.array(self.bandMap)
    elif attr in ["beat"]:
      return np.array(self.beats)
    elif attr in ["lastBeatTime"]:
      return np.array(self.lastBeatTimes)
    else:
      raise ValueError("Invalid attribute")



cdef class FindBeats(object):
  cdef FindBeatData d, groupD
  cdef int32 [:] groupBandMap, numGroupOnsets
  cdef params
  
  def __init__(self, params):
    self.params = params

    self.d = FindBeatData(self.params, type="calc")
    self.groupD = FindBeatData(self.params, type="group")

    self.groupBandMap = np.repeat(np.arange(params.numGroupBands,dtype=np.int32),
                                  params.numCalcBands/params.numGroupBands)
    self.numGroupOnsets = np.zeros(params.numGroupBands, np.int32)

    self._loadBandMap()
    self._getFreqs()
  
  
  def _loadBandMap(self):
    nf = self.params.numFrames/2+1
    ncb = self.params.numCalcBands
    
    if not self.params.logScale:
      bandWidth = int(np.ceil(float(nf)/ncb))
      self.d.bandCounts = np.array([bandWidth for b in range(ncb)], np.int32)
    else:
      # logbandwidth = log2(1024)/20 = 10/20 = 1/2
      # 2^0.5 = 1.4, 2^1=2, 2^1.5=2.8, 2^2=4
      logBandWidth = log2(nf) / ncb

      # create array of size ncb with number of frames per band in each entry
      bandCounts = np.power(2,logBandWidth*np.arange(ncb))
      bandCounts = bandCounts * nf / bandCounts.sum()
      bandCounts = np.ceil(bandCounts).astype(np.int32)
      bandCounts[bandCounts==0] = 1
      diff = bandCounts.sum() - nf
      while diff != 0:
        assert(diff >= 0)
        #print "post: ", bandCounts
        bandCounts[-diff:] -= 1
        bandCounts[bandCounts==0] = 1
        #print "post2: ", bandCounts
        diff = bandCounts.sum() - nf

      #print "numbands: " , bandCounts.sum()
      assert(bandCounts.sum() == nf)
      self.d.bandCounts = bandCounts
    
    calcBands = np.arange(self.params.numCalcBands,dtype=np.int32)
    self.d.bandMap = np.repeat(calcBands, self.d.bandCounts)
    
    
  def _getFreqs(self):
    cdef int32 [:] bandMap = self.d.bandMap
  
    # numpy.fft.rfftfreq is not in numpy v1.7 or previously
    fftFreqs = scipy.fftpack.rfftfreq(self.params.numFrames, 1.0/self.params.sampleRate)
    fftFreqs = fftFreqs.take([0]+range(1,fftFreqs.shape[0],2))

    #print "cython channels:", [a for a in self.d.bandMap]
    freqs = np.zeros(self.params.numCalcBands, np.int32)
    cdef int b
    for b in range(self.params.numCalcBands):
      count = self.d.bandCounts[b]
      finds = np.where(np.array(bandMap,np.int32) == b) # this cast is necessary b/c typed mem views don't work well with numpy ops
      freqs[b] = np.median(fftFreqs[finds])
      
    self.d.freqs = freqs
    

  def findBeats(self, int16 [:] data):
    # if the data is corrupt, return the same state
    if self._dataIsEmpty(data):
      return self.groupD, self.d
  
    #with Timer("FFT"):
    self._calcFFT(data)
    #with Timer("Energies"):
    self._getEnergies()
    #with Timer("Onsets"):
    self._findOnsets()
    #with Timer("Group"):
    self._groupResults()
    #with Timer("Beats"):
    self._findGroupBeats()
          
    return self.groupD, self.d
  
  
  cdef _dataIsEmpty(self, data):
    cdef int i
    for i in range(data.shape[0]):
      if data[i] != 0:
        return False
    return True


  cdef _findGroupBeats(self):
    cdef int cb, gb
    cdef int32 [:] groupBandMap = self.groupBandMap
    cdef unsigned char [:] beats = self.d.beats
    cdef unsigned char [:] groupBeats = self.groupD.beats

    self._findBeats(self.groupD.newOnsets, self.groupD.lastBeatTimes,
                    self.groupD.beats)


    if self.params.numCalcBands != self.params.numGroupBands:
      for cb in range(self.params.numCalcBands):
        gb = groupBandMap[cb]
        beats[cb] = groupBeats[gb]
    else:
      self._findBeats(self.d.newOnsets, self.d.lastBeatTimes, self.d.beats)
    #results["beatIntensity"] = ramp(currTime - self.lastBeatTimesCalc, self.params.beatHoldTime, 1.0)


  cdef _calcFFT(self, int16 [:] data):
    cdef int i
    cdef int n = data.shape[0]
    
    IF FFT_TYPE == "numpy":
      cdef complex [:] fftOut

      # exposed numpy calls
      with Timer("FFT Inner"):
        fftOut = np.fft.rfft(data)
    ELIF FFT_TYPE == "numpy_inner":
      cdef complex [:] fftOut
      cdef np.ndarray[np.float32_t, ndim=1] dataIn = self.dataIn
      for i in range(n):
        dataIn[i] = data[i]
      #fftOut =  np_fftpack._raw_fft(dataIn, n, -1, np_fftpack.fftpack.rffti, np_fftpack.fftpack.rfftf, np_fftpack._real_fft_cache)
      fftOut =  np_fftpack._raw_fft(dataIn, n, -1, np_fftpack.fftpack.rffti, np_fftpack.fftpack.rfftf, np_fftpack._real_fft_cache)
        
    ELIF FFT_TYPE == "double":
      cdef double *rptr = <double *>self.d.fftpack_ret
      for i in range(n):
        rptr[i+1] = data[i]

      #with Timer("FFTinner"):
      rfftf(n, rptr+1, self.d.fftpack_wsave)
      rptr[0] = rptr[1]
      rptr[1] = 0.0
      cdef complex *fftOut = <complex *> rptr
    ELIF FFT_TYPE == "float":
      cdef float *rptr = <float *>self.d.fftpack_ret
      for i in range(n):
        rptr[i+1] = data[i]

      with Timer("FFTinner"):
        rfftf(n, rptr+1, self.d.fftpack_wsave)
      rptr[0] = rptr[1]
      rptr[1] = 0.0
      cdef float complex *fftOut = <float complex *> rptr
    
    cdef int32 fs2 = self.params.numFrames ** 2
    cdef float complex cVal
    cdef float [:] fftData = self.d.fftData
    for i in range(fftData.shape[0]):
      cVal = <float complex>fftOut[i]
      fftData[i] = (cVal.real**2 + cVal.imag**2) / fs2


  cdef _getEnergies(self):
    cdef float [:] fftData = self.d.fftData
    cdef float [:] energies = self.d.energies
    cdef int32 [:] bandMap = self.d.bandMap
    cdef int32 [:] bandCounts = self.d.bandCounts
    cdef int32 b, c, fi
    cdef float scale = self.params.energyScale

    for b in range(energies.shape[0]):
      energies[b] = 0

    for fi in range(fftData.shape[0]):
      b = bandMap[fi]
      c = bandCounts[b]
      energies[b] += fftData[fi] / c
    
    for b in range(energies.shape[0]):
      energies[b] = log10(energies[b]) / scale
      if energies[b] < 0:
        energies[b] = 0

    
  cdef _findOnsets(self):
    cdef int32 ncb = self.params.numCalcBands
    cdef int32 b, h
  
    cdef float [:] energies = self.d.energies
    cdef float [:] lastMeanEs = self.d.meanEs[self.d.meanEsIndex,:]
    cdef unsigned char [:,:] onsets = self.d.onsets
    
    # use moving average to find the new mean energies
    cdef float meanWindow = self.params.avgWindow
    cdef float [:] newMeanEs = self.d.newMeanEs
    for b in range(ncb):
      newMeanEs[b] = lastMeanEs[b] + (energies[b] - lastMeanEs[b]) / meanWindow

    # calculate the instantaneous change in mean energies
    cdef float [:,:] allMeanEs = self.d.meanEs
    cdef float [:] fluxEs = self.d.fluxEs
    cdef int32 hmax = allMeanEs.shape[0]
    for b in range(ncb):
       fluxEs[b] = 0
       for h in range(hmax):
         fluxEs[b] += newMeanEs[b] - allMeanEs[h,b]
       if fluxEs[b] < 0:
         fluxEs[b] = 0

    # add the newMeanEs to the history
    self.d.meanEsIndex = (self.d.meanEsIndex + 1) % self.d.meanEs.shape[0]
    allMeanEs[self.d.meanEsIndex,:] = newMeanEs

    # if an onset has occurred recently, don't record this flux
    cdef unsigned char [:] hasRecentOnsets = self.d.hasRecentOnsets
    hmax = onsets.shape[0]
    if self.params.onsetWait > 0:
      for b in range(ncb):
        hasRecentOnsets[b] = 0
        for h in range(hmax):
          hasRecentOnsets[b] = onsets[h,b] or hasRecentOnsets[b]
        if hasRecentOnsets[b]:
          fluxEs[b] = 0

    # determine an onset before recalculating the fluxEs
    cdef float [:] stdFluxEs = self.d.stdFluxEs
    cdef float [:] lastStdFluxEs = self.d.lastStdFluxEs
    cdef float [:] fluxCutoffEs = self.d.fluxCutoffEs
    cdef float sensitivity = self.params.sensitivity
    cdef unsigned char [:] newOnsets = self.d.newOnsets
    self.d.onsetsIndex = (self.d.onsetsIndex + 1) % self.d.onsets.shape[0]
    for b in range(ncb):
      # keep track of the last stdFluxEs
      lastStdFluxEs[b] = stdFluxEs[b]

      fluxCutoffEs[b] = sensitivity * stdFluxEs[b]
      newOnsets[b] = fluxEs[b] > fluxCutoffEs[b]
      onsets[self.d.onsetsIndex,b] = newOnsets[b]

    # now update the moving stdFluxEs with the new fluxEs
    cdef float a = 1-self.params.stdAlpha
    cdef float [:] movVar = self.d.movVar
    cdef float [:] movMean = self.d.movMean
    cdef float diff, incr
    for b in range(ncb):
      diff = fluxEs[b] - movMean[b]
      incr = (1 - a) * diff
      movMean[b] += incr
      movVar[b] = a * (movVar[b] + diff * incr)
      stdFluxEs[b] = sqrt(movVar[b])


  cdef _findBeats(self, unsigned char [:] onsets, float [:] lastBeatTimes,
                 unsigned char [:] beats):
    cdef double currTime = time()
    currTime = self.d.getCurrentTime()
    cdef float timeDiff
    cdef int32 holding, waiting, enoughOnsets
    cdef float holdTime = self.params.beatHoldTime
    cdef float waitTime = self.params.beatWaitTime
    cdef int32 minOnsets = self.params.minOnsetsOut
  
    # count total onsets
    cdef int32 numOnsets = 0
    cdef int32 b
    for b in range(onsets.shape[0]):
      numOnsets += onsets[b]
    
    for b in range(onsets.shape[0]):
      timeDiff = currTime - lastBeatTimes[b]
      holding = timeDiff <= holdTime
      waiting = timeDiff > holdTime and timeDiff <= waitTime

      enoughOnsets = <int>onsets[b] and <int>(numOnsets >= minOnsets)
      #print "startTime=%.3f timediff=%.3f currtime=%.3f lastbeattime=%.3f holding=%d waiting=%d enoughOnsets=%d" % (self.startTime, timeDiff, currTime, lastBeatTimes[b], holding, waiting, enoughOnsets)
      if holding:
        beats[b] = True
        #print "band %d is holding" % (b)
      elif waiting:
        beats[b] = False
      elif enoughOnsets:
        beats[b] = True
        lastBeatTimes[b] = currTime
        #print lastBeatTimes[b],
      else:
        beats[b] = False
    #print
    #print


  cdef _groupResults(self):
    cdef int32 ncb = self.params.numCalcBands
    cdef int32 ngb = self.params.numGroupBands
    if ncb == ngb: return self.d
    assert(ngb < ncb and ncb % ngb == 0)
    cdef int32 bandsPerGroup = ncb / ngb
    cdef int32 gb, cb

    # create map of cb to gb
    cdef int32 [:] groupBandMap = self.groupBandMap

    # count total onsets
    cdef unsigned char [:] newOnsets = self.d.newOnsets
    cdef int32 numOnsetsCalc = 0
    for cb in range(ncb):
      if newOnsets[cb]:
        numOnsetsCalc += 1

    # group onsets
    cdef unsigned char [:] groupOnsets = self.groupD.newOnsets
    cdef int32 [:] numGroupOnsets = np.zeros(ngb, np.int32)
    cdef int minOnsetsGroup = self.params.minOnsetsGroup

    # count group onsets
    for gb in range(ngb):
      numGroupOnsets[gb] = 0
      groupOnsets[gb] = 0

    if numOnsetsCalc >= self.params.minOnsetsPreGroup:
      
      for cb in range(ncb):
        gb = groupBandMap[cb]
        numGroupOnsets[gb] += newOnsets[cb]
      
      for gb in range(ngb):
        groupOnsets[gb] = numGroupOnsets[gb] >= minOnsetsGroup
    #print "newOnsets", np.array(newOnsets)
    #print "numgroupOnsets", np.array(numGroupOnsets)

    # set group freqs  FIX slow
    cdef int32 [:] freqs = self.d.freqs
    cdef int32 [:] groupFreqs = self.groupD.freqs
    for gb in range(ngb):
      for cb in range(ncb):
        if groupBandMap[cb] == gb:
          groupFreqs[gb] = freqs[cb+bandsPerGroup/2]
          break
      

