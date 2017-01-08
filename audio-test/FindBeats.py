from CircularBuffer import CircularBuffer2D
import time
import numpy, scipy.fftpack
import os
import cPickle
import numexpr

USE_NUMEXPR = False
LINE_PROFILE = False
TIMING = False
FFT_METHOD = "numpy"

if FFT_METHOD == "ftw":
  import pyfftw
  WISDOM_FILE = "pyfftw-wisdom.txt"
  pyfftw.interfaces.cache.enable()

# otherwise the @profile decorators will get errors
if not LINE_PROFILE:
  def profile(ob):
    return ob


class Timer:
  def __init__(self, name):
    self.name = name
  
  def __enter__(self):
    if TIMING:
      self.cpu = time.clock()
      self.wall = time.time()
  
  def __exit__(self, *args):
    if TIMING:
      print "%10s: %4.1fms" % (self.name, 1000*(time.time() - self.wall))



def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def constrain(val, low, hi):
  return min(max(low, val), hi)


def ma(prev, new, n):
  return prev + 1.0/n * (new - prev) # prev + new/n - prev/n = prev*((n-1)/n) + new/n = (prev*(n-1) + new)/n


def ema(prev, new, a):
  return a*prev + (1 - a) * new

#def triangle_wave(length, amplitude):
#  mid = (length+1)/2
#  rampUp = np.linspace(0, amplitude, mid)
#
#  a = numpy.array.zeros(length, numpy.float32)
#  a[:mid] = rampUp
#  a[mid:] = rampUp[::-1]
#
#  return a

def ramp(x, maxx, maxy):
  # boundaries val=0: return=0; val=length/4: return amplitude/2; val=length: return 0
  peakx = maxx/2.0
  y = numpy.where(x <= peakx, maxy*x/peakx, maxy*(peakx-x)/peakx)
  return numpy.clip(y, 0, maxy)


class MovingStd(object):
  def __init__(self, shape, a):
    self.a = a
    self.var = numpy.zeros(shape, numpy.float32)
    self.ema = numpy.zeros(shape, numpy.float32)

  def update(self, vals):
    #emaNew = ema(self.ema, vals, self.a)
    diff = vals - self.ema
    incr = (1 - self.a) * diff
    self.ema += incr
    
    self.var = self.a * (self.var + diff * incr)
  
  def get(self):
    return numpy.sqrt(self.var)



def chunks(l, n):
  """ Yield successive n-sized chunks from l. """
  for i in xrange(0, len(l), n):
    yield l[i:i+n]




class FindBeatParams(object):
  def __init__(self, numCalcBands, numGroupBands, **kwargs):
    self.numCalcBands = numCalcBands
    self.numGroupBands = numGroupBands
    self.minOnsetsGroup = 1    # how many onests needed within groups
    self.minOnsetsOut = 1      # how many onsets needed after grouping
    self.minOnsetsPreGroup = 1 # how many total onsets needed before grouping (applied first)
    self.beatHoldTime = 0.25
    self.beatWaitTime = 0.4
    self.avgWindow = 3
    self.stdAlpha = 0.01 # weight of contribution of new values to exponentionally weighted moving std (lower means slower decay of previous value)
    self.derivDist = 2
    self.derivActions = []
    self.fftWindow = None
    self.logScale = True
    self.sensitivity = 3
    #self.onsetCheck = None
    self.energyScale = 7
    self.onsetWait = 0 # how many cycles to wait before the next onset is allowed
    
    self.numFrames = 1024
    self.sampleRate = 44100
    self.bundleSize = 1

    self.showBands = range(self.numGroupBands)
    self.__dict__.update(kwargs)
  
    assert(self.beatWaitTime >= self.beatHoldTime)


  def copy(self):
    return FindBeatParams(**self.__dict__)


  def __eq__(self, other):
    if other is None: return False
    else:
      return self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not self.__eq__(other)
    
  
class FindBeats(object):
  def __init__(self, params):
    self.startTime = time.time()
  
    self.params = params
    self.meanEnergies = CircularBuffer2D(self.params.derivDist, params.numCalcBands)
    self.onsets = CircularBuffer2D(self.params.onsetWait, params.numCalcBands)
    self.derivEMStd = MovingStd((params.numCalcBands), 1-params.stdAlpha)
    
    self.lastBeatTimesCalc = numpy.zeros(params.numCalcBands, numpy.float64) - 2*params.beatWaitTime
    self.lastBeatTimesGroup = numpy.zeros(params.numGroupBands, numpy.float64) - 2*params.beatWaitTime


    self.channelMap = self._getChannelMap()

#    self.audioInput.start()

    self.freqs = self._getFreqs()
    
    if FFT_METHOD == "fftw" and os.path.exists(WISDOM_FILE):
      wisdom = cPickle.load(open(WISDOM_FILE))
      pyfftw.import_wisdom(wisdom)


#  def stop(self):
#    self.audioInput.stop()
  
  
  # 80% (5%)
  @profile
  def findBeats(self, data):
    assert(len(data) == self.params.numFrames)
      
    #with Timer("Wait"):
    #data = self.audioInput.get()

    #with Timer("FFT"):
    fftData = self._calcFFT(data, fftWindow=self.params.fftWindow)
    #return {"freq":self.freqs}, {} # DEBUG
    
    #with Timer("Energies"):
    energies = self._getEnergies(fftData)

    # transform energies to log scale
    if USE_NUMEXPR:
      scale = self.params.energyScale
      energies = numexpr.evaluate("log10(energies) / scale")
      energies = numexpr.evaluate("where(energies<0, energies, 0)")
    else:
      energies = numpy.log10(energies) / self.params.energyScale
      energies[energies<0] = 0
    #return {"freq":self.freqs}, {} # DEBUG

    #with Timer("Onsets"):
    results = self._findOnsets(energies)
    results["freq"] = self.freqs
    results["fft"] = fftData

    channels, mask = self.channelMap
    results["bandMap"] = numpy.where(~mask)[0]
    results["lastBeatTime"] = self.lastBeatTimesCalc

    #return {"freq":self.freqs}, {} # DEBUG

    #with Timer("Group"):
    reducedResults = self._groupResults(results)
    reducedResults["lastBeatTime"] = self.lastBeatTimesGroup
    #return {"freq":self.freqs}, {} # DEBUG

    #with Timer("Beat"):
    reducedResults["beat"] = self._findBeats(reducedResults["onset"], self.lastBeatTimesGroup)

    currTime = time.time() - self.startTime
    reducedResults["beatIntensity"] = ramp(currTime - self.lastBeatTimesGroup, self.params.beatHoldTime, 1.0)
      
    if self.params.numCalcBands != self.params.numGroupBands:
      results["beat"] = numpy.repeat(reducedResults["beat"], self.params.numCalcBands/self.params.numGroupBands)
    else:
      results["beat"] = self._findBeats(results["onset"], self.lastBeatTimesCalc)
    results["beatIntensity"] = ramp(currTime - self.lastBeatTimesCalc, self.params.beatHoldTime, 1.0)
    
    return reducedResults, results
  
  # 13%(3%)
  @profile
  def _calcFFT(self, data, fftWindow=None):
    if fftWindow == "hann":
      data = numpy.hanning(len(data)) * data

    if FFT_METHOD == "fftw":
      fft = pyfftw.interfaces.numpy_fft.rfft(data, overwrite_input=False, planner_effort='FFTW_PATIENT')#FFTW_MEASURE')
      if not os.path.exists(WISDOM_FILE):
        wisdom = pyfftw.export_wisdom()
        cPickle.dump(wisdom, open(WISDOM_FILE,'w'))
      real, imag = numpy.real(fft).astype(numpy.float32), numpy.imag(fft).astype(numpy.float32)

      fft = numpy.square(real) + numpy.square(imag)
      fft /= self.params.numFrames**2
    elif FFT_METHOD == "numpy": # 0.00317211329/call on bbb
      fft = numpy.fft.rfft(data).astype(numpy.complex64)
      #print "numypy fft size:", fft.shape, data.shape
      
      if USE_NUMEXPR:
        #fft.dtype = numpy.complex64
        fs2 = self.params.numFrames**2
        fft = numexpr.evaluate("(real(fft)**2 + imag(fft)**2) / fs2")
      else:
        #fft.dtype = numpy.complex64
        real, imag = numpy.real(fft), numpy.imag(fft)

        fft = numpy.square(real) + numpy.square(imag)
        fft /= self.params.numFrames**2
    else: # 0.003704184704/call on bbb
      fft = scipy.fftpack.rfft(data).astype(numpy.float32)
      if data.shape[0] % 2 == 0:
        real = numpy.concatenate([fft[:1], fft[1::2]])
        imag = numpy.concatenate([[0], fft[2::2], [0]])
      else:
        real = numpy.concatenate([fft[:1], fft[1::2]])
        imag = numpy.concatenate([[0], fft[2::2]])

      fft = numpy.square(real) + numpy.square(imag)
      fft /= self.params.numFrames**2

    return fft


  def _getFreqs(self):
    channels, mask = self.channelMap
    #print "python channels:", numpy.where(~mask)[0].shape[0],channels, numpy.where(~mask)[0]
    
    if FFT_METHOD in ("numpy", "scipy"):
      # numpy.fft.rfftfreq is not in numpy v1.7 or previously
      fftFreqs = scipy.fftpack.rfftfreq(self.params.numFrames, 1.0/self.params.sampleRate)
      fftFreqs = fftFreqs.take([0]+range(1,len(fftFreqs),2))
    else:
      raise Exception("Not implemented")
    freqs = fftFreqs.take(channels)
    return numpy.ma.median(numpy.ma.masked_array(freqs, mask),axis=1).data.astype(numpy.int32)


  def _getChannelMap(self):
    nf = self.params.numFrames/2+1
    ncb = self.params.numCalcBands
  
    if not self.params.logScale:
      bandWidth = int(numpy.ceil(float(nf)/self.params.numCalcBands))
      channels = numpy.arange(self.params.numCalcBands*bandWidth).reshape((self.params.numCalcBands, bandWidth))
      mask = channels>=n
      channels[mask] = 0
      
      #mask.dtype = numpy.int8
      return channels, mask
    else:
      # logbandwidth = log2(1024)/20 = 10/20 = 1/2
      # 2^0.5 = 1.4, 2^1=2, 2^1.5=2.8, 2^2=4
      logBandWidth = numpy.log2(nf) / ncb
      
      # create array of size ncb with number of frames per band in each entry
      bandCounts = numpy.power(2,logBandWidth*numpy.arange(ncb))
      bandCounts = bandCounts * nf / bandCounts.sum()
      bandCounts = bandCounts.astype(numpy.int32)
      bandCounts[bandCounts==0] = 1
      diff = bandCounts.sum() - nf
      assert(diff >= 0)
      bandCounts[-diff:] -= 1
    
      calcBands = numpy.arange(self.params.numCalcBands,dtype=numpy.int32)
      bandMap = numpy.repeat(calcBands, bandCounts)
    

      channels = numpy.zeros((self.params.numCalcBands, bandCounts.max()), numpy.int)
      mask = numpy.ones((self.params.numCalcBands, bandCounts.max()), numpy.bool)
      currChannel = 0
      for band in range(self.params.numCalcBands):
        channelsThisBand = max(1, int(round(bandCounts[band])))
        endChannel = min(currChannel + channelsThisBand, nf)
        chs = range(currChannel, endChannel)
        currChannel = endChannel
        channels[band,:len(chs)] = chs
        mask[band,:len(chs)] = False

      #mask.dtype = numpy.int8
      #print mask
      return channels, mask

  # 6% (3%)
  @profile
  def _getEnergies(self, fft):
    if self.params.numCalcBands == None: self.params.numCalcBands = len(fft)
    
    channels, mask = self.channelMap
    energies = fft.take(channels)

    if USE_NUMEXPR:
      energies = numexpr.evaluate("sum((1-mask) * energies, axis=1)")
      count = numexpr.evaluate("sum(1-mask, axis=1)")
      energies = numexpr.evaluate("energies / count")
    else:
      mask2 = ~mask
      energiesSum = (mask2*energies).sum(axis=1)
      energiesCount = mask2.sum(axis=1)
      energies = energiesSum/energiesCount
    
    #print energiesCount, energiesSum
    #print "python energies pre log:", energies.shape, energies

    return energies


  # Strategy: find the rate of change of the mean relative to a multiple of the rolling standard deviation
  # and allow requirement that multiple onsets occur at once
  # apply windowing to the spectral fluxes (will return 0 for edge bands)
  # 9% (3%)
  @profile
  def _findOnsets(self, energies):
    # determine spectral fluxes of the moving avg of the energies
    mas = ma(self.meanEnergies.get(), energies, self.params.avgWindow)
    
    if "max" in self.params.derivActions:
      derivEs = mas - self.meanEnergies.getData()
      derivEs = derivEs.max(axis=0)
    elif "sum" in self.params.derivActions:
      if USE_NUMEXPR:
        currMa, pastMas = mas, self.meanEnergies.getData()
        derivEs = numexpr.evaluate("sum(mas - pastMas, axis=0)")
      else:
        derivEs = mas - self.meanEnergies.getData()
        derivEs = derivEs.sum(axis=0)
    else:
      derivEs = self.meanEnergies.deriv(self.params.derivDist)

    self.meanEnergies.append(mas)

    if USE_NUMEXPR:
      derivEs = numexpr.evaluate("where(derivEs>0, derivEs, 0)")
      
      if self.params.onsetWait > 0:
        pastOnsets = self.onsets.getData().astype(numpy.int)
        recentOnsetsCount = numexpr.evaluate("sum(pastOnsets, axis=0)")
        derivEs = numexpr.evaluate("where(recentOnsetsCount >= 1, 0, derivEs)")

      stdDerivEs = self.derivEMStd.get()
      self.derivEMStd.update(derivEs)
      sensitivity = self.params.sensitivity

      derivCutoffEs = numexpr.evaluate("sensitivity * stdDerivEs")
      onsets = numexpr.evaluate("derivEs > derivCutoffEs")

    else:
      # first find the derivEs (the flux) and its recent variance
      derivEs = numpy.clip(derivEs, 0, numpy.inf)

      # if an onset has occurred recently, don't record this derivE
      if self.params.onsetWait > 0:
        pastOnsets = self.onsets.getData()
        hasRecentOnsets = pastOnsets.any(axis=0)
        derivEs *= ~hasRecentOnsets

      # get the moving standard deviation, then update it with the new values
      stdDerivEs = self.derivEMStd.get()
      self.derivEMStd.update(derivEs)

      derivCutoffEs = self.params.sensitivity * stdDerivEs
      onsets = derivEs > derivCutoffEs
    
    self.onsets.append(onsets)

    return {"energy":energies, "meanE":self.meanEnergies.get(), "derivE":derivEs,
      "derivCutoffE":derivCutoffEs, "onset":onsets, "stdDerivE":stdDerivEs}

  # 3% (2.5%)
  @profile
  def _findBeats(self, onsets, lastBeatTimes):
    if USE_NUMEXPR:
      minOnsets = self.params.minOnsetsOut
      numOnsets = onsets.sum()
      enoughOnsets = numexpr.evaluate("onsets & (numOnsets >= minOnsets)")
      
      currentTime = time.time() - self.startTime
      timeDiff = currentTime - lastBeatTimes
      holding = timeDiff <= self.params.beatHoldTime
      waiting = (timeDiff > self.params.beatHoldTime) & (timeDiff <= self.params.beatWaitTime)
      
      beats = numexpr.evaluate("(holding & ~waiting) | enoughOnsets")
      lastBeatTimes = numexpr.evaluate("where(enoughOnsets & ~holding & ~waiting, currentTime, lastBeatTimes)")
    else:
      # then determine beats
      currentTime = time.time() - self.startTime
      timeDiff = currentTime - lastBeatTimes
      holding = timeDiff <= self.params.beatHoldTime
      waiting = timeDiff <= self.params.beatWaitTime

      minOnsets = self.params.minOnsetsOut
      numOnsets = onsets.sum()
      if self.params.onsetCheck == "neighbor" and self.params.minOnsets > 1:
        # 3: ooo111011ooo & oo110110oooo & oooo011101o = ooo010000ooo
        # 2: oo111011oo & o110110ooo = oo110010oo
        before, after = minOnsets/2, minOnsets-minOnsets/2-1
        onsetsPad = numpy.pad(onsets, (minOnsets, minOnsets), 'constant', constant_values=(0,0))
        
        enoughOnsets = numpy.roll(onsetsPad, -before)
        for i in range(-before+1, after+1):
          enoughOnsets = enoughOnsets & numpy.roll(onsetsPad, i)
        enoughOnsets = enoughOnsets[minOnsets:-minOnsets]
      else:
        enoughOnsets = onsets & (numOnsets >= minOnsets)
    
      beats = numpy.zeros(onsets.shape, numpy.bool)
      #print "pyonsets", onsets
      #print "pybeats1", beats
      beats[holding] = True
      #print "pybeats2", beats
      waitingNotHolding = ~holding & waiting
      beats[waitingNotHolding] = False
      #print "pybeats3", beats
      enoughOnsetsNotWaitingNotHolding = ~holding & ~waiting & enoughOnsets
      beats[enoughOnsetsNotWaitingNotHolding] = True
      #print "pybeats4", beats
      lastBeatTimes[enoughOnsetsNotWaitingNotHolding] = currentTime

    return beats

  # 3.5% (1%)
  @profile
  def _groupResults(self, results):
    ncb, ngb = self.params.numCalcBands, self.params.numGroupBands
    
    if ncb == ngb:
      return results
    assert(ngb < ncb and ncb % ngb == 0)

    bandsPerGroup = ncb / ngb
    # split the calculated bands evenly into groups
    bandGroups = numpy.arange(ncb).reshape((ngb, bandsPerGroup))
    mids = bandGroups[:,bandsPerGroup/2]
    onsetsCalc = results["onset"].astype(numpy.int)

    # look for numOnsets > minOnsetsGroup to determine if there's an onset for this outBand
    if USE_NUMEXPR:
      numOnsetsCalc = numexpr.evaluate("sum(onsetsCalc, axis=0)")
    else:
      numOnsetsCalc = onsetsCalc.sum(axis=0)

    # look for numOnsets > minOnsetsGroup to determine if there's an onset for this outBand
    if numOnsetsCalc >= self.params.minOnsetsPreGroup:
      groupOnsets = numpy.take(onsetsCalc, bandGroups).astype(numpy.int)
      
      if USE_NUMEXPR:
        numGroupOnsets = numexpr.evaluate("sum(groupOnsets, axis=1)")
      else:
        numGroupOnsets = groupOnsets.sum(axis=1)

      groupOnsets = numGroupOnsets >= self.params.minOnsetsGroup
    else:
      groupOnsets = numpy.zeros(ngb, numpy.bool)
    
    # speed up
    reducedResults = {}
    for key, vals in results.iteritems():
      reducedResults[key] = results[key].take(mids)
    reducedResults["onset"] = groupOnsets

    return reducedResults
    


