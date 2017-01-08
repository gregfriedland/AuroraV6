PLOT = True
PLOT_GROUPS = True
USE_CYTHON = False
CYTHON_COMPARE = False
USE_PYXIMPORT = False

import sys, time, numpy
if PLOT: import pygame

from AudioInput import AudioInput
from FindBeats import FindBeatParams
from collections import OrderedDict
import os

if USE_PYXIMPORT:
  import pyximport
  pyximport.install(inplace=True)

from FindBeats import FindBeats
# from AuroraViz.cFindBeats import FindBeats as FindBeats
# from Timer import Timer

# if CYTHON_COMPARE:
#   from AuroraViz.FindBeats import FindBeats


# songs:
# 1) hey ya
# 2) heart of gold
# 3) everything in its right place
# 4) black dog
# 5) power of love
# 6) yer so bad
# 7) use me
# 8) terrapin station
# 9) every breath you take
#10) like a rolling stone
#11) one (warren haynes)
#12) green onions
#13) superstition
#14) harrisburg
#15) u can't touch this
#16) take on me
#17) more than words
#18) desert eagle (ratatat)
#19) free falling
#20) forever
#21) don't think twice it's alright (joan baez)
#22) blackbird (csn)
#23) landslide
#24) tiny dancer

# Best so far:
# 8:good, 9:mostly very good, 2:excellent, 1:very good (mostly all-channel beats?), 5:mostly very good, 12:excellent
# 14:ok, 15:very good, 11: mostly very good, 16:very good, 17: very good, 10:good 6:moslty good?, 18:good
# increasing only, higher minonsets, lower sensitivity
#PLOT_BANDS = (3, 8, 15, 30, 50)
#params = FindBeatParams(64, 64, minOnsetsOut=15, minOnsetsGroup=1, beatHoldTime=0.25, beatWaitTime=0.25,
#                        logScale=True, frameSize=1024, energyScale=6,
#                        avgWindow=3, derivDist=3, derivActions=["sum", "increasing"], onsetCheck=None, sensitivity=3,
#                        onsetWait=5)


# linear scale, simulatenous onsets with using lots of non log scale channels; Frequencies: [172, 947, 4823]
# 1) excellent (misses a few beats), 5) excellent, 6) pretty good?
# minOnsets=5 seems to be important
#PLOT_BANDS = [0,2,12]
#params = FindBeatParams(64, 64, minOnsetsOut=5, minOnsetsGroup=1, beatHoldTime=0.25, beatWaitTime=0.25,
#                        logScale=False, frameSize=1024, energyScale=6,
#                        avgWindow=2, derivDist=3, derivActions=["sum"], onsetCheck=None, sensitivity=3.5)



# onset grouping
# on mac: 1:excellent, 6:poor, 14:poor, 12:excellent, 19:good, 7:good, 2:very good, 20:very good, 21:good, 22:fair, 23:poor
if PLOT_GROUPS:
  PLOT_BANDS = range(6)
else:
  PLOT_BANDS = range(100)
params = FindBeatParams(120, 6, minOnsetsPreGroup=15, minOnsetsGroup=7, beatHoldTime=0.25, beatWaitTime=0.25,
                        logScale=True, numFrames=940, energyScale=6, bundleSize=1,
                        avgWindow=3, derivDist=3, derivActions=["sum", "increasing"], onsetCheck=None, sensitivity=3,
                        onsetWait=5)


if PLOT_GROUPS:
  PARAM_COLORS = OrderedDict([#("energy",(0,128,0)),
                            ("derivCutoffE",(0,255,255)),
                            ("onset",(255,0,0)),
                            ("beat",(255,255,255)),
                            ("meanE",(0,0,255)),
                            ("derivE",(255,255,0)),
                            ])
else:
  PARAM_COLORS = OrderedDict([("onset",(255,0,0)),
                              ("beat",(255,255,255)),
                              ])

WIDTH = 1200
HEIGHT = 801
BEAT_AREA_WIDTH = 200
BEAT_CIRCLE_RADIUS = HEIGHT / len(PLOT_BANDS) / 3
XINCR = 6



if __name__ == "__main__":
  if PLOT:
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    display.fill((0,0,0))

  audioInput = AudioInput(numFrames=params.numFrames, sampleRate=params.sampleRate, bundleSize=params.bundleSize)
  audioInput.start()

  # if CYTHON_COMPARE:
  #   fbpy = FindBeats(params)
  fb = FindBeats(params)

  #data = audioInput.get()
  #beatResultsGroup, beatResults = fb.findBeats(data)
  #print "Frequencies:", beatResultsGroup["freq"]

#  if CYTHON_COMPARE:
#    beatResultsGroupPy, beatResultsPy = fbpy.findBeats(data)
#    print "FrequenciesPy:", beatResultsGroupPy["freq"].astype(int)


  panelHeight = HEIGHT/len(PLOT_BANDS)
  lastBeatResults = None
  x = 0
  pause = False

  while True:
    #with Timer("Audio"):
    data = audioInput.get()
    #print "data", data.dtype, data[:10]

    beatResultsGroup, beatResultsAll = fb.findBeats(data)
    if PLOT_GROUPS:
      beatResults = beatResultsGroup
    else:
      beatResults = beatResultsAll

    if CYTHON_COMPARE:
      beatResultsGroupPy, beatResultsAllPy = fbpy.findBeats(data)

      numpy.set_printoptions(suppress=False, precision=3, linewidth=150)
      allParams = ["bandMap", "freq", "fft", "energy", "meanE", "derivE", "stdDerivE", "derivCutoffE", "onset", "beat", "lastBeatTime"]
      groupParams = ["onset", "beat", "lastBeatTime"]

      for type, params, beatResults, beatResultsPy in \
          zip(["All", "Group"], [allParams, groupParams],
          [beatResultsAll, beatResultsGroup],
          [beatResultsAllPy, beatResultsGroupPy]):
        print "#####", type
        for param in params:
          c, p = beatResults[param], beatResultsPy[param]
          #if c.dtype == numpy.bool: c = c.astype(numpy.uint8)
          if p.dtype == numpy.bool: p = p.astype(numpy.uint8)
          same = numpy.allclose(c,p, rtol=1)
          print "param=%s same=%s numSame=%d maxDiff=%.8f" % \
            (param, same, numpy.isclose(c,p,rtol=1).sum(),
            numpy.absolute(c-p).max())
          if True or not same:
            print "cython: len=", len(c)
            print c
            print "python: len=", len(p)
            print p
        print
      print



    # plot the params over time
    if not PLOT:
      print "".join(["*" if o else " " for o in beatResults.getBeats()]) + "    ",
      print "".join(["%d"%(i*9) for i in beatResults.getBeatIntensity(type="triangle")])
    else:
      # pause the plotting if any key is pressed
      events = pygame.event.get()
      if pygame.KEYUP in [event.type for event in events]: pause = False
      if pygame.KEYDOWN in [event.type for event in events]: pause = True
      if pause:
        time.sleep(0.01)
        continue

      if lastBeatResults is not None:
        for panelNum, band in enumerate(PLOT_BANDS):
          #print calcBeatsResult[band]
          for param, color in PARAM_COLORS.items():
            lastVal, currVal = lastBeatResults[param][band], beatResults[param][band]
            if param == "beat": lastVal, currVal = lastVal/2.0, currVal/2.0 # make beats shorter
            
            #print param, lastVal, currVal
            pygame.draw.line(display, color, (x-XINCR,panelHeight*(panelNum+1) - lastVal*panelHeight),
                               (x,panelHeight*(panelNum+1) - currVal*panelHeight))

          beatIntensity = beatResults["beat"][band]
          pygame.draw.circle(display, (0, 255*beatIntensity,0), (WIDTH-BEAT_AREA_WIDTH/2, int(panelHeight*(panelNum+0.5))), BEAT_CIRCLE_RADIUS)
    
      pygame.display.flip()
      x += XINCR
      if x >= WIDTH - BEAT_AREA_WIDTH:
        x = XINCR
        display.fill((0,0,0))

    lastBeatResults = beatResults

  print "DONE"