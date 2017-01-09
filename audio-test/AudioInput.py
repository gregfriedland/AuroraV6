import numpy
import time, sys


if sys.platform == "linux2":
  import alsaaudio

  class AudioInput(object):
    def __init__(self, numFrames=1024, sampleRate=44100, bundleSize=1):
      print "Cards found:", alsaaudio.cards()
    
      self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, card='default')
      self.inp.setchannels(1)
      self.inp.setrate(sampleRate)
      self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
      self.inp.setperiodsize(numFrames)

      self.sampleRate = sampleRate
      self.numFrames = numFrames
      self.bundleSize = bundleSize

      self.lastGetTime = time.time()

    def get(self):
      """ Blocks until data is received from the audio input """
      datas = []
      while True:
        nframes, data = self.inp.read()
        sys.stdout.flush()

        if nframes > 0:
          data = numpy.fromstring(data,dtype=numpy.int16)
        else:
          print "Expected %d at %d: %d, %d" % (self.numFrames, self.sampleRate, nframes, len(data))
          data = numpy.zeros(self.numFrames, numpy.int16)
          #time.sleep(0.1)
          
        datas.append(data)
          
        if len(datas) >= self.bundleSize:
          break

      data = numpy.concatenate(datas)
      #print "##### Data (%d bundles: %d): %.3f" % (len(datas), data.shape[0], time.time() - self.lastGetTime)
      self.lastGetTime = time.time()
    
      return data


    def start(self):
      pass


    def stop(self):
      pass

else:
  import pyaudio
  from threading import Lock
 
  class AudioInput(object):
    def __init__(self, numFrames=1024, sampleRate=44100, bundleSize=1):
      self.numFrames = numFrames
      self.bundleSize = bundleSize
      self.sampleRate = sampleRate
      
      self.stream = None
      self.streamData = []
      self.sampleRate = sampleRate
      self.lastGetTime = time.time()
      self.lock = Lock()
    
      self.start()
    
    
    def start(self, filename=""):
      p = pyaudio.PyAudio()

      self.isFromFile = filename != ""
      if filename == "":
        defaultDevice = p.get_default_input_device_info()
        print "Default input device:", defaultDevice
        print "Is desired audio format supported: ", \
            p.is_format_supported(self.sampleRate, input_device=defaultDevice['index'],
               input_channels=1, input_format=pyaudio.paInt16,   output_device=None, output_channels=None, output_format=None)
        self.stream = p.open(format=pyaudio.paInt16,channels=1,rate=self.sampleRate,input=True,
                            frames_per_buffer=self.numFrames,
                            stream_callback=lambda id,fc,ti,f: self.inputCallback(id,fc,ti,f))
      else:
        import wave

        self.wf = wave.open(filename, 'rb')
        self.stream = p.open(format=p.get_format_from_width(self.wf.getsampwidth()),
                            channels = self.wf.getnchannels(), rate = self.wf.getframerate(), output=True)
        # print "fileData:", len(self.fileData)
    
    
    def inputCallback(self, in_data, frame_count, time_info, flags):
      if flags != 0:
        if flags & pyaudio.paInputOverflow:   print "Input Overflow"
        if flags & pyaudio.paInputUnderflow:  print "Input Underflow"
        if flags & pyaudio.paOutputOverflow:  print "Output Overflow"
        if flags & pyaudio.paOutputUnderflow: print "Output Underflow"
        if flags & pyaudio.paPrimingOutput:   print "Priming Output"
      
      with self.lock:
        self.streamData.append(numpy.fromstring(in_data,dtype=numpy.int16))
      
      return (None, pyaudio.paContinue)
    
    
    def stop(self):
      self.stream.close()
    

    def get(self):
      """ Blocks until data is received from the audio input """
      if not self.isFromFile:
        while True:
          n = len(self.streamData)
          if n >= self.bundleSize:
            break
          time.sleep(0.003)
      
      with self.lock:
        if self.isFromFile:
          data = self.wf.readframes(self.numFrames)
        else:
          data = self.streamData[:self.bundleSize]
          self.streamData = [] #self.streamData[self.bundleSize:]

      if self.isFromFile:
        if data != "":
          self.stream.write(data)
          # print len(data)
        data = numpy.fromstring(data,dtype=numpy.int16)
        if self.wf.getnchannels() == 2:
          data = data[::2]
        # print "data:", data.shape
      else:
        data = numpy.concatenate(data)
      
      # print "##### Data (%d bundles: %d): %.3f" % (n, data.shape[0], time.time() - self.lastGetTime)
      
      self.lastGetTime = time.time()
      return data
