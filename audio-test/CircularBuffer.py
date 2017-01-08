import numpy

def mad(arr):
  return numpy.median(numpy.abs(arr - numpy.median(arr)))


class CircularBuffer2D(object):
  def __init__(self, capacity, width):
    self.shape = (capacity, width)
    self.buf = numpy.zeros(self.shape, numpy.float)
    self.index = 0


  def clear(self):
    self.buf = numpy.zeros(self.shape, numpy.float)
    self.index = 0
  

  def getIndex(self):
    return self.index
  

  def getCapacity(self):
    return self.shape[0]


  def getWidth(self):
    return self.shape[1]


  def getData(self):
    return self.buf


  # index increments before adding an entry so index points to current entry
  def append(self, vals):
    self.index = (self.index+1) % self.shape[0]
    self.buf[self.index,:] = vals


  def get(self, n=1):
    """ Get the last N elements before current index """
    if n == 1:
      return self.buf[self.index]
    elif n is None:
      return self.buf
    else:
      return numpy.take(self.buf, range(self.index+1-n, self.index+1), axis=0, mode="wrap")


  def __str__(self):
    return "CircularBuffer: index=%d\n%s" % (self.index, str(self.buf))


  def op(self, func, n=None, end=0):
    """ Apply a function to the last N elements before current index """
    end = self.index + end + 1
    if n is None:
      tmp = self.buf
    else:
      tmp = numpy.take(self.buf, range(end-n, end), axis=0, mode="wrap")
    
    return func(tmp, axis=0)


  def median(self, n=None):
    """ Median of last N elements before current index """
    return self.op(numpy.median, n=n)


  def std(self, n=None):
    """ Standard deviation of last N elements before current index """
    return self.op(numpy.std, n=n)


  def mad(self, n=None):
    """ MAD of last N elements before current index """
    return self.op(mad, n=n)


  def min(self, n=None):
    """ Min of last N elements before current index """
    return self.op(numpy.min, n=n)


  def max(self, n=None):
    """ Max of last N elements before current index """
    return self.op(numpy.max, n=n)


  def mean(self, n=None):
    """ Mean of last N elements before current index """
    return self.op(numpy.mean, n=n)


  def deriv(self, n):
    return self.buf[self.index,:] - self.buf[(self.index-n)%self.shape[0],:]





class CircularBuffer(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.buf = numpy.zeros(capacity, numpy.float)
    self.index = 0
  
  
  def clear(self):
    self.buf = numpy.zeros(capacity, numpy.float)
    self.index = 0
  
  
  def getIndex(self):
    return self.index
  
  
  def getCapacity(self):
    return self.capacity
  
  
  def getData(self):
    return self.buf
  
  
  # index increments before adding an entry so index points to current entry
  def append(self, val):
    self.index = (self.index+1) % self.capacity
    self.buf[self.index] = val
  
  
  def get(self, index=None, offset=None):
    if index is not None:
      index %= self.capacity
    elif offset is not None:
      index = (self.index + offset) % self.capacity
    else:
      index = self.index
    
    return self.buf[index]
  
  
  def __str__(self):
    return "CircularBuffer: index=%d\n%s" % (self.index, str(self.buf))
  
  
  def op(self, func, n=None):
    """ Apply a function to the last N elements before current index """
    if n is None:
      tmp = self.buf
    else:
      end = self.index + 1
      tmp = numpy.take(self.buf, range(end-n, end), mode="wrap")
    
    return func(tmp)
  
  
  def median(self, n=None):
    """ Median of last N elements before current index """
    return self.op(numpy.median, n=n)
  
  
  def std(self, n=None):
    """ Standard deviation of last N elements before current index """
    return self.op(numpy.std, n=n)
  
  
  def mad(self, n=None):
    """ MAD of last N elements before current index """
    return self.op(mad, n=n)
  
  
  def min(self, n=None):
    """ Min of last N elements before current index """
    return self.op(numpy.min, n=n)
  
  
  def max(self, n=None):
    """ Max of last N elements before current index """
    return self.op(numpy.max, n=n)
  
  
  def mean(self, n=None):
    """ Mean of last N elements before current index """
    return self.op(numpy.mean, n=n)
  
  
  def deriv(self, n):
    return self.buf[self.index] - self.buf[(self.index-n)%self.capacity]
  
  
  def ema(self, n):
    tmp = numpy.take(self.buf, range(self.index-n+1, self.index+1), mode="wrap")
    return pandas.stats.moments.ewma(tmp, span=n)[-1]
  
  
  def emstd(self, n):
    tmp = numpy.take(self.buf, range(self.index-n+1, self.index+1), mode="wrap")
    return pandas.stats.moments.ewmstd(tmp, span=n)[-1]
  
  
  def ma(self, n):
    tmp = numpy.take(self.buf, range(self.index-n+1, self.index+1), mode="wrap")
    return pandas.stats.moments.rolling_mean(tmp, n)[-1]


#  def mstd(self, n, end=0):
#    end = self.index + end + 1
#
#    tmp = numpy.take(self.buf, range(end-n, end), mode="wrap")
#    return pandas.stats.moments.rolling_std(tmp, n)[-1]


