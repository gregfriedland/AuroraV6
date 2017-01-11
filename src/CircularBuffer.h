/*
  CircularBuffer.h - A circular array implementation for Arduino
  Created by Greg Friedland on 10/8/2011
  Adapted from ByteBuffer by Sigurdur Orn.
 */
 
#ifndef CircularBuffer_h
#define CircularBuffer_h

#include <Arduino.h>

template <class T>
static void split (T *a, int n, T x, T *i, T *j) {
  //do the left and right scan until the pointers cross
  do {
    //scan from the left then scan from the right
    while (a[*i] < x) (*i)++;
    while (x < a[*j]) (*j)--;
    //now swap values if they are in the wrong part:
    if (*i <= *j) {
      T t = a[*i];
      a[*i] = a[*j];
      a[*j] = t;
      (*i)++; (*j)--;
    }
  //and continue the scan until the pointers cross:
  } while (*i <= *j);
}

// quick median from http://www.i-programmer.info/babbages-bag/505-quick-median.html
// modifies input array
template <class T>
static T quickMedian(T *a, int n) {
  int k = n / 2;

  int L = 0;
  int R = n-1;
  int i, j;
  while (L < R) {
    int x = a[k];
    i = L; j = R;
    split(a, n, x, &i, &j);
    if (j < k)  L = i;
    if (k < i)  R = j;
  }
  
  return a[k];
}


//// uses a lot of RAM in order to achieve speed
//class RunningMedian {
//public:
//  RunningMedian(int _maxVal, int _historySize) {
//    counts = (int*) malloc(sizeof(int) * maxVal);
//    lastMedian = lastMedianIndex = 0;
//    historySize = _historySize;
//  }
//  
//  int replaceVal(int oldVal, int newVal) {
//    counts[oldVal] = MAX(0, counts[oldVal]-1);
//    counts[newVal]++;
//    
//    if (newVal > oldVal) {
//      // increment the median index
//      lastMedianIndex++;
//      if (counts[lastMedian] > lastMedianIndex) {
//        // advance the median to the next value
//        lastMedianIndex = 0;
//        while(counts[++lastMedian] == 0);
//      }
//    }
//  }
//  
//  ~RunningMedian() {
//    free(vals);
//  }
//  
//private:
//  int *counts;
//  int lastMedian, historySize;
//  int lastMedianIndex; // the index into the 'list' of values at the median entry
//}



template <class T>
class CircularBuffer {
public:
  CircularBuffer() {
    data = NULL;
  }

  
  // This method initializes the datastore of the array to a certain size; the array should NOT be used before this call is made
  void init(unsigned int bufSize) {
    if (data != NULL) {
      free(data);
    }
    
    data = (T*)malloc(sizeof(T)*bufSize);
    capacity = bufSize;
    index = 0;
    for (unsigned int i=0; i<bufSize; i++) data[i] = 0;
  }

  
  // This method resets the array into an original state (with no data)	
  void clear() { 
    index = 0; 
    for (unsigned int i=0; i<capacity; i++) data[i] = 0;
  }

  
  // This releases resources for this array, after this has been called the array should NOT be used
  ~CircularBuffer() {
    if (data != NULL) free(data);
  }

  
  // Returns the active and other indices in the array
  unsigned int getIndex() { return index; }
  unsigned int getOffsetIndex(int offset) { 
#if 1
    int i = index + offset;
    int result;
    if (i<0) result = i+capacity;
    else if (i>=capacity) result = i-capacity;
    else result = i;

    //int result2 = (index+capacity+offset)%capacity; 
    //p2("ind=%d cap=%d off=%d i=%d res=%d res2=%d\n", index, capacity, offset, i, result, result2);
    return result;
#else
    return (index+capacity+offset)%capacity; 
#endif
  }

  
  // Returns the size of the array
  unsigned int getCapacity() { return capacity; }

  T* getArray() { return data; }

  
  // Set methods
  // index increments before adding an entry
  void append(T in) {
    index = index==capacity-1 ? 0 : index+1;
    data[index] = in;
  }
  void set(T in, unsigned int atIndex) { data[atIndex] = in; }
  void set(T in) { data[index] = in; }

  
  // Get methods
  T get() { return data[index]; }
  T get(unsigned int atIndex) { return data[atIndex]; }
  T getPrev() { return data[getOffsetIndex(-1)]; }
  
  
  void print() {
    Serial.print(index, DEC); Serial.print(": ");
    for (unsigned int i=0; i<capacity; i++) {
      Serial.print(data[i], DEC); Serial.print(" ");
    }
    Serial.println();
  }

  
  T median() {
    T *tmp = (T*)malloc(sizeof(T)*capacity);
    for (int i=0; i<capacity; i++) tmp[i] = data[i];
    
    T med = quickMedian(tmp, capacity);

    free(tmp);
    return med;
  }
  
  
  T mad(T med) {
    T *tmp = (T*)malloc(sizeof(T)*capacity);
    for (int i=0; i<capacity; i++) tmp[i] = abs(data[i] - med);

    T mad = quickMedian(tmp, capacity);

    free(tmp);
    return mad;
  }
  
  T mean(int startIndex, int endIndex) {
    if (endIndex < startIndex) endIndex += capacity;
    
    T sum = 0;
    int num = 0;
    for (int i=startIndex; i<=endIndex; i++) {
      sum += data[(i+capacity)%capacity];
      num++;
    }
    if (num == 0) return 0;
    else return sum/num;
  }
  
private:
  T* data;
  
  unsigned int capacity;
  unsigned int index;
};

#endif

