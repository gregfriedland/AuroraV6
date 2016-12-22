#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <iomanip>
#include <functional>
#include <chrono>
#include <random>
#include <array>
#include <cassert>
#include <ostream>

#ifdef __arm__
  #include <arm_neon.h>
#endif

using namespace std::chrono;


struct Color24 {
    Color24(int col) 
    : r((col >> 16) & 255), g((col >> 8) & 255), b(col & 255)
    {}

    Color24(unsigned char _r, unsigned char _g, unsigned char _b)
    : r(_r), g(_g), b(_b)
    {}

    unsigned char r, g, b;
};


inline float mapValue(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

inline int mapValue(int x, int in_min, int in_max, int out_min, int out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

inline unsigned long millis() {
	/* struct timeval tv; */

	/* gettimeofday(&tv, NULL); */

	/* return (unsigned long)(tv.tv_sec) * 1000 + */
    /* 	   (unsigned long)(tv.tv_usec) / 1000; */
	return duration_cast< milliseconds >(
    	system_clock::now().time_since_epoch()).count();
}

static unsigned long int startTime = millis();
inline void fail() {
    std::cout << "Exit after " << ((millis() - startTime) / 1000) << "s\n";
    exit(1);
}

static std::minstd_rand0 randGen(millis());
inline int random2() {
	return randGen();
}

inline float randomFloat(float min, float max) {
	static std::random_device rd;
	static std::mt19937 gen(rd());

	return std::uniform_real_distribution<>(min, max)(gen);
}

class IntervalTimer {
public:
	IntervalTimer(unsigned int interval) 
	: m_lastTime(millis()), m_interval(interval) 
	{}

	void reset() {
		m_lastTime = millis();
	}

	bool tick(unsigned int* timeLeft) {
		unsigned long currTime = millis();
		if (timeLeft != NULL)
			*timeLeft = std::max(0, (int)currTime - (int)(m_lastTime + m_interval));
		//cout << "currTime:" << currTime << " lastTime:" << m_lastTime << endl;
		if (currTime - m_lastTime > m_interval) {
			m_lastTime = currTime;
			return true;
		} else
			return false;
	}

private:
	unsigned long m_lastTime;
	unsigned int m_interval;
};


class FpsCounter {
public:
	FpsCounter(unsigned int outputInterval, const std::string& name);
	void tick();

private:
	unsigned int m_count;
	unsigned long m_lastTime;
	unsigned int m_interval;
	std::string m_name;
};


class FrameTimer {
public:
    FrameTimer();
    void tick(unsigned int intervalMs, std::function<void()> func);

private:
    unsigned long m_lastTime;
};


template<typename T>
class Array2D {
 public:
 	Array2D(size_t width, size_t height)
 	: m_width(width), m_height(height) {
 		m_data = new T[width*height];
 	}

 	~Array2D() {
 		delete[] m_data;
 	}

 	T& get(int x, int y) {
		if (x < 0)
 			x += m_width;
 		else if (x >= m_width)
 			x -= m_width;

 		if (y < 0)
 			y += m_height;
 		else if (y >= m_height)
 			y -= m_height;

 		return m_data[x + y * m_width];
  	}

 	const T& get(int x, int y) const {
		if (x < 0)
 			x += m_width;
 		else if (x >= m_width)
 			x -= m_width;

 		if (y < 0)
 			y += m_height;
 		else if (y >= m_height)
 			y -= m_height;

 		return m_data[x + y * m_width];
  	}

 	T& get(size_t index) {
 		return m_data[index];
 	}

 	const T& get(size_t index) const {
 		return m_data[index];
 	}

 	T& operator[](size_t index) {
 		return m_data[index];
 	}

 	const T& operator[](size_t index) const {
 		return m_data[index];
 	}

 	void random() {
 		for (size_t x = 0; x < m_width; ++x) {
 			for (size_t y = 0; y < m_height; ++y) {
 				get(x, y) = (random2() % 10000) / 10000.0;
 			}
 		}
 	}

 	T sum() const {
 		T sum = 0;
 		for (size_t x = 0; x < m_width; ++x) {
 			for (size_t y = 0; y < m_height; ++y) {
 				sum += get(x, y);
 			}
 		}
 		return sum;
 	}

 	void constrain(T min, T max) {
 		for (size_t x = 0; x < m_width; ++x) {
 			for (size_t y = 0; y < m_height; ++y) {
 				get(x, y) = std::min(min, std::max(max, get(x, y)));
 			}
 		}
 	}

 	const T* rawData() const {
 		return m_data;
 	}

 	size_t width() const {
 		return m_width;
 	}

 	size_t height() const {
 		return m_height;
 	}

	friend std::ostream& operator <<(std::ostream& os, const Array2D<T>& arr) {
 		for (size_t x = 0; x < arr.m_width; ++x) {
 			for (size_t y = 0; y < arr.m_height; ++y) {
 				os << std::setprecision(3) << std::setw(4) << arr.get(x, y) << " ";
 			}
 			os << std::endl;
 		}
 		return os;
	}
	
 protected:
 	size_t m_width, m_height;
 	T* m_data;
};

#ifdef __arm__
template <typename T, typename NEON_TYPE, int REGISTER_N>
class Array2DNeon {
 public:
 	Array2DNeon(size_t width, size_t height)
 	: m_width(width), m_height(height) {
            m_numVectors = width * height / REGISTER_N;
	    m_data = new NEON_TYPE[m_numVectors];
       }

 	~Array2DNeon() {
 		delete[] m_data;
 	}

 	T get(size_t index) const {
 		T out[REGISTER_N];
 		vst1q_f32(out, m_data[index / REGISTER_N]);
 		return out[index % REGISTER_N];
 	}

 	void set1(size_t index, T val) {
 		T tmp[REGISTER_N];
 		vst1q_f32(tmp, m_data[index / REGISTER_N]);
 		tmp[index % REGISTER_N] = val;
 		m_data[index / REGISTER_N] = vld1q_f32(tmp);
 	}

 	void setN(size_t index, T val) {
	  assert(index % REGISTER_N == 0);
	  m_data[index / REGISTER_N] = vdupq_n_f32(val);
 	}

 	void setN(size_t index, const NEON_TYPE& val) {
	  assert(index % REGISTER_N == 0);
	  m_data[index / REGISTER_N] = val;
 	}

	template <bool CHECK_BOUNDS = false>
 	NEON_TYPE getN(size_t index) const {
	  size_t remainder = index % REGISTER_N;
	  if (remainder == 0) {
	    return m_data[index / REGISTER_N];
	  }

	  T tmp[REGISTER_N * 2];
	  size_t indexN = index / REGISTER_N;
	  size_t indexNextN;
	  if (CHECK_BOUNDS) {
	    indexNextN = (indexN + 1) % m_numVectors;
	  } else {
	    indexNextN = indexN + 1;
	  }
	   
	  vst1q_f32(tmp, m_data[indexN]);
	  vst1q_f32(tmp + REGISTER_N, m_data[indexNextN]);
	  return vld1q_f32(tmp + remainder);
	}

 	size_t width() const {
 		return m_width;
 	}

 	size_t height() const {
 		return m_height;
 	}

	friend std::ostream& operator <<(std::ostream& os, const Array2DNeon<T,NEON_TYPE,REGISTER_N>& arr) {
	  for (size_t y = 0; y < arr.m_height; ++y) {
	    for (size_t x = 0; x < arr.m_width; ++x) {
	      size_t i = x + y * arr.m_width;
	      os << std::setw(4) << std::round(arr.get(i) * 100.0) / 100.0 << " ";
	    }
	    os << std::endl;
	  }
	  return os;
	}

 protected:
 	size_t m_width, m_height;
 	NEON_TYPE* m_data;
	size_t m_numVectors;
};

#endif

template <typename T>
void convolve(const Array2D<T>* convArr, const Array2D<T>* inputArr, Array2D<T>* outputArr) {
	assert(inputArr->width() == outputArr->width() && inputArr->height() == outputArr->height());

	int xConvMid = convArr->width() / 2;
	int yConvMid = convArr->height() / 2;
	T convSum = convArr->sum();

	for (int x = 0; x < inputArr->width(); ++x) {
		for (int y = 0; y < inputArr->height(); ++y) {

			T val = 0;
			for (int yy = 0; yy < convArr->height(); ++yy) {
				for (int xx = 0; xx < convArr->width(); ++xx) {
					val += convArr->get(xx, yy) * inputArr->get(x + xx - xConvMid, y + yy - yConvMid);
					// if (x == 0 && y == 0) {
					// 	std::cout << xx << "/" << yy << "=" << convArr->get(xx,yy) << " " << 
					// 		x + xx - xConvMid << "/" << y + yy - yConvMid << "=" << inputArr->get(x - xx - xConvMid, y - yy - yConvMid) << std::endl;
					// }
				}
			}
			val /= convSum;
			outputArr->get(x, y) = val;
		}
	}
}

#endif
