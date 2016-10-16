#ifndef HZELLER_RPI_MATRIX_H
#define HZELLER_RPI_MATRIX_H

#include "Matrix.h"
#include "led-matrix.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include "threaded-canvas-manipulator.h"

using namespace rgb_matrix;

class HzellerRpiMatrix : public Matrix {
 public:
 HzellerRpiMatrix(size_t width, size_t height) : Matrix(width, height) {
    RGBMatrix::Options options;
    RuntimeOptions runtime;
    
    options.hardware_mapping = "regular";
    options.rows = 32;
    options.chain_length = width / 32;
    options.parallel = height / 32;
    options.show_refresh_rate = true;
    m_matrix = CreateMatrixFromOptions(options, runtime);
    if (m_matrix == NULL) {
        std::cout << "Unable to create hzeller rpi matrix\n";
        exit(1);
    }

    m_offscreenCanvas = m_matrix->CreateFrameCanvas();
 }

  virtual ~HzellerRpiMatrix() {
      delete m_matrix;
  }
  
  virtual void setPixel(size_t x, size_t y, char r, char g, char b) {
      m_offscreenCanvas->SetPixel(x, y, r, g, b);
  }
  
  virtual void update() {   
      m_offscreenCanvas = m_matrix->SwapOnVSync(m_offscreenCanvas);
  }
  
  virtual const unsigned char* rawData(size_t& size) const {
      std::cout << "HzellerRpiMatrix rawData() not implemented\n";
      exit(1);
      return nullptr;
  }
  
 private:
      RGBMatrix* m_matrix;
      FrameCanvas* m_offscreenCanvas;  
#endif
};

#endif
