// Arduino code to listen for pixel data in serial format from a PC/RaspberryPi,
// and relay it to an LED matrix or strip

#define ADAFRUIT_MATRIX 77
#define LED_TYPE ADAFRUIT_MATRIX // 2801 | 2811 | ADAFRUIT_MATRIX
#define WIDTH 64
#define HEIGHT 32
#define COLOR_DEPTH 48 // internal color depth to use in SmartMatrix: 24 or 48bit
#define BLINK_PIN 13
#define OUTPUT_FPS_INTERVAL 5000
#define USE_GAMMA_CONVERSION

#define p(...) Serial.print(__VA_ARGS__)

#if (LED_TYPE==2811)
  #include <OctoWS2811.h>
  #define LEDS_PER_STRIP (WIDTH*HEIGHT/8)

  DMAMEM int displayMemory[LEDS_PER_STRIP*6];
  int drawingMemory[LEDS_PER_STRIP*6];
  OctoWS2811 leds(LEDS_PER_STRIP, displayMemory, drawingMemory, WS2811_GRB | WS2811_800kHz);
#elif (LED_TYPE==2801)
  #include "FastLED.h"
  #define CLOCK_PIN 2 // data line 1
  #define DATA_PIN 14 // data line 2
  CRGB leds[WIDTH*HEIGHT];
#elif (LED_TYPE==ADAFRUIT_MATRIX)
  #include <SmartMatrix3.h>

  const uint8_t kMatrixWidth = 64;        // known working: 32, 64, 96, 128
  const uint8_t kMatrixHeight = 32;       // known working: 16, 32, 48, 64
  const uint8_t kRefreshDepth = 36;       // known working: 24, 36, 48
  const uint8_t kDmaBufferRows = 4;       // known working: 2-4, use 2 to save memory, more to keep from dropping frames and automatically lowering refresh rate
  const uint8_t kPanelType = SMARTMATRIX_HUB75_32ROW_MOD16SCAN;   // use SMARTMATRIX_HUB75_16ROW_MOD8SCAN for common 16x32 panels
  const uint8_t kMatrixOptions = (SMARTMATRIX_OPTIONS_NONE);      // see http://docs.pixelmatix.com/SmartMatrix for options
  const uint8_t kBackgroundLayerOptions = (SM_BACKGROUND_OPTIONS_NONE);

  SMARTMATRIX_ALLOCATE_BUFFERS(matrix, kMatrixWidth, kMatrixHeight, kRefreshDepth, kDmaBufferRows, kPanelType, kMatrixOptions);
  SMARTMATRIX_ALLOCATE_BACKGROUND_LAYER(backgroundLayer, kMatrixWidth, kMatrixHeight, COLOR_DEPTH, kBackgroundLayerOptions);
#endif

#define BUFFER_SIZE 6500 * 3
static byte buffer[BUFFER_SIZE];

#ifdef USE_GAMMA_CONVERSION
//gamma=4
//static const uint16_t gammaTable[] = {0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x1,0x1,0x1,0x2,0x2,0x2,0x3,0x4,0x4,0x5,0x6,0x7,0x8,0x9,0xb,0xc,0xe,0x10,0x12,0x14,0x17,0x1a,0x1d,0x20,0x23,0x27,0x2b,0x2f,0x34,0x39,0x3f,0x44,0x4a,0x51,0x58,0x5f,0x67,0x70,0x78,0x82,0x8c,0x96,0xa1,0xad,0xb9,0xc6,0xd3,0xe1,0xf0,0x100,0x110,0x122,0x133,0x146,0x15a,0x16e,0x184,0x19a,0x1b1,0x1ca,0x1e3,0x1fd,0x218,0x235,0x252,0x271,0x291,0x2b2,0x2d4,0x2f8,0x31d,0x343,0x36a,0x393,0x3bd,0x3e9,0x416,0x445,0x475,0x4a7,0x4db,0x510,0x547,0x57f,0x5ba,0x5f6,0x634,0x674,0x6b5,0x6f9,0x73f,0x786,0x7d0,0x81c,0x86a,0x8ba,0x90c,0x961,0x9b8,0xa11,0xa6d,0xacb,0xb2b,0xb8e,0xbf4,0xc5c,0xcc7,0xd34,0xda4,0xe17,0xe8d,0xf06,0xf81,0x1000,0x1081,0x1106,0x118e,0x1218,0x12a6,0x1338,0x13cc,0x1464,0x14ff,0x159e,0x1640,0x16e6,0x178f,0x183c,0x18ed,0x19a1,0x1a59,0x1b15,0x1bd5,0x1c99,0x1d61,0x1e2d,0x1efd,0x1fd1,0x20a9,0x2186,0x2267,0x234d,0x2437,0x2525,0x2618,0x2710,0x280c,0x290d,0x2a13,0x2b1e,0x2c2e,0x2d42,0x2e5c,0x2f7b,0x309f,0x31c8,0x32f7,0x342a,0x3564,0x36a3,0x37e7,0x3931,0x3a80,0x3bd6,0x3d31,0x3e92,0x3ff9,0x4166,0x42d9,0x4452,0x45d1,0x4757,0x48e3,0x4a75,0x4c0e,0x4dad,0x4f53,0x5100,0x52b3,0x546d,0x562e,0x57f6,0x59c5,0x5b9c,0x5d79,0x5f5e,0x614a,0x633d,0x6538,0x673a,0x6944,0x6b56,0x6d6f,0x6f91,0x71ba,0x73eb,0x7624,0x7866,0x7aaf,0x7d01,0x7f5c,0x81bf,0x842a,0x869e,0x891b,0x8ba0,0x8e2e,0x90c6,0x9366,0x960f,0x98c2,0x9b7e,0x9e43,0xa112,0xa3ea,0xa6cc,0xa9b7,0xacac,0xafab,0xb2b5,0xb5c8,0xb8e5,0xbc0c,0xbf3e,0xc27a,0xc5c0,0xc911,0xcc6d,0xcfd3,0xd344,0xd6c1,0xda48,0xddda,0xe177,0xe520,0xe8d4,0xec93,0xf05e,0xf435,0xf817,0xfc05};
//gamma=3.5
static const uint16_t gammaTable[] = {0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x1,0x1,0x1,0x2,0x3,0x3,0x4,0x5,0x6,0x7,0x9,0xa,0xc,0xe,0x11,0x13,0x16,0x19,0x1c,0x20,0x24,0x28,0x2d,0x32,0x38,0x3e,0x44,0x4b,0x53,0x5a,0x63,0x6c,0x75,0x7f,0x8a,0x95,0xa1,0xae,0xbb,0xc9,0xd8,0xe7,0xf8,0x109,0x11a,0x12d,0x141,0x155,0x16b,0x181,0x198,0x1b1,0x1ca,0x1e5,0x200,0x21d,0x23a,0x259,0x279,0x29a,0x2bd,0x2e0,0x305,0x32b,0x353,0x37c,0x3a6,0x3d2,0x3ff,0x42e,0x45e,0x490,0x4c3,0x4f8,0x52e,0x566,0x5a0,0x5dc,0x619,0x658,0x698,0x6db,0x71f,0x766,0x7ae,0x7f8,0x844,0x892,0x8e3,0x935,0x989,0x9e0,0xa39,0xa93,0xaf1,0xb50,0xbb2,0xc16,0xc7c,0xce5,0xd50,0xdbe,0xe2e,0xea1,0xf16,0xf8e,0x1008,0x1085,0x1105,0x1188,0x120d,0x1296,0x1321,0x13af,0x143f,0x14d3,0x156a,0x1604,0x16a1,0x1740,0x17e4,0x188a,0x1933,0x19e0,0x1a90,0x1b43,0x1bfa,0x1cb4,0x1d71,0x1e32,0x1ef7,0x1fbe,0x208a,0x2159,0x222c,0x2302,0x23dd,0x24bb,0x259c,0x2682,0x276b,0x2859,0x294a,0x2a40,0x2b39,0x2c37,0x2d38,0x2e3e,0x2f48,0x3056,0x3169,0x3280,0x339b,0x34bb,0x35df,0x3707,0x3834,0x3966,0x3a9c,0x3bd7,0x3d17,0x3e5b,0x3fa4,0x40f2,0x4245,0x439d,0x44f9,0x465b,0x47c2,0x492e,0x4a9e,0x4c14,0x4d90,0x4f10,0x5096,0x5221,0x53b1,0x5547,0x56e3,0x5883,0x5a2a,0x5bd6,0x5d88,0x5f3f,0x60fc,0x62bf,0x6487,0x6656,0x682a,0x6a05,0x6be5,0x6dcb,0x6fb8,0x71aa,0x73a3,0x75a2,0x77a8,0x79b3,0x7bc5,0x7ddd,0x7ffc,0x8222,0x844e,0x8680,0x88b9,0x8af9,0x8d3f,0x8f8d,0x91e1,0x943c,0x969e,0x9907,0x9b77,0x9dee,0xa06c,0xa2f1,0xa57e,0xa811,0xaaac,0xad4f,0xaff9,0xb2aa,0xb563,0xb823,0xbaeb,0xbdba,0xc092,0xc371,0xc657,0xc946,0xcc3c,0xcf3b,0xd241,0xd550,0xd866,0xdb85,0xdeac,0xe1db,0xe513,0xe853,0xeb9b,0xeeeb,0xf245,0xf5a6,0xf910,0xfc83};
//gamma=3
//static const uint16_t gammaTable[] = {0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x1,0x2,0x3,0x4,0x5,0x7,0x9,0xb,0xd,0x10,0x13,0x17,0x1b,0x1f,0x24,0x2a,0x30,0x36,0x3d,0x45,0x4d,0x56,0x5f,0x69,0x74,0x80,0x8c,0x9a,0xa7,0xb6,0xc6,0xd6,0xe8,0xfa,0x10d,0x121,0x137,0x14d,0x164,0x17c,0x196,0x1b0,0x1cc,0x1e8,0x206,0x225,0x246,0x267,0x28a,0x2ae,0x2d3,0x2fa,0x322,0x34c,0x377,0x3a3,0x3d1,0x400,0x431,0x463,0x497,0x4cc,0x503,0x53c,0x576,0x5b2,0x5f0,0x62f,0x670,0x6b3,0x6f7,0x73e,0x786,0x7d0,0x81c,0x86a,0x8ba,0x90b,0x95f,0x9b5,0xa0c,0xa66,0xac2,0xb20,0xb80,0xbe2,0xc46,0xcac,0xd15,0xd80,0xded,0xe5c,0xece,0xf42,0xfb9,0x1031,0x10ac,0x112a,0x11aa,0x122c,0x12b1,0x1339,0x13c3,0x144f,0x14de,0x1570,0x1604,0x169b,0x1735,0x17d1,0x1870,0x1912,0x19b7,0x1a5e,0x1b08,0x1bb5,0x1c65,0x1d18,0x1dcd,0x1e86,0x1f41,0x2000,0x20c1,0x2186,0x224d,0x2318,0x23e6,0x24b7,0x258b,0x2662,0x273c,0x281a,0x28fb,0x29df,0x2ac6,0x2bb1,0x2c9f,0x2d90,0x2e85,0x2f7d,0x3078,0x3177,0x3279,0x337f,0x3489,0x3596,0x36a6,0x37ba,0x38d2,0x39ee,0x3b0d,0x3c2f,0x3d56,0x3e80,0x3fae,0x40df,0x4215,0x434e,0x448b,0x45cc,0x4711,0x485a,0x49a6,0x4af7,0x4c4c,0x4da4,0x4f01,0x5062,0x51c7,0x5330,0x549d,0x560e,0x5783,0x58fd,0x5a7b,0x5bfd,0x5d83,0x5f0e,0x609d,0x6230,0x63c7,0x6563,0x6704,0x68a9,0x6a52,0x6c00,0x6db2,0x6f69,0x7124,0x72e4,0x74a8,0x7671,0x783f,0x7a12,0x7be9,0x7dc4,0x7fa5,0x818a,0x8374,0x8563,0x8757,0x894f,0x8b4d,0x8d4f,0x8f56,0x9163,0x9374,0x958a,0x97a5,0x99c5,0x9beb,0x9e15,0xa045,0xa279,0xa4b3,0xa6f2,0xa936,0xab7f,0xadce,0xb022,0xb27b,0xb4da,0xb73d,0xb9a7,0xbc15,0xbe89,0xc103,0xc382,0xc606,0xc890,0xcb1f,0xcdb4,0xd04f,0xd2ef,0xd595,0xd840,0xdaf2,0xdda8,0xe065,0xe327,0xe5ef,0xe8bd,0xeb91,0xee6a,0xf14a,0xf42f,0xf71a,0xfa0b,0xfd02};
#endif

void updateLEDs()
{
#if (LED_TYPE==2811)
    leds.show();
#elif (LED_TYPE==2801)
    FastLED.show();
    //memset(leds, 0, sizeof(leds));
#elif (LED_TYPE==ADAFRUIT_MATRIX)
    backgroundLayer.swapBuffers();
#endif
}

void setup() {
  pinMode(BLINK_PIN, OUTPUT);

	// sanity check delay - allows reprogramming if accidently blowing power w/leds
  delay(2000);

  Serial.begin(115200);
  Serial.setTimeout(0);

#if (LED_TYPE==2811)
  leds.begin();
#elif (LED_TYPE==2801)
  FastLED.addLeds<WS2801, DATA_PIN, CLOCK_PIN, BRG, DATA_RATE_MHZ(4)>(leds, WIDTH*HEIGHT);
  memset(leds, 0, sizeof(leds));
#elif (LED_TYPE==ADAFRUIT_MATRIX)
  matrix.addLayer(&backgroundLayer);
  matrix.setRefreshRate(180);
  matrix.begin();
  matrix.setBrightness(255);
  backgroundLayer.enableColorCorrection(false);
#endif
}


void outputFPS() {
  static uint32_t lastFpsOutputTime = millis();
  static int32_t fpsOutputCount = 0;

  // output effective FPS every so often
  fpsOutputCount++;
  uint32_t fpsOutputTimeDiff = millis() - lastFpsOutputTime;
  if (fpsOutputTimeDiff > OUTPUT_FPS_INTERVAL) {
    digitalWrite(BLINK_PIN, HIGH);
    delayMicroseconds(1000);
    digitalWrite(BLINK_PIN, LOW);

    int32_t fps = fpsOutputCount * 1000UL / fpsOutputTimeDiff;
    p(fps);

// #if (LED_TYPE==ADAFRUIT_MATRIX)
//     char value[] = "00";
//     value[0] = '0' + fps / 100;
//     value[1] = '0' + (fps % 100) / 10;
//     value[2] = '0' + fps % 10;    
//     matrix.drawForegroundString(12, matrix.getScreenHeight()-1 -5, value, true);
//     matrix.displayForegroundDrawing();
//     matrix.clearForeground();
// #endif
    fpsOutputCount = 0;
    lastFpsOutputTime = millis();
  }
}

static int pix=0;

void loop() {
  int nbytes = Serial.readBytes((char*)buffer, BUFFER_SIZE);
  if (nbytes > WIDTH*HEIGHT*3+1) {
    Serial.print("Bytes:"); 
    Serial.println(nbytes);
  }

  for (int i=0; i<nbytes; i++) {
    int c = (int)buffer[i];
    if (c == 255) {
      updateLEDs();
      //outputFPS();
      //Serial.print(1);

      pix = 0;
      break; // may skip some frames that are backed up
    } else {
#if (LED_TYPE==2811)
#elif (LED_TYPE==2801)
    #if COLOR_DEPTH == 24
      leds[pix/3] += (c) << (8*(pix%3));
    #else // 48 bit
      leds[pix/6] += (c) << (8*(pix%6));
    #endif
#elif (LED_TYPE==ADAFRUIT_MATRIX)
    #if COLOR_DEPTH == 24
      // the order expected is:
      // 24bit:
      // RGBRGB
      // 111222    <- pixel#
      int pos = pix / 3;
      RGB_TYPE(COLOR_DEPTH)& col = backgroundLayer.backBuffer()[pos];
      switch (pix % 3) {
        case 0:
          col.red = c;
          break;
        case 1:
          col.green = c;
          break;
        case 2:
          col.blue = c;
          break;
      }
    #elif COLOR_DEPTH == 48 && defined(USE_GAMMA_CONVERSION)
      // the order expected is:
      // 24bit:
      // RGBRGB
      // 111222    <- pixel#
      int pos = pix / 3;
      RGB_TYPE(COLOR_DEPTH)& col = backgroundLayer.backBuffer()[pos];
      switch (pix % 3) {
        case 0:
          col.red = gammaTable[c];
          break;
        case 1:
          col.green = gammaTable[c];
          break;
        case 2:
          col.blue = gammaTable[c];
          break;
      }
    #else // 48 bit
      // the order expected is:
      // 48bit:
      // RRGGBBRRGGBB
      // 1 1 1 2 2 2    <- pixel#
      int pos = pix / 6;
      RGB_TYPE(COLOR_DEPTH)& col = backgroundLayer.backBuffer()[pos];
      switch (pix % 6) {
        case 0:
          col.red = (col.red & 0xFF00) + c;
          break;
        case 1:
          col.red = (c << 8) + (col.red & 0xFF);
          break;
        case 2:
          col.green = (col.green & 0xFF00) + c;
          break;
        case 3:
          col.green = (c << 8) + (col.green & 0xFF);
          break;
        case 4:
          col.blue = (col.blue & 0xFF00) + c;
          break;
        case 5:
          col.blue = (c << 8) + (col.blue & 0xFF);
          break;
      }
    #endif      
#endif

      pix++;
    }
  }
}
