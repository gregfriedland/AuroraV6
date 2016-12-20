#ifndef PALETTE_H
#define PALETTE_H

#include <math.h>
#include <assert.h>
#include <iostream>
#include "Util.h"

class Palettes {
public:
    Palettes(int palSize, int* baseColors, int numBaseColors, int baseColorsPerPalette, float gamma);

    int size();

    Color24 get(int paletteIndex, int gradientIndex);
    
private:
    int m_palSize;
    int* m_baseColors;
    int m_numBaseColors;
    int m_baseColorsPerPalette;
    unsigned char m_gammaTable[256];
};

#endif
