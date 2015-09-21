#include "Palette.h"


Palettes::Palettes(int palSize, int* baseColors, int numBaseColors, int baseColorsPerPalette)
: m_palSize(palSize), m_baseColors(baseColors), m_numBaseColors(numBaseColors),
  m_baseColorsPerPalette(baseColorsPerPalette)
{}

int Palettes::size() {
	return m_numBaseColors / m_baseColorsPerPalette;
}

Color24 Palettes::get(int paletteIndex, int gradientIndex) {
    assert(paletteIndex < m_numBaseColors);

    gradientIndex = gradientIndex % m_palSize;
    int subGradientSize = ceil(m_palSize / (float)m_baseColorsPerPalette);

    int baseColIndex1 = floor(gradientIndex / subGradientSize);
    int baseColIndex2 = (baseColIndex1 + 1) % m_baseColorsPerPalette;
    //cout << "gradInd=" << gradientIndex << " subGradSize=" << subGradientSize << " baseColIndex1=" << baseColIndex1 << " baseColIndex2=" << baseColIndex2 << endl;

    Color24 col1(m_baseColors[baseColIndex1 + paletteIndex * m_baseColorsPerPalette]);
    Color24 col2(m_baseColors[baseColIndex2 + paletteIndex * m_baseColorsPerPalette]);

    gradientIndex = gradientIndex % subGradientSize;

    return Color24(floor(col1.r + gradientIndex * (col2.r - col1.r) / subGradientSize),
                 floor(col1.g + gradientIndex * (col2.g - col1.g) / subGradientSize),
                 floor(col1.b + gradientIndex * (col2.b - col1.b) / subGradientSize));
}
