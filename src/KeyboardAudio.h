#ifndef KEYBOARDAUDIO_H
#define KEYBOARDAUDIO_H

#include "Drawer.h"

class Audio;

class KeyboardAudioDrawer : public Drawer {
public:
	KeyboardAudioDrawer(size_t width, size_t height, size_t palSize, Audio* audio);

	virtual void reset();
	virtual void draw(int* colIndices);

private:
	Audio* m_audio;
};

#endif