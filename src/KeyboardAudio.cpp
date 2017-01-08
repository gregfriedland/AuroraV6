#include "Audio.h"
#include "KeyboardAudio.h"
#include "Util.h"

KeyboardAudioDrawer::KeyboardAudioDrawer(size_t width, size_t height, size_t palSize, Audio* audio) 
: Drawer("KeyboardAudio", width, height, palSize), m_audio(audio)
{}

void KeyboardAudioDrawer::reset() {
}

void KeyboardAudioDrawer::draw(int* colIndices) {
	auto& currentNotes = m_audio->currentNotes();
	size_t numNotes = currentNotes.size();

	fillRect(colIndices, 0, 0, m_width, m_height, 0);

#if 1
	size_t noteWidth = m_width / numNotes;
	size_t noteHeight = 10;
	size_t y = m_height / 2 - noteHeight / 2;

	for (size_t i = 0; i < numNotes; ++i) {
		if (currentNotes[i].m_velocity != 0) {
			size_t x = mapValue(i, 0, currentNotes.size() - 1, 0, m_width);
			std::cout << "note " << i << ": at " << x << " " << y << " velocity=" << currentNotes[i].m_velocity << std::endl;
			fillRect(colIndices, x, y, noteWidth, noteHeight, m_palSize / 2);
		}
	}
#else
	if (currentNotes[0].m_velocity > 0) {
		fillRect(colIndices, 0, 0, m_width, m_height, m_palSize / 2);
	}
#endif	
}
