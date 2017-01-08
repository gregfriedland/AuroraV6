#ifndef AUDIO_H
#define AUDIO_H

#include <cstddef>
#include <portaudio.h>
#include <array>

// #define USE_VAMP
#define USE_AUBIO

#ifdef USE_AUBIO
	#include <aubio.h>
#elif defined(USE_VAMP)
	#define VAMP_PLUGIN "qm-vamp-plugins:qm-transcription"
	// #define VAMP_PLUGIN "vamp-aubio:aubionotes"
	#include "VampPluginHost.h"
#endif

// Audio class for managing the audio device and doing note detection
// using aubio and portaudio. Keeps track of one note at time.

struct AudioSettings {
	size_t m_numChannels = 1;
	size_t m_sampleRate = 44100;
	size_t m_bufferSize = 4096;
	size_t m_hopSize = 4096;
	size_t m_minOnsetInterval = 30;
};

struct AudioNote {
	int m_velocity = 0;
	unsigned long m_startTimeMs = 0;
};

class Audio {
 public:
	Audio(const AudioSettings& settings);
	~Audio();

	void start();
	void stop();

	void callback(const void* inputBuffer);

	const std::array<AudioNote,128>& currentNotes() const;

 protected:
 	AudioSettings m_settings;

#ifdef USE_AUBIO
    fvec_t *m_inputBuf;
#if 1
    fvec_t *m_outputBuf;
	aubio_notes_t *m_notes;
#else
  	fvec_t *m_onset;
  	aubio_onset_t *m_onsets;
#endif
#elif defined(USE_VAMP)
	float* m_stepBuffer;
	VampPluginHost* m_vampPluginHost;
#endif

    PaStream* m_paStream;
    PaStreamParameters* m_paInParams;

    std::array<AudioNote,128> m_currentNotes;
};

#endif
