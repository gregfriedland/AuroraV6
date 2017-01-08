#include "Audio.h"
#include "Util.h"
#include <iostream>

int staticCallback(const void *inputBuffer, void *outputBuffer,
              unsigned long framesPerBuffer,
              const PaStreamCallbackTimeInfo* timeInfo,
              PaStreamCallbackFlags statusFlags,
              void *userData) {
	Audio* audio = (Audio*) userData;
	audio->callback(inputBuffer);

	return 0;
}

void Audio::callback(const void* inputBuffer) {
#ifdef USE_AUBIO
#if 1
	m_inputBuf->data = (smpl_t*)inputBuffer;

	aubio_notes_do(m_notes, m_inputBuf, m_outputBuf);
	/*
	  The notes output is a vector of length 3 containing:
	   - 0. the midi note value, or 0 if no note was found
	   - 1. the note velocity
	   - 2. the midi note to turn off
	*/

	// note end
	if (m_outputBuf->data[2] != 0) {
		size_t midi = m_outputBuf->data[2];

		std::cout << "Note end: " << midi << " duration=" << millis() - m_currentNotes[midi].m_startTimeMs << std::endl;
		m_currentNotes[midi].m_velocity = 0;
	}

	// note begin
	if (m_outputBuf->data[0] != 0) {
		size_t midi = m_outputBuf->data[0];
		size_t velocity = m_outputBuf->data[1];

		std::cout << "Note start: " << midi << "(" << velocity << ")\n";
		m_currentNotes[midi].m_velocity = velocity;
		m_currentNotes[midi].m_startTimeMs = millis();
	}
#else
	m_inputBuf->data = (smpl_t*)inputBuffer;

	aubio_onset_do(m_onsets, m_inputBuf, m_onset);
	smpl_t isOnset = fvec_get_sample(m_onset, 0);

	if (isOnset) {
		std::cout << "Onset start.\n";
		m_currentNotes[0].m_velocity = 100;
	} else {
		m_currentNotes[0].m_velocity = 0;		
	}
#endif

#elif defined(USE_VAMP)
	auto notes = m_vampPluginHost->process((float*)inputBuffer);

	for (size_t i = 0; i < m_currentNotes.size(); ++i) {
		m_currentNotes[i].m_velocity = 0;
	}
	for (auto& note: notes) {
		m_currentNotes[note.m_midiNote].m_velocity = 255;
	}
#endif
}


Audio::Audio(const AudioSettings& settings)
: m_settings(settings) {
	std::cout << "Creating audio\n";

	PaError err = Pa_Initialize();
	if (err != paNoError) {
		std::cerr << "Error: Unable to initialize audio\n";
		exit(1);
	}

	m_paInParams = new PaStreamParameters();
    m_paInParams->device = Pa_GetDefaultInputDevice();
    if (m_paInParams->device == paNoDevice) {
        std::cerr << "Error: No default input device.\n";
        exit(1);
    }
    m_paInParams->channelCount = m_settings.m_numChannels;
    m_paInParams->sampleFormat = paFloat32;
    m_paInParams->suggestedLatency = Pa_GetDeviceInfo(m_paInParams->device)->defaultLowInputLatency;
    m_paInParams->hostApiSpecificStreamInfo = NULL;

#ifdef USE_AUBIO
	m_inputBuf = new_fvec(m_settings.m_hopSize);

#if 1
	m_outputBuf = new_fvec(m_settings.m_hopSize);
	m_notes = new_aubio_notes("default", m_settings.m_bufferSize, m_settings.m_hopSize, m_settings.m_sampleRate);
    aubio_notes_set_minioi_ms(m_notes, m_settings.m_minOnsetInterval);
#else
	m_onset = new_fvec(1);
	m_onsets = new_aubio_onset("phase", m_settings.m_bufferSize, m_settings.m_hopSize, m_settings.m_sampleRate);
    aubio_onset_set_threshold(m_onsets, 0.6);
    aubio_onset_set_silence(m_onsets, -80);
    aubio_onset_set_minioi_s(m_onsets, 0.02);
   #endif
#elif defined(USE_VAMP)
    m_stepBuffer = new float[m_settings.m_hopSize];
    m_vampPluginHost = new VampPluginHost(VAMP_PLUGIN, m_settings.m_sampleRate, m_settings.m_hopSize, m_settings.m_bufferSize);
#endif
}

Audio::~Audio() {
	std::cout << "Freeing audio\n";
	delete m_paInParams;

#ifdef USE_AUBIO
	del_fvec(m_inputBuf);
#if 1
	del_fvec(m_outputBuf);
	del_aubio_notes(m_notes);
#else
	del_aubio_onset(m_onsets);
	del_fvec(m_onset);
#endif
	aubio_cleanup();
#elif defined(USE_VAMP)
	delete[] m_stepBuffer;
	delete m_vampPluginHost;
#endif
}

void Audio::start() {
	std::cout << "Starting audio\n";
    PaError err = Pa_OpenStream(
              &m_paStream,
              m_paInParams,
              NULL,                  /* &outputParameters, */
              m_settings.m_sampleRate,
              m_settings.m_hopSize * m_settings.m_numChannels,
              paClipOff,      /* we won't output out of range samples so don't bother clipping them */
              staticCallback,
              (void*)this);
	if (err != paNoError) {
		std::cerr << "Error: Unable to open audio stream\n";
		exit(1);
	}

	err = Pa_StartStream(m_paStream);    
	if (err != paNoError) {
		std::cerr << "Error: Unable to start audio stream\n";
		exit(1);
	}
 }

void Audio::stop() {
	std::cout << "Stopping Audio\n";
	Pa_StopStream(m_paStream); 	
}

const std::array<AudioNote,128>& Audio::currentNotes() const {
	return m_currentNotes;
}
