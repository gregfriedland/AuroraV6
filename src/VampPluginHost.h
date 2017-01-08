#ifndef VAMP_PLUGIN_HOST_H
#define VAMP_PLUGIN_HOST_H

#include <vector>
#include <vamp-hostsdk/PluginHostAdapter.h>
#include <vamp-hostsdk/PluginInputDomainAdapter.h>
#include <vamp-hostsdk/PluginLoader.h>

using namespace Vamp;
using namespace Vamp::HostExt;

class VampPluginHost {
public:
	struct NoteInfo {
		NoteInfo(size_t midiNote) : m_midiNote(midiNote) {}
		size_t m_midiNote;
	};

	VampPluginHost(const std::string& pluginName, size_t sampleRate, size_t stepSize, size_t blockSize);
	~VampPluginHost();

	std::vector<NoteInfo> process(float* stepBuffer);

private:
	size_t m_sampleRate, m_stepSize, m_blockSize;
	float** m_blockBuffer;
	bool m_firstStepLoaded;
	Plugin *m_plugin;
	size_t m_count;
};

#endif
