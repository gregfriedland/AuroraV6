#include "VampPluginHost.h"

#define NUM_CHANNELS 1

VampPluginHost::VampPluginHost(const std::string& pluginPath, size_t sampleRate, size_t stepSize, size_t blockSize)
: m_sampleRate(sampleRate), m_firstStepLoaded(false), m_count(0) {
    // load plugin
    std::string pluginId, pluginLib;
    std::string::size_type sep = pluginPath.find(':');
    if (sep == std::string::npos) {
        std::cerr << "ERROR: Invalid Vamp plugin path: " << pluginPath << std::endl;
        exit(1);
    }

    pluginId = pluginPath.substr(sep + 1);
    pluginLib = pluginPath.substr(0, sep);

    PluginLoader *loader = PluginLoader::getInstance();

    PluginLoader::PluginKey key = loader->composePluginKey(pluginLib, pluginId);

    m_plugin = loader->loadPlugin(key, m_sampleRate, PluginLoader::ADAPT_ALL_SAFE);
    if (!m_plugin) {
        std::cerr << "ERROR: Failed to load Vamp plugin \"" << pluginId
             << "\" from library \"" << pluginLib << "\"" << std::endl;
        exit(1);
    }    

    std::cout << "Running plugin: \"" << m_plugin->getIdentifier() << "\"..." << std::endl;


    // use preferred block/step size if available
    m_blockSize = m_plugin->getPreferredBlockSize();
    m_stepSize = m_plugin->getPreferredStepSize();

    if (m_blockSize == 0) {
        m_blockSize = blockSize;
    }
    if (m_stepSize == 0) {
        if (m_plugin->getInputDomain() == Plugin::FrequencyDomain) {
            m_stepSize = m_blockSize/2;
        } else {
            m_stepSize = m_blockSize;
        }
    } else if (m_stepSize > m_blockSize) {
        std::cerr << "WARNING: stepSize " << m_stepSize << " > blockSize " << m_blockSize << ", resetting blockSize to ";
        if (m_plugin->getInputDomain() == Plugin::FrequencyDomain) {
            m_blockSize = m_stepSize * 2;
        } else {
            m_blockSize = m_stepSize;
        }
        std::cerr << m_blockSize << std::endl;
    }

    std::cout << "Vamp plugin using block size = " << blockSize << ", step size = "
              << stepSize << std::endl;


    // get output descriptor
    Plugin::OutputList outputs = m_plugin->getOutputDescriptors();
    if (outputs.empty()) {
        std::cerr << "ERROR: Vamp plugin has no outputs!" << std::endl;
        exit(1);
    }

    Plugin::OutputDescriptor& output = outputs[0];
    std::cerr << "Vamp output is: \"" << output.identifier << "\"" << std::endl;



    if (!m_plugin->initialise(NUM_CHANNELS, m_stepSize, m_blockSize)) {
        std::cerr << "ERROR: Vamp plugin initialise (channels = " << NUM_CHANNELS
             << ", stepSize = " << m_stepSize << ", blockSize = "
             << m_blockSize << ") failed." << std::endl;
        exit(1);
   }

    m_blockBuffer = new float*[NUM_CHANNELS];
    for (size_t c = 0; c < NUM_CHANNELS; ++c) {
        m_blockBuffer[c] = new float[m_blockSize + 2];
    }
}

VampPluginHost::~VampPluginHost() {
    delete[] m_blockBuffer;
    delete m_plugin;
}

std::vector<VampPluginHost::NoteInfo> VampPluginHost::process(float* stepData) {
    // memmove(m_blockBuffer, stepData, m_stepSize * NUM_CHANNELS * sizeof(float));

    for (size_t c = 0; c < NUM_CHANNELS; ++c) {
        size_t j = 0;
        while (j < m_blockSize) {
            m_blockBuffer[c][j] = stepData[j * NUM_CHANNELS + c];
            ++j;
        }
    }


    if (m_stepSize == m_blockSize) {
        m_firstStepLoaded = true;
    }

    std::vector<VampPluginHost::NoteInfo> notes; 
    if (m_firstStepLoaded) {
        RealTime rt = RealTime::frame2RealTime(m_count * m_stepSize, m_sampleRate);
        Plugin::FeatureSet features = m_plugin->process(m_blockBuffer, rt);
        Plugin::FeatureSet remainingFeatures = m_plugin->getRemainingFeatures();

        std::cout << "# features: " << features.size() << " # remaining features: " << remainingFeatures.size() << std::endl;

        for (auto& feature: features[0]) {
            std::cout << "Got note: " << feature.values[0] << std::endl;
            notes.push_back(VampPluginHost::NoteInfo(feature.values[0]));
        }
        for (auto& feature: remainingFeatures[0]) {
            std::cout << "Got note: " << feature.values[0] << std::endl;
            notes.push_back(VampPluginHost::NoteInfo(feature.values[0]));            
        }
    }

    if (m_stepSize != m_blockSize) {
        memmove(m_blockBuffer + m_stepSize * NUM_CHANNELS, m_blockBuffer, (m_blockSize - m_stepSize) * NUM_CHANNELS * sizeof(float));
        m_firstStepLoaded = true;
    }

    ++m_count;
    return notes;
}