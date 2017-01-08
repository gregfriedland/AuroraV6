#ifndef GINZBURGLANDAU_H
#define GINZBURGLANDAU_H

#include "ReactionDiffusion.h"

class Audio;

class GinzburgLandauDrawer : public ReactionDiffusionDrawer {
public:
    GinzburgLandauDrawer(int width, int height, int palSize, Audio* audio);

    virtual void reset();

 protected:
    virtual void setParams();

    float m_alpha, m_beta, m_gamma, m_delta;
};

#endif
