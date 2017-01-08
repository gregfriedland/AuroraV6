#ifndef GrayScott_H
#define GrayScott_H

#include "ReactionDiffusion.h"

class Audio;

class GrayScottDrawer : public ReactionDiffusionDrawer {
public:
    GrayScottDrawer(size_t width, size_t height, size_t palSize, Audio* audio);

    virtual void reset();

 protected:
    virtual void setParams();

    float m_F, m_k;
};

#endif
