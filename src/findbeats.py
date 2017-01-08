#!python

# find beats in audio

import subprocess as sp
import numpy as np
import sys
import pygame as pg
from pygame.locals import FULLSCREEN, DOUBLEBUF
import math

CAPTION = "FindBeats: logFFT"
SCREEN_SIZE = (1024, 640)
BACKGROUND_COLOR = (0, 0, 0)

SAMPLE_RATE = 44100
CHANNELS = 2
BUFFER_SIZE = 2048

NUM_BANDS = 5

def loadSound(filename, sampleRate, channels):
    FFMPEG_BIN = "ffmpeg"
    command = [ FFMPEG_BIN,
            '-i', filename,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(sampleRate),
            '-ac', str(channels),
            '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=2**8)

    raw_audio = pipe.stdout.read() #88200*4)

    # Reorganize raw_audio as a Numpy array
    audioArray = np.fromstring(raw_audio, dtype="int16")

    return audioArray

# pygame seems unable to play at 44100 on my laptop and
# seems to need 22050
def startSound(audioArray, in_sampleRate, in_channels, out_sampleRate=22050, out_channels=2):
    print "Starting sound"

    import audioop

    audioArray2 = audioop.ratecv(audioArray, 2, in_channels, in_sampleRate, out_sampleRate, None)[0]
    if out_channels == 1:
        audioArray2 = audioop.tomono(audioArray2, 2, 0.5, 0.5)[0]
    audioArray3 = np.frombuffer(audioArray2, np.int16)

    if out_channels > 1:
        audioArray3 = audioArray3.reshape((len(audioArray3)/out_channels,out_channels))

    pg.mixer.init(frequency=out_sampleRate, size=-16, channels=out_channels)
#    if DEBUG: print pg.mixer.get_init()
    sound = pg.sndarray.make_sound(audioArray3)
    playing = sound.play()    


def analyzeAudio(audioArray, numBands):
    out = np.fft.fft(audioArray)

    # group into logaraithmic bands
    # logSpacing = logspace(1, audioArray.shape[0] + 1, NUM_BANDS + 1)
    logSpacing = 10 ** np.linspace(np.log10(5), np.log10(audioArray.shape[0]), NUM_BANDS)
    # print "logSpacing:", logSpacing

    outLog = np.zeros(NUM_BANDS, np.complex)
    logInd = 0
    for i in range(out.shape[0]):
        if i > logSpacing[logInd]:
            logInd += 1
        outLog[logInd] += out[i]

    return outLog

if __name__ == "__main__":
    pg.init()
    pg.display.set_caption(CAPTION)
    display = pg.display.set_mode(SCREEN_SIZE, 0, 32)

    snd = loadSound(sys.argv[1], SAMPLE_RATE, CHANNELS)
    print "snd:", snd.shape
    startSound(snd, SAMPLE_RATE, CHANNELS)

    display.fill(BACKGROUND_COLOR)
    pg.draw.rect(display,(0,0,255),(200,150,100,50))
    clock = pg.time.Clock()
    clock.tick(1)

    maxOut = 0
    i = 0
    while i + BUFFER_SIZE < snd.shape[0]:
        print i

        for event in pg.event.get():
            keys = pg.key.get_pressed()
            if event.type == pg.QUIT or keys[pg.K_ESCAPE]:
                sys.exit()

        display.fill(BACKGROUND_COLOR)
        # pg.draw.rect(surface, (0,0,0), (0,0, SCREEN_SIZE[0], SCREEN_SIZE[1]))

        buf = snd[i:i+BUFFER_SIZE]
        out = analyzeAudio(buf, NUM_BANDS)
        out = np.abs(out)
        print "out:", out
        maxOut = max(maxOut, out.max())

        for j in range(len(out)):
            y = int(SCREEN_SIZE[1] * out[j] / maxOut)
            # print y,
            rect = (j * SCREEN_SIZE[0] / 5, 0, SCREEN_SIZE[0] / 5, y)
            # print rect
            display.fill((255,255,255), rect)
        print

        pg.display.flip()
        pg.display.update()
        clock.tick(SAMPLE_RATE / BUFFER_SIZE)

        i += BUFFER_SIZE
