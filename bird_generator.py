# https://people.ece.cornell.edu/land/courses/ece4760/labs/f2021/lab1birdsong/Birdsong_keypad.html#Deconstructing-the-song
# https://people.ece.cornell.edu/land/courses/ece4760/labs/f2021/Birds-serial/Birdsong_serial.html#Software
# https://vanhunteradams.com/Birds-serial/Birdsong_synthesis.html
# https://www.sciencedirect.com/science/article/pii/S2212017313005252
# https://github.com/Uberi/speech_recognition/blob/master/examples/background_listening.py
# https://gist.github.com/ryanbekabe/0a2c840134f9b7dfb635429e5e17c6ed
# https://people.csail.mit.edu/hubert/pyaudio/docs/#example-callback-mode-audio-i-o
# https://stackoverflow.com/questions/47189624/maintain-a-streaming-microphone-input-in-python
from random import randrange, random
import numpy as np
import simpleaudio as sa

import librosa

Fs = 44100  # audio sample rate
sintable = np.sin(np.linspace(0, 2 * np.pi, 256))  # sine table for DDS
two32 = 2**32  # 2^32
silence = np.zeros(5720)


def build_swoop():
    swoop = list(
        np.zeros(randrange(5719, 5720))
    )  # a 5720-length array (130ms @ 44kHz) that will hold swoop audio samples
    DDS_phase = 0  # current phase
    for i in range(len(swoop)):
        r1 = -260  # randrange(-100, 1000)  # -260
        f1 = 1740  # randrange(1000, 3480) # 1740
        frequency = -r1 * np.sin((-np.pi / len(swoop)) * i) + f1  # calculate frequency
        DDS_increment = frequency * two32 / Fs  # update DDS increment
        DDS_phase += DDS_increment  # update DDS phase by increment
        DDS_phase = DDS_phase % (
            two32 - 1
        )  # need to simulate overflow in python, not necessary in C
        swoop[i] = sintable[int(DDS_phase / (2**24))]  # can just shift in C

    # Amplitude modulate with a linear envelope to avoid clicks
    amplitudes = list(np.ones(len(swoop)))
    amplitudes[0:1000] = list(np.linspace(0, 1, len(amplitudes[0:1000])))
    amplitudes[-1000:] = list(np.linspace(0, 1, len(amplitudes[-1000:]))[::-1])
    amplitudes = np.array(amplitudes)

    # Finish with the swoop
    swoop = swoop * amplitudes

    return swoop


def build_chirp():
    chirp = list(
        np.zeros(5720)
    )  # a 5720-length array (130ms @ 44kHz) that will hold chirp audio samples
    DDS_phase = 0  # current phase
    for i in range(len(chirp)):
        frequency = (1.53e-4) * (i**2.0) + 2000  # update DDS frequency
        DDS_increment = frequency * two32 / Fs  # update DDS increment
        DDS_phase += DDS_increment  # update DDS phase
        DDS_phase = DDS_phase % (
            two32 - 1
        )  # need to simulate overflow in python, not necessary in C
        chirp[i] = sintable[int(DDS_phase / (2**24))]  # can just shift in C

    # Amplitude modulate with a linear envelope to avoid clicks
    amplitudes = list(np.ones(len(chirp)))
    amplitudes[0:1000] = list(np.linspace(0, 1, len(amplitudes[0:1000])))
    amplitudes[-1000:] = list(np.linspace(0, 1, len(amplitudes[-1000:]))[::-1])
    amplitudes = np.array(amplitudes)

    # Finish with the chirp
    chirp = chirp * amplitudes

    return chirp


swoop = build_swoop()
chirp = build_chirp()


def add_sound(song, pitch, sr):
    r = random()
    if r > 0.4:
        sound = swoop
    else:
        sound = chirp

    sound = librosa.effects.pitch_shift(sound, sr=sr, n_steps=-12 + 24 * pitch)
    song.extend(list(sound))

    return song


# https://zulko.github.io/blog/2014/03/29/soundstretching-and-pitch-shifting-in-python/
def build_song(pitches, sr):
    song = []
    for i in pitches:
        song = add_sound(song=song, pitch=i, sr=sr)

        r = random()
        if r > 0.5:
            song.extend(list(silence))
        else:
            song = add_sound(song=song, pitch=i, sr=sr)

    return np.array(song)


def play_audio(audio):
    # normalize to 16-bit range
    audio *= 32767 / np.max(np.abs(audio))
    # convert to 16-bit data
    audio = audio.astype(np.int16)

    play_obj = sa.play_buffer(audio, 1, 2, Fs)
    play_obj.wait_done()

    return audio
