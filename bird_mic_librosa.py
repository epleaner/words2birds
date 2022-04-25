#!/usr/bin/env python3
import numpy as np
import os
import tempfile
import librosa

import speech_recognition as sr

import bird_generator

# this is called from the background thread
def callback(recognizer, audio):
    print("got some audio")

    samplerate = 16000

    tmp = tempfile.NamedTemporaryFile(delete=False)

    try:
        print("writing to temp file")
        tmp.write(audio.get_wav_data())

        print("loading into librosa")
        y, sr = librosa.load(tmp.name, sr=samplerate)

        print("getting pitches...")
        pitches, magnitudes = np.array(librosa.piptrack(y=y, sr=sr))
        length = min(np.shape(pitches)[0], np.shape(magnitudes)[0])

        arr = []
        for t in range(length - 5):
            try:
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]

                if pitch != 0:
                    arr.append(pitch)
            except IndexError:
                continue

        result = (arr - np.min(arr)) / np.ptp(arr)

        n = 10
        end = n * int(len(result) / n)
        result = np.mean(result[:end].reshape(-1, n), 1)

        print("got %i pitches, building song" % len(result))

        song = bird_generator.build_song(pitches=result, sr=sr)

        print("playing song")
        bird_generator.play_audio(song)
        print("listening!")

    finally:
        tmp.close()
        os.unlink(tmp.name)


r = sr.Recognizer()
m = sr.Microphone()

with m as source:
    print("adjusting for ambient noise...")
    r.adjust_for_ambient_noise(
        source
    )  # we only need to calibrate once, before we start listening

print("listening!")
# start listening in the background (note that we don't have to do this inside a `with` statement)
stop_listening = r.listen_in_background(m, callback)

while True:
    1
