from gtts import gTTS
import pyglet

def speak_str(blabla):

# blabla = ("The average is 0.98")
    tts = gTTS(text=blabla, lang='en')
    tts.save("tts_speech/test.mp3")

    path = 'tts_speech/test.mp3'
    foo=pyglet.media.load(path)
    foo.play()

    def exiter(dt):
        pyglet.app.exit()
        print "Song length is: %f" % foo.duration
        # foo.duration is the song length
    pyglet.clock.schedule_once(exiter, foo.duration)

    pyglet.app.run()


