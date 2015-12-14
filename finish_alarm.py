__author__ = 'mac'
import pyglet

def ring(file):

    path = 'audio_sample_old/' + file + '.wav'

    foo=pyglet.media.load(path)
    foo.play()

    def exiter(dt):
        pyglet.app.exit()
        print "Song length is: %f" % foo.duration
        # foo.duration is the song length
    pyglet.clock.schedule_once(exiter, foo.duration)

    pyglet.app.run()
