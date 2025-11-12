import winsound

class audioService:
    def __init__(self, frequency=1000,duration=300):
        self.frequency = frequency
        self.duration = duration
    
    def play(self):
        winsound.Beep(self.frequency, self.duration)
