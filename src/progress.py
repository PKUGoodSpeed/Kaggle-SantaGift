from sys import stdout

class ProgressBar:
    '''
    Here I implement a progress bar program by myself.
    '''
    __bar = 50
    __blk = 0
    __n = 0
    def __init__(self):
        pass
        
    def setBar(self, num_iteration, bar_size = 50, bracket = '[]'):
        assert len(bracket) == 2
        self.__blk = (num_iteration + bar_size - 1)/bar_size
        self.__bar = num_iteration/self.__blk
        self.__n = num_iteration
        stdout.write("{1}{0}{2}".format(' '*self.__bar, bracket[0], bracket[1]))
        stdout.flush()
        stdout.write("\b" * (self.__bar + 1))
        
    def show(self, i):
        if((i+1) % self.__blk == 0):
            stdout.write("=")
            stdout.flush()
        if(i+1 == self.__n):
            stdout.write('\n')
        
    