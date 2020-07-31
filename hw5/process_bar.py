import sys, time

class ShowProcess():
    _i = 0
    _max_arrow = 50
    _max_steps = 50
    _info = ''
    
    def __init__(self, max_steps, max_arrow=50, info='', verbose=0):
        self._max_steps = max_steps
        self._max_arrow = max_arrow
        self._info = info
        self.verbose = verbose
        self.time = 0

    def show_process(self, i=None, other=''):
        if self.time == 0:
            self.time = time.time()
        if self._i == 0 and self._info != '':
            t = '='*((self._max_arrow - len(self._info)) // 2) + self._info
            t += '=' * (self._max_arrow - len(t))
            print(t)
        if i is not None:
            self._i = i
        else:
            self._i += 1
        num_arrow = int(self._i * self._max_arrow / self._max_steps)
        num_line = self._max_arrow - num_arrow
        per = '%.2f'%(self._i * 100.0 / self._max_steps)+'%'
        time_str = ', %.2fs' % (time.time() - self.time)
        process_bar = '['+'>'*num_arrow+'-'*num_line+']'+ per + time_str + other + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()

        if self._i >= self._max_steps:
            self.close()
    def close(self):
        print('')
        self.time = 0
        if self.verbose:
            print('Done')
        self._i = 0

if __name__=='__main__':
    max_steps = 100
    process_bar = ShowProcess(max_steps)
    for i in range(max_steps):
        process_bar.show_process()
        time.sleep(0.01)

