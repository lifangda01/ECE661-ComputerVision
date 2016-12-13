from pylab import *
a = [1,2,3]
b = [1,2,3]
line, = plot(a,b,label="lol")
legend(handles=[line])
show()