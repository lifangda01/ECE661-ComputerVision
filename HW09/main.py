import levmar
import scipy.optimize
from numpy.testing import *
import numpy


## This is y-data:
y_data = numpy.array([0.2867, 0.1171, -0.0087, 0.1326, 0.2415, 0.2878, 0.3133, 0.3701, 0.3996, 0.3728, 0.3551, 0.3587, 0.1408, 0.0416, 0.0708, 0.1142, 0, 0, 0])

## This is t-data:
t = numpy.array([67., 88, 104, 127, 138, 160, 169, 188, 196, 215, 240, 247, 271, 278, 303, 305, 321, 337, 353])

def fitfunc(p, t):
    """This is the equation"""
    return p[0] + (p[1] -p[0]) * ((1/(1+numpy.exp(-p[2]*(t-p[3])))) + (1/(1+numpy.exp(p[4]*(t-p[5])))) -1)
   
def errfunc(p, t, y):
    return fitfunc(p,t) -y
    
guess = numpy.array([0, max(y_data), 0.1, 140, -0.1, 270])
p2, C, info, msg, success = scipy.optimize.leastsq(errfunc, guess, args=(t, y_data), full_output=1)

print p2

print numpy.sum(errfunc(p2, t, y_data)**2)
print numpy.sum(errfunc(p3, t, y_data)**2)
# p2, C, info, msg, success = scipy.optimize(

    
#iters, result = levmar.ddif(func, initial, measurement, 5000,
#                            data = Ab)

# if result:
#     ex, ey, ez, et = result
#     tx, ty, tz, tt = map(float, bits[16:20])
#     print '%.4f %.4f %.4f (%.4f %.4f %.4f) %d' % \
#           (ex, ey, ez,  tx, ty, tz, iters)

# ##chk = levmar.dchkjac(func, jacf, (0.5, 0.5, 0.5, 10.0), 4, Ab)

            
        