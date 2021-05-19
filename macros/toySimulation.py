#!/usr/bin/python

'''
  This macro performs simple simulation of muon passing through ATLAS RPC detector,
  following these assumptions:

     - Use 6 cylindrical layers of RPC without gaps and holes
     - Uniform toroidal magnetic field of 0.5 T
     - Muons originate at (0, 0)
     - Uncorrelate random noise 

  Authors: Rustem Ospanov <rustem@cern.ch>, Wenhao Feng, Shining Yang

'''

import os
import time
import sys
import random as rand
import math
import matplotlib
import matplotlib.pyplot as plt
import logging

from optparse import OptionParser

p = OptionParser()

p.add_option('--nevent', '-n',       type='int',   default = 10000, help = 'number of events to simulate')
p.add_option('--prob',               type='float', default = 0.25,  help = 'noise probability per strip')

p.add_option('-d', '--debug', action = 'store_true', default = False, help='print debug info')
p.add_option('-p', '--plot',  action = 'store_true', default = False, help='draw histograms')


(options, args) = p.parse_args()

#----------------------------------------------------------------------------------------------
def getLog(name, level='INFO', debug=False, print_time=False):

    if print_time:
        f = logging.Formatter('%(asctime)s - %(name)s: %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        f = logging.Formatter('%(name)s: %(levelname)s - %(message)s')
        
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(f)
    
    log = logging.getLogger(name)
    log.addHandler(h)
    
    if debug:
        log.setLevel(logging.DEBUG)
    else:
        if level == 'DEBUG':   log.setLevel(logging.DEBUG)
        if level == 'INFO':    log.setLevel(logging.INFO)
        if level == 'WARNING': log.setLevel(logging.WARNING)    
        if level == 'ERROR':   log.setLevel(logging.ERROR)

    return log

log = getLog(os.path.basename(__file__), debug=options.debug)


#----------------------------------------------------------------------------------------------    
def main():

    log.info('Starting main() of %s macro' %os.path.basename(__file__))

    log.info('Macro options: %s' %str(options))
    log.info('Macro arguments: %s' %str(args))    

    icount = 0
    width = 0.03
    prob = options.prob

    for i in range(options.nevent):
        z = rand.uniform(0.0, width)

        p = prob*z/width

        if rand.uniform(0.0, 1.0) < p:
            icount += 1
    
    log.info('nevent = {}, icount = {}, nevent/icount = {}'.format(options.nevent, icount, icount/options.nevent))

#----------------------------------------------------------------------------------------------    
if __name__ == "__main__":
    main()
