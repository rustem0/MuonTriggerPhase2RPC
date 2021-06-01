#!/usr/bin/python3

'''
  This macro performs simple simulation of muon passing through ATLAS RPC detector,
  following these assumptions:

     - Use 6 cylindrical layers of RPC without gaps and holes
     - Uniform toroidal magnetic field of 0.5 T
     - Muons originate at (0, 0)
     - Uncorrelate random noise

  Authors: Rustem Ospanov <rustem at cern dot ch>, Wenhao Feng, Shining Yang

'''

import os
import time
import sys
import random as rand
import math
import csv
import itertools
import pickle
import logging
import collections

import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
import torch
import torch.nn as nn

from scipy.optimize import root
from collections import defaultdict
from enum import Enum
from optparse import OptionParser

p = OptionParser()

p.add_option('--nevent', '-n',       type='int',   default =  0,    help = 'number of events to simulate')
p.add_option('--min-angle',          type='int',   default = 85,    help = 'minimum muon angle in degrees')
p.add_option('--max-angle',          type='int',   default = 40,    help = 'maximum muon angle in degrees')
p.add_option('--seed',               type='int',   default = 42,    help = 'random seed')

p.add_option('--min-pt',             type='float', default = 3.0,   help = 'minimum muon pT')
p.add_option('--max-pt',             type='float', default = 30.0,  help = 'maximum muon pT')
p.add_option('--rpc-radius',         type='float', default = 6.0,   help = 'radius of RPC region where magnetic field begins')
p.add_option('--noise-prob',         type='float', default = 0.001, help = 'noise probability per strip')
p.add_option('--rpc-eff',            type='float', default = 0.95,  help = 'RPC signal efficiency')
p.add_option('--cluster-prob-2hit',  type='float', default = 0.25,  help = 'Probability to produce RPC cluster with 2 strips')
p.add_option('--cluster-prob-3hit',  type='float', default = 0.05,  help = 'Probability to produce RPC cluster with 3 strips')
p.add_option('--strip-width',        type='float', default = 0.03,  help = 'RPC strip width in meters')
p.add_option('--rpc1-max-deltaz',    type='float', default = 0.15,  help = 'Maximum RPC1 delta z for making candidates')
p.add_option('--rpc3-max-deltaz',    type='float', default = 0.60,  help = 'Maximum RPC3 delta z for making candidates')

p.add_option('--input-dim',             type='int',   default= 3)
p.add_option('--output-dim',            type='int',   default= 1)
p.add_option('--fc1',                   type='int',   default=20)
p.add_option('--fc2',                   type='int',   default=20)
p.add_option('--fc3',                   type='int',   default=20)

p.add_option('-d', '--debug',     action = 'store_true', default = False, help='print debug info')
p.add_option('-v', '--verbose',   action = 'store_true', default = False, help='print debug info')
p.add_option('--draw',            action = 'store_true', default = False, help='draw event display')
p.add_option('--draw-line',       action = 'store_true', default = False, help='draw muon direction line')
p.add_option('-p', '--plot',      action = 'store_true', default = False, help='plot histograms')
p.add_option('-w', '--wait',      action = 'store_true', default = False, help='wait for click on figures to continue')

p.add_option('--logy',            action = 'store_true', default = False, help='draw histograms with log Y')
p.add_option('--veto-noise-cand', action = 'store_true', default = False, help='veto candidates containing noise hits')
p.add_option('--do-pdf',          action = 'store_true', default = False, help='save figures as PDF')
p.add_option('--has-rpc2-noise',  action = 'store_true', default = False, help='require noise hits in RPC2 later for event display')
p.add_option('--no-out',          action = 'store_true', default = False, help='do not create default output directory and do not write output')

p.add_option('--outdir', '-o',    type='string',  default = None, help = 'output directory')
p.add_option('--in-pickle',       type='string',  default = None, help = 'input path for pickle output file')
p.add_option('--torch-model',     type='string',  default = None, help = 'input path for torch model')
p.add_option('--plot-dir',        type='string',  default = None, help = 'output directory for plots')

(options, args) = p.parse_args()

#----------------------------------------------------------------------------------------------
def getLog(name, level='INFO', debug=False, print_time=False, verbose=False):

    if print_time:
        f = logging.Formatter('%(asctime)s - %(name)s: %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    elif options.debug:
        f = logging.Formatter('%(name)s: %(levelname)s - %(message)s')
    else:
        f = logging.Formatter('%(message)s')

    if verbose:
        h = logging.StreamHandler(filename=name)
        h.setFormatter(f)
    else:
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

#----------------------------------------------------------------------------------------------
log = getLog(os.path.basename(__file__), debug=options.debug)

if options.verbose:
    logVerbose = getLog('verbose.log', debug=options.debug)
else:
    logVerbose = None

#----------------------------------------------------------------------------------------------
def verbose(*args):

    if logVerbose:
        out = ''

        for arg in args:
            out += '{} '.format(arg)

        logVerbose.debug(out)

#----------------------------------------------------------------------------------------------
def getOutPath(filename=None):

    if options.no_out:
        return None

    if options.outdir:
        outdir = options.outdir
    else:
        outdir = 'rpc-sim_{:06d}-events_{:04d}-noise_{:06d}-seed'.format(options.nevent, int(10000*options.noise_prob), options.seed)

    if not os.path.isdir(outdir):
        log.info('getOutPath - create directory: "{}"'.format(outdir))
        os.makedirs(outdir)

    if filename:
        return '{}/{}'.format(outdir, filename)
        
    return outdir

#----------------------------------------------------------------------------------------------
def getPlotPath(plotName=None, doPDF=options.do_pdf):

    if options.plot_dir:
        return options.plot_dir

    if options.in_pickle and options.torch_model:
        pdir = '{}/plot-model-{}/'.format(os.path.dirname(options.in_pickle), os.path.dirname(options.torch_model).replace('/', ''))

        if not os.path.isdir(pdir):
            log.info('getPlotPath - make new directory for plots: {}'.format(pdir))
            os.makedirs(pdir)

        if plotName:
            plotPath = '{}{}'.format(pdir, plotName)

            if doPDF:
                plotPath = plotPath.replace('.png', '.pdf')

            return plotPath

        return pdir

    return None

#----------------------------------------------------------------------------------------------
class Origin(Enum):

    '''Enumerate different sources of raw hits'''

    MUON = 1
    NEARBY = 2
    NOISE = 3

#----------------------------------------------------------------------------------------------
class Layer(Enum):

    '''Enumerate different RPC layers'''

    RPC1 = 1
    RPC2 = 2
    RPC3 = 3

    RPC11 = 11
    RPC12 = 12

    RPC21 = 21
    RPC22 = 22

    RPC31 = 31
    RPC32 = 32

    @staticmethod
    def isFirstDoubletLayer(layer):
        return layer in [Layer.RPC11, Layer.RPC21, Layer.RPC31]

    @staticmethod
    def getDoublet(layer):

        if layer in [Layer.RPC1, Layer.RPC2, Layer.RPC3]:
            return layer

        if layer in [Layer.RPC11, Layer.RPC12]:
            return Layer.RPC1

        if layer in [Layer.RPC21, Layer.RPC22]:
            return Layer.RPC2

        if layer in [Layer.RPC31, Layer.RPC32]:
            return Layer.RPC3

        raise Exception('getDoublet - unknown enum: %s' %layer)

#----------------------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(options.input_dim, options.fc1)
        self.fc2 = nn.Linear(options.fc1, options.fc2)
        self.fc3 = nn.Linear(options.fc2, options.fc3)
        self.fc4 = nn.Linear(options.fc3, options.output_dim)

    def forward(self,x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#----------------------------------------------------------------------------------------------
Strip = collections.namedtuple('Strip', ['layer', 'layerObject', 'zcenter', 'number'])

#----------------------------------------------------------------------------------------------
class RawHit:
    '''RawHit - hit on one RPC strip'''

    def __init__(self, strip, origin, muonZ=None) -> None:

        self.strip  = strip
        self.origin = origin
        self.muonZ  = muonZ

    def __str__(self) -> str:
        return self.printHit()

    def getLayer(self):
        return self.strip.layer

    def getY(self):
        return self.strip.layerObject.y

    def getZ(self):
        return self.strip.zcenter

    def getAbsDeltaZ(self, otherHit):
        return abs(self.strip.zcenter - otherHit.strip.zcenter)

    def printHit(self):
        return 'RawHit {} with strip center z = {:.3f}, due to {}'.format(self.strip.layer, self.strip.zcenter, self.origin)

#----------------------------------------------------------------------------------------------
class SingleLayer:
    '''SingleLayer - data for one RPC singlet layer'''


    def __init__(self, layer, z1, z2, y) -> None:

        self.layer  = layer
        self.z1     = z1
        self.z2     = z2
        self.y      = y
        self.nstrip = int((z2 - z1)//options.strip_width)

    def plotLayer(self, ax, minz=None, maxz=None, npoint=100, color='black', yoffset=0.0, showStrips=False, stripOffset=0.0):

        zarr = []
        yarr = []

        for i in range(0, npoint+1):
            z = self.z1 + i*(self.z2 - self.z1)/npoint

            if minz != None and z < minz:
                continue
            if maxz != None and z > maxz:
                continue

            zarr += [z]
            yarr += [self.y + yoffset]

        ax.plot(zarr, yarr, color=color)

        if showStrips:
            stripY = []
            stripZ = []

            for i in range(self.nstrip+1):
                stripZ += [self.z1 + i*options.strip_width]
                stripY += [self.y + stripOffset]

            ax.scatter(stripZ, stripY, color='green', marker='|')

        return zarr, yarr

    def getDoublet(self):
        return Layer.getDoublet(self.layer)

    def getStripNumber(self, z):
        if z < self.z1 or z > self.z2:
            return None

        return int((z - self.z1)//options.strip_width)

    def getStripCenter(self, z):
        stripN = self.getStripNumber(z)

        if stripN == None:
            return None

        stripZ = (stripN + 0.5)*options.strip_width

        log.debug('z={}, n={:d}, center={:.3f}'.format(z, stripN, stripZ))

        return stripZ

    def getClosestStrip(self, z):
        stripN = self.getStripNumber(z)
        stripZ = self.getStripCenter(z)

        if stripN == None or stripZ == None:
            return None

        return Strip(layer=self.layer, layerObject=self, zcenter=stripZ, number=stripN)

    def getSecondClosestStrip(self, z):
        strip = self.getClosestStrip(z)

        if strip == None:
            return None

        stripL = self.getClosestStrip(strip.zcenter - 0.9*options.strip_width)
        stripR = self.getClosestStrip(strip.zcenter + 0.9*options.strip_width)

        if stripL and stripR:
            if abs(stripL.zcenter - z) < abs(stripR.zcenter - z):
                return stripL
            return stripR
        elif stripR:
            return stripR
        elif stripL:
            return stripL

        raise Exception('getSecondClosestStrip - failed to find closest strip for z={}'.format(z))

    def getStrip(self, number):

        if number < 0 or number > self.nstrip:
            return None

        return Strip(layer=self.layer, layerObject=self, zcenter=(number + 0.5)*options.strip_width, number=number)

    def printLayer(self):
        return 'layer {} y = {}'.format(self.layer, self.y)

    def __str__(self) -> str:
        return self.printLayer()

layerRPC32 = SingleLayer(layer=Layer.RPC32, z1=0, z2=12.0, y=9.82)
layerRPC31 = SingleLayer(layer=Layer.RPC31, z1=0, z2=12.0, y=9.80)

layerRPC22 = SingleLayer(layer=Layer.RPC22, z1=0, z2=9.60, y=7.52)
layerRPC21 = SingleLayer(layer=Layer.RPC21, z1=0, z2=9.60, y=7.50)

layerRPC12 = SingleLayer(layer=Layer.RPC12, z1=0, z2=9.60, y=6.82)
layerRPC11 = SingleLayer(layer=Layer.RPC11, z1=0, z2=9.60, y=6.80)

allLayers = [layerRPC11, layerRPC12, layerRPC21, layerRPC22, layerRPC31, layerRPC32]

#----------------------------------------------------------------------------------------------
class Trajectory:
    '''Trajectory - solve muon trajectory y = f(z) under these constraints:
            - Muon trajectory starts at (y, z) = (0, 0)
            - Zero magnetic field for y < rpc_radius
            - Uniform toroidal magnetic field for y > rpc_radius

    '''

    def __init__(self, muonPt, muonAngleDeg, muonSign):

        self.muonPt       = muonPt
        self.muonAngleDeg = muonAngleDeg
        self.muonAngle    = math.pi*self.muonAngleDeg/180.0
        self.muonSign     = muonSign
        self.muonRadius   = self.muonPt*6.6

        #
        # Coordinates of the point where muon enters magnetic field
        #
        self.yb = options.rpc_radius
        self.zb = options.rpc_radius/math.tan(self.muonAngle)

        #
        # Solve for equation of circle with centre at (z0, y0),
        # such that the line parallel to cicrle is straight line passing through (zb, yb)
        #

        self.a0 = math.tan(self.muonAngle + math.pi/2.0)
        self.b0 = self.yb - math.tan(self.muonAngle + math.pi/2.0)*self.zb

        self.aq = (1 + self.a0**2)
        self.bq = self.a0*(self.b0 - self.yb) - self.zb
        self.cq = self.zb**2 + (self.b0 - self.yb)**2 - self.muonRadius**2

        self.z0 = (-self.bq + self.muonSign*math.sqrt(self.bq**2 - self.aq*self.cq))/self.aq
        self.y0 = self.z0*math.tan(self.muonAngle + math.pi/2.0) + self.yb - self.zb*math.tan(self.muonAngle + math.pi/2.0)

    def __str__(self):
        return self.printTrajectory()

    def getYatZ(self, z, returnPair=False):

        if z <= self.zb:
            return self.getLinearYatZ(z)

        diff = self.muonRadius**2 - (z - self.z0)**2

        if diff < 0:
            return None

        y1 = self.y0 + math.sqrt(diff)
        y2 = self.y0 - math.sqrt(diff)

        if returnPair:
            return (y1, y2)

        if self.muonSign > 0:
            return max([y1, y2])
        else:
            return min([y1, y2])

        raise Exception('getYatZ - y out of range: z={}, y1={}, y2={}'.format(z, y1, y2))

    def getZatY(self, y):

        if y <= self.yb:
            return self.getLinearZatY(y)

        diff = self.muonRadius**2 - (y - self.y0)**2

        if diff < 0.0:
            return None

        z1 = self.z0 - math.sqrt(diff)
        z2 = self.z0 + math.sqrt(diff)

        if self.muonSign > 0: return min([z1, z2])
        if self.muonSign < 0: return max([z1, z2])

        raise Exception('getZatY - z out of range: y={}, z1={}, z2={}'.format(y, z1, z2))


    def getLinearYatZ(self, z):
        return z*math.tan(self.muonAngle)

    def getLinearZatY(self, y):
        return y/math.tan(self.muonAngle)

    def getOrthogonalLineYatZ(self, z):
        return (z-self.zb)*math.tan(self.muonAngle + math.pi/2.0) + self.yb

    def printTrajectory(self):

        s = 'muon pT = {0:.1f}, angle = {1:.1f} degrees, R = {2:.2f} meter'.format(self.muonPt*self.muonSign, 
                                                                                   self.muonAngleDeg, 
                                                                                   self.muonRadius)

        s += '\n   zb = {0}, yb = {1}'.format(self.zb, self.yb)
        s += '\n   a0 = {0}, b0 = {1}'.format(self.a0, self.b0)
        s += '\n   aq = {0}, bq = {1}, cq = {2}'.format(self.aq, self.bq, self.cq)
        s += '\n   z0 = {0}, y0 = {1}'.format(self.z0, self.y0)

        return s

#----------------------------------------------------------------------------------------------
class PredPath:
    '''PredPath - solve predicted muon trajectory y = f(z) under these constraints:
            - Muon passes through seed point zs, ys
            - Muon passes through point at zr, yr = rpc_radius - where zr to be determined
            - Muon trajectory tangent at zr, yr = artan(yr/zr) - straight line passing (0, 0) and (zr, yr)
    '''

    def __init__(self, seedz, seedy, predMuonQPt) -> None:
        self.seedz          = seedz
        self.seedy          = seedy
        self.predMuonQPt    = predMuonQPt
        self.predMuonRadius = abs(self.predMuonQPt)*6.6

        self.result = None

    def __str__(self) -> str:
        return self.printPredPath()

    def hasConverged(self):
        return self.getStatus() == 1

    def getStatus(self):
        if self.result == None:
            return 0
        return self.result.status

    def getPredAngle(self):
        if self.result == None:
            return None

        return math.atan(options.rpc_radius/self.result.x[0])

    def getPredAngleDeg(self):
        angle = self.getPredAngle()

        if angle == None:
            return None

        return 180*angle/math.pi

    def solveTrajectory(self, initx):
        self.result = root(self.getFunc, initx, tol=1e-9)

        return self.result

    def getFunc(self, x):
        zr = x[0]
        z0 = x[1]
        y0 = x[2]

        zs = self.seedz
        ys = self.seedy
        yr = options.rpc_radius

        f0 = (zs - z0)**2 + (ys - y0)**2 - self.predMuonRadius**2
        f1 = (zr - z0)**2 + (yr - y0)**2 - self.predMuonRadius**2
        f2 = (yr**2)*(self.predMuonRadius**2 - (zr - z0)**2) -4*(zr**2)*((zr - z0)**2)

        return (f0, f1, f2)

    def printPredPath(self):
        s = 'PredPath - seedz = {:.4f}, seedy = {:.4f}, predMuonQPt = {:.1f}, predMuonRadius = {:.1f}'.format(
            self.seedz,
            self.seedy,
            self.predMuonQPt,
            self.predMuonRadius)

        return s

#----------------------------------------------------------------------------------------------
class CandEvent:

    def __init__(self, simEvent, seedCluster):
        self.seedCluster = seedCluster
        self.seedAngle   = math.atan(seedCluster.getY()/seedCluster.getMeanZ())

        self.rpc1Cluster = None
        self.rpc3Cluster = None

        self.rpc1DeltaZ  = None
        self.rpc3DeltaZ  = None

        self.muonSign    = simEvent.muonSign
        self.muonPt      = simEvent.muonPt
        self.muonAngle   = simEvent.muonAngle

        self.predPt         = None
        self.predSign       = None
        self.predPath       = None
        self.predTrajectory = None

        self._hasNoiseCluster = None
        self._passQualityCuts = None

    def getMuonQPt(self):
        return self.muonPt*self.muonSign

    def getPredQPt(self):
        return self.predPt*self.predSign

    def getPredSign(self):
        return self.predSign

    def getPredPt(self):
        return self.predPt

    def getMuonAngleDegrees(self):
        return 180*self.muonAngle/math.pi

    def getSeedLineZ(self, y):
        return y/math.tan(self.seedAngle)

    def getSeedLineY(self, z):
        return z*math.tan(self.seedAngle)

    def passDeltaZCuts(self):
        return abs(self.rpc1DeltaZ) < options.rpc1_max_deltaz and abs(self.rpc3DeltaZ) < options.rpc3_max_deltaz

    def getNHit(self, origin=None):
        return self.seedCluster.getNHit(origin) + self.rpc1Cluster.getNHit(origin) + self.rpc3Cluster.getNHit(origin)

    def getNLayer(self, origin=None):
        return self.seedCluster.getNLayer(origin) + self.rpc1Cluster.getNLayer(origin) + self.rpc3Cluster.getNLayer(origin)

    def getSeedZ(self):
        return self.seedCluster.getMeanZ()

    def isSeedNoise(self):
        return self.seedCluster.getNHit() > 0 and self.seedCluster.getNHit() == self.seedCluster.getNHit(Origin.NOISE)

    def isRPC1Noise(self):
        return self.rpc1Cluster.getNHit() > 0 and self.rpc1Cluster.getNHit() == self.rpc1Cluster.getNHit(Origin.NOISE)

    def isRPC3Noise(self):
        return self.rpc3Cluster.getNHit() > 0 and self.rpc3Cluster.getNHit() == self.rpc3Cluster.getNHit(Origin.NOISE)

    def hasNoiseCluster(self):
        if self._hasNoiseCluster == None:
            self._hasNoiseCluster = (self.isSeedNoise() or self.isRPC1Noise() or self.isRPC3Noise())

        return self._hasNoiseCluster

    def passQualityCuts(self):
        if self._passQualityCuts == None:
            self._passQualityCuts = (self.getNHit() >= 6 and self.getNLayer() >= 4 )

        return self._passQualityCuts

    def setModelPred(self, value):
        self.predPt   = abs(10.0/float(value))
        self.predSign = int(math.copysign(1, value))
        self.predQPt  = self.predPt*self.predSign

    def getCandVars(self):
        '''Return variables for NN training and testing, also truth information.
           Variables are normalised to produced distributions with std variation of ~1 and centred at ~0.
        '''
        return [(self.getSeedZ()-4.0)/2.0,
                self.rpc1DeltaZ/0.04,
                self.rpc3DeltaZ/0.16,
                10.0*self.muonSign/self.muonPt,
                self.muonAngle,
                self.getNHit(),
                self.getNHit(origin=Origin.NOISE),
                int(self.isSeedNoise()),
                int(self.hasNoiseCluster())]

    def makeSeedLine(self):
        '''Draw straight line through the RPC2 seed cluster centre
        '''
        zl = []
        yl = []

        for i in range(1300):
            zl += [i*0.01]
            yl += [zl[-1]*math.tan(self.seedAngle)]

        return zl, yl

    def makePredLine(self):
        '''Draw trajectory corresponding to predicted muon pT, charge and angle
        '''
        zp = []
        yp = []

        if self.predTrajectory:
            for i in range(0, 1200):
                y = i*0.01
                z = self.predTrajectory.getZatY(y)

                if z != None:
                    zp += [z]
                    yp += [y]

        return (zp, yp)

#----------------------------------------------------------------------------------------------
class SimEvent:

    def __init__(self, event_number):

        self.eventNumber  = event_number
        self.muonPt       = rand.uniform(options.min_pt,    options.max_pt)
        self.muonAngleDeg = rand.uniform(options.min_angle, options.max_angle)
        self.muonAngle    = math.pi*self.muonAngleDeg/180.0
        self.muonSign     = rand.choice([-1, 1])
        self.muonRadius   = self.muonPt*6.6
        self.muonEta      = -math.log(math.tan(self.muonAngle/2.0))
        self.hits         = []
        self.candEvents   = []

        self.singleClusters = None
        self.superClusters  = None


        self.path = Trajectory(self.muonPt, self.muonAngleDeg, self.muonSign)

    def __str__(self):
        return self.printSimEvent()

    def printSimEvent(self):
        s = 'Event #{0:6d} - pT={1: >5.1f}, angle={2: >3.0f}, R={3:.2f}, eta={4:.2f}'.format(
                    self.eventNumber,
                    self.muonPt*self.muonSign,
                    self.muonAngleDeg,
                    self.muonRadius,
                    self.muonEta)

        return s

    def printMuonPosAtY(self, y):
        zm = self.path.getZatY(y)
        zl = self.path.getLinearZatY(y)

        s = 'y = {0: >5.2f}, z_muon - z_linear = {1: >5.3f} - {2: >5.3f} = {3: >5.3f} meter'.format(y, zm, zl, zm - zl)

        return s

    def countHits(self, origin=None):
        icount = 0

        for hit in self.hits:
            if origin == None or hit.origin == origin:
                icount += 1

        return icount

#----------------------------------------------------------------------------------------------
class Cluster:
    '''Cluster - reconstructed cluster in one RPC single layer
                       or in RPC double layer
    '''

    def __init__(self, layer, hits) -> None:

        self.layer = layer
        self.hits  = hits

    def __str__(self) -> str:
        return self.printCluster()

    def printCluster(self):
        return 'Cluster {} - nhit = {:d}, mean z = {:.3f}, y = {:.3f}, noise hits = {:d}'.format(self.layer, 
                                                                                                len(self.hits), 
                                                                                                self.getMeanZ(), 
                                                                                                self.getY(), 
                                                                                                self.getNHit(Origin.NOISE))

    def getMinHitDeltaZ(self, otherHit):
        if len(self.hits) == 0:
            return None

        minHit = min(self.hits, key = lambda hit: hit.getAbsDeltaZ(otherHit))

        return minHit.getAbsDeltaZ(otherHit)

    def getMinClusterDeltaZ(self, otherCluster):
        if len(self.hits) == 0 or len(otherCluster.hits) == 0:
            return None

        result = None

        for otherHit in otherCluster.hits:
            for hit in self.hits:
                minz = hit.getAbsDeltaZ(otherHit)

                if result == None:
                    result = minz
                else:
                    result = min([minz, result])

        return result

    def getDoublet(self):
        return Layer.getDoublet(self.layer)

    def getMeanZ(self):
        if len(self.hits) == 0:
            return None

        meanz = 0

        for hit in self.hits:
            meanz += hit.getZ()

        return meanz/len(self.hits)

    def getY(self):
        doublet = self.getDoublet()

        if doublet == Layer.RPC1:
            return 0.5*(layerRPC11.y + layerRPC12.y)
        if doublet == Layer.RPC2:
            return 0.5*(layerRPC21.y + layerRPC22.y)
        if doublet == Layer.RPC3:
            return 0.5*(layerRPC31.y + layerRPC32.y)

        raise Exception('Layer.getY - unknown doublet value: {}'.format(doublet))

    def getNHit(self, origin=None):
        return sum(map(lambda x: x.origin == origin or origin == None, self.hits))

    def getNLayer(self, origin=None):
        layers = set()

        for hit in self.hits:
            if hit.origin == origin or origin == None:
                layers.add(hit.strip.layer)

        return len(layers)

#----------------------------------------------------------------------------------------------
def collectClusterHits(cluster, hits):

    icount = 0

    while icount < len(hits):
        hit = hits[icount]

        if len(cluster.hits) == 0 or cluster.getMinHitDeltaZ(hit) < 1.1*options.strip_width:
            cluster.hits.append(hit)
            hits.remove(hit)
            icount = 0
        else:
            icount += 1

    return len(cluster.hits)

#----------------------------------------------------------------------------------------------
def collectSuperClusterConstituents(superCluster, clusters):

    icount = 0

    while icount < len(clusters):
        cluster = clusters[icount]

        if len(superCluster.hits) == 0 or superCluster.getMinClusterDeltaZ(cluster) < 1.1*options.strip_width:
            superCluster.hits += cluster.hits
            clusters.remove(cluster)
            icount = 0
        else:
            icount += 1

    return len(superCluster.hits)


#----------------------------------------------------------------------------------------------
def reconstructClusters(event):

    '''reconstructClusters(event) - first reconstruct single-layer, then super-clusters
    '''

    log.debug('reconstructClusters - process {} hits'.format(len(event.hits)))

    #
    # Collect all hits in each layer and then reconstruct clusters in each layer
    #
    hitDict = defaultdict(list)

    for hit in event.hits:
        hitDict[hit.getLayer()].append(hit)

    clusterDict = defaultdict(list)

    for layer, hits in hitDict.items():

        icount = 0

        while len(hits):
            clust = Cluster(layer, hits=[])

            if collectClusterHits(clust, hits):
                clusterDict[clust.getDoublet()] += [clust]
            else:
                break

    #
    # Reconstruct super-clusters
    #
    superDict = defaultdict(list)

    for doublet, clusters in clusterDict.items():
        log.debug('{} has {:d} single-layer clusters'.format(doublet, len(clusters)))

        for cluster in clusters:
            log.debug('   cluster   {} with {:d} hits: '.format(cluster.layer, len(cluster.hits)))

            for hit in cluster.hits:
                log.debug('      {}'.format(hit))

        tmpClusters = clusters.copy()

        while len(tmpClusters):
            superCluster = Cluster(doublet, hits=[])

            if collectSuperClusterConstituents(superCluster, tmpClusters):
                superDict[superCluster.getDoublet()] += [superCluster]
            else:
                break

    log.debug('Print all super clusters')

    for doublet, clusters in superDict.items():
        log.debug('{} has {:d} super clusters'.format(doublet, len(clusters)))

        for cluster in clusters:
            log.debug('   cluster   {} with {:d} hits: '.format(cluster.layer, len(cluster.hits)))

            for hit in cluster.hits:
                log.debug('      {}'.format(hit))

    event.singleClusters = clusterDict
    event.superClusters  = superDict

#----------------------------------------------------------------------------------------------
def waitForClick(figname=None):

    plotPath = getPlotPath(figname)

    if not options.wait:
        if figname and plotPath:
            log.info('waitForClick - save figure: {}'.format(plotPath))
            plt.savefig(plotPath)    
            plt.close()
        return

    log.info('Click on figure to continue, close to exit programme...')

    while True:
        try:
            result = plt.waitforbuttonpress()

            if result == True and figname and plotPath:
                log.info('waitForClick - save figure: {}'.format(plotPath))
                plt.savefig(plotPath)
            
            plt.close()
            break
        except:
            log.info('waitForClick - exiting programme on canvas close')
            sys.exit(0)

#----------------------------------------------------------------------------------------------
def plotSimulatedHits(events):

    log.info('plotSimulatedHits - plot hit info foor {:d} events'.format(len(events)))

    hitAll = []
    hitMuon = []
    hitNearby = []
    hitNoise = []

    for event in events:
        hitAll += [event.countHits()]
        hitMuon += [event.countHits(Origin.MUON)]
        hitNearby += [event.countHits(Origin.NEARBY)]
        hitNoise += [event.countHits(Origin.NOISE)]

    bins  = [b + 0.5 for b in range(-1, 10)]
    abins = [b + 0.5 for b in range( 1, 17)]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.25, bottom=0.08, left=0.07)

    npAll    = np.array(hitAll)
    npMuon   = np.array(hitMuon)
    npNearby = np.array(hitNearby)
    npNoise  = np.array(hitNoise)

    hAll    = ax[0, 0].hist(npAll,    bins=abins, log=options.logy, )
    hMuon   = ax[0, 1].hist(npMuon,   bins=bins,  log=options.logy)
    hNearby = ax[1, 1].hist(npNearby, bins=bins,  log=options.logy)
    hNoise  = ax[1, 0].hist(npNoise,  bins=bins,  log=options.logy)

    ax[0, 0].axvline(npAll.mean(),    color='k', linestyle='dashed', linewidth=1)
    ax[0, 1].axvline(npMuon.mean(),   color='k', linestyle='dashed', linewidth=1)
    ax[1, 1].axvline(npNearby.mean(), color='k', linestyle='dashed', linewidth=1)
    ax[1, 0].axvline(npNoise.mean(),  color='k', linestyle='dashed', linewidth=1)

    ax[0, 0].set_xlabel('Number of all hits')
    ax[0, 0].set_ylabel('Simulated events')

    ax[0, 1].set_xlabel('Number of muon hits')
    ax[0, 1].set_ylabel('Simulated events')

    ax[1, 1].set_xlabel('Number of muon cluster hits')
    ax[1, 1].set_ylabel('Simulated events')

    ax[1, 0].set_xlabel('Number of noise hits')
    ax[1, 0].set_ylabel('Simulated events')

    log.info('Mean all    hits: {:.3f}'.format(npAll.mean()))
    log.info('Mean muon   hits: {:.3f}'.format(npMuon.mean()))
    log.info('Mean nearby hits: {:.3f}'.format(npNearby.mean()))
    log.info('Mean noise  hits: {:.3f}'.format(npNoise.mean()))

    log.info('All    bins: {}'.format(hAll[0]))
    log.info('Muon   bins: {}'.format(hMuon[0]))
    log.info('Nearby bins: {}'.format(hNearby[0]))

    fig.show()

    plotPath = getPlotPath('hits.png')

    if plotPath:
        plt.savefig(plotPath)

    waitForClick()

#----------------------------------------------------------------------------------------------
def plotRecoClusters(events):

    log.info('plotSimulatedHits - plot hit info for {:d} events'.format(len(events)))

    superc = []
    single = []

    hitSuperc = []
    hitSingle = []

    for event in events:
        countSuperc = 0
        countSingle = 0

        countSupercHits = 0
        countSingleHits = 0

        for clusters in itertools.chain(event.superClusters.values()):
            countSuperc += len(clusters)
            for cluster in clusters:
                countSupercHits += len(cluster.hits)

        for clusters in itertools.chain(event.singleClusters.values()):
            countSingle += len(clusters)

            for cluster in clusters:
                countSingleHits += len(cluster.hits)

        single += [countSingle]
        superc += [countSuperc]

        hitSingle += [countSingleHits]
        hitSuperc += [countSupercHits]

    cbins = [b + 0.5 for b in range(1, 14)]
    hbins = [b + 0.5 for b in range(2, 17)]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.25, bottom=0.08, left=0.07)

    npSuper     = np.array(superc)
    npSingle    = np.array(single)
    npSuperHit  = np.array(hitSuperc)
    npSingleHit = np.array(hitSingle)

    hSuperc    = ax[0, 0].hist(npSuper,     bins=cbins, log=options.logy, )
    hSingle    = ax[0, 1].hist(npSingle,    bins=cbins, log=options.logy)
    hSuperHit  = ax[1, 1].hist(npSuperHit,  bins=hbins, log=options.logy)
    hSingleHit = ax[1, 0].hist(npSingleHit, bins=hbins, log=options.logy)

    ax[0, 0].axvline(npSuper.mean(),     color='k', linestyle='dashed', linewidth=1)
    ax[0, 1].axvline(npSingle.mean(),    color='k', linestyle='dashed', linewidth=1)
    ax[1, 1].axvline(npSuperHit.mean(),  color='k', linestyle='dashed', linewidth=1)
    ax[1, 0].axvline(npSingleHit.mean(), color='k', linestyle='dashed', linewidth=1)

    ax[0, 0].set_xlabel('Number of super clusters')
    ax[0, 0].set_ylabel('Simulated events')

    ax[0, 1].set_xlabel('Number of single-layer clusters')
    ax[0, 1].set_ylabel('Simulated events')

    ax[1, 1].set_xlabel('Number of super cluster hits')
    ax[1, 1].set_ylabel('Simulated events')

    ax[1, 0].set_xlabel('Mumber of single-layer cluster hits')
    ax[1, 0].set_ylabel('Simulated events')

    log.info('Mean number of super        clusters: {:.3f}'.format(npSuper.mean()))
    log.info('Mean number of single-layer clusters: {:.3f}'.format(npSingle.mean()))

    log.info('Mean number of super        cluster hits: {:.4f}'.format(npSuperHit.mean()))
    log.info('Mean number of single-layer cluster hits: {:.4f}'.format(npSingleHit.mean()))


    log.info('Super  cluster bins: {}'.format(hSuperc[0]))
    log.info('Single cluster bins: {}'.format(hSingle[0]))

    fig.show()

    plotPath = getPlotPath('clusters.png')

    if plotPath:
        plt.savefig(plotPath)

    waitForClick()

#----------------------------------------------------------------------------------------------
def getMU20Eff():

    bins = [2, 3,    4,    5,    6,    7,    8,  10,   12,   14,   16,   18,   20,   25,   30,   35,   40,   50,   60,   70,   80, 85]
    effs = [0.54, 0.68, 1.04, 1.31, 1.66, 2.28, 3.7, 10.1, 26.8, 46.5, 59.4, 65.7, 69.0, 70.3, 70.4, 70.1, 69.9, 69.8, 69.9, 69.9, 70.1]
    cens = []

    for i in range(len(effs)):
        cens += [0.5*(bins[i] + bins[i+1])]

        log.debug('Bin {: >2d} - {: >2d} eff = {}'.format(bins[i], bins[i+1], effs[i]))

    return (cens, effs)

#----------------------------------------------------------------------------------------------
def prepEffPlot(denMuonQPt, numDictPt, plot=False, scaleToATLAS=True):

    ebins = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 30]
    ebins = [b for b in range(3, 31)]

    resultBins = None
    resultEffs = None

    for mev in sorted(numDictPt.keys(), reverse=True):
        numMuonQPt = numDictPt[mev]

        aden = np.abs(np.array(denMuonQPt))
        anum = np.abs(np.array(numMuonQPt))

        denCounts, denBins = np.histogram(aden, ebins)
        numCounts, numBins = np.histogram(anum, ebins)

        leff = []
        beff = []
        eff20 = None # efficiency at first pT>20 GeV bin
        eff25 = None # efficiency at first pT>25 GeV bin

        for i in range(len(denCounts)):
            den = denCounts[i]
            num = numCounts[i]

            beff += [0.5*(ebins[i]+ebins[i+1])]

            if den > 0.0:
                leff += [100.0*num/den]
            else:
                leff += [0.0]

            if eff20 == None and beff[-1] > 20.1:
                eff20 = leff[-1]
                log.debug('prepEffPlot - eff20 efficiency = {:d}/{:d} = {:.4f} for pT > {:d} MeV, beff[-1] = {}'.format(num, den, leff[-1], mev, beff[-1]))

            if eff25 == None and beff[-1] > 25.1:
                eff25 = leff[-1]
                log.debug('prepEffPlot - eff25 efficiency = {:d}/{:d} = {:.4f} for pT > {:d} MeV, beff[-1] = {}'.format(num, den, leff[-1], mev, beff[-1]))

        if eff20 != None and eff25 != None and eff20 > 0.95*eff25 and resultEffs == None:
            log.info('prepEffPlot - record efficiency at 20 GeV = {:.4f} for pT > {:d} MeV'.format(eff20, mev))
            resultBins = beff
            resultEffs = leff

        verbose('ebins: ', ebins)
        verbose('denCounts:', denCounts)
        verbose('denBins:', denBins, type(denBins))

        verbose('numCounts:', numCounts)
        verbose('numBins:', numBins, type(numBins))

        verbose('beff', len(beff), beff)
        verbose('leff', len(leff), leff)

        verbose('Length denCounts={}'.format(len(denCounts)))
        verbose('Length numCounts={}'.format(len(numCounts)))

        verbose('Length denBins={}'.format(len(denBins)))
        verbose('Length numBins={}'.format(len(numBins)))

        verbose('Length leff ={}'.format(len(leff)))
        verbose('Length ebins={}'.format(len(ebins)))

        if plot:
            log.info('Plot efficiency for mev = {}'.format(mev))

            fig, ax = plt.subplots(3, figsize=(6, 10))

            plt.subplots_adjust(hspace=0.28, bottom=0.08, left=0.08, top=0.97, right=0.97)

            ax[0].hist(aden, bins=ebins, histtype='stepfilled')
            ax[1].hist(anum, bins=ebins, histtype='stepfilled')

            ax[2].plot(beff, leff, linewidth=2)

            waitForClick()

    if resultBins == None or resultEffs == None:
        log.info('PrepEff - failed to find 95% efficiency at 20 GeV')
        return (None, None, None)

    if scaleToATLAS:
        plateau = []

        for i, bin in enumerate(resultBins):
            if bin > 25.0:
                plateau += [resultEffs[i]]

        scale = 70.0/stat.mean(plateau)

        resultEffs = [eff*scale for eff in resultEffs]

        log.info('prepEff - scaled efficiency by {}'.format(scale))
    else:
        scale = None

    return resultBins, resultEffs, scale

#----------------------------------------------------------------------------------------------
def plotModelResults(events):

    if options.torch_model == None:
        return

    log.info('plotModelResults - plot NN model results for {:d} simulated events'.format(len(events)))

    muonQPt = []
    predQPt = []

    realMuonQPt = []
    realPredQPt = []

    noiseMuonQPt = []
    noisePredQPt = []

    passCandQPt = defaultdict(list)
    passRealQPt = defaultdict(list)
    passBestQPt = defaultdict(list)

    for event in events:
        for cand in event.candEvents:
            muonQPt += [cand.getMuonQPt()]
            predQPt += [cand.getPredQPt()]

            if cand.hasNoiseCluster():
                noiseMuonQPt += [cand.getMuonQPt()]
                noisePredQPt += [cand.getPredQPt()]
            else:
                realMuonQPt += [cand.getMuonQPt()]
                realPredQPt += [cand.getPredQPt()]

            #
            # Fill numerators and denominators for efficiency plots
            #
            for mev in range(15000, 20000, 100):
                if cand.predPt > 0.001*mev:
                    passCandQPt[mev] += [cand.getMuonQPt()]

                    if not cand.hasNoiseCluster():
                        passRealQPt[mev] += [cand.getMuonQPt()]

                    if cand.passQualityCuts():
                        passBestQPt[mev] += [cand.getMuonQPt()]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    plt.subplots_adjust(hspace=0.30, bottom=0.08, left=0.08, top=0.97, right=0.97)

    amAll = np.array(muonQPt)
    apAll = np.array(predQPt)

    amReal = np.array(realMuonQPt)
    apReal = np.array(realPredQPt)

    amNoise = np.array(noiseMuonQPt)
    apNoise = np.array(noisePredQPt)

    dpAll   = (amAll   - apAll)  /np.abs(amAll)
    dpReal  = (amReal  - apReal) /np.abs(amReal)
    dpNoise = (amNoise - apNoise)/np.abs(amNoise)

    dbins = [b*0.008 for b in range(-100, 100)]

    muonColor  = 'royalblue'
    noiseColor = 'yellowgreen'
    bestColor  = 'magenta'
    atlasColor = 'tab:orange'

    ax[0, 0].scatter(amReal,  apReal,  s=1, label=r'Pure $\mu$', c=muonColor)
    ax[0, 0].scatter(amNoise, apNoise, s=1, label=r'Noise $\mu$', c=noiseColor)

    ax[0, 1].scatter(np.abs(amReal),  dpReal,  s=1, c=muonColor)
    ax[0, 1].scatter(np.abs(amNoise), dpNoise, s=1, c=noiseColor)

    ax[1, 0].hist(dpReal,  bins=dbins, log=options.logy, label=r'Pure $\mu$', color=muonColor)
    ax[1, 0].hist(dpNoise, bins=dbins, log=options.logy, label=r'Noise $\mu$', color=noiseColor)

    effRealBins, effReal, scaleReal = prepEffPlot(muonQPt, passRealQPt)
    effCandBins, effCand, scaleCand = prepEffPlot(muonQPt, passCandQPt)
    effBestBins, effBest, scaleBest = prepEffPlot(muonQPt, passBestQPt, plot=False)

    effMU20Bins, effMU20 = getMU20Eff()

    if effRealBins and effReal and effCandBins and effCand:
        ax[1, 1].plot(effRealBins, effReal, label=r'Pure $\mu \times {:.3f}$'.format(scaleReal), color=muonColor)
        ax[1, 1].plot(effCandBins, effCand, label=r'Noise $\mu \times {:.3f}$'.format(scaleCand), color=noiseColor)
        ax[1, 1].plot(effBestBins, effBest, label=r'Pass $\mu \times {:.3f}$'.format(scaleBest), color=bestColor)
        ax[1, 1].plot(effMU20Bins, effMU20, label='ATLAS MU20', color=atlasColor)

    ax[0, 0].set_xlabel(r'$q \cdot p_{\mathrm{T}}^{\mathrm{sim.}}$ [GeV]',  fontsize=14)
    ax[0, 0].set_ylabel(r'$q \cdot p_{\mathrm{T}}^{\mathrm{pred.}}$ [GeV]', fontsize=14)

    ax[0, 1].set_xlabel(r'$p_{\mathrm{T}}^{\mathrm{sim.}}$ [GeV]',  fontsize=14)
    ax[0, 1].set_ylabel(r'($q \cdot p_{\mathrm{T}}^{\mathrm{sim.}} - q \cdot p_{\mathrm{T}}^{\mathrm{pred.}})/p_{\mathrm{T}}^{\mathrm{sim.}}$', fontsize=14)

    ax[1, 0].set_ylabel('Candidate events',  fontsize=14)
    ax[1, 0].set_xlabel(r'($q \cdot p_{\mathrm{T}}^{\mathrm{sim.}} - q \cdot p_{\mathrm{T}}^{\mathrm{pred.}})/p_{\mathrm{T}}^{\mathrm{sim.}}$', fontsize=14)
    ax[1, 0].legend(loc='best', prop={'size': 12}, frameon=False)

    ax[1, 1].set_ylabel('Efficiency [%]',  fontsize=14)
    ax[1, 1].set_xlabel(r'$p_{\mathrm{T}}^{\mathrm{sim.}}$ [GeV]', fontsize=14)
    ax[1, 1].legend(loc='best', prop={'size': 12}, frameon=False)
    ax[1, 1].xaxis.grid(True)
    ax[1, 1].yaxis.grid(True)

    limPt = 30.0
    limdp = 0.7

    ax[0, 0].set_xlim(-limPt, limPt)
    ax[0, 0].set_ylim(-limPt, limPt)
    ax[0, 1].set_ylim(-limdp, limdp)
    ax[1, 1].set_xlim(   3.0, limPt)

    fig.show()

    plotPath = getPlotPath('results.png')

    if plotPath:
        plt.savefig(plotPath)

    waitForClick()

    if effRealBins and effReal and effCandBins and effCand:
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(hspace=0.30, bottom=0.08, left=0.08, top=0.97, right=0.97)

        ax.plot(effRealBins, effReal, label=r'Pure $\mu \times {:.2f}$'.format(scaleReal), color=muonColor)
        ax.plot(effCandBins, effCand, label=r'Noise $\mu \times {:.2f}$'.format(scaleCand), color=noiseColor)
        ax.plot(effBestBins, effBest, label=r'Pass $\mu \times {:.2f}$'.format(scaleBest), color=bestColor)
        ax.plot(effMU20Bins, effMU20, label='ATLAS MU20', color=atlasColor)

        ax.set_ylabel('Efficiency [%]',  fontsize=14)
        ax.set_xlabel(r'$p_{\mathrm{T}}^{\mathrm{sim.}}$ [GeV]', fontsize=14)
        ax.legend(loc='best', prop={'size': 12}, frameon=False)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)

        plt.xlim(3.0, 15.0)
        plt.ylim(0.0, 15.0)

        plotPath = getPlotPath('results_effzoom.png')

        if plotPath:
            plt.savefig(plotPath)

        waitForClick()

#----------------------------------------------------------------------------------------------
def plotLineDifferences(events):

    log.info('plotLineDifferences - plot differences with respect to straight seed line for {:d} simulated events'.format(len(events)))

    muonPtDict  = collections.defaultdict(list)
    deltaz1Dict = collections.defaultdict(list)
    deltaz3Dict = collections.defaultdict(list)

    for event in events:
        for cand in event.candEvents:

            if cand.getMuonAngleDegrees() > 70:
                angleGroup = 1
            elif cand.getMuonAngleDegrees() > 50:
                angleGroup = 2
            else:
                angleGroup = 3

            if cand.hasNoiseCluster():
                angleGroup *= 100

            deltaz1Dict[angleGroup] += [cand.rpc1DeltaZ]
            deltaz3Dict[angleGroup] += [cand.rpc3DeltaZ]
            muonPtDict[angleGroup]  += [cand.muonPt*cand.muonSign]

            #log.info('Angle group={}, muon={}'.format(angleGroup, event.printSimEvent()))

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.28, bottom=0.08, left=0.10, top=0.97, right=0.97)

    labelDict = {1:r'$85^{\circ} < \angle_{\mathrm{sim.}~\mu} < 70^{\circ}$',
                 2:r'$70^{\circ} < \angle_{\mathrm{sim.}~\mu} < 50^{\circ}$',
                 3:r'$50^{\circ} < \angle_{\mathrm{sim.}~\mu} < 40^{\circ}$',
                 100:r'$85^{\circ} < \angle_{\mathrm{sim.}~\mu} < 70^{\circ}$',
                 200:r'$70^{\circ} < \angle_{\mathrm{sim.}~\mu} < 50^{\circ}$',
                 300:r'$50 ^{\circ}< \angle_{\mathrm{sim.}~\mu} < 40^{\circ}$',
                }

    for i in [1, 2, 3]:
        log.info('Angle group = {} contains {} events label={}'.format(i, len(muonPtDict[i]), labelDict[i]))

        ax[0, 0].scatter(muonPtDict[i], deltaz1Dict[i], marker='.', s=10, label=labelDict[i])
        ax[0, 1].scatter(muonPtDict[i], deltaz3Dict[i], marker='.', s=10, label=labelDict[i])

    ax[0, 0].set_xlabel(r'Pure muon $q\times p_{\mathrm{T}}$', fontsize=14)
    ax[0, 0].set_ylabel('RPC1 $z_{\mathrm{line}} - z_{\mathrm{cluster}}$ [m]', fontsize=14)
    ax[0, 0].legend(loc='best', prop={'size': 12}, frameon=False)

    ax[0, 1].set_xlabel(r'Pure muon $q\times p_{\mathrm{T}}$', fontsize=14)
    ax[0, 1].set_ylabel('RPC3 $z_{\mathrm{line}} - z_{\mathrm{cluster}}$ [m]', fontsize=14)
    ax[0, 1].legend(loc='best', prop={'size': 12}, frameon=False)

    for i in [100, 200, 300]:
        log.info('Angle group = {} contains {} events label={}'.format(i, len(muonPtDict[i]), labelDict[i]))

        ax[1, 0].scatter(muonPtDict[i], deltaz1Dict[i], marker='.', s=10)
        ax[1, 1].scatter(muonPtDict[i], deltaz3Dict[i], marker='.', s=10)

    ax[1, 0].set_xlabel(r'Muon with noise $q\times p_{\mathrm{T}}$', fontsize=14)
    ax[1, 0].set_ylabel('RPC1 $z_{\mathrm{line}} - z_{\mathrm{cluster}}$ [m]', fontsize=14)
    ax[1, 0].legend(loc='best', prop={'size': 12}, frameon=False)

    ax[1, 1].set_xlabel(r'Muon with noise $q\times p_{\mathrm{T}}$', fontsize=14)
    ax[1, 1].set_ylabel('RPC3 $z_{\mathrm{line}} - z_{\mathrm{cluster}}$ [m]', fontsize=14)
    ax[1, 1].legend(loc='best', prop={'size': 12}, frameon=False)

    fig.show()

    plotPath = getPlotPath('differences.png')

    if plotPath:
        plt.savefig(plotPath)

    waitForClick()

#----------------------------------------------------------------------------------------------
def plotCandEvents(events):

    log.info('plotCandEvents - plot candidates for {:d} simulated events'.format(len(events)))

    ncand = []
    ncandNoise = []

    deltaz1    = []
    deltaz1Pos = []
    deltaz1Neg = []
    deltaz1Noise = []

    deltaz3    = []
    deltaz3Pos = []
    deltaz3Neg = []
    deltaz3Noise = []

    seedz = []

    for event in events:
        ncand += [len(event.candEvents)]
        ncandNoise += [len([x for x in event.candEvents if x.getNHit(Origin.NOISE) > 0])]

        for cand in event.candEvents:
            seedz   += [cand.seedCluster.getMeanZ()]
            deltaz1 += [cand.rpc1DeltaZ]
            deltaz3 += [cand.rpc3DeltaZ]

            if cand.hasNoiseCluster():
                deltaz1Noise += [cand.rpc1DeltaZ]
                deltaz3Noise += [cand.rpc3DeltaZ]
            else:
                if cand.muonSign > 0:
                    deltaz1Pos += [cand.rpc1DeltaZ]
                    deltaz3Pos += [cand.rpc3DeltaZ]
                else:
                    deltaz1Neg += [cand.rpc1DeltaZ]
                    deltaz3Neg += [cand.rpc3DeltaZ]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.28, bottom=0.08, left=0.08, top=0.97, right=0.97)

    cbins  = [b+0.5   for b in range(-1,   4)]
    zbins1 = [b*0.004 for b in range(-50, 50)]
    zbins3 = [b*0.014 for b in range(-50, 50)]
    sbins  = [b*0.25  for b in range(  0, 40)]

    ax[0, 0].hist(ncand, log=options.logy, bins=cbins, label=r'All $\mu$')
    ax[0, 1].hist(seedz, log=options.logy, bins=sbins, label=r'All $\mu$')

    ax[1, 0].hist(deltaz1Neg,   log=options.logy, bins=zbins1, color='blue', label=r'Pure $+\mu$')
    ax[1, 0].hist(deltaz1Pos,   log=options.logy, bins=zbins1, color='red',  label=r'Pure $-\mu$')
    ax[1, 0].hist(deltaz1Noise, log=options.logy, bins=zbins1, color='yellowgreen', label=r'Noise $\mu$', histtype='step', linewidth=2)

    ax[1, 1].hist(deltaz3Neg,   log=options.logy, bins=zbins3, color='blue', label=r'Pure $+\mu$')
    ax[1, 1].hist(deltaz3Pos,   log=options.logy, bins=zbins3, color='red',  label=r'Pure $-\mu$')
    ax[1, 1].hist(deltaz3Noise, log=options.logy, bins=zbins3, color='yellowgreen', label=r'Noise $\mu$', histtype='step', linewidth=2)

    ax[0, 0].axvline(stat.mean(ncand), color='k', linestyle='dashed', linewidth=1)
    ax[0, 1].axvline(stat.mean(seedz), color='k', linestyle='dashed', linewidth=1)

    ax[0, 0].set_xlabel('Number of muon candidates', fontsize=14)
    ax[0, 0].set_ylabel('Number of simulated events', fontsize=14)
    ax[0, 0].legend(loc='best', prop={'size': 12}, frameon=False)

    ax[0, 1].set_xlabel(r'RPC2 seed $z_{\mathrm{cluster}}$ [m]', fontsize=14)
    ax[0, 1].set_ylabel('Muon candidates', fontsize=14)
    ax[0, 1].legend(loc='best', prop={'size': 12}, frameon=False)

    ax[1, 1].set_xlabel(r'RPC3 $z_{\mathrm{line}} - z_{\mathrm{cluster}}$ [m]', fontsize=14)
    ax[1, 1].set_ylabel('Muon candidates', fontsize=14)
    ax[1, 1].legend(loc='best', prop={'size': 12}, frameon=False)

    ax[1, 0].set_xlabel(r'RPC1 $z_{\mathrm{line}} - z_{\mathrm{cluster}}$ [m]', fontsize=14)
    ax[1, 0].set_ylabel('Muon candidates', fontsize=14)
    ax[1, 0].legend(loc='best', prop={'size': 12}, frameon=False)

    log.info('Mean ncand:   {:.4f}'.format(stat.mean(ncand)))
    log.info('Mean seed z:  {:.4f}'.format(stat.mean(seedz)))
    log.info('Mean rpc1 dz: {:.4f}'.format(stat.mean(deltaz1)))
    log.info('Mean rpc3 dz: {:.4f}'.format(stat.mean(deltaz3)))

    log.info('Std seed z:  {:.4f}'.format(stat.stdev(seedz)))
    log.info('Std rpc1 dz: {:.4f}'.format(stat.stdev(deltaz1)))
    log.info('Std rpc3 dz: {:.4f}'.format(stat.stdev(deltaz3)))

    fig.show()

    plotPath = getPlotPath('candidates.png')

    if plotPath:
        plt.savefig(plotPath)

    waitForClick()

#----------------------------------------------------------------------------------------------
def plotCandQuality(events):

    log.info('plotCandQuality - plot candidates for {:d} simulated events'.format(len(events)))

    rmHits = []
    nmHits = []

    rmLayers = []
    nmLayers = []

    for event in events:
        for cand in event.candEvents:

            if cand.hasNoiseCluster():
                nmHits += [cand.getNHit()]
                nmLayers += [cand.getNLayer()]
            else:
                rmHits += [cand.getNHit()]
                rmLayers += [cand.getNLayer()]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    plt.subplots_adjust(hspace=0.28, bottom=0.12, left=0.08, top=0.97, right=0.97)

    hbins = [b+0.5 for b in range(1,  10)]
    lbins = [b+0.5 for b in range(1,   7)]

    ax[0].hist(rmHits, log=options.logy, bins=hbins, label=r'Pure $\mu$')
    ax[0].hist(nmHits, log=options.logy, bins=hbins, label=r'Noise $\mu$')

    ax[1].hist(rmLayers, log=options.logy, bins=lbins, label=r'Pure $\mu$')
    ax[1].hist(nmLayers, log=options.logy, bins=lbins, label=r'Noise $\mu$')

    ax[0].set_ylabel('Number of muon candidates', fontsize=14)
    ax[0].set_xlabel('Number of reconstructed hits', fontsize=14)
    ax[0].legend(loc='best', prop={'size': 12}, frameon=False)

    ax[1].set_ylabel('Number of muon candidates', fontsize=14)
    ax[1].set_xlabel('Number of layers', fontsize=14)
    ax[1].legend(loc='best', prop={'size': 12}, frameon=False)

    fig.show()

    plotPath = getPlotPath('candidates_quality.png')

    if plotPath:
        plt.savefig(plotPath)

    waitForClick()

#----------------------------------------------------------------------------------------------
def makeCandidates(event):
    '''makeCandidates - make muon candidates
    '''

    seeds = event.superClusters[Layer.RPC2]
    rpc1s = event.superClusters[Layer.RPC1]
    rpc3s = event.superClusters[Layer.RPC3]

    if len(seeds) == 0:
        log.debug('makeCandidates - zero seed clusters in RPC2, make no candidates for this event')
        return []

    if len(rpc1s) == 0 or len(rpc3s) == 0:
        log.debug('makeCandidates - zero seed clusters in RPC1 and/or RPC3, make no candidates for this event')
        return []

    for seedCluster in seeds:
        cand = CandEvent(event, seedCluster)

        cand.rpc1Cluster = min(rpc1s, key=lambda x: abs(cand.getSeedLineZ(x.getY()) - x.getMeanZ()))
        cand.rpc3Cluster = min(rpc3s, key=lambda x: abs(cand.getSeedLineZ(x.getY()) - x.getMeanZ()))

        cand.rpc1DeltaZ = cand.getSeedLineZ(cand.rpc1Cluster.getY()) - cand.rpc1Cluster.getMeanZ()
        cand.rpc3DeltaZ = cand.getSeedLineZ(cand.rpc3Cluster.getY()) - cand.rpc3Cluster.getMeanZ()

        log.debug('CandEvent seed {}'.format(seedCluster))
        log.debug('               {}, delta z = {:.3f}'.format(cand.rpc1Cluster, cand.rpc1DeltaZ))
        log.debug('               {}, delta z = {:.3f}'.format(cand.rpc3Cluster, cand.rpc3DeltaZ))

        if options.veto_noise_cand and cand.getNHit(Origin.NOISE):
            continue

        if cand.passDeltaZCuts():
            event.candEvents += [cand]

#----------------------------------------------------------------------------------------------
def simulateHits(event):
    '''simulateHits - simulate muon hits and noise hits
    '''

    event.hits = []

    for layer in allLayers:
        log.debug('-----------------------------------------------------------------')

        #
        # Make noise hits
        #
        for istrip in range(0, layer.nstrip+1):
            if rand.random() < options.noise_prob:
                event.hits += [RawHit(layer.getStrip(istrip), Origin.NOISE)]

        #
        # Make muon hits
        #
        muonZ = event.path.getZatY(layer.y)
        strip = layer.getClosestStrip(muonZ)

        if strip == None:
            log.debug('Muon position {:.3f} outside active layer {}'.format(muonZ, layer.layer))
            continue

        log.debug('Muon position {:.3f} at {}, y={}, strip = {}'.format(muonZ, strip.layer, layer.y, strip.number))

        if rand.random() > options.rpc_eff:
            log.debug('This strip is inefficient - continue')
            continue

        log.debug('   created muon hit')

        event.hits += [RawHit(strip, Origin.MUON, muonZ)]

        stripNearby = layer.getSecondClosestStrip(muonZ)

        if not stripNearby:
            continue

        probNearby = 2*options.cluster_prob_2hit*abs(strip.zcenter - muonZ)/(0.5*options.strip_width)

        log.debug('   stripNearby = {}, probNearby = {:.4f}, delta z = {}'.format(stripNearby.number, probNearby, abs(strip.zcenter - muonZ)))

        if rand.random() < probNearby:
            #
            # Make another muon hit in nearby strip
            #
            event.hits += [RawHit(stripNearby, Origin.NEARBY, muonZ=None)]
            log.debug('   created nearby hit')

    log.debug('simulateHits - created {:d} hits'.format(len(event.hits)))

#----------------------------------------------------------------------------------------------
def drawEvent(event, candEvent=None):

    '''DrawEvent - draw event display
    '''

    miny =  6.5
    maxy = 10.1
    yoff = 0.011
    labelSize = 15

    zatminy = min([event.path.getZatY(miny), event.path.getLinearZatY(miny)])
    zatmaxy = max([event.path.getZatY(maxy), event.path.getLinearZatY(maxy)])

    minz = max([0.0, min([zatmaxy, zatminy]) - 0.1])
    maxz = max([zatmaxy, zatminy]) + 0.1

    eventName = 'event_{:05d}'.format(event.eventNumber)

    log.info('drawEvent - {} (miny, maxy) = ({:.2f}, {:.2f}), minz, maxz = ({:.2f}, {:.2f})'.format(eventName, miny, maxy, minz, maxz))

    log.info('   RPC1 muon {}'.format(event.printMuonPosAtY(layerRPC11.y)))
    log.info('   RPC2 muon {}'.format(event.printMuonPosAtY(layerRPC21.y)))
    log.info('   RPC3 muon {}'.format(event.printMuonPosAtY(layerRPC31.y)))

    zarr = []
    yarr = []
    zlin = []
    ylin = []

    hitNoiseCounts = collections.Counter()

    for hit in event.hits:
        if hit.origin == Origin.NOISE and minz < hit.strip.zcenter and hit.strip.zcenter < maxz:
            hitNoiseCounts[Layer.getDoublet(hit.strip.layer)] += 1

    if options.has_rpc2_noise:
        if hitNoiseCounts[Layer.RPC2] == 0:
            return

    #
    # Draw RPC layers
    #
    fig, ax = plt.subplots(figsize=(15, 11))
    plt.subplots_adjust(hspace=0.0, bottom=0.06, left=0.06, top=0.97, right=0.99)

    fig.canvas.set_window_title(eventName) 
    ax.set_title('RPC toy simulation')

    layerRPC12.plotLayer(ax, yoffset=+yoff)
    layerRPC11.plotLayer(ax, yoffset=-yoff, showStrips=True, stripOffset=0.01)

    layerRPC22.plotLayer(ax, yoffset=+yoff)
    layerRPC21.plotLayer(ax, yoffset=-yoff, showStrips=True, stripOffset=0.01)

    layerRPC32.plotLayer(ax, yoffset=+yoff)
    layerRPC31.plotLayer(ax, yoffset=-yoff, showStrips=True, stripOffset=0.01)

    ax.annotate('RPC1 doublet', (minz + 0.007*(maxz-minz), layerRPC12.y + 0.04), fontsize=labelSize)
    ax.annotate('RPC2 doublet', (minz + 0.007*(maxz-minz), layerRPC22.y + 0.04), fontsize=labelSize)
    ax.annotate('RPC3 doublet', (minz + 0.007*(maxz-minz), layerRPC32.y + 0.04), fontsize=labelSize)

    #
    # Draw event
    #
    for i in range(0, 1200):
        y = i*0.01
        zm = event.path.getZatY(y)
        zl = event.path.getLinearZatY(y)

        if zm != None:
            zarr += [zm]
            yarr += [y]

            if options.verbose:
                log.debug('y={0: >5.2f}, zm - zlin = {1} - {2} = {3}'.format(y, zm, zl, zm-zl))

        zlin += [zl]
        ylin += [y]

    hitCounts = collections.Counter()

    for hit in event.hits:
        marker = 'v'
        label = ''
        size = 40

        if Layer.isFirstDoubletLayer(hit.strip.layer):
            marker = '^'
            hoff = 0
        else:
            hoff = yoff

        hitCounts[hit.origin] += 1

        if hit.origin == Origin.MUON:
            color = 'red'
            if hitCounts[hit.origin] == 1:
                label = 'Muon hit'
        elif hit.origin == Origin.NEARBY:
            color = 'blue'
            if hitCounts[hit.origin] == 1:
                label = 'Muon cluster hit'
        elif hit.origin == Origin.NOISE:
            marker = '.'
            color = 'orange'
            size = 160
            if hitCounts[hit.origin] == 1:
                label='Noise hit'

        ax.scatter([hit.getZ()], [hit.getY()+hoff], color=color, marker=marker, label=label, s=size)

    ax.plot(zarr, yarr, label=r'Sim. muon $q \times p_{\mathrm{T}}$' + r' = {0: >4.1f} GeV'.format(event.muonPt*event.muonSign), color='royalblue', linewidth=2)

    if options.debug or options.draw_line:
        ax.plot(zlin, ylin, linewidth=1, linestyle='--', color='orange', label='Sim. muon direction')

    ax.set_xlabel('Beam axis z [m]',  fontsize=labelSize)
    ax.set_ylabel('Radial y [m]',  fontsize=labelSize)

    if candEvent:
        if getattr(candEvent, 'predTrajectory', None):
            zpred, ypred = candEvent.makePredLine()
            ax.plot(zpred, ypred, linewidth=1, linestyle='-', color='red', label=r'Pred. muon $q \times p_{\mathrm{T}}$' + r' = {0: >4.1f} GeV'.format(candEvent.predPt*candEvent.predSign))

            zseed, yseed = candEvent.makeSeedLine()
            ax.plot(zseed, yseed, linewidth=1, linestyle='--', color='green', label='RPC2 seed cluster line')

    plt.xlim(minz, maxz)
    plt.ylim(miny, maxy)

    plt.legend(loc='center right', prop={'size': 14}, frameon=False)

    fig.show()

    waitForClick(eventName)

#----------------------------------------------------------------------------------------------
def prepEvents():

    startTime = time.time()

    if options.in_pickle:
        #
        # First try to read pickled events from file
        #
        log.info('Will read pickle data from: {}'.format(options.in_pickle))

        with open(options.in_pickle, 'rb') as f:
            events = pickle.load(f)

        log.info('prepEvents - read {} pickled events in {:4.1}s'.format(len(events), time.time() - startTime))
    else:
        #
        # Simulate events
        #
        events = []

        for i in range(options.nevent):

            event = SimEvent(i)
            events += [event]

            if i % 100 == 0 or options.debug or options.draw:
                log.info('-------------------------------------------------------------------------------------')
                log.info(event)
                log.debug(event.path)

            simulateHits(event)

            reconstructClusters(event)

            makeCandidates(event)

    #
    # Load and evaluate torch model
    #
    if options.torch_model:
        log.info('Load torch model from: {}'.format(options.torch_model))

        net = Net()
        net.load_state_dict(torch.load(options.torch_model))
        net.eval()
    else:
        net = None

    if options.nevent and len(events) > options.nevent:
        events = events[:options.nevent]

    for event in events:
        if net:
            for cand in event.candEvents:
                ''' Compute predicted candidate muon pT and charge using previously trained NN 
                '''
                data = np.array([cand.getCandVars()[0:3]], np.float32)
                dataTr = torch.from_numpy(data)
                pred = net(dataTr).detach().numpy()

                cand.setModelPred(pred[0][0])

                '''Use predicted pT and charge, and the angle of the line passing through RPC1 cluster
                '''
                angleRPC1 = 180*math.atan(cand.rpc1Cluster.getY()/cand.rpc1Cluster.getMeanZ())/math.pi

                cand.predTrajectory = Trajectory(cand.getPredPt(), angleRPC1, cand.getPredSign())

                if options.debug:
                    log.info('-------------------------------------------------------------------')
                    log.info('Muon true q*pT = {:.1f}, predicted q*pT = {:.1f}'.format(cand.muonPt*cand.muonSign, cand.getPredQPt()))


                    ''' Solved for predicted muon trajectory passing through seed cl
                    '''
                    predPath = PredPath(cand.seedCluster.getMeanZ(), cand.seedCluster.getY(), cand.getPredQPt())
                    log.info(predPath)

                    initAngle = math.atan(cand.seedCluster.getY()/cand.seedCluster.getMeanZ())

                    initTrajectory = Trajectory(cand.getPredPt(), initAngle, cand.getPredSign())

                    initValues = (cand.seedCluster.getMeanZ(), initTrajectory.z0, initTrajectory.y0)

                    result = predPath.solveTrajectory(initValues)

                    cand.predPath = predPath

                    log.info('Muon true zr = {:.4f}, z0 = {:.4f}, y0 = {:.4f}'.format(event.path.getZatY(options.rpc_radius), event.path.z0, event.path.y0))
                    log.info('Initial   zr = {:.4f}, z0 = {:.4f}, y0 = {:.4f}'.format(initValues[0], initValues[1], initValues[2]))
                    log.info('Solution  zr = {:.4f}, z0 = {:.4f}, y0 = {:.4f}'.format(result.x[0],   result.x[1],   result.x[2]))
                    log.info('Muon angle true = {:.4f}, predicted = {:.4f}'.format(event.muonAngle, predPath.getPredAngle()))

                    log.info(dir(result))
                    log.info(result)


        if options.draw:
            drawEvent(event, cand)

    log.info('prepEvents - processed {} events in {:.1f}s'.format(len(events), time.time() - startTime))

    return events

#----------------------------------------------------------------------------------------------
def writeCandEvents(events):

    if options.in_pickle != None:
        log.info('writeCandEvents - input pickle file specified {} - do not write events again'.format(options.in_pickle))
        return

    outcsv = getOutPath('events.csv')

    if outcsv:
        if os.path.isfile(outcsv):
            log.info('writeCandEvents - output file will be replaced: {}'.format(outcsv))

        log.info('writeCandEvents - save {} events to {}'.format(len(events), outcsv))

        with open(outcsv, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')

            for event in events:
                for cand in event.candEvents:
                    writer.writerow(cand.getCandVars())

    outpickle = getOutPath('events.pickle')

    if outpickle:
        log.info('writeCandEvents - pickle {} events to {}'.format(len(events), outpickle))

        with open(outpickle, 'wb') as f:
            pickle.dump(events, f, pickle.HIGHEST_PROTOCOL)

    outopts = getOutPath('options.txt')

    if outopts:
        log.info('writeCandEvents - write macro options to {}'.format(outopts))

        with open(outopts, 'w') as f:
            f.write('Options: {}\n\n'.format(str(options)))
            f.write('Arguments: {}\n'.format(args))

#----------------------------------------------------------------------------------------------
def main():

    log.info('Starting main() of {} macro at {}'.format(os.path.basename(__file__), time.localtime()))

    log.info('Macro options: {}'.format(options))
    log.info('Macro arguments: {}'.format(args))

    rand.seed(options.seed)

    events = prepEvents()

    writeCandEvents(events)

    if options.plot:
        plotCandQuality(events)

        plotModelResults(events)

        plotCandEvents(events)

        plotLineDifferences(events)

        plotSimulatedHits(events)

        plotRecoClusters(events)

    log.info('All done')

#----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
