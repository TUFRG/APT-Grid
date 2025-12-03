#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:58:35 2025

@author: adekola
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import model_function as mf
import TransfiniteInterpolation as tf
import findLastQuadPointFunction as fq
from scipy.interpolate import interp1d, CubicSpline, splprep, splev
from scipy.optimize import fsolve, minimize

dispDir=1  # -1 to flip vertically so it looks like the real geometry
linedist=0.1

XX=m

plt.figure()
plt.plot(blade1PS2D[XX][:,0], dispDir*blade1PS2D[XX][:,1], 'k--')
plt.plot(blade2PS2D[XX][:,0], dispDir*blade2PS2D[XX][:,1], 'k--')
plt.plot(blade1SS2D[XX][:,0], dispDir*blade1SS2D[XX][:,1], 'k--')
plt.plot(blade2SS2D[XX][:,0], dispDir*blade2SS2D[XX][:,1], 'k--')
plt.plot(offsetBlade12D[XX][:,0], dispDir*offsetBlade12D[XX][:,1], 'k.')
plt.plot(offsetBlade22D[XX][:,0], dispDir*offsetBlade22D[XX][:,1], 'k.')
plt.plot(offsetSplinedBlade12D[XX][:,0], dispDir*offsetSplinedBlade12D[XX][:,1], 'b')
plt.plot(offsetSplinedBlade22D[XX][:,0], dispDir*offsetSplinedBlade22D[XX][:,1], 'b')
plt.plot(upstreamCamber1[XX][-5:,0], dispDir*upstreamCamber1[XX][-5:,1], 'k')
plt.plot(upstreamCamber2[XX][-5:,0], dispDir*upstreamCamber2[XX][-5:,1], 'k')
plt.plot(downstreamCamber1[XX][0:5:,0], dispDir*downstreamCamber1[XX][0:5:,1], 'k')
plt.plot(downstreamCamber2[XX][0:5:,0], dispDir*downstreamCamber2[XX][0:5:,1], 'k')
plt.plot(LECurveRot[XX][:,0], dispDir*LECurveRot[XX][:,1], 'r')
plt.plot(TECurveRot[XX][:,0], dispDir*TECurveRot[XX][:,1], 'r')

plt.plot(np.array([offsetBlade12D[XX][0][0],offsetBlade12D[XX][0][0]-linedist]), np.array([offsetBlade12D[XX][0][1], offsetBlade12D[XX][0][1]-linedist*curveSlopeLE1]), '--r')
plt.plot(np.array([offsetBlade22D[XX][0][0],offsetBlade22D[XX][0][0]-linedist]), np.array([offsetBlade22D[XX][0][1], offsetBlade22D[XX][0][1]-linedist*curveSlopeLE2]), '--r')

plt.plot(np.array([offsetBlade12D[XX][-1][0],offsetBlade12D[XX][-1][0]+linedist]), np.array([offsetBlade12D[XX][-1][1], offsetBlade12D[XX][-1][1]+linedist*curveSlopeTE1]), '--r')
plt.plot(np.array([offsetBlade22D[XX][-1][0],offsetBlade22D[XX][-1][0]+linedist]), np.array([offsetBlade22D[XX][-1][1], offsetBlade22D[XX][-1][1]+linedist*curveSlopeTE2]), '--r')

plt.plot(crossPassageLEp1[0], crossPassageLEp1[1], 'r.')
plt.plot(crossPassageTEp1[0], crossPassageTEp1[1], 'ro')

plt.plot(np.array([offsetBlade12D[XX][0][0],offsetBlade12D[XX][0][0]+linedist]), np.array([offsetBlade12D[XX][0][1], offsetBlade12D[XX][0][1]+linedist*offsetSlopeLE1]), '--b')
plt.plot(np.array([offsetBlade22D[XX][0][0],offsetBlade22D[XX][0][0]+linedist]), np.array([offsetBlade22D[XX][0][1], offsetBlade22D[XX][0][1]+linedist*offsetSlopeLE2]), '--b')

plt.plot(np.array([offsetBlade12D[XX][-1][0],offsetBlade12D[XX][-1][0]-linedist]), np.array([offsetBlade12D[XX][-1][1], offsetBlade12D[XX][-1][1]-linedist*offsetSlopeTE1]), '--b')
plt.plot(np.array([offsetBlade22D[XX][-1][0],offsetBlade22D[XX][-1][0]-linedist]), np.array([offsetBlade22D[XX][-1][1], offsetBlade22D[XX][-1][1]-linedist*offsetSlopeTE2]), '--b')

plt.axis('equal')
plt.show()
