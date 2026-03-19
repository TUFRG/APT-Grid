#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbomachinery Blade Passage Block Boundary Surface Generator
for use in a system for creating OH meshes for periodic and
aperiodic blade rows.
Authors:
    Adekola Adeyemi
    Justin Smart
    Tony Woo
    Jeff Defoe
Started in 2024
Active development ongoing as of January 2026
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import fsolve
import model_function as mf
import TransfiniteInterpolation as tf
import findLastQuadPointFunction as fq  # F = fq.getFvertex(A, B, C, D, E, meridCurve)
import subprocess
import importlib

# FUNCTION DEFINITIONS

def CartToCyl(arr):
    """
    Converts data in 3D arrays from Cartesian to Cylindrical
    Z is assumed to be the cylindrical axis
    indices of arr are [points, section, X-Y-Z]
    outputs arrCyl with indices [points, section, R-Theta-Z]
    """
    arrCyl = np.zeros(arr.shape)
    for i in range(arr.shape[1]):
        arrCyl[:,i,:] = np.array(mf.cart2pol(arr[:,i,0], arr[:,i,1], arr[:,i,2])).T
    return arrCyl


def trimProfilesToGasPath(profile1, profile2, lowerTrim, upperTrim, res):
    """
    Trims cylindrical coordinate profile sets to lie between
    lowerTrim and upperTrim meridional (R, Z) curves
    outputs new profile1 and profile2 and number of sections they contain
    """
    # Now I need to ensure that the blade profiles protudes beyond the hub and casing. This is important to get a profile that lie exactly on hub and casing
   
    hub2D = lowerTrim[:,[1,0]]
    cas2D = upperTrim[:,[1,0]]
    numOldProfiles = profile1.shape[1]
    N = int(profile1.shape[0]/2)
    Nr = int(2*N)
    M = profile1.shape[0]
    #profile12D = np.zeros([numOldProfiles,N*2,2])
    #profile22D = np.zeros([numOldProfiles,N*2,2])
    profile12D = np.zeros([numOldProfiles,M,2])
    profile22D = np.zeros([numOldProfiles,M,2])
    for i in range(numOldProfiles):
        profile12D[i,:,:] = np.array([profile1[:,i,2], profile1[:,i,1]]).T
        profile22D[i,:,:] = np.array([profile2[:,i,2], profile2[:,i,1]]).T

    # Get current LE and TE points using min/max Z coordinates of profiles
    LE2D = np.zeros((numOldProfiles,2))
    TE2D = np.zeros((numOldProfiles,2))
    for i in range(numOldProfiles):
        # combprof = np.concatenate((profile1[:, i, :], profile2[:, i, :]), axis=0) #I am not sure how this works, why combine both blades to find LE and TE?
        combprof = profile1[:, i, :]
        LEind = np.argmin(combprof[:, 2])
        TEind = np.argmax(combprof[:, 2])         
        # LEelement = combprof[LEind, :] #wrong
        # TEelement = combprof[TEind, :]
        LEelement = combprof[LEind]
        TEelement = combprof[TEind]
        LE2D[i, 0] = LEelement[2]
        LE2D[i, 1] = LEelement[1]
        TE2D[i, 0] = TEelement[2]
        TE2D[i, 1] = TEelement[1]
        # print(LEind, TEind)
    # print(LE2D)
    # print(TE2D)

    #Extend the LE and TE. First Evaluate the slope then use that to extend the LE and TE. The slope should be evaluated at hub and casing
    leHubSlope = mf.Slope(LE2D[0][0], LE2D[0][1], LE2D[1][0], LE2D[1][1]) #the slope is basically the last point and the preceding point 
    leCasSlope = mf.Slope(LE2D[-1][0], LE2D[-1][1], LE2D[-2][0], LE2D[-2][1])
    teHubSlope = mf.Slope(TE2D[0][0], TE2D[0][1], TE2D[1][0], TE2D[1][1])
    teCasSlope = mf.Slope(TE2D[-1][0], TE2D[-1][1], TE2D[-2][0], TE2D[-2][1])

    leHubInterX = LE2D[0][1] - leHubSlope*LE2D[0][0]
    teHubInterX = TE2D[0][1] - teHubSlope*TE2D[0][0]
    leCasInterX = LE2D[-1][1] - leCasSlope*LE2D[-1][0]
    teCasInterX = TE2D[-1][1] - teCasSlope*TE2D[-1][0]

    #There are instances where the blade data does not extend beyond the hub and casing and does not lie on the hub and casing either. This portion of the code extends the Le and te
    # curve beyond the hub and casing
    if leHubSlope > 0:
        leHubExtn = np.array([LE2D[0][0]-res*res, (LE2D[0][0]-res*res)*leHubSlope + leHubInterX])
    else:
        leHubExtn = np.array([LE2D[0][0]+res*res, (LE2D[0][0]+res*res)*leHubSlope + leHubInterX])
        
    if leCasSlope > 0:
        leCasExtn = np.array([LE2D[-1][0]+res*res, (LE2D[-1][0]+res*res)*leCasSlope + leCasInterX])
    else:
        leCasExtn = np.array([LE2D[-1][0]-res*res, (LE2D[-1][0]-res*res)*leCasSlope + leCasInterX])
        
    if teHubSlope > 0:
        teHubExtn = np.array([TE2D[0][1]-res*res, (TE2D[0][1]-res*res)*teHubSlope + teHubInterX])
    else:
        teHubExtn = np.array([TE2D[0][1]+res*res, (TE2D[0][1]+res*res)*teHubSlope + teHubInterX])
        
    if teCasSlope > 0:
        teCasExtn = np.array([TE2D[-1][1]+res*res, (TE2D[-1][1]+res*res)*teCasSlope + teCasInterX])
    else:
        teCasExtn = np.array([TE2D[-1][1]-res*res, (TE2D[-1][1]-res*res)*teCasSlope + teCasInterX])

    # repeat for profile 2
    LE2D2 = np.zeros((numOldProfiles,2))
    TE2D2 = np.zeros((numOldProfiles,2))
    for i in range(numOldProfiles):
        # combprof = np.concatenate((profile1[:, i, :], profile2[:, i, :]), axis=0)
        combprof = profile2[:, i, :]
        LEind = np.argmin(combprof[:, 2])
        TEind = np.argmax(combprof[:, 2]) 
        LEelement = combprof[LEind]
        TEelement = combprof[TEind]
        LE2D2[i, 0] = LEelement[2]
        LE2D2[i, 1] = LEelement[1]
        TE2D2[i, 0] = TEelement[2]
        TE2D2[i, 1] = TEelement[1]

    leHubSlope2 = mf.Slope(LE2D2[0][0], LE2D2[0][1], LE2D2[1][0], LE2D2[1][1])
    leCasSlope2 = mf.Slope(LE2D2[-1][0], LE2D2[-1][1], LE2D2[-2][0], LE2D2[-2][1])
    teHubSlope2 = mf.Slope(TE2D2[0][0], TE2D2[0][1], TE2D2[1][0], TE2D2[1][1])
    teCasSlope2 = mf.Slope(TE2D2[-1][0], TE2D2[-1][1], TE2D2[-2][0], TE2D2[-2][1])

    leHubInterX2 = LE2D2[0][1] - leHubSlope2*LE2D2[0][0]
    teHubInterX2 = TE2D2[0][1] - teHubSlope2*TE2D2[0][0]
    leCasInterX2 = LE2D2[-1][1] - leCasSlope2*LE2D2[-1][0]
    teCasInterX2 = TE2D2[-1][1] - teCasSlope2*TE2D2[-1][0]

    if leHubSlope2 > 0:
        leHubExtn2 = np.array([LE2D2[0][0]-res*res, (LE2D2[0][0]-res*res)*leHubSlope2 + leHubInterX2])
    else:
        leHubExtn2 = np.array([LE2D2[0][0]+res*res, (LE2D2[0][0]+res*res)*leHubSlope2 + leHubInterX2])
        
    if leCasSlope2 > 0:
        leCasExtn2 = np.array([LE2D2[-1][0]+res*res, (LE2D2[-1][0]+res*res)*leCasSlope2 + leCasInterX2])
    else:
        leCasExtn2 = np.array([LE2D2[-1][0]-res*res, (LE2D2[-1][0]-res*res)*leCasSlope2 + leCasInterX2])
        
    if teHubSlope2 > 0:
        teHubExtn2 = np.array([TE2D2[0][1]-res*res, (TE2D2[0][1]-res*res)*teHubSlope2 + teHubInterX2])
    else:
        teHubExtn2 = np.array([TE2D2[0][1]+res*res, (TE2D2[0][1]+res*res)*teHubSlope2 + teHubInterX2])
        
    if teCasSlope2 > 0:
        teCasExtn2 = np.array([TE2D2[-1][1]+res*res, (TE2D2[-1][1]+res*res)*teCasSlope2 + teCasInterX2])
    else:
        teCasExtn2 = np.array([TE2D2[-1][1]-res*res, (TE2D2[-1][1]-res*res)*teCasSlope2 + teCasInterX2])

    extnLE2D = np.vstack((leHubExtn, LE2D, leCasExtn))
    extnTE2D = np.vstack((teHubExtn, TE2D, teCasExtn))
    hubLE = mf.TwoLinesIntersect(hub2D, extnLE2D)
    casLE = mf.TwoLinesIntersect(cas2D, extnLE2D)
    hubTE = mf.TwoLinesIntersect(hub2D, extnTE2D)
    casTE = mf.TwoLinesIntersect(cas2D, extnTE2D)

    extnLE2D2 = np.vstack((leHubExtn2, LE2D2, leCasExtn2))
    extnTE2D2 = np.vstack((teHubExtn2, TE2D2, teCasExtn2))
    hubLE2 = mf.TwoLinesIntersect(hub2D, extnLE2D2)
    casLE2 = mf.TwoLinesIntersect(cas2D, extnLE2D2)
    hubTE2 = mf.TwoLinesIntersect(hub2D, extnTE2D2)
    casTE2 = mf.TwoLinesIntersect(cas2D, extnTE2D2)

    hubLEIdx = np.searchsorted(hub2D[:,0], LE2D[0][0])
    hubTEIdx = np.searchsorted(hub2D[:,0], TE2D[0][0])
    casLEIdx = np.searchsorted(cas2D[:,0], LE2D[-1][0])
    casTEIdx = np.searchsorted(cas2D[:,0], TE2D[-1][0])

    hubBladeRegion = np.vstack((hubLE, hub2D[hubLEIdx:hubTEIdx],hubTE)) #This is the portion of the hub that intersects with the blade
    hubRegionFunc = interp1d(hubBladeRegion[:,0], hubBladeRegion[:,1])
    hubRegionZ = np.linspace(hubBladeRegion[0,0], hubBladeRegion[-1,0], Nr*2-2) #Sampled same number of points on the hub as the blade
    hubRegionPts = np.array([hubRegionZ, hubRegionFunc(hubRegionZ)]).T
    casBladeRegion = np.vstack((casLE, cas2D[casLEIdx:casTEIdx],casTE))
    casRegionFunc = interp1d(casBladeRegion[:,0], casBladeRegion[:,1])
    casRegionZ = np.linspace(casBladeRegion[0,0], casBladeRegion[-1,0], Nr*2-2)
    casRegionPts = np.array([casRegionZ, casRegionFunc(casRegionZ)]).T

    hubBladeRegion2 = np.vstack((hubLE2, hub2D[hubLEIdx:hubTEIdx],hubTE2)) #This is the portion of the hub that intersects with the blade
    hubRegionFunc2 = interp1d(hubBladeRegion2[:,0], hubBladeRegion2[:,1])
    hubRegionZ2 = np.linspace(hubBladeRegion2[0,0], hubBladeRegion2[-1,0], Nr*2-2) #Sampled same number of points on the hub as the blade
    hubRegionPts2 = np.array([hubRegionZ2, hubRegionFunc2(hubRegionZ2)]).T
    casBladeRegion2 = np.vstack((casLE2, cas2D[casLEIdx:casTEIdx],casTE2))
    casRegionFunc2 = interp1d(casBladeRegion2[:,0], casBladeRegion2[:,1])
    casRegionZ2 = np.linspace(casBladeRegion2[0,0], casBladeRegion2[-1,0], Nr*2-2)
    casRegionPts2 = np.array([casRegionZ2, casRegionFunc2(casRegionZ2)]).T

    #% Now I will try to obtain the profile that lies exactly on the blade surface. To do this I will scale the last blade profile that is closest to both the hub and casing. 
    #Then I will use an interpolation function to obtain the other dimensions. 
    #Firstly get the index of the point that lies on the hub and casing 
    leHubIdx = np.searchsorted(extnLE2D[:,1], hubLE[1])
    teHubIdx = np.searchsorted(extnTE2D[:,1], hubTE[1])
    leCasIdx = np.searchsorted(extnLE2D[:,1], casLE[1])
    teCasIdx = np.searchsorted(extnTE2D[:,1], casTE[1])

    leHubIdx2 = np.searchsorted(extnLE2D2[:,1], hubLE2[1])
    teHubIdx2 = np.searchsorted(extnTE2D2[:,1], hubTE2[1])
    leCasIdx2 = np.searchsorted(extnLE2D2[:,1], casLE2[1])
    teCasIdx2 = np.searchsorted(extnTE2D2[:,1], casTE2[1])

    # Now at the hub, check which index is greater at the LE and TE, then use the higher index, then check if the hub intersect with next blade profile, if the answer is no use 
    #that blade profile as the profile to scale, if the answer is yes, check with the next blade profile until you get a no as an answer then use the profile as the one to scale
    hubIdx = max(leHubIdx, teHubIdx)
    casIdx = min(leCasIdx, teCasIdx)
    hubIntersect = False
    casIntersect = False
    hubIdx2 = max(leHubIdx2, teHubIdx2)
    casIdx2 = min(leCasIdx2, teCasIdx2)
    hubIntersect2 = False
    casIntersect2 = False
    """
    Here I try to check which profile of the blade is contained in the domain enclosed by hub and the casing. This is done firstly by checking the 
    LE and TE intersection of the blade with the hub and casing and note the index of the point on the LE TE curve. Then at the hub I chose the
    greater of the index and for the casing I chose the lesser of the index. Now it is possible that the curve intersect with another profile
    along the streamwise direction so i run a while loop to confirm there is no intersection. 

    Now when I have the right profile. I map the point distribution of the profile to the hub and casing. This data is in z and r. This will 
    give the blade profile that lies on the hub and casing. To get the theta direction, i basically copy the last the theta of the last profile. 
    Think of it as an extrusion in the radial direction.
    """
    while hubIdx < numOldProfiles and not hubIntersect: #
        profile1Surface = np.column_stack((profile1[:,hubIdx,2], profile1[:,hubIdx,1]))
        if not mf.TwoLinesIntersect(hubBladeRegion, profile1Surface): 
            hubIntersect = True
        else:
            hubIdx += 1
    if hubIntersect:   
        hubSpanFrac = mf.compute_span_fractions(profile12D[hubIdx-1,:,:], extnLE2D[hubIdx], extnTE2D[hubIdx])
        newHubBladeRegion = mf.rearrange_curve_by_arc_length(hubRegionPts, hubSpanFrac)
    else:
        None

    while casIdx-2 < numOldProfiles and not casIntersect: #The reson for subtracting 2 here is because I added extensions at both ends
        profile1Surface = np.column_stack((profile1[:,casIdx-2,2], profile1[:,casIdx-2,1]))
        if not mf.TwoLinesIntersect(casBladeRegion, profile1Surface):
            casIntersect = True
        else:
            casIdx -= 1
        print(casIntersect)
    if casIntersect:
        casSpanFrac = mf.compute_span_fractions(profile12D[numOldProfiles-1,:,:], extnLE2D[casIdx], extnTE2D[casIdx])
        newCasBladeRegion = mf.rearrange_curve_by_arc_length(casRegionPts, casSpanFrac)
        
    else:
        None


    ## For Blade 2
    while hubIdx2 < numOldProfiles and not hubIntersect2: #
        profile2Surface = np.column_stack((profile2[:,hubIdx2,2], profile2[:,hubIdx2,1])) # 
        if not mf.TwoLinesIntersect(hubBladeRegion2, profile2Surface):
            hubIntersect2 = True
        else:
            hubIdx2 += 1
    if hubIntersect2:   
        hubSpanFrac2 = mf.compute_span_fractions(profile22D[hubIdx2-1,:,:], extnLE2D2[hubIdx], extnTE2D2[hubIdx2])
        newHubBladeRegion2 = mf.rearrange_curve_by_arc_length(hubRegionPts2, hubSpanFrac2)
      
    else:
        None


    while casIdx2-2 < numOldProfiles and not casIntersect2:
        profile2Surface = np.column_stack((profile2[:,casIdx2-2,2], profile2[:,casIdx2-2,1]))
        if not mf.TwoLinesIntersect(casBladeRegion2, profile2Surface):
            casIntersect2 = True
        else:
            casIdx2 -= 1
    if casIntersect2:
        casSpanFrac2 = mf.compute_span_fractions(profile22D[numOldProfiles-1,:,:], extnLE2D2[casIdx2], extnTE2D2[casIdx2])
        newCasBladeRegion2 = mf.rearrange_curve_by_arc_length(casRegionPts2, casSpanFrac2)
        
    else:
        None
    #% Now get the profile on the hub and casing 

    hubBladeProfile = np.zeros([M,3])
    hubBladeProfile[:,0] = profile1[:,hubIdx-1,0]
    hubBladeProfile[:,[2,1]] = newHubBladeRegion
    # hubBladeProfile = np.insert(hubBladeProfile, 0, hubBladeProfile[-1])

    casBladeProfile = np.zeros([M,3])
    casBladeProfile[:,0] = profile1[:,casIdx-2,0]
    # casBladeProfile[:,0] = interpolate_theta_parametric(blade1Cyl[casIdx-1], blade1Cyl[casIdx-2], newCasBladeRegion[:,1], newCasBladeRegion[:,0])
    casBladeProfile[:,[2,1]] = newCasBladeRegion
    # casBladeProfile = np.insert(casBladeProfile, 0, casBladeProfile[-1])

    hubBladeProfile2 = np.zeros([M,3])
    hubBladeProfile2[:,0] = profile2[:,hubIdx2-1,0]
    hubBladeProfile2[:,[2,1]] = newHubBladeRegion2
    

    casBladeProfile2 = np.zeros([M,3])
    casBladeProfile2[:,0] = profile2[:,casIdx2-2,0]
    # casBladeProfile2[:,0] = interpolate_theta_parametric(blade2Cyl[casIdx2-1], blade2Cyl[casIdx2-2], newCasBladeRegion2[:,1], newCasBladeRegion2[:,0])
    casBladeProfile2[:,[2,1]] = newCasBladeRegion2
    # print(casBladeProfile)
    #The blade was splitted here, I need to know why that was done, in a moment. Found it.
    #The hub and casing profiles needs to be closed.
    
    #I made changes here!!!
    hub1 = mf.densify_curve_robust(hubBladeProfile, M -1)  
    cas1 = mf.densify_curve_robust(casBladeProfile, M -1)
    
    hub2 = mf.densify_curve_robust(hubBladeProfile2, M -1)
    cas2 = mf.densify_curve_robust(casBladeProfile2, M -1)
    
    #Stacking blade for new Blade definition 
    if casIdx-2 > numOldProfiles:
        newNsection = numOldProfiles - hubIdx + 3 # Remember that i added two points for the extension both at hub and casing. So for cases where 
        # the blade did not touch the casing. so this is the number of points in original radial direction + the two new points i added (hub nad cas)
        # then minus the point already excluded at the hub - 1. For example imagine there are 7 points originally, at the hub I excluded 2 points
        # but the casing is not touching. By definition my hubIdx is 3 (not 2 because of how i defined it) so 7 + 2 - 3 + 1 = 7 - 3 + 3.
    else:
        newNsection = numOldProfiles - hubIdx + (numOldProfiles - casIdx-2) + 4  #In this case the casing is touching so 

    # I will compute 6 distances
    hubMidIdx = int(0.5*N)
    # Check the distance of the profile before the hub profile and the profile after the hub profile from original
    distHubALE = mf.dist2D(LE2D[hubIdx,0], LE2D[hubIdx,1], LE2D[hubIdx-2,0], LE2D[hubIdx-2,1]) # blade data at the LE
    distHubATE = mf.dist2D(TE2D[hubIdx,0], TE2D[hubIdx,1], TE2D[hubIdx-2,0], TE2D[hubIdx-2,1]) # blade data at the TE
    distHubAMid = mf.dist2D(profile1[hubMidIdx,hubIdx,2], profile1[hubMidIdx,hubIdx,1], profile1[hubMidIdx,hubIdx-2,2], profile1[hubMidIdx,hubIdx-2,1]) # after the hub profile from original blade data at the midChord
    distHubBLE = mf.dist2D(LE2D[hubIdx,0], LE2D[hubIdx,1], hub1[0,2], hub1[0,1]) # blade data at the LE
    distHubBTE = mf.dist2D(TE2D[hubIdx,0], TE2D[hubIdx,1], hub1[hub1[:,2].argmax(),2], hub1[hub1[:,2].argmax(),1]) # blade data at the TE
    distHubBMid = mf.dist2D(profile1[hubMidIdx,hubIdx,2], profile1[hubMidIdx,hubIdx,1], hub1[hubMidIdx,2], hub1[hubMidIdx,1]) #after the hub profile from original after the hub profile from original blade data at the midChord

    distHubA = min(distHubALE, distHubATE, distHubAMid)  # The minimum distance is the closest 
    distHubB = min(distHubBLE, distHubBTE, distHubBMid)

    distCasALE = mf.dist2D(LE2D[casIdx-1,0], LE2D[casIdx-1,1], LE2D[casIdx-2,0], LE2D[casIdx-2,1])
    distCasATE = mf.dist2D(TE2D[casIdx-1,0], TE2D[casIdx-1,1], TE2D[casIdx-2,0], TE2D[casIdx-2,1])
    distCasAMid = mf.dist2D(profile1[hubMidIdx,casIdx-1,2], profile1[hubMidIdx,casIdx-1,1], profile1[hubMidIdx,casIdx-2,2], profile1[hubMidIdx,casIdx-2,1])
    distCasBLE = mf.dist2D(LE2D[casIdx-1,0], LE2D[casIdx-1,1], cas1[0,2], cas1[0,1])
    distCasBTE = mf.dist2D(TE2D[casIdx-1,0], TE2D[casIdx-1,1], cas1[cas1[:,2].argmax(),2], cas1[cas1[:,2].argmax(),1])
    distCasBMid = mf.dist2D(profile1[hubMidIdx,casIdx-1,2], profile1[hubMidIdx,casIdx-1,1], cas1[hubMidIdx,2], cas1[hubMidIdx,1])

    distCasA = min(distCasALE, distCasATE, distCasAMid)  # The minimum distance is the closest
    distCasB = min(distCasBLE, distCasBTE, distCasBMid)
    if distHubB > 0.5*distHubA:
        newNsection = newNsection
    else:
        newNsection = newNsection - 1
    if distCasB > 0.5*distCasA:
        newNsection = newNsection - 1
    else:
        newNsection = newNsection
        
    hub1 = np.insert(hub1, 0, hub1[-1], axis=0) #I need to guanrantee that this forms a close loop, initially this was ensured by splitting the curve.
    cas1 = np.insert(cas1, 0, cas1[-1], axis=0)
    # print(hub1[-1])
    hub2 = np.insert(hub2, 0, hub2[-1], axis=0)
    cas2 = np.insert(cas2, 0, cas2[-1], axis=0)    
    oldBlade1 = np.zeros([M, newNsection, 3])  # (theta, r, z)
    oldBlade2 = np.zeros([M, newNsection, 3])  # (theta, r, z)
    for f in range(newNsection):
        if f == 0:
            oldBlade1[:, f, :] = hub1
            oldBlade2[:, f, :] = hub2
        elif f == newNsection - 1:
            oldBlade1[:, f, :] = cas1
            oldBlade2[:, f, :] = cas2
        elif distHubB < 0.5*distHubA:
            oldBlade1[:, f, :] = profile1[:, f+hubIdx-1, :]
            oldBlade2[:, f, :] = profile2[:, f+hubIdx-1, :]
        else:
            oldBlade1[:, f, :] = profile1[:, f+hubIdx-2, :]
            oldBlade2[:, f, :] = profile2[:, f+hubIdx-2, :]

    return oldBlade1, oldBlade2, newNsection


def getLETEandSplit(profile, N):
    # get LE and TE points from a closed loop
    # Define "LE" and "TE" here as simply the furthest-forward
    # and further-backward axial (Z) coordinates
    # Input data is in cylindrical coordinates, theta-R-Z
    # Output: two numpy arrays with the coordinates of the LE/TE
    # on each section, and two more arrays with profiles split between
    # high and low theta (p and n) sides at a resolution of N each

    # number of sections/profiles
    nsec = profile.shape[1]

    # split profiles and add points
    profilep = np.zeros([N, nsec, 3])  # (theta, r, z)
    profilen = np.zeros([N, nsec, 3])

    # first get LE/TE arrays
    LE = np.zeros((nsec, 3))
    TE = np.zeros((nsec, 3))
    for i in range(nsec):
        LEind = profile[:, i, 2].argmin()
        TEind = profile[:, i, 2].argmax()
        LE[i] = profile[LEind, i, :]
        TE[i] = profile[TEind, i, :]

        # determine which "side" has larger theta coordinates
        if TEind < LEind:
            indices1 = np.r_[TEind:LEind+1]
            avgTheta1 = np.average(profile[indices1, i, 0])
            indices2 = np.r_[0:TEind+1, LEind:profile.shape[0]]
            avgTheta2 = np.average(profile[indices2, i, 0])
        else:
            indices1 = np.r_[LEind:TEind+1]
            avgTheta1 = np.average(profile[indices1, i, 0])
            indices2 = np.r_[0:LEind+1, TEind:profile.shape[0]]
            avgTheta2 = np.average(profile[indices2, i, 0])
        if avgTheta2 > avgTheta1:
            ptsp = profile[indices2, i, :]
            ptsn = profile[indices1, i, :]
        else:
            ptsp = profile[indices1, i, :]
            ptsn = profile[indices2, i, :]

        # order ptsp and ptsn in ascending order by Z coordinate
        sortIndp = ptsp[:, 2].argsort()
        sortIndn = ptsn[:, 2].argsort()
        ptsp = ptsp[sortIndp]
        ptsn = ptsn[sortIndn]

        # assign new data
        profilep[:, i, :] = mf.densify_curve_robust(remove_duplicates(ptsp), N)
        profilen[:, i, :] = mf.densify_curve_robust(remove_duplicates(ptsn), N)

    return LE, TE, profilep, profilen


def getMeridCurve(LE1, TE1, LE2, TE2, hub, cas, res):
    # Produces meridional curves (R-Z) on all sections
    # requires LE and TE data for each blade to ensure these are included
    # Output: array of meridional curves, the first and last of which are just
    # the hub and casing again but with the LE and TE points of both blades
    # added. Note that basically the LE and TE points in meridional coordinates
    # of the two blades defining the passage must be the same!
    nsec = LE1.shape[0]
    tol = 1e-10
    if np.abs(LE1[:,1]-LE2[:,1]).max() > tol:
        raise ValueError("LE points don't line up between blades!")
    if np.abs(TE1[:,1]-TE2[:,1]).max() > tol:
        raise ValueError("TE points don't line up between blades!")
    # if no errors, just use LE1/TE1 for the rest
    LE = LE1
    TE = TE1

    # create meridional curves for inlet and outlet, and map interior points
    # on them to the same span fractions as the LE/TE points
    inletR = np.linspace(hub[0, 0], cas[0, 0], res)
    inletFunc = interp1d([hub[0, 0], cas[0, 0]], [hub[0, 1], cas[0, 1]])
    inletZ = inletFunc(inletR)
    inlet = np.column_stack((inletZ, inletR))
    scaledNewInletR = mf.scale(min(LE[:,1]), max(LE[:,1]), hub[0,0], cas[0,0], LE[:,1])
    inletFunc = interp1d(inlet[:,1], inlet[:,0], bounds_error=False, fill_value=(inletZ[0], inletZ[-1]))
    scaledNewInletZ = inletFunc(scaledNewInletR)

    outletR = np.linspace(hub[-1, 0], cas[-1, 0], res)
    outletFunc = interp1d([hub[-1, 0], cas[-1, 0]], [hub[-1, 1], cas[-1, 1]])
    outletZ = outletFunc(outletR)
    outlet = np.column_stack((outletZ, outletR))
    scaledNewOutletR = mf.scale(min(TE[:,1]), max(TE[:,1]), hub[-1,0], cas[-1,0], TE[:,1])
    outletFunc = interp1d(outlet[:,1], outlet[:,0], bounds_error=False, fill_value=(outletZ[0], outletZ[-1]))
    scaledNewOutletZ = outletFunc(scaledNewOutletR)

    adjInlet = np.column_stack((scaledNewInletZ, scaledNewInletR))
    adjOutlet = np.column_stack((scaledNewOutletZ, scaledNewOutletR))
    tempHub = np.zeros([len(hub), 2])
    tempCas = np.zeros([len(cas), 2])
    if min(hub[:,1]) == min(cas[:,1]):
        tempHub = hub
        tempCas = cas
    elif min(hub[:,1]) < min(cas[:,1]):
        tempHub[:,:] = hub[:,:]
        tempCas[:,:] = cas[:,:]
        tempCas[0] = np.array([cas[0][0], min(hub[:,1])])
    elif min(hub[:,2]) > min(cas[:,2]):
        tempHub[:,:] = hub[:,:]
        tempHub[0] = np.array([cas[0][0], min(cas[:,1])])
        tempCas[:,:] = cas[:,:]
        
    if max(hub[:,1])  == max(cas[:,1]):
        tempHub[:,:] = tempHub[:,:]
        tempCas[:,:] = tempCas[:,:]   
    elif max(hub[:,1]) < max(cas[:,1]):
        tempHub[-1] = np.array([hub[-1][0], max(cas[:,1])])
    elif max(hub[:,1]) > max(cas[:,1]):
        tempCas[-1] = np.array([cas[-1][0], max(hub[:,1])])
    
    tempInletR = np.linspace(tempHub[0][0], tempCas[0][0], res)
    tempInletFunc = interp1d([tempHub[0][1], tempCas[0][0]], [tempHub[0][1], tempCas[0][1]])
    tempInletZ = tempInletFunc(tempInletR)
    inletTemp = np.column_stack((tempInletZ, tempInletR))
    tempOutletR = np.linspace(tempHub[::-1][0][0], tempCas[::-1][0][0], res)
    tempOutletFunc = interp1d([tempHub[::-1][0][0], tempCas[::-1][0][0]], [tempHub[::-1][0][1], tempCas[::-1][0][1]])
    tempOutletZ = tempOutletFunc(tempOutletR)
    outletTemp = np.column_stack((tempOutletZ, tempOutletR))
    
    scaledTempNewInletR = mf.scale(min(LE[:,1]), max(LE[:,1]), tempHub[0][0], tempCas[0][0],LE[:,1])
    scaledTempNewOutletR = mf.scale(min(TE[:,1]), max(TE[:,1]), tempHub[::-1][0][0], tempCas[::-1][0][0],TE[:,1])
    scaledInletFunc = interp1d(inletTemp[:,1], inletTemp[:,0], bounds_error=False, fill_value=(tempInletZ[0], tempInletZ[-1]))
    scaledOutletFunc = interp1d(outletTemp[:,1], outletTemp[:,0], bounds_error=False, fill_value=(tempOutletZ[0], tempOutletZ[-1]))
    scaledTempNewInletZ = scaledInletFunc(scaledTempNewInletR)
    scaledTempNewOutletZ = scaledOutletFunc(scaledTempNewOutletR)
    tempInlet = np.column_stack((scaledTempNewInletZ, scaledTempNewInletR))
    tempOutlet = np.column_stack((scaledTempNewOutletZ, scaledTempNewOutletR))

    # now have a "closed box" of hub, cas, inlet, and outlet
    # inlet/outlet are guaranteed to have the same number of points
    # but hub and cas could still have a different number of points
    # this needs to be fixed by "densifying" the sparser curve
    if hub.shape[0] > cas.shape[0]:
        cas = mf.densify_curve_robust(cas, hub.shape[0])
    elif cas.shape[0] > hub.shape[0]:
        hub = mf.densify_curve_robust(hub, cas.shape[0])

    #Using transfinite interpolation to get the meridional curves
    meridionalNodes = tf.transfinite(tempInlet, tempOutlet, hub[:, [1, 0]], cas[:, [1, 0]])
    meridCurve = meridionalNodes.reshape(nsec, len(hub), 2)
    meridCurve = meridCurve[:, :, [1, 0]]

    return meridCurve


def getInitExtAngles(LE, TE, bladep, bladen):
    # estimate the direction each section "wants" to go for upstream/
    # downstream extensions
    # inputs: LE points, TE points, blade side profiles (p and n)
    # outputs: LE angles (directions), TE angles (directions)
    # Key note: operates in m'-theta!

    nsec = LE.shape[0]
    angleLE = np.zeros(nsec)
    angleTE = np.zeros(nsec)

    for i in range(nsec):
        chordlengthBlade = mf.dist2D(LE[i][0], LE[i][1], TE[i][0], TE[i][1])
        # Add 5% of chord length to the axial distance of the LE and TE to get vertical line that intersect blade surface
        Pfunc = interp1d(bladep[i, :, 0], bladep[i, :, 1])  # Interpolation function pos side
        Nfunc = interp1d(bladen[i, :, 0], bladen[i, :, 1])  # Interpolation function neg side
        # Determine the slope at the LE and TE of blade
        zLE = LE[i][0] + 0.05*chordlengthBlade
        rThLEP = Pfunc(zLE)
        rThLEN = Nfunc(zLE)
        rThLE = 0.5*(rThLEP+rThLEN)

        zTE = TE[i][0] - 0.05*chordlengthBlade
        rThTEP = Pfunc(zTE)
        rThTEN = Nfunc(zTE)
        rThTE = 0.5*(rThTEP + rThTEN)

        slopeLE = mf.Slope(LE[i][0], LE[i][1], zLE, rThLE)
        slopeTE = mf.Slope(TE[i][0], TE[i][1], zTE, rThTE)
        angleLE[i] = np.arctan(slopeLE)
        angleTE[i] = np.arctan(slopeTE)

    return angleLE, angleTE


def getOffsetVertices(bladep, bladen, meridCurve, LE, TE, delta):
    # takes in split blade, meridional curves, LE/TE, and delta (offset distance)
    # computes offset vertices which guarantee valid
    # unprojected blocks during grid generation
    # output: vertices (4), an LE and TE one and midchord ones

    newNsection = LE.shape[0]
    Nr = bladep.shape[0]
    
    offsetVertex1Cart = np.zeros([newNsection,4,3])  # LE,Mid,TE,Mid
    offsetVertex1Cyl = np.zeros([newNsection,4,3])  # LE,Mid,TE,Mid
    newBlade1PCylM = np.zeros([Nr+1,newNsection,3])  # so i added the midPopints values here 
    newBlade1NCylM = np.zeros([Nr+1,newNsection,3])
    mid1P = np.zeros((newNsection, 3))
    mid1N = np.zeros((newNsection, 3))
    for d in range(newNsection):
        newBlade1PCart = np.array(mf.pol2cart(bladep[:,d,0], bladep[:,d,1], bladep[:,d,2])).T
        newBlade1NCart = np.array(mf.pol2cart(bladen[:,d,0], bladen[:,d,1], bladen[:,d,2])).T
        midZ1Point = 0.5*(newBlade1PCart[0,2]+ newBlade1PCart[-1,2]) # This is the mid point of the axial chord
        xP1Func = CubicSpline(newBlade1PCart[:,2], newBlade1PCart[:,0])
        yP1Func = CubicSpline(newBlade1PCart[:,2], newBlade1PCart[:,1])
        xN1Func = CubicSpline(newBlade1NCart[:,2], newBlade1NCart[:,0])
        yN1Func = CubicSpline(newBlade1NCart[:,2], newBlade1NCart[:,1])
        leBlade1 = newBlade1PCart[0] #This is point E
        teBlade1 = newBlade1PCart[-1] #This is also point E
        midPBlade1 = np.array([xP1Func(midZ1Point), yP1Func(midZ1Point), midZ1Point]) #This is the point C
        midPBlade1Cyl = np.array(mf.cart2pol(midPBlade1[0], midPBlade1[1], midPBlade1[2])).T
        newBlade1PCylM[:,d,:] = insertPoint(bladep[:,d,:], midPBlade1Cyl)
        midNBlade1 = np.array([xN1Func(midZ1Point), yN1Func(midZ1Point), midZ1Point]) #This is the point B
        midNBlade1Cyl = np.array(mf.cart2pol(midNBlade1[0], midNBlade1[1], midNBlade1[2])).T
        newBlade1NCylM[:,d,:] = insertPoint(bladen[:,d,:], midNBlade1Cyl) 
        mid1P[d] = midPBlade1Cyl
        mid1N[d] = midNBlade1Cyl
        # Here i made an assumption that the offset at midpoint is at constant radius with the point close to the blade surface (this is not really an assumption, it must be true!)
        rPBlade1 = np.sqrt(midPBlade1[0]**2 + midPBlade1[1]**2)
        rNBlade1 = np.sqrt(midNBlade1[0]**2 + midNBlade1[1]**2)
        rBlade1 = 0.5*(rPBlade1 + rNBlade1)
        # Using the definition of arc length, I can determine the theta shift and I can determine the x and y value from that. Assuming the BL thickness is the arc length
        thPBlade1 = delta/rBlade1 #s=r*theta
        thNBlade1 = -delta/rBlade1 # Negative is added here to move in the opposite direction
        thOrigPBlade1 = np.arctan2(midPBlade1[1], midPBlade1[0])
        thOrigNBlade1 = np.arctan2(midNBlade1[1], midNBlade1[0])
        midPOffset1 = np.array([rBlade1*np.cos(thPBlade1 + thOrigPBlade1), rBlade1*np.sin(thPBlade1+thOrigPBlade1), midZ1Point]) # This is the point D
        midNOffset1 = np.array([rBlade1*np.cos(thNBlade1 + thOrigNBlade1), rBlade1*np.sin(thNBlade1+thOrigNBlade1 ), midZ1Point]) # This is the point A  
        leOffset1 = fq.getFvertex(midNOffset1, midNBlade1, midPBlade1, midPOffset1, leBlade1, meridCurve[d])  
        teOffset1 = fq.getFvertex(midNOffset1, midNBlade1, midPBlade1, midPOffset1, teBlade1, meridCurve[d])  
        offsetVertex1Cart[d,:] = np.vstack((leOffset1, midPOffset1, teOffset1, midNOffset1))
        leOffset1Cyl = np.array(mf.cart2pol(leOffset1[0], leOffset1[1], leOffset1[2])).T
        teOffset1Cyl = np.array(mf.cart2pol(teOffset1[0], teOffset1[1], teOffset1[2])).T
        midPOffset1Cyl = np.array(mf.cart2pol(midPOffset1[0], midPOffset1[1], midPOffset1[2])).T
        midNOffset1Cyl = np.array(mf.cart2pol(midNOffset1[0], midNOffset1[1], midNOffset1[2])).T
        offsetVertex1Cyl[d,:] = np.vstack((leOffset1Cyl, midPOffset1Cyl, teOffset1Cyl, midNOffset1Cyl))

    return offsetVertex1Cyl, newBlade1PCylM, newBlade1NCylM, mid1P, mid1N


def cylToMPT(blade1p, blade1n, blade2p, blade2n, offsetVertex1Cyl, offsetVertex2Cyl, LE1, TE1, LE2, TE2, meridCurve, bladeRes, res):
    # takes in blade side data for both blades and offset vertices (all in
    # cylindrical coordinates) + LE/TE + meridional curves and
    # increased blade resolution values
    # output: blade side data, offset points in m'-theta

    newNsection = LE1.shape[0]
    
    blade1PMprime = np.zeros([newNsection,bladeRes,4]) # (theta, r, z, mprime)
    blade1NMprime = np.zeros([newNsection,bladeRes,4])
    blade2PMprime = np.zeros([newNsection,bladeRes,4])
    blade2NMprime = np.zeros([newNsection,bladeRes,4])
    
    for d in range(newNsection):
        blade1PMprime[d,:,0:3] = mf.densify_curve_robust(blade1p[:,d,:], bladeRes)
        blade1NMprime[d,:,0:3] = mf.densify_curve_robust(blade1n[:,d,:], bladeRes)
        blade2PMprime[d,:,0:3] = mf.densify_curve_robust(blade2p[:,d,:], bladeRes)
        blade2NMprime[d,:,0:3] = mf.densify_curve_robust(blade2n[:,d,:], bladeRes)
        index = 0
        for e in range(bladeRes):
            if index == 0:
                blade1PMprime[d,e,3] = 0  #initialize 0 m' at the first profile point
                blade1NMprime[d,e,3] = 0  #initialize 0 m' at the first profile point
                blade2PMprime[d,e,3] = 0  #initialize 0 m' at the first profile point
                blade2NMprime[d,e,3] = 0  #initialize 0 m' at the first profile point               
            else:
                mPrev1P = blade1PMprime[d,:,3][e-1]
                rPrev1P = blade1PMprime[d,:,1][e-1]
                rCurr1P = blade1PMprime[d,:,1][e]
                zPrev1P = blade1PMprime[d,:,2][e-1]
                zCurr1P = blade1PMprime[d,:,2][e]
                blade1PMprime[d,e,3] = mPrev1P + ((2/(rCurr1P + rPrev1P)) * np.sqrt((rCurr1P-rPrev1P)**2 + (zCurr1P-zPrev1P)**2))
                mPrev1N = blade1NMprime[d,:,3][e-1]
                rPrev1N = blade1NMprime[d,:,1][e-1]
                rCurr1N = blade1NMprime[d,:,1][e]
                zPrev1N = blade1NMprime[d,:,2][e-1]
                zCurr1N = blade1NMprime[d,:,2][e]
                blade1NMprime[d,e,3] = mPrev1N + ((2/(rCurr1N + rPrev1N)) * np.sqrt((rCurr1N-rPrev1N)**2 + (zCurr1N-zPrev1N)**2))
                mPrev2P = blade2PMprime[d,:,3][e-1]
                rPrev2P = blade2PMprime[d,:,1][e-1]
                rCurr2P = blade2PMprime[d,:,1][e]
                zPrev2P = blade2PMprime[d,:,2][e-1]
                zCurr2P = blade2PMprime[d,:,2][e]
                blade2PMprime[d,e,3] = mPrev2P + ((2/(rCurr2P + rPrev2P)) * np.sqrt((rCurr2P-rPrev2P)**2 + (zCurr2P-zPrev2P)**2))
                mPrev2N = blade2NMprime[d,:,3][e-1]
                rPrev2N = blade2NMprime[d,:,1][e-1]
                rCurr2N = blade2NMprime[d,:,1][e]
                zPrev2N = blade2NMprime[d,:,2][e-1]
                zCurr2N = blade2NMprime[d,:,2][e]
                blade2NMprime[d,e,3] = mPrev2N + ((2/(rCurr2N + rPrev2N)) * np.sqrt((rCurr2N-rPrev2N)**2 + (zCurr2N-zPrev2N)**2))  
            index += 1

    tempHub = meridCurve[0,:,:]
    tempCas = meridCurve[-1,:,:]
    tempInlet = meridCurve[:,0,:]
    tempOutlet = meridCurve[:,-1,:]
    
    hubLEIdx = np.searchsorted(tempHub[:,1], LE1[0][2])
    hubTEIdx = np.searchsorted(tempHub[:,1], TE1[0][2])
    casLEIdx = np.searchsorted(tempCas[:,1], LE1[newNsection-1][2])
    casTEIdx = np.searchsorted(tempCas[:,1], TE1[newNsection-1][2])
    
    newLE2D = np.column_stack((LE1[:,2], LE1[:,1]))
    newTE2D = np.column_stack((TE1[:,2], TE1[:,1]))
    
    newLE2D2 = np.column_stack((LE2[:,2], LE2[:,1]))
    newTE2D2 = np.column_stack((TE2[:,2], TE2[:,1]))
    
    upHub = np.vstack((tempHub[0:hubLEIdx][:,[1,0]],newLE2D[0]))#hubLE))
    upHFunc = interp1d(upHub[:,0], upHub[:,1])
    upHub = np.column_stack((np.linspace(upHub[0][0], upHub[::-1][0][0],res), upHFunc(np.linspace(upHub[0][0], upHub[::-1][0][0],res))))
    dwHub = np.vstack((newTE2D[0], tempHub[hubTEIdx:len(tempHub)][:,[1,0]]))
    dwHFunc = interp1d(dwHub[:,0], dwHub[:,1])
    dwHub = np.column_stack((np.linspace(dwHub[0][0], dwHub[::-1][0][0],res), dwHFunc(np.linspace(dwHub[0][0], dwHub[::-1][0][0],res))))
    upCas = np.vstack((tempCas[0:casLEIdx][:,[1,0]],newLE2D[-1]))#casLE))
    upCFunc = interp1d(upCas[:,0], upCas[:,1])
    upCas = np.column_stack((np.linspace(upCas[0][0], upCas[::-1][0][0],res), upCFunc(np.linspace(upCas[0][0], upCas[::-1][0][0],res))))
    dwCas = np.vstack((newTE2D[-1], tempCas[casTEIdx:len(tempCas)][:,[1,0]]))
    dwCFunc = interp1d(dwCas[:,0], dwCas[:,1])
    dwCas = np.column_stack((np.linspace(dwCas[0][0], dwCas[::-1][0][0],res), dwCFunc(np.linspace(dwCas[0][0], dwCas[::-1][0][0],res))))
    
    upNodes = tf.transfinite(tempInlet, newLE2D, upHub, upCas)
    dwNodes = tf.transfinite(newTE2D, tempOutlet, dwHub, dwCas)
    
    upstreamMprime = np.zeros([newNsection,res,4]) #This contains the hub and casing 
    dwstreamMprime = np.zeros([newNsection,res,4]) 
    offsetUpstreamMprime = np.zeros([newNsection,res,4])
    offsetDwstreamMprime = np.zeros([newNsection,res,4])
    for ef in range(newNsection):
        upstreamMprime[ef,:,0] = np.zeros(res)
        dwstreamMprime[ef,:,0] = np.zeros(res)
        offsetUpstreamMprime[ef,:,0] = np.zeros(res)
        offsetDwstreamMprime[ef,:,0] = np.zeros(res)
        tempUp = np.zeros([res,2])  
        tempDw = np.zeros([res,2])  
        
        count = 0
        for f in range(res):
            tempUp[f] = upNodes[ef*res+f]
            tempDw[f] = dwNodes[ef*res+f] 
            upstreamMprime[ef,f,2] = tempUp[f,0]
            dwstreamMprime[ef,f,2] = tempDw[f,0]
            rFunc1 = interp1d(tempUp[:,0], tempUp[:,1])
            rFunc2 = interp1d(tempDw[:,0], tempDw[:,1])
            upstreamMprime[ef,f,1] = rFunc1(upstreamMprime[ef,f,2])
            dwstreamMprime[ef,f,1] = rFunc2(dwstreamMprime[ef,f,2])       
            if count == 0:
                upstreamMprime[ef,:,3] = blade1PMprime[ef][0][3]#Initializing with the first value on the blade in mprime theta
                dwstreamMprime[ef,:,3] = blade1PMprime[ef][::-1][0][3] #Initializing with the last value on the blade in mprime theta
            else:
                mPrevUh = upstreamMprime[ef,:,3][f-1]
                rPrevUh = upstreamMprime[ef,:,1][f-1]
                rCurrUh = upstreamMprime[ef,:,1][f]
                zPrevUh = upstreamMprime[ef,:,2][f-1]
                zCurrUh = upstreamMprime[ef,:,2][f]
                upstreamMprime[ef,f,3] = mPrevUh - ((2/(rCurrUh + rPrevUh)) * np.sqrt((rCurrUh-rPrevUh)**2 + (zCurrUh-zPrevUh)**2))
                mPrevDh = dwstreamMprime[ef,:,3][f-1]
                rPrevDh = dwstreamMprime[ef,:,1][f-1]
                rCurrDh = dwstreamMprime[ef,:,1][f]
                zPrevDh = dwstreamMprime[ef,:,2][f-1]
                zCurrDh = dwstreamMprime[ef,:,2][f]
                dwstreamMprime[ef,f,3] = mPrevDh + ((2/(rCurrDh + rPrevDh)) * np.sqrt((rCurrDh-rPrevDh)**2 + (zCurrDh-zPrevDh)**2))         
            count +=1      
        upstreamMprime[ef,:,3] = upstreamMprime[ef,:,3][::-1]
    #%% Convert the offset vertices to mPrime 
    ## This is not a straight forward approach as I have to ensure that I make reference to the mprime definition of the blade data
    ## To do this, will use the interpolation to get the m prime value of the LE offset point then work from there 
    offsetVertex1Mprime = np.zeros([newNsection, 4, 4])
    offsetVertex2Mprime = np.zeros([newNsection, 4, 4])
    offsetVertex1MprimeP = np.zeros([newNsection, 3, 4])
    offsetVertex1MprimeN = np.zeros([newNsection, 3, 4])
    offsetVertex2MprimeP = np.zeros([newNsection, 3, 4])
    offsetVertex2MprimeN = np.zeros([newNsection, 3, 4])
    count = 0
    for g in range(newNsection):
        combinedMprime = np.concatenate((upstreamMprime[g][:-1], blade1NMprime[g], dwstreamMprime[g][1:]))
        combinedMprime2 = np.concatenate((upstreamMprime[g][:-1], blade2PMprime[g], dwstreamMprime[g][1:]))
        deltaFunc = CubicSpline(combinedMprime[:,2], combinedMprime[:,3])
    
        offsetVertex1MprimeP[g,:,0:3] = np.array([offsetVertex1Cyl[g][0],offsetVertex1Cyl[g][1],offsetVertex1Cyl[g][2]])
        offsetVertex1MprimeN[g,:,0:3] = np.array([offsetVertex1Cyl[g][0],offsetVertex1Cyl[g][3],offsetVertex1Cyl[g][2]])
        offsetVertex2MprimeP[g,:,0:3] = np.array([offsetVertex2Cyl[g][0],offsetVertex2Cyl[g][1],offsetVertex2Cyl[g][2]])
        offsetVertex2MprimeN[g,:,0:3] = np.array([offsetVertex2Cyl[g][0],offsetVertex2Cyl[g][3],offsetVertex2Cyl[g][2]])    
        for gg in range(3):
            if gg == 0:
                offsetVertex1Mprime[g,:,0:3] = offsetVertex1Cyl[g]
                offsetVertex2Mprime[g,:,0:3] = offsetVertex2Cyl[g]
                offsetVertex1Mprime[g,gg,3] = deltaFunc(offsetVertex1Cyl[g,0,2]) #to initialize the mprime value, I have to figure out where it lies on the 
                offsetVertex2Mprime[g,gg,3] = deltaFunc(offsetVertex2Cyl[g,0,2]) #meriodional curve in mprime
                offsetVertex1MprimeP[g,gg,3] = deltaFunc(offsetVertex1Cyl[g,0,2])
                offsetVertex1MprimeN[g,gg,3] = deltaFunc(offsetVertex1Cyl[g,0,2])
                offsetVertex2MprimeP[g,gg,3] = deltaFunc(offsetVertex2Cyl[g,0,2])
                offsetVertex2MprimeN[g,gg,3] = deltaFunc(offsetVertex2Cyl[g,0,2])
            else:
                mPrev1P = offsetVertex1MprimeP[g,:,3][gg-1]
                rPrev1P = offsetVertex1MprimeP[g,:,1][gg-1]
                rCurr1P = offsetVertex1MprimeP[g,:,1][gg]
                zPrev1P = offsetVertex1MprimeP[g,:,2][gg-1]
                zCurr1P = offsetVertex1MprimeP[g,:,2][gg]
                offsetVertex1MprimeP[g,gg,3] = mPrev1P + ((2/(rCurr1P + rPrev1P)) * np.sqrt((rCurr1P-rPrev1P)**2 + (zCurr1P-zPrev1P)**2))
                mPrev2P = offsetVertex2MprimeP[g,:,3][gg-1]
                rPrev2P = offsetVertex2MprimeP[g,:,1][gg-1]
                rCurr2P = offsetVertex2MprimeP[g,:,1][gg]
                zPrev2P = offsetVertex2MprimeP[g,:,2][gg-1]
                zCurr2P = offsetVertex2MprimeP[g,:,2][gg]
                offsetVertex2MprimeP[g,gg,3] = mPrev2P + ((2/(rCurr2P + rPrev2P)) * np.sqrt((rCurr2P-rPrev2P)**2 + (zCurr2P-zPrev2P)**2))
        
                mPrev1 = offsetVertex1MprimeN[g,:,3][gg-1]
                rPrev1 = offsetVertex1MprimeN[g,:,1][gg-1]
                rCurr1 = offsetVertex1MprimeN[g,:,1][gg]
                zPrev1 = offsetVertex1MprimeN[g,:,2][gg-1]
                zCurr1 = offsetVertex1MprimeN[g,:,2][gg]
                offsetVertex1MprimeN[g,gg,3] = mPrev1 + ((2/(rCurr1 + rPrev1)) * np.sqrt((rCurr1-rPrev1)**2 + (zCurr1-zPrev1)**2))
                mPrev2 = offsetVertex2MprimeN[g,:,3][gg-1]
                rPrev2 = offsetVertex2MprimeN[g,:,1][gg-1]
                rCurr2 = offsetVertex2MprimeN[g,:,1][gg]
                zPrev2 = offsetVertex2MprimeN[g,:,2][gg-1]
                zCurr2 = offsetVertex2MprimeN[g,:,2][gg]
                offsetVertex2MprimeN[g,gg,3] = mPrev2 + ((2/(rCurr2 + rPrev2)) * np.sqrt((rCurr2-rPrev2)**2 + (zCurr2-zPrev2)**2)) 
            if gg == 1:
                offsetVertex1Mprime[g,gg,:] = offsetVertex1MprimeP[g,gg,:]  
                offsetVertex2Mprime[g,gg,:] = offsetVertex2MprimeP[g,gg,:]  
            elif gg == 2:
                offsetVertex1Mprime[g,gg,:] = offsetVertex1MprimeN[g,gg-1,:]  
                offsetVertex2Mprime[g,gg,:] = offsetVertex2MprimeN[g,gg-1,:] 
            offsetVertex1Mprime[g,3,:] = offsetVertex1MprimeP[g,gg,:]  
            offsetVertex2Mprime[g,3,:] = offsetVertex2MprimeP[g,gg,:]    
            # count +=1  
    
    #%% Now working in 2D mprime-theta
    blade1P2D = np.zeros([newNsection,bladeRes,2]) #mprime theta
    blade1N2D = np.zeros([newNsection,bladeRes,2])
    blade2P2D = np.zeros([newNsection,bladeRes,2])
    blade2N2D = np.zeros([newNsection,bladeRes,2])
    offsetBlade12D = np.zeros([newNsection, 4, 2])
    offsetBlade22D = np.zeros([newNsection, 4, 2])
    upstream2D = np.zeros([newNsection,res,2])
    dwstream2D = np.zeros([newNsection,res,2])
    allLEBlade12D = np.zeros([newNsection,2])
    allLEBlade22D = np.zeros([newNsection,2])
    allTEBlade12D = np.zeros([newNsection,2])
    allTEBlade22D = np.zeros([newNsection,2])
    
    for h in range(newNsection):
        blade1P2D[h,:,:] = np.column_stack((blade1PMprime[h,:,3], blade1PMprime[h,:,0]))
        blade1N2D[h,:,:] = np.column_stack((blade1NMprime[h,:,3], blade1NMprime[h,:,0]))
        blade2P2D[h,:,:] = np.column_stack((blade2PMprime[h,:,3], blade2PMprime[h,:,0]))
        blade2N2D[h,:,:] = np.column_stack((blade2NMprime[h,:,3], blade2NMprime[h,:,0]))   
        upstream2D[h,:,:] = np.column_stack((upstreamMprime[h,:,3], upstreamMprime[h,:,0]))
        dwstream2D[h,:,:] = np.column_stack((dwstreamMprime[h,:,3], dwstreamMprime[h,:,0]))
        offsetBlade12D[h,:,:] = np.column_stack((offsetVertex1Mprime[h,:,3], offsetVertex1Mprime[h,:,0]))
        offsetBlade22D[h,:,:] = np.column_stack((offsetVertex2Mprime[h,:,3], offsetVertex2Mprime[h,:,0]))
        allLEBlade12D[h] = blade1P2D[h][0]
        allLEBlade22D[h] = blade2N2D[h][0]
        allTEBlade12D[h] = blade1P2D[h][::-1][0]
        allTEBlade22D[h] = blade2N2D[h][::-1][0] 

    return (blade1PMprime, blade1NMprime, blade2PMprime, blade2NMprime,
            upstreamMprime, dwstreamMprime, offsetUpstreamMprime,
            offsetDwstreamMprime, offsetVertex1Mprime, offsetVertex2Mprime,
            blade1P2D, blade1N2D, blade2P2D, blade2N2D, upstream2D,
            dwstream2D, offsetBlade12D, offsetBlade22D,
            allLEBlade12D, allLEBlade22D, allTEBlade12D, allTEBlade22D)


def defineExt(blade1pmpt, blade1nmpt, blade2pmpt, blade2nmpt, offset1mpt, offset2mpt, LE1mpt, LE2mpt, TE1mpt, TE2mpt, upstreamMprime, downstreamMprime, offsetUpstreamMprime, offsetDownstreamMprime, angleLE1, angleTE1, angleLE2, angleTE2, res):
    # Define "optimal" upstream/downstream extensions for both sides of the passage
    # inputs: blade sides in mpt, offset vertices in mpt, LEs/TEs, upstream/downstreams, angles, resolution
    # outputs: blade1 upstream/downstream, blade2 upstream/downstream

    newNsection = blade1pmpt.shape[0]
    nl = 2

    # JD: code below pasted from prior script

    # Set up arrays to hold up/downstream extensions
    
    upstreamCamber1 = np.zeros([newNsection,res,2]) # upstream line for blade1
    upstreamCamber2 = np.zeros([newNsection,res,2]) # upstream line for blade1
    downstreamCamber1 = np.zeros([newNsection,res,2]) # downstream line for blade1
    downstreamCamber2 = np.zeros([newNsection,res,2]) # downstream line for blade1
    
    
    upstreamExtnCamber1 = np.zeros([newNsection,nl+1,2])
    upstreamExtnCamber2 = np.zeros([newNsection,nl+1,2])
    downstreamExtnCamber1 = np.zeros([newNsection,nl+1,2]) #Default downsteam line extension 
    downstreamExtnCamber2 = np.zeros([newNsection,nl+1,2]) #Default downsteam line extension 
    
    upstreamExtnCamber1Adj = np.zeros([newNsection,nl+1,2]) #Temporary storage 
    upstreamExtnCamber2Adj = np.zeros([newNsection,nl+1,2]) #Temporary storage 
    downstreamExtnCamber1Adj = np.zeros([newNsection,nl+1,2]) #Temporary storage 
    downstreamExtnCamber2Adj = np.zeros([newNsection,nl+1,2]) #Temporary storage 
    
    # Loop to get the m'-theta coordinates of the offset midchord points
    midchordP1 = np.zeros([newNsection,2])
    midchordN1 = np.zeros([newNsection,2])
    midchordP2 = np.zeros([newNsection,2])
    midchordN2 = np.zeros([newNsection,2])
    for i in range(newNsection):
        blade1 = np.concatenate((blade1pmpt[i][:-1], blade1nmpt[i][::-1])) #Joing both surfaces to create a closed loop
        distLE1 = mf.dist2D(offset1mpt[i][0,0], offset1mpt[i][0,1], LE1mpt[i][0], LE2mpt[i][1]) #compute the distance of offsetLE to bladeLE
        distTE1 = mf.dist2D(offset1mpt[i][-1,0], offset1mpt[i][-1,1], TE1mpt[i][0], TE1mpt[i][1]) #compute the distance between offsetTE to bladeTE
        midLine1 = np.vstack((offset1mpt[i][1], offset1mpt[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
        nInterX, pInterX = mf.TwoLinesIntersect(midLine1, blade1) #Find the intersections of the line on the blade curve at both pressure and suction side 
        if nInterX[1] > pInterX[1]:
            nInterX, pInterX = pInterX, nInterX
        # nInterX[0] = pInterX[0] = offset1mpt[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
        distN1 = mf.dist2D(nInterX[0], nInterX[1], offset1mpt[i][2,0], offset1mpt[i][2,1])
        distP1 = mf.dist2D(pInterX[0], pInterX[1], offset1mpt[i][1,0], offset1mpt[i][1,1])
        dist1 = np.array([distLE1, distP1, distTE1])

        # Do the same for blade2
        blade2 = np.concatenate((blade2pmpt[i][:-1], blade2nmpt[i][::-1])) #Joing both surfaces to create a closed loop
        distLE2 = mf.dist2D(offset2mpt[i][0,0], offset2mpt[i][0,1], LE2mpt[i][0], LE2mpt[i][1]) #compute the distance of offsetLE to bladeLE
        distTE2 = mf.dist2D(offset2mpt[i][-1,0], offset2mpt[i][-1,1], TE2mpt[i][0], TE2mpt[i][1]) #compute the distance between offsetTE to bladeTE
        midLine2 = np.vstack((offset2mpt[i][1], offset2mpt[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
        n2InterX, p2InterX = mf.TwoLinesIntersect(midLine2, blade2) #Find the intersections of the line on the blade curve at both pressure and suction side 
        if n2InterX[1] > p2InterX[1]:
            n2InterX, p2InterX = p2InterX, n2InterX
        # n2InterX[0] = p2InterX[0] = offset2mpt[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
        distN2 = mf.dist2D(n2InterX[0], n2InterX[1], offset2mpt[i][2,0], offset2mpt[i][2,1])
        distP2 = mf.dist2D(p2InterX[0], p2InterX[1], offset2mpt[i][1,0], offset2mpt[i][1,1])
        dist2 = np.array([distLE2, distN2, distTE2])

        center = mf.MidPts(np.vstack((nInterX, p2InterX))) #this is the mid point between the midpoints on the suction side of blade1 and pressure side of blade2
        midchordP1[i] = pInterX
        midchordN1[i] = nInterX

        midchordP2[i] = p2InterX
        midchordN2[i] = n2InterX   

    # Loop to initialize the upstream/downstream extension directions
    for m in range(newNsection):
        lePtBlade1 = LE1mpt[m]  # LE of the high blade (low)
        lePtBlade2 = LE2mpt[m]  # LE of the low blade (high)
    
        midPt1 = angle_bisector_line(midchordN1[m],lePtBlade1, midchordP1[m]) #Compute the angle between SS,LE,PS
        leLine1Slope = compute_bisector_slope(midchordN1[m],lePtBlade1, midchordP1[m]) #determine the slope of the bisector
        tanLine1LEInterX = lePtBlade1[1] - leLine1Slope*lePtBlade1[0]  #Trying to get the equation of the line
        tanLine1LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*leLine1Slope + tanLine1LEInterX], lePtBlade1)) 
        lePtCam1 = offset1mpt[m][0]
        upLine1LEInterX = lePtCam1[1] - np.tan(angleLE1[m])*lePtCam1[0] 
        upLine1LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*np.tan(angleLE1[m]) + upLine1LEInterX], lePtCam1, lePtBlade1))
    
        midPt2 = angle_bisector_line(midchordN2[m],lePtBlade2, midchordP2[m])
        leLine2Slope = compute_bisector_slope(midchordN2[m],lePtBlade2, midchordP2[m])
        tanLine2LEInterX = lePtBlade2[1] - leLine2Slope*lePtBlade2[0] 
        tanLine2LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*leLine2Slope + tanLine2LEInterX],lePtBlade2)) 
        lePtCam2 = offset2mpt[m][0]
        upLine2LEInterX = lePtCam2[1] - np.tan(angleLE2[m])*lePtCam2[0] 
        upLine2LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*np.tan(angleLE2[m]) + upLine2LEInterX], lePtCam2, lePtBlade2))
     
        angle1Deviation = np.deg2rad(mf.find_angle(upLine1LE[0], lePtCam1, lePtCam2)) #This determines the angle between the tangent and line connecting LE pts on blade 1 and 2
        upstream1 = np.linspace(upLine1LE[0], upLine1LE[-1], res-1)
        upstream1 = insertPoint(upstream1, upLine1LE[1])
        upstreamCamber1[m] = upstream1
        upstreamExtnCamber1[m] = np.vstack((upLine1LE[0], lePtCam1, lePtBlade1))
        upstreamExtnCamber1Adj[m] = upstreamExtnCamber1[m]
        
        angle2Deviation = np.deg2rad(mf.find_angle(upLine2LE[0], lePtCam2, lePtCam1)) #This determines the angle between the tangent and line connecting LE pts on blade 1 and 2
        upstream2 = np.linspace(upLine2LE[0], upLine2LE[-1], res-1)
        upstream2 = insertPoint(upstream2,upLine2LE[1] )
        upstreamCamber2[m] = upstream2
        upstreamExtnCamber2[m] = np.vstack((upLine2LE[0], lePtCam2, lePtBlade2))  
        upstreamExtnCamber2Adj[m] = upstreamExtnCamber2[m]
        
        #Now do the same for TE
        tePtBlade1 = TE1mpt[m]
        tePtBlade2 = TE2mpt[m]
        
        midPtTE1 = angle_bisector_line(midchordN1[m],tePtBlade1, midchordP1[m])
        teLine1Slope = compute_bisector_slope(midchordN1[m],tePtBlade1, midchordP1[m])
        tanLine1TEInterX = tePtBlade1[1] - teLine1Slope*tePtBlade1[0] 
        tanLine1TE = np.vstack(([downstreamMprime[m,-1,3], downstreamMprime[m,-1,3]*teLine1Slope + tanLine1TEInterX], tePtBlade1))
        tePtCam1 = offset1mpt[m][-1]
        dwLine1TEInterX = tePtCam1[1] - np.tan(angleTE1[m])*tePtCam1[0]
        dwLine1TE = np.vstack(([downstreamMprime[m,-1,3], downstreamMprime[m,-1,3]*np.tan(angleTE1[m]) + dwLine1TEInterX], tePtCam1, tePtBlade1))
      
        midPtTE2 = angle_bisector_line(midchordN2[m],tePtBlade2, midchordP2[m])
        teLine2Slope = compute_bisector_slope(midchordN2[m],tePtBlade2, midchordP2[m])
        tanLine2TEInterX = tePtBlade2[1] - teLine2Slope*tePtBlade2[0] 
        tanLine2TE = np.vstack(([downstreamMprime[m,-1,3], downstreamMprime[m,-1,3]*teLine2Slope + tanLine2TEInterX], tePtBlade2))
        tePtCam2 = offset2mpt[m][-1]
        dwLine2TEInterX = tePtCam2[1] - np.tan(angleTE2[m])*tePtCam2[0]
        dwLine2TE = np.vstack(([downstreamMprime[m,-1,3], downstreamMprime[m,-1,3]*np.tan(angleTE2[m]) + dwLine2TEInterX], tePtCam2, tePtBlade2))
    
        angle1DeviationTE = np.deg2rad(mf.find_angle(dwLine1TE[0], tePtCam1, tePtCam2)) #This determines the angle between the tangent and line connecting LE pts on blade 1 and 2
        downstream1 = np.linspace(dwLine1TE[-1], dwLine1TE[0], res-1)
        downstream1 = insertPoint(downstream1, dwLine1TE[1])
        downstreamCamber1[m] = downstream1
        downstreamExtnCamber1[m] = np.vstack(( dwLine1TE[0], tePtCam1, tePtBlade1))
        downstreamExtnCamber1Adj[m] = downstreamExtnCamber1[m]

        angle2DeviationTE = np.deg2rad(mf.find_angle(dwLine2TE[0], tePtCam2, tePtCam1)) #This determines the angle between the tangent and line connecting LE pts on blade 1 and 2
        downstream2 = np.linspace(dwLine2TE[-1], dwLine2TE[0], res-1)
        downstream2 = insertPoint(downstream2,dwLine2TE[1])
        downstreamCamber2[m] = downstream2
        downstreamExtnCamber2[m] = np.vstack((dwLine2TE[0], tePtCam2, tePtBlade2))
        downstreamExtnCamber2Adj[m] = downstreamExtnCamber2[m]

    deltaMprimeLE1 = np.zeros(newNsection)
    alphaLE1 = np.zeros(newNsection)
    deltaMprimeLE2 = np.zeros(newNsection)
    alphaLE2 = np.zeros(newNsection)
    deltaMprimeTE1 = np.zeros(newNsection)
    alphaTE1 = np.zeros(newNsection)
    deltaMprimeTE2 = np.zeros(newNsection)
    alphaTE2 = np.zeros(newNsection)
    for n in range(newNsection):
        deltaMprimeLE1[n] = upstreamExtnCamber1[n,0,0] - upstreamExtnCamber1[n,-1,0]
        alphaLE1[n] = np.deg2rad(mf.find_angle(upstreamExtnCamber1[n,0], upstreamExtnCamber1[n,-1], np.array([upstreamExtnCamber1[n,-1,0]-res, upstreamExtnCamber1[n,-1,1]])))
        deltaMprimeLE2[n] = upstreamExtnCamber2[n,0,0] - upstreamExtnCamber2[n,-1,0]
        alphaLE2[n] = np.deg2rad(mf.find_angle(upstreamExtnCamber2[n,0], upstreamExtnCamber2[n,-1], np.array([upstreamExtnCamber2[n,-1,0]-res, upstreamExtnCamber2[n,-1,1]])))

        deltaMprimeTE1[n] = downstreamExtnCamber1[n,0,0] - downstreamExtnCamber1[n,-1,0]
        alphaTE1[n] = np.deg2rad(mf.find_angle(downstreamExtnCamber1[n,0], downstreamExtnCamber1[n,-1], np.array([downstreamExtnCamber1[n,-1,0]+res, downstreamExtnCamber1[n,-1,1]])))
        deltaMprimeTE2[n] = downstreamExtnCamber2[n,0,0] - downstreamExtnCamber2[n,-1,0]
        alphaTE2[n] = np.deg2rad(mf.find_angle(downstreamExtnCamber2[n,0], downstreamExtnCamber2[n,-1], np.array([downstreamExtnCamber2[n,-1,0]+res, downstreamExtnCamber2[n,-1,1]])))

    deltaThetaLE1 = fsolve(dy_dx, x0=0, args=(deltaMprimeLE1, alphaLE1))
    deltaThetaLE2 = fsolve(dy_dx, x0=0, args=(deltaMprimeLE2, alphaLE2))
    deltaThetaTE1 = fsolve(dy_dx, x0=0, args=(deltaMprimeTE1, alphaTE1))
    deltaThetaTE2 = fsolve(dy_dx, x0=0, args=(deltaMprimeTE2, alphaTE2))

    #%% Define final extensions
    thetaInlet1 = upstreamExtnCamber1[:,-1,1] - deltaThetaLE1
    thetaInlet2 = upstreamExtnCamber2[:,-1,1] - deltaThetaLE2
    thetaOutlet1 = downstreamExtnCamber1[:,-1,1] - deltaThetaTE1
    thetaOutlet2 = downstreamExtnCamber2[:,-1,1] - deltaThetaTE2
    upstreamExtnCamber1Adj[:,0,1] = thetaInlet1
    upstreamExtnCamber2Adj[:,0,1] = thetaInlet2
    downstreamExtnCamber1Adj[:,0,1] = thetaOutlet1
    downstreamExtnCamber2Adj[:,0,1] = thetaOutlet2

    for nn in range(newNsection):
        leFunc1 = interp1d(upstreamExtnCamber1Adj[nn,:,0], upstreamExtnCamber1Adj[nn,:,1])
        upstreamCamber1[nn,:,1] = leFunc1(upstreamCamber1[nn,:,0])
        leFunc2 = interp1d(upstreamExtnCamber2Adj[nn,:,0], upstreamExtnCamber2Adj[nn,:,1])
        upstreamCamber2[nn,:,1] = leFunc2(upstreamCamber2[nn,:,0])
        teFunc1 = interp1d(downstreamExtnCamber1Adj[nn,:,0], downstreamExtnCamber1Adj[nn,:,1])
        downstreamCamber1[nn,:,1] = teFunc1(downstreamCamber1[nn,:,0])
        teFunc2 = interp1d(downstreamExtnCamber2Adj[nn,:,0], downstreamExtnCamber2Adj[nn,:,1])
        downstreamCamber2[nn,:,1] = teFunc2(downstreamCamber2[nn,:,0])  

    return upstreamCamber1, upstreamCamber2, downstreamCamber1, downstreamCamber2


# JD: UP TO HERE:
# This is the function we need to work on, where we
# need to use the same idea of how the arc length maps
# are generated to get the points on the offsets in
# such a way that when we do transfinite interpolation
# we get points that always lie inside the bounding
# curves.
def getCurvesAndMaps(offsetVertices2mpt, offsetVertices1mpt,
                     LE2mpt, LE1mpt, TE2mpt, TE1mpt,
                     blade2pmpt, blade2nmpt, blade1pmpt, blade1nmpt,
                     blade2UpExtmpt, blade2DnExtmpt, blade1UpExtmpt, blade1DnExtmpt,
                     angConstraintCurves, angConstraintOffsets, bladeRes, passageRes,
                     percentVal, percentValNonCutLE, percentValNonCutTE):
    # Create cross-passage and offset curves + do arclength mapping
    # between blade surfaces and offsets
    # Inputs: all in mpt: offset vertices, extensions, blades
    # Outputs: cross-passage curves, offset curves, arclength maps

    newNsection = offsetVertices1mpt.shape[0]
    LECurveRot = np.zeros([newNsection,passageRes,2]) #Section of the ellipse that defines the LE curve
    TECurveRot = np.zeros([newNsection,passageRes,2]) #section of the ellipse that defines the TE curve
    offsetSplinedBlade12D = np.zeros([newNsection, bladeRes, 2])
    offsetSplinedBlade22D = np.zeros([newNsection, bladeRes, 2])
    midchordPS1 = np.zeros([newNsection,2])
    midchordSS1 = np.zeros([newNsection,2])
    midchordPS2 = np.zeros([newNsection,2])
    midchordSS2 = np.zeros([newNsection,2])
    for m in range(newNsection):
        # Now we must figure out the angles/slopes/directions for the LE/TE
        # curves as well as the offsets.
        #
        # 1) get 3 points that define the subtended angle -- LE/TE, offset end,
        #    and the next point on the extension
        # 2) compute the subtended angle (in m'-theta)
        # 3) divide into 3 equal parts and get the angles of the cross-passage
        #    curve and the offset. Convert these to slopes.
        # 4) generate the cross-passage curves
        # 5) generate the offset curves
        # 6) do arclength mapping between blades and offsets

        # 1) get points
        offsetBlade1LEpt = offsetVertices1mpt[m][0]
        offsetBlade2LEpt = offsetVertices2mpt[m][0]
        offsetBlade1TEpt = offsetVertices1mpt[m][-1]
        offsetBlade2TEpt = offsetVertices2mpt[m][-1]
    
        lePtBlade1 = LE1mpt[m]
        lePtBlade2 = LE2mpt[m]
        tePtBlade1 = TE1mpt[m]
        tePtBlade2 = TE2mpt[m]
    
        farPtBlade1LE = blade1UpExtmpt[m][-3]
        farPtBlade2LE = blade2UpExtmpt[m][-3]
        farPtBlade1TE = blade1DnExtmpt[m][2]
        farPtBlade2TE = blade2DnExtmpt[m][2]
    
        # 2) get end angles to divvy up
        # angles are measured from the +m' direction, +angle is moving from the
        # positive m' axis towards the positive theta axis
        anglesLE1 = angles_between_points(farPtBlade1LE, offsetBlade1LEpt, lePtBlade1)
        anglesLE2 = angles_between_points(farPtBlade2LE, offsetBlade2LEpt, lePtBlade2)
        anglesTE1 = angles_between_points(farPtBlade1TE, offsetBlade1TEpt, tePtBlade1)
        anglesTE2 = angles_between_points(farPtBlade2TE, offsetBlade2TEpt, tePtBlade2)
    
        # 3) get the 2 new interior angles and convert to slopes
        inAnglesLE1 = np.linspace(anglesLE1[0], anglesLE1[1], 4)[1:3]
        inAnglesLE2 = np.linspace(anglesLE2[0], anglesLE2[1], 4)[1:3]
        inAnglesTE1 = np.linspace(anglesTE1[0], anglesTE1[1], 4)[1:3]
        inAnglesTE2 = np.linspace(anglesTE2[0], anglesTE2[1], 4)[1:3]
    
        # Alternating indexing takes care of the fact
        # we want complimentary angles on one side
        # compared to the other.
        #
        # Constrain angles to avoid cases that don't work
        # properly.
        # The offset constraints are really just because later,
        # cubic splines are constructed to make extra points on those
        # curves and those fail if the points aren't monotonic.
        # Not sure it would really be a problem for the actual grid.
        # But the problem for the cross-passage curves is real: they
        # must be outwardly-curved in order to achieve a proper block
        # structure.
        curveAngleLE1 = max(np.deg2rad(angConstraintCurves), inAnglesLE1[1])
        curveAngleLE2 = max(np.deg2rad(90+angConstraintCurves), inAnglesLE2[0])
        curveAngleTE1 = min(np.deg2rad(90+angConstraintCurves), inAnglesTE1[1])
        curveAngleTE2 = min(np.deg2rad(90-angConstraintCurves), inAnglesTE2[0])
    
        offsetAngleLE1 = max(np.deg2rad(90+angConstraintOffsets), inAnglesLE1[0])
        offsetAngleLE2 = min(np.deg2rad(90-angConstraintOffsets), inAnglesLE2[1])
        offsetAngleTE1 = min(np.deg2rad(90-angConstraintOffsets), inAnglesTE1[0])
        offsetAngleTE2 = max(np.deg2rad(90+angConstraintOffsets), inAnglesTE2[1])
    
        curveSlopeLE1 = np.tan(curveAngleLE1)
        curveSlopeLE2 = np.tan(curveAngleLE2)
        curveSlopeTE1 = np.tan(curveAngleTE1)
        curveSlopeTE2 = np.tan(curveAngleTE2)
    
        offsetSlopeLE1 = np.tan(offsetAngleLE1)
        offsetSlopeLE2 = np.tan(offsetAngleLE2)
        offsetSlopeTE1 = np.tan(offsetAngleTE1)
        offsetSlopeTE2 = np.tan(offsetAngleTE2)
    
        #4) define the cross-passage curves
        # Now using quadratic Bezier curves
        # create curves, densify
        crossPassageLEp1 = getBezierControlPoint(offsetBlade1LEpt, offsetBlade2LEpt, curveSlopeLE1, curveSlopeLE2)
        crossPassageTEp1 = getBezierControlPoint(offsetBlade1TEpt, offsetBlade2TEpt, curveSlopeTE1, curveSlopeTE2)
        # and m' as the dependent, but for later use we want to swap that back
        # the flipud is because the curve is constructed from low theta to high theta, but later
        # code expects it to go the other way.
        LECurve = quadratic_bezier_curve(p0=offsetBlade1LEpt, p1=crossPassageLEp1, p2=offsetBlade2LEpt)
        LECurveRot[m] = densifyCurve(LECurve, passageRes)
       
        TECurve = quadratic_bezier_curve(p0=offsetBlade1TEpt, p1=crossPassageTEp1, p2=offsetBlade2TEpt)
        TECurveRot[m] = densifyCurve(TECurve, passageRes)
    
        # Old Adekola comment: Please note at this point that the part of the code profile that lies in the domain is the SS for blade 1 and PS for blade2.
        i = m  # this was a separate loop from here down with different indexing, so to avoid changing all the code for now I just set i=m
        blade1 = np.concatenate((blade1pmpt[i][:-1], blade1nmpt[i][::-1])) #Joing both surfaces to create a closed loop
        distLE1 = mf.dist2D(offsetVertices1mpt[i][0,0], offsetVertices1mpt[i][0,1], LE1mpt[i][0], LE1mpt[i][1]) #compute the distance of offsetLE to bladeLE
        distTE1 = mf.dist2D(offsetVertices1mpt[i][-1,0], offsetVertices1mpt[i][-1,1], TE1mpt[i][0], TE1mpt[i][1]) #compute the distance between offsetTE to bladeTE
        midLine1 = np.vstack((offsetVertices1mpt[i][1], offsetVertices1mpt[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
        ssInterX, psInterX = mf.TwoLinesIntersect(midLine1, blade1) #Find the intersections of the line on the blade curve at both pressure and suction side 
        if ssInterX[1] > psInterX[1]:
            ssInterX, psInterX = psInterX, ssInterX
        # ssInterX[0] = psInterX[0] = offsetBlade12D[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
        distSS1 = mf.dist2D(ssInterX[0], ssInterX[1], offsetVertices1mpt[i][2,0], offsetVertices1mpt[i][2,1])
        distPS1 = mf.dist2D(psInterX[0], psInterX[1], offsetVertices1mpt[i][1,0], offsetVertices1mpt[i][1,1])
        dist1 = np.array([distLE1, distSS1, distTE1])
        
        #Do the same for blade2
        blade2 = np.concatenate((blade2pmpt[i][:-1], blade2nmpt[i][::-1])) #Joing both surfaces to create a closed loop
        distLE2 = mf.dist2D(offsetVertices2mpt[i][0,0], offsetVertices2mpt[i][0,1], LE2mpt[i][0], LE2mpt[i][1]) #compute the distance of offsetLE to bladeLE
        distTE2 = mf.dist2D(offsetVertices2mpt[i][-1,0], offsetVertices2mpt[i][-1,1], TE2mpt[i][0], TE2mpt[i][1]) #compute the distance between offsetTE to bladeTE
        midLine2 = np.vstack((offsetVertices2mpt[i][1], offsetVertices2mpt[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
        ss2InterX, ps2InterX = mf.TwoLinesIntersect(midLine2, blade2) #Find the intersections of the line on the blade curve at both pressure and suction side 
        if ss2InterX[1] > ps2InterX[1]:
            ss2InterX, ps2InterX = ps2InterX, ss2InterX
        # ss2InterX[0] = ps2InterX[0] = offsetBlade22D[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
        distSS2 = mf.dist2D(ss2InterX[0], ss2InterX[1], offsetVertices2mpt[i][2,0], offsetVertices2mpt[i][2,1])
        distPS2 = mf.dist2D(ps2InterX[0], ps2InterX[1], offsetVertices2mpt[i][1,0], offsetVertices2mpt[i][1,1])
        dist2 = np.array([distLE2, distPS2, distTE2])
        
        center = mf.MidPts(np.vstack((ssInterX, ps2InterX))) #this is the mid point between the midpoints on the suction side of blade1 and pressure side of blade2
    
        # This now uses quadratic Bezier curves
        # curve1 = createOffsetCurve(bladePr=blade1SS2D[i], offsetPt=offsetBlade12D[i], interX=ssInterX, dist=dist1, center=center, LEslope=offsetSlopeLE1, TEslope=offsetSlopeTE1, side='hi')
        curve1 = createOffsetCurve(bladePr=blade1nmpt[i], offsetPt=offsetVertices1mpt[i], interX=ssInterX, dist=dist1, center=center, LEslope=offsetSlopeLE1, TEslope=offsetSlopeTE1, side='hi')

        offsetSplinedBlade12D[i] = mf.densify_curve_robust(curve1, bladeRes)
        #densifyCurve(curve1, bladeRes, 'both')
        midchordPS1[i] = psInterX
        midchordSS1[i] = ssInterX

        # This now uses quadratic Bezier curves
        # curve2 = createOffsetCurve(bladePr=blade2PS2D[i], offsetPt=offsetBlade22D[i], interX=ps2InterX, dist=dist2, center=center, LEslope=offsetSlopeLE2, TEslope=offsetSlopeTE2, side='lo')
        curve2 = createOffsetCurve(bladePr=blade2pmpt[i], offsetPt=offsetVertices2mpt[i], interX=ps2InterX, dist=dist2, center=center, LEslope=offsetSlopeLE2, TEslope=offsetSlopeTE2, side='lo')
        
        offsetSplinedBlade22D[i] = mf.densify_curve_robust(curve2, bladeRes)
        #densifyCurve(curve2, bladeRes, 'both')
        midchordPS2[i] = ps2InterX
        midchordSS2[i] = ss2InterX   
    
        blade1Func = CubicSpline(blade1nmpt[i][:,0], blade1nmpt[i][:,1]) # Interpolation function for high theta blade
        blade2Func = CubicSpline(blade2pmpt[i][:,0], blade2pmpt[i][:,1]) # Interpolation function for low theta blade
        offset1Func = CubicSpline(offsetSplinedBlade12D[i][:,0], offsetSplinedBlade12D[i][:,1])
        offset2Func = CubicSpline(offsetSplinedBlade22D[i][:,0], offsetSplinedBlade22D[i][:,1])
    
        # 6) get arclength mapping for hub and casing sections
        if i == 0:
            #H is high, L is low, 1 is upstream and 2 is downstream. I am trying to replace naming convention from here
            mBladeH1 = cosineSpace(bladeRes+1, blade1nmpt[i][0,0], ssInterX[0]) #upstream portion of the high theta blade
            mBladeH2 = cosineSpace(bladeRes+1, ssInterX[0], blade1nmpt[i][-1,0]) #downstream portion of the high theta blade
            mBladeL1 = cosineSpace(bladeRes+1, blade2pmpt[i][0,0], ps2InterX[0]) #upstream portion of the low theta blade
            mBladeL2 = cosineSpace(bladeRes+1, ps2InterX[0], blade2pmpt[i][-1,0]) #downstream portion of the low theta blade 
            thBladeH1 = blade1Func(mBladeH1)
            thBladeH2 = blade1Func(mBladeH2)
            thBladeL1 = blade2Func(mBladeL1)
            thBladeL2 = blade2Func(mBladeL2)
            bladeH1Pt = np.column_stack((mBladeH1, thBladeH1))  #upstream portion of the high theta blade
            bladeL1Pt = np.column_stack((mBladeL1, thBladeL1))  #downstream portion of the high theta blade
            bladeH2Pt = np.column_stack((mBladeH2, thBladeH2))  #upstream portion of the low theta blade
            bladeL2Pt = np.column_stack((mBladeL2, thBladeL2))  #downstream portion of the low theta blade    
            mOffsetH1 = cosineSpace(bladeRes+1, curve1[0,0], ssInterX[0]) #upstream portion of the high theta blade
            mOffsetH2 = cosineSpace(bladeRes+1, ssInterX[0], curve1[-1,0]) #downstream portion of the high theta blade
            mOffsetL1 = cosineSpace(bladeRes+1, curve2[0,0], ps2InterX[0]) #upstream portion of the low theta blade
            mOffsetL2 = cosineSpace(bladeRes+1, ps2InterX[0], curve2[-1,0]) #downstream portion of the low theta blade 
            thOffsetH1 = offset1Func(mOffsetH1)
            thOffsetH2 = offset1Func(mOffsetH2)
            thOffsetL1 = offset2Func(mOffsetL1)
            thOffsetL2 = offset2Func(mOffsetL2)        
            offsetOrigH1Pt = np.column_stack((mOffsetH1, thOffsetH1)) #upstream portion of the high theta blade
            offsetOrigL1Pt = np.column_stack((mOffsetL1, thOffsetL1))  #downstream portion of the high theta blade
            offsetOrigH2Pt = np.column_stack((mOffsetH2, thOffsetH2)) #upstream portion of the low theta blade
            offsetOrigL2Pt = np.column_stack((mOffsetL2, thOffsetL2))  #downstream portion of the low theta blade         
            lowHub1Slopes, lowHub1MidPt = slopeAndMidPtsLoop(bladeL1Pt) # Get the normal slope and the midPoint on the blade surface
            lowHub2Slopes, lowHub2MidPt = slopeAndMidPtsLoop(bladeL2Pt)
            highHub1Slopes, highHub1MidPt = slopeAndMidPtsLoop(bladeH1Pt)
            highHub2Slopes, highHub2MidPt = slopeAndMidPtsLoop(bladeH2Pt)        
            offsetL1Pt = pointAtDistLoop(lowHub1MidPt, lowHub1Slopes, np.full(bladeRes,1.5*dist1.max()), 'PS')[:-1] # This is offseting the point on the blade surface in the normal direction at some made up dist
            offsetL2Pt = pointAtDistLoop(lowHub2MidPt, lowHub2Slopes, np.full(bladeRes,1.5*dist1.max()), 'PS')[:-1]
            offsetH1Pt = pointAtDistLoop(highHub1MidPt, highHub1Slopes, np.full(bladeRes,1.5*dist1.max()), 'SS')[:-1]
            offsetH2Pt = pointAtDistLoop(highHub2MidPt, highHub2Slopes, np.full(bladeRes,1.5*dist1.max()), 'SS')[:-1]        
            lowHub1OffsetPt = np.zeros((bladeRes+1, 2)) # lowHub1. 
            lowHub2OffsetPt = np.zeros((bladeRes+1, 2)) # lowHub2
            highHub1OffsetPt = np.zeros((bladeRes+1, 2)) # highHub1
            highHub2OffsetPt = np.zeros((bladeRes+1, 2)) #highHub2  
            for ii in range(bladeRes-1):
                # Connects the corresponding points to make a line
                lineH1 = np.vstack((bladeH1Pt[ii+1], offsetH1Pt[ii])) # high theta upstream
                lineL1 = np.vstack((bladeL1Pt[ii+1], offsetL1Pt[ii]))  # high theta downsteam
                lineH2 = np.vstack((bladeH2Pt[ii+1], offsetH2Pt[ii])) # low theta upstream
                lineL2 = np.vstack((bladeL2Pt[ii+1], offsetL2Pt[ii])) #low theta downstream        
                highHub1OffsetPt[ii+1] = mf.TwoLinesIntersect(lineH1, offsetSplinedBlade12D[i]) # Get the points of intersection on the offset curve 
                highHub2OffsetPt[ii+1] = mf.TwoLinesIntersect(lineH2, offsetSplinedBlade12D[i])
                lowHub1OffsetPt[ii+1] = mf.TwoLinesIntersect(lineL1, offsetSplinedBlade22D[i])
                lowHub2OffsetPt[ii+1] = mf.TwoLinesIntersect(lineL2, offsetSplinedBlade22D[i])
            highHub1OffsetPt[0] = curve1[0] #Insert the first point on the offset curve
            highHub1OffsetPt[-1] = offsetVertices1mpt[i][2] # insert the mid point of the offset curve 
            highHub2OffsetPt[0] = offsetVertices1mpt[i][2] #insert mid point of the offset curve as the last point
            highHub2OffsetPt[-1] = curve1[-1]
            
            lowHub1OffsetPt[0] = curve2[0]
            lowHub1OffsetPt[-1] = offsetVertices2mpt[i][1]
            lowHub2OffsetPt[0] = offsetVertices2mpt[i][1]  
            lowHub2OffsetPt[-1] = curve2[-1]
            
            #Now find the angle to determine the points to ignore
            
            lowHub1OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][0], bladeL1Pt[0], bladeL1Pt[1]) #Angle at LE
            lowHub1OffsetPtAngle2 =  mf.find_angle(offsetVertices2mpt[i][1], bladeL1Pt[-1], bladeL1Pt[-2]) #Angle at Mid
            lowHub2OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][-1], bladeL2Pt[-1], bladeL2Pt[-2]) #Angle at TE
            lowHub2OffsetPtAngle2 =  mf.find_angle(offsetVertices2mpt[i][1], bladeL2Pt[0], bladeL2Pt[1]) #Angle at Mid
            
            highHub1OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][0], bladeH1Pt[0], bladeH1Pt[1]) #Angle at LE
            highHub1OffsetPtAngle2 = mf.find_angle(offsetVertices1mpt[i][2], bladeH1Pt[-1], bladeH1Pt[-2]) #Angle at Mid
            highHub2OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][-1], bladeH2Pt[-1], bladeH2Pt[-2]) #Angle at TE
            highHub2OffsetPtAngle2 = mf.find_angle(offsetVertices1mpt[i][2], bladeH2Pt[0], bladeH2Pt[1]) #Angle at Mid    
            
            #Now decide on which offset to truncate points from based on angle less than 90 degrees
            percentLowH1 = np.array([(lowHub1OffsetPtAngle1 < 90),(lowHub1OffsetPtAngle2 < 90)])*percentVal
            lowHub1 = curveFrac(bladeL1Pt, offsetOrigL1Pt, lowHub1OffsetPt, percentLowH1).T
            lowHub1[0,1] = 0.0
            lowHub1 = cutArcLenMaps(lowHub1, lower_bound=percentValNonCutLE, upper_bound=1.0)
            percentLowH2 = np.array([(lowHub2OffsetPtAngle1 < 90),(lowHub2OffsetPtAngle2 < 90)])*percentVal
            lowHub2 = curveFrac(bladeL2Pt, offsetOrigL2Pt, lowHub2OffsetPt, percentLowH2).T
            lowHub2[0,1] = 0.0
            lowHub2 = cutArcLenMaps(lowHub2, lower_bound=0.0, upper_bound=1.0-percentValNonCutTE)
            percentHighH1 = np.array([(highHub1OffsetPtAngle1 < 90),(highHub1OffsetPtAngle2 < 90)])*percentVal
            highHub1 = curveFrac(bladeH1Pt, offsetOrigH1Pt, highHub1OffsetPt, percentHighH1).T
            highHub1[0,1] = 0.0
            highHub1 = cutArcLenMaps(highHub1, lower_bound=percentValNonCutLE, upper_bound=1.0)
            percentHighH2 = np.array([(highHub2OffsetPtAngle1 < 90),(highHub2OffsetPtAngle2 < 90)])*percentVal
            highHub2 = curveFrac(bladeH2Pt, offsetOrigH2Pt, highHub2OffsetPt, percentHighH2).T
            highHub2[0,1] = 0.0
            highHub2 = cutArcLenMaps(highHub2, lower_bound=0.0, upper_bound=1.0-percentValNonCutTE)
           
        elif i == newNsection - 1:
            mBladeH1 = cosineSpace(bladeRes+1, blade1nmpt[i][0,0], ssInterX[0]) #upstream portion of the high theta blade
            mBladeH2 = cosineSpace(bladeRes+1, ssInterX[0], blade1nmpt[i][-1,0]) #downstream portion of the high theta blade
            mBladeL1 = cosineSpace(bladeRes+1, blade2pmpt[i][0,0], ps2InterX[0]) #upstream portion of the low theta blade
            mBladeL2 = cosineSpace(bladeRes+1, ps2InterX[0], blade2pmpt[i][-1,0]) #downstream portion of the low theta blade 
            thBladeH1 = blade1Func(mBladeH1)
            thBladeH2 = blade1Func(mBladeH2)
            thBladeL1 = blade2Func(mBladeL1)
            thBladeL2 = blade2Func(mBladeL2)        
            bladeH1Pt = np.column_stack((mBladeH1, thBladeH1)) #upstream portion of the high theta blade
            bladeL1Pt = np.column_stack((mBladeL1, thBladeL1))  #downstream portion of the high theta blade
            bladeH2Pt = np.column_stack((mBladeH2, thBladeH2)) #upstream portion of the low theta blade
            bladeL2Pt = np.column_stack((mBladeL2, thBladeL2))  #downstream portion of the low theta blade  
            mOffsetH1 = cosineSpace(bladeRes+1, curve1[0,0], ssInterX[0]) #upstream portion of the high theta blade
            mOffsetH2 = cosineSpace(bladeRes+1, ssInterX[0], curve1[-1,0]) #downstream portion of the high theta blade
            mOffsetL1 = cosineSpace(bladeRes+1, curve2[0,0], ps2InterX[0]) #upstream portion of the low theta blade
            mOffsetL2 = cosineSpace(bladeRes+1, ps2InterX[0], curve2[-1,0]) #downstream portion of the low theta blade 
            thOffsetH1 = offset1Func(mOffsetH1)
            thOffsetH2 = offset1Func(mOffsetH2)
            thOffsetL1 = offset2Func(mOffsetL1)
            thOffsetL2 = offset2Func(mOffsetL2)        
            offsetOrigH1Pt = np.column_stack((mOffsetH1, thOffsetH1)) #upstream portion of the high theta blade
            offsetOrigL1Pt = np.column_stack((mOffsetL1, thOffsetL1))  #downstream portion of the high theta blade
            offsetOrigH2Pt = np.column_stack((mOffsetH2, thOffsetH2)) #upstream portion of the low theta blade
            offsetOrigL2Pt = np.column_stack((mOffsetL2, thOffsetL2))  #downstream portion of the low theta blade    
            
            lowCas1Slopes, lowCas1MidPt = slopeAndMidPtsLoop(bladeL1Pt) # Get the normal slope and the midPoint on the blade surface
            lowCas2Slopes, lowCas2MidPt = slopeAndMidPtsLoop(bladeL2Pt)
            highCas1Slopes, highCas1MidPt = slopeAndMidPtsLoop(bladeH1Pt)
            highCas2Slopes, highCas2MidPt = slopeAndMidPtsLoop(bladeH2Pt)
            offsetL1Pt = pointAtDistLoop(lowCas1MidPt, lowCas1Slopes, np.full(bladeRes,1.5*dist1.max()), 'PS')[:-1] # This is offseting the point on the blade surface in the normal direction at some made up dist
            offsetL2Pt = pointAtDistLoop(lowCas2MidPt, lowCas2Slopes, np.full(bladeRes,1.5*dist1.max()), 'PS')[:-1]
            offsetH1Pt = pointAtDistLoop(highCas1MidPt, highCas1Slopes, np.full(bladeRes,1.5*dist1.max()), 'SS')[:-1]
            offsetH2Pt = pointAtDistLoop(highCas2MidPt, highCas2Slopes, np.full(bladeRes,1.5*dist1.max()), 'SS')[:-1]        
            lowCas1OffsetPt = np.zeros((bladeRes+1, 2)) # lowCas1. 
            lowCas2OffsetPt = np.zeros((bladeRes+1, 2)) # lowCas2
            highCas1OffsetPt = np.zeros((bladeRes+1, 2)) # highCas1
            highCas2OffsetPt = np.zeros((bladeRes+1, 2)) #highCas2  
            for ii in range(bladeRes-1):
                # Connects the corresponding points to make a line
                lineH1 = np.vstack((bladeH1Pt[ii+1], offsetH1Pt[ii])) # high theta upstream
                lineL1 = np.vstack((bladeL1Pt[ii+1], offsetL1Pt[ii]))  # high theta downsteam
                lineH2 = np.vstack((bladeH2Pt[ii+1], offsetH2Pt[ii])) # low theta upstream
                lineL2 = np.vstack((bladeL2Pt[ii+1], offsetL2Pt[ii])) #low theta downstream        
                # JD: why do the front and back parts of the blade using the same "splined
                # blade" array? Don't we already have front and back part blade arrays?
                # I think this is PART of the problem, this is finding intersections that
                # end up on the wrong half of the offset, which is a problem
                # We get (0,0) return values when no intersection is found but we get intersections
                # in places we ought not to because of the use of the full offset
                highCas1OffsetPt[ii+1] = mf.TwoLinesIntersect(lineH1, offsetSplinedBlade12D[i]) # Get the points of intersection on the offset curve 
                highCas2OffsetPt[ii+1] = mf.TwoLinesIntersect(lineH2, offsetSplinedBlade12D[i])
                lowCas1OffsetPt[ii+1] = mf.TwoLinesIntersect(lineL1, offsetSplinedBlade22D[i])
                lowCas2OffsetPt[ii+1] = mf.TwoLinesIntersect(lineL2, offsetSplinedBlade22D[i])
            highCas1OffsetPt[0] = curve1[0] #Insert the first point on the offset curve
            highCas1OffsetPt[-1] = offsetVertices1mpt[i][2] # insert the mid point of the offset curve 
            highCas2OffsetPt[0] = offsetVertices1mpt[i][2] #insert mid point of the offset curve as the last point
            highCas2OffsetPt[-1] = curve1[-1]
            
            lowCas1OffsetPt[0] = curve2[0]
            lowCas1OffsetPt[-1] = offsetVertices2mpt[i][1]
            lowCas2OffsetPt[0] = offsetVertices2mpt[i][1]  
            lowCas2OffsetPt[-1] = curve2[-1]
            
            #Now find the angle to determine the points to ignore
            
            lowCas1OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][0], bladeL1Pt[0], bladeL1Pt[1]) #Angle at LE
            lowCas1OffsetPtAngle2 =  mf.find_angle(offsetVertices2mpt[i][1], bladeL1Pt[-1], bladeL1Pt[-2]) #Angle at Mid
            lowCas2OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][-1], bladeL2Pt[-1], bladeL2Pt[-2]) #Angle at TE
            lowCas2OffsetPtAngle2 =  mf.find_angle(offsetVertices2mpt[i][1], bladeL2Pt[0], bladeL2Pt[1]) #Angle at Mid
    
            highCas1OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][0], bladeH1Pt[0], bladeH1Pt[1]) #Angle at LE
            highCas1OffsetPtAngle2 = mf.find_angle(offsetVertices1mpt[i][2], bladeH1Pt[-1], bladeH1Pt[-2]) #Angle at Mid
            highCas2OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][-1], bladeH2Pt[-1], bladeH2Pt[-2]) #Angle at TE
            highCas2OffsetPtAngle2 = mf.find_angle(offsetVertices1mpt[i][2], bladeH2Pt[0], bladeH2Pt[1]) #Angle at Mid   
            
            percentLowH1 = np.array([(lowCas1OffsetPtAngle1 < 90),(lowCas1OffsetPtAngle2 < 90)])*percentVal
            lowCas1 = curveFrac(bladeL1Pt, offsetOrigL1Pt, lowCas1OffsetPt, percentLowH1).T
            lowCas1[0,1] = 0.0
            lowCas1 = cutArcLenMaps(lowCas1, lower_bound=percentValNonCutLE, upper_bound=1.0)
            percentLowH2 = np.array([(lowCas2OffsetPtAngle1 < 90),(lowCas2OffsetPtAngle2 < 90)])*percentVal
            lowCas2 = curveFrac(bladeL2Pt, offsetOrigL2Pt, lowCas2OffsetPt, percentLowH2).T
            lowCas2[0,1] = 0.0
            lowCas2 = cutArcLenMaps(lowCas2, lower_bound=0.0, upper_bound=1.0-percentValNonCutTE)
            percentHighH1 = np.array([(highCas1OffsetPtAngle1 < 90),(highCas1OffsetPtAngle2 < 90)])*percentVal
            highCas1 = curveFrac(bladeH1Pt, offsetOrigH1Pt, highCas1OffsetPt, percentHighH1).T
            highCas1[0,1] = 0.0
            highCas1 = cutArcLenMaps(highCas1, lower_bound=percentValNonCutLE, upper_bound=1.0)
            percentHighH2 = np.array([(highCas2OffsetPtAngle1 < 90),(highCas2OffsetPtAngle2 < 90)])*percentVal
            highCas2 = curveFrac(bladeH2Pt, offsetOrigH2Pt, highCas2OffsetPt, percentHighH2).T       
            highCas2[0,1] = 0.0 
            highCas2 = cutArcLenMaps(highCas2, lower_bound=0.0, upper_bound=1.0-percentValNonCutTE)

            blade1hubUpArclenmap = lowHub1
            blade1hubDnArclenmap = lowHub2
            blade1casUpArclenmap = lowCas1
            blade1casDnArclenmap = lowCas2
            blade2hubUpArclenmap = highHub1
            blade2hubDnArclenmap = highHub2
            blade2casUpArclenmap = highCas1
            blade2casDnArclenmap = highCas2

    """
    # Finally, resample the offset curves using the arclength maps
    # The objective is to make it so that lines going from each point on the
    # blade surfaces to the corresponding point (equal index) on the offset
    # surfaces will be quasi-normal to the blade surface. The arclength maps
    # just created can help with this, though there are a few extra steps
    # involved:
    #
    # 1. arc length maps are split between front and back halfs, and
    # are only defined on the hub and casing. So, a first step is to
    # combine the arc length distributions for blades and offsets,
    # but as these are fractions, we need to go back to actual
    # arc lengths to help us. Need to know where the split happens...
    midchordmprime = np.average((midchordSS1[:, 0], midchordSS2[:, 0], midchordPS1[:, 0], midchordPS2[:, 0]), axis=0)
    # this varies by section, but is constant for both blades and offsets
    # at each section.

    # this process doesn't need to be exact -- find closest point

    b2uS = np.zeros(newNsection)
    b1uS = np.zeros(newNsection)
    o2uS = np.zeros(newNsection)
    o1uS = np.zeros(newNsection)
    
    b2dS = np.zeros(newNsection)
    b1dS = np.zeros(newNsection)
    o2dS = np.zeros(newNsection)
    o1dS = np.zeros(newNsection)

    # Get arclength of each half of every curve
    for i in range(newNsection):
        # Find the index of the minimum difference
        b2splitIndex = np.abs(blade2pmpt[i, :, 0] - midchordmprime[i]).argmin()
        b1splitIndex = np.abs(blade2pmpt[i, :, 0] - midchordmprime[i]).argmin()
        o2splitIndex = np.abs(offsetSplinedBlade22D[i, :, 0] - midchordmprime[i]).argmin()
        o1splitIndex = np.abs(offsetSplinedBlade12D[i, :, 0] - midchordmprime[i]).argmin()
        # Get total arclength of each part
        b2uS[i] = np.sum(np.linalg.norm(np.diff(blade2pmpt[i, 0:b2splitIndex, :], axis=0), axis=1))
        b1uS[i] = np.sum(np.linalg.norm(np.diff(blade2pmpt[i, 0:b1splitIndex, :], axis=0), axis=1))
        o2uS[i] = np.sum(np.linalg.norm(np.diff(offsetSplinedBlade22D[i, 0:o2splitIndex, :], axis=0), axis=1))
        o1uS[i] = np.sum(np.linalg.norm(np.diff(offsetSplinedBlade12D[i, 0:o1splitIndex, :], axis=0), axis=1))
        
        b2dS[i] = np.sum(np.linalg.norm(np.diff(blade2pmpt[i, b2splitIndex:, :], axis=0), axis=1))
        b1dS[i] = np.sum(np.linalg.norm(np.diff(blade2pmpt[i, b1splitIndex:, :], axis=0), axis=1))
        o2dS[i] = np.sum(np.linalg.norm(np.diff(offsetSplinedBlade22D[i, o2splitIndex:, :], axis=0), axis=1))
        o1dS[i] = np.sum(np.linalg.norm(np.diff(offsetSplinedBlade12D[i, o1splitIndex:, :], axis=0), axis=1))

    # renormalize arclength fractions (hub and casing)
    b2hubMap = np.concatenate((blade2hubUpArclenmap[:, 0]*b2uS[0], b2uS[0]+blade2hubDnArclenmap[:, 0]*b2dS[0])) / (b2uS[0] + b2dS[0])
    b1hubMap = np.concatenate((blade1hubUpArclenmap[:, 0]*b1uS[0], b1uS[0]+blade1hubDnArclenmap[:, 0]*b1dS[0])) / (b1uS[0] + b1dS[0])
    o2hubMap = np.concatenate((blade2hubUpArclenmap[:, 1]*o2uS[0], o2uS[0]+blade2hubDnArclenmap[:, 1]*o2dS[0])) / (o2uS[0] + o2dS[0])
    o1hubMap = np.concatenate((blade1hubUpArclenmap[:, 1]*o1uS[0], o1uS[0]+blade1hubDnArclenmap[:, 1]*o1dS[0])) / (o1uS[0] + o1dS[0])

    b2casMap = np.concatenate((blade2casUpArclenmap[:, 0]*b2uS[-1], b2uS[-1]+blade2casDnArclenmap[:, 0]*b2dS[-1])) / (b2uS[-1] + b2dS[-1])
    b1casMap = np.concatenate((blade1casUpArclenmap[:, 0]*b1uS[-1], b1uS[-1]+blade1casDnArclenmap[:, 0]*b1dS[-1])) / (b1uS[-1] + b1dS[-1])
    o2casMap = np.concatenate((blade2casUpArclenmap[:, 1]*o2uS[-1], o2uS[-1]+blade2casDnArclenmap[:, 1]*o2dS[-1])) / (o2uS[-1] + o2dS[-1])
    o1casMap = np.concatenate((blade1casUpArclenmap[:, 1]*o1uS[-1], o1uS[-1]+blade1casDnArclenmap[:, 1]*o1dS[-1])) / (o1uS[-1] + o1dS[-1])

    # make sure all have same number of points
    b2hubMap = mf.densify_curve_robust(b2hubMap, bladeRes)
    b1hubMap = mf.densify_curve_robust(b1hubMap, bladeRes)
    o2hubMap = mf.densify_curve_robust(o2hubMap, bladeRes)
    o1hubMap = mf.densify_curve_robust(o1hubMap, bladeRes)

    b2casbMap = mf.densify_curve_robust(b2casMap, bladeRes)
    b1casMap = mf.densify_curve_robust(b1casMap, bladeRes)
    o2casMap = mf.densify_curve_robust(o2casMap, bladeRes)
    o1casMap = mf.densify_curve_robust(o1casMap, bladeRes)

    # Interpolate offset points at ALL sections to follow
    # new arc length maps. This is a 2-step interpolation.
    # First, interpolate the maps between hub and casing
    # to the current section. This can be done crudely by
    # section number rather than distance since in m'-
    # theta, we have no idea how close/far apart the sections
    # are.
    # Second, once we have a "local" map, interpolate (linear)
    # the existing points to have the same number of points,
    # but following the specified arclength fraction distribution.
    for i in range(newNsection):
        o2locMap = ((newNsection - 1 - i)*o2hubMap + i*o2casMap)/(newNsection - 1)
        o1locMap = ((newNsection - 1 - i)*o1hubMap + i*o1casMap)/(newNsection - 1)

        offsetSplinedBlade22D[i] = matchArcLengthFractions2(o2locMap, offsetSplinedBlade22D[i])
        offsetSplinedBlade12D[i] = matchArcLengthFractions2(o1locMap, offsetSplinedBlade12D[i])


    # Relevant variables besides arc length maps:
    # offsetSplinedBlade22D -- LOW THETA OFFSET
    # offsetSplinedBlade12D -- HIGH THETA OFFSET
    # Note: variable names above and below are opposite of 
    # what actual data contains.
    # blade2pmpt -- LOW THETA BLADE
    # blade1nmpt -- HIGH THETA BLADE
    

    bob = alice
    """
    return (LECurveRot, TECurveRot, offsetSplinedBlade22D, offsetSplinedBlade12D,
            blade1hubUpArclenmap, blade1hubDnArclenmap,
            blade1casUpArclenmap, blade1casDnArclenmap,
            blade2hubUpArclenmap, blade2hubDnArclenmap,
            blade2casUpArclenmap, blade2casDnArclenmap)


def mptToCyl(arrmpt, upstreamMprime, blade1PMprime, downstreamMprime, hub, cas):
    """
    Take an array in m'-theta and convert to cylindrical coordinates
    """
    newNsection = arrmpt.shape[0]

    # steps:
    # 1. get a continuous array from inlet to outlet of (r, z, m')
    # 2. construct a cubic spline interpolant with m' data as the 'x' and z data as 'y'
    # 3. use interpolant to create z data for arr from arr's m' data
    # 4. construct a cubic spline interpolant with z data as the 'x' and r data as 'y'; use hub/cas data for extremes
    # 5. user interpolantt o create r data for arr from interpoalted z data
    # 6. assemble into arrCyl (theta, r, z)

    arrCyl = np.zeros((arrmpt.shape[0], arrmpt.shape[1], 3))
    for p in range(newNsection):
        fullMprime = np.concatenate((upstreamMprime[p][:-1], blade1PMprime[p], downstreamMprime[p][1:]))
        funcZ = CubicSpline(fullMprime[:, 3], fullMprime[:, 2])
        arrZ = funcZ(arrmpt[p, :, 0])

        fullR = np.concatenate((upstreamMprime[p][:-1], blade1PMprime[p], downstreamMprime[p][1:])) 
        # special treatment for hub/casing to ensure
        # points lie on hub/casing
        if p == 0:  # hub
            funcR = CubicSpline(hub[:, 1], hub[:, 0])
        elif p == newNsection-1:  # casing
            funcR = CubicSpline(cas[:, 1], cas[:, 0])
        else:
            funcR = CubicSpline(fullR[:, 2], fullR[:, 1])
        arrR = funcR(arrZ)
        arrCyl[p][:, 0] = arrmpt[p][:, 1]
        arrCyl[p][:, 1] = arrR
        arrCyl[p][:, 2] = arrZ

    return arrCyl

# JD: is something wrong with this function?
def bladeToOffset(pt1, pt2, res, hub, cas):
    """
    Create a new curve via interpolation between two points
    (with 'res' points total)
    Z coordinate of pt2 must be < Z coordinate of pt1
    pt1 and pt2 are assumed to have nNewSections sets of points
    hub and cas are bounding curves where linear interpolation
    can't be used at the bounding ends.
    """
    newNsection = pt1.shape[0]
    curve = np.zeros([newNsection, res, 3])
    for p in range(newNsection):
        z = np.linspace(pt1[p][2], pt2[p][2], res)  # range of Z values where we want to get values
        xinterp = np.array([pt1[p][2], pt2[p][2]])  # Z values of ends of range
        yinterp = np.array([pt1[p][0:2], pt2[p][0:2]])  # theta, R values of ends of range
        func1 = interp1d(xinterp, yinterp, axis=0)  # interpolation object for theta, R values as a function of Z
        curve[p][:, 2] = z  # output Z values, just use linear range
        curve[p][:, 0:2] = func1(z)  # output theta, R values, use linear interpolator

        # That works except on hub and casing, where the shape may not be
        # linear in the meridional plane. Note the theta values from the
        # linear interpolation should be OK.
        if p == 0:  # hub
            funcR = CubicSpline(hub[:, 1], hub[:, 0])
            curve[p][:, 1] = funcR(z)
        elif p == newNsection-1:  # casing
            funcR = CubicSpline(cas[:, 1], cas[:, 0])
            curve[p][:, 1] = funcR(z)
    return curve


def trimAndRefineExt(extCyl, res, mul, endZ, keep, hub, cas):
    """
    trims an extension to the endZ values, keeping
    either the extCyl Z values >= endZ ('hi') or
    the extCyl Z values <= endZ ('lo')
    'hi' used for upstream extensions
    'lo' used for downstream extensions
    """
    newNsection = len(endZ)
    trimmedExtCyl = np.zeros([newNsection, res*mul, 3])
    if keep == "hi":
        for p in range(newNsection):
            # pol variables were (z, theta)
            func1 = interp1d(extCyl[p, :, 2], extCyl[p, :, 0], fill_value="extrapolate")
            zTrimmed = np.linspace(endZ[p], extCyl[p, -2, 2], mul*res)
            thTrimmed = func1(zTrimmed)
            fullR = extCyl[p, 0:-1, 1:]
            # special treatment for hub/casing to ensure
            # points lie on hub/casing
            if p == 0:  # hub
                funcR = CubicSpline(hub[:, 1], hub[:, 0])
            elif p == newNsection-1:  # casing
                funcR = CubicSpline(cas[:, 1], cas[:, 0])
            else:
                funcR = CubicSpline(fullR[:, 1], fullR[:, 0])
            rTrimmed = funcR(zTrimmed)
            trimmedExtCyl[p, :, 0] = thTrimmed
            trimmedExtCyl[p, :, 1] = rTrimmed
            trimmedExtCyl[p, :, 2] = zTrimmed
    elif keep == "lo":
        for p in range(newNsection):
            # pol variables were (z, theta)
            func1 = interp1d(extCyl[p, :, 2], extCyl[p, :, 0], fill_value="extrapolate")
            zTrimmed = np.linspace(extCyl[p, 1, 2], endZ[p], mul*res)
            thTrimmed = func1(zTrimmed)
            fullR = extCyl[p, 1:, 1:]
            # special treatment for hub/casing to ensure
            # points lie on hub/casing
            if p == 0:  # hub
                funcR = CubicSpline(hub[:, 1], hub[:, 0])
            elif p == newNsection-1:  # casing
                funcR = CubicSpline(cas[:, 1], cas[:, 0])
            else:
                funcR = CubicSpline(fullR[:, 1], fullR[:, 0])
            rTrimmed = funcR(zTrimmed)
            trimmedExtCyl[p, :, 0] = thTrimmed
            trimmedExtCyl[p, :, 1] = rTrimmed
            trimmedExtCyl[p, :, 2] = zTrimmed

    return trimmedExtCyl


def splitBladesAndOffsets(blade1Cyl, blade2Cyl, offset1Cyl, offset2Cyl, offsetVertices1Cyl, offsetVertices2Cyl, mid1, mid2, crossPassageRes, passageRes):
    """
    Split blade and offset curves at mid-chord to upstream and downstream parts
    Returns split blades, offsets, as well as the 3 mid-chord curve sections
    """
    newNsection = blade1Cyl.shape[0]
    bladeRes = blade1Cyl.shape[1]
    
    blade1UpCyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of blade surface
    offset1UpCyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of offset surface
    blade1DnCyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of blade surface
    offset1DnCyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of offset surface
    
    blade2UpCyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of blade surface
    offset2UpCyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of offset surface
    blade2DnCyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of blade surface
    offset2DnCyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of offset surface
    
    midCurveMidCyl =  np.zeros([newNsection,passageRes,3])  # midCurve in between offsets
    midCurve1Cyl =  np.zeros([newNsection,int(crossPassageRes),3])  # midCurve between low theta blade and offset
    midCurve2Cyl =  np.zeros([newNsection,int(crossPassageRes),3])  # midCurve between offset and high theta blade
    
    for v in range(newNsection):
        blade1Idx = np.argmin(np.abs(blade1Cyl[v][:, 2] - mid1[v][2]))
        highThetaB1 = blade1Cyl[v][0:blade1Idx+1]
        blade1UpCyl[v] = densifyCurve(highThetaB1, bladeRes, 'LE')
        highThetaB2 = blade1Cyl[v][blade1Idx:]
        blade1DnCyl[v] = densifyCurve(highThetaB2, bladeRes, 'TE')

        blade2Idx = np.argmin(np.abs(blade2Cyl[v][:, 2] - mid2[v][2]))
        lowThetaB1 = blade2Cyl[v][0:blade2Idx+1]
        blade2UpCyl[v] = densifyCurve(lowThetaB1, bladeRes, 'LE')
        lowThetaB2 = blade2Cyl[v][blade2Idx:]
        blade2DnCyl[v] = densifyCurve(lowThetaB2, bladeRes, 'TE')

        offset1CylM = insertPoint(offset1Cyl[v], offsetVertices1Cyl[v][1])
        offset1Idx = np.argmin(np.abs(offset1CylM[:, 2] - offsetVertices1Cyl[v][1][2]))
        highThetaO1 = offset1CylM[0:offset1Idx+1]
        offset1UpCyl[v] = densifyCurve(highThetaO1, bladeRes, 'LE')
        highThetaO2 = offset1CylM[offset1Idx:]
        offset1DnCyl[v] = densifyCurve(highThetaO2, bladeRes, 'TE')

        offset2CylM = insertPoint(offset2Cyl[v], offsetVertices2Cyl[v][3])
        offset2Idx = np.argmin(np.abs(offset2CylM[:, 2] - offsetVertices2Cyl[v][3][2]))
        lowThetaO1 = offset2CylM[0:offset2Idx+1]
        offset2UpCyl[v] = densifyCurve(lowThetaO1, bladeRes, 'LE')
        lowThetaO2 = offset2CylM[offset2Idx:]
        offset2DnCyl[v] = densifyCurve(lowThetaO2, bladeRes, 'TE')

        # "midCurve" --> just the set of four points blade-offset-offset-blade on this section
        # (to clarify: these are in order of increasing theta)
        midCurve = np.vstack((highThetaB2[0], highThetaO2[0], lowThetaO2[0], lowThetaB2[0]))
        # these arrays hold the surface points for the 3 sections of the midChord surface:
        midCurveMidCyl[v][:,2] = np.linspace(midCurve[0,2], midCurve[-1,2], passageRes)
        midCurve1Cyl[v][:,2] = np.linspace(midCurve[0,2], midCurve[-1,2], int(crossPassageRes))
        midCurve2Cyl[v][:,2] = np.linspace(midCurve[0,2], midCurve[-1,2], int(crossPassageRes))
        midCurveMidCyl[v][:,1] = np.linspace(midCurve[0,1], midCurve[-1,1], passageRes)
        midCurve1Cyl[v][:,1] = np.linspace(midCurve[0,1], midCurve[-1,1], int(crossPassageRes))
        midCurve2Cyl[v][:,1] = np.linspace(midCurve[0,1], midCurve[-1,1],int(crossPassageRes))
        midCurve2Cyl[v][:,0]= np.linspace(midCurve[:,0][0], midCurve[:,0][1], int(crossPassageRes))
        midCurveMidCyl[v][:,0] = np.linspace(midCurve[:,0][1], midCurve[:,0][2], passageRes)
        midCurve1Cyl[v][:,0] = np.linspace(midCurve[:,0][2], midCurve[:,0][3], int(crossPassageRes))

    # JD: adding to try to clean up mapping

    blade1UpCart = cylToCart(blade1UpCyl)
    blade1DnCart = cylToCart(blade1DnCyl)
    blade2UpCart = cylToCart(blade2UpCyl)
    blade2DnCart = cylToCart(blade2DnCyl)

    offset1UpCart = cylToCart(offset1UpCyl)
    offset1DnCart = cylToCart(offset1DnCyl)
    offset2UpCart = cylToCart(offset2UpCyl)
    offset2DnCart = cylToCart(offset2DnCyl)

    for v in range(newNsection):
        offset1UpCart[v] = matchArcLengthFractions(blade1UpCart[v], offset1UpCart[v])
        offset1DnCart[v] = matchArcLengthFractions(blade1DnCart[v], offset1DnCart[v])
        offset2UpCart[v] = matchArcLengthFractions(blade2UpCart[v], offset2UpCart[v])
        offset2DnCart[v] = matchArcLengthFractions(blade2DnCart[v], offset2DnCart[v])

    offset1UpCyl = CartToCyl(offset1UpCart)
    offset1DnCyl = CartToCyl(offset1DnCart)
    offset2UpCyl = CartToCyl(offset2UpCart)
    offset2DnCyl = CartToCyl(offset2DnCart)

    return blade1UpCyl, blade1DnCyl, blade2UpCyl, blade2DnCyl, offset1UpCyl, offset1DnCyl, offset2UpCyl, offset2DnCyl, midCurveMidCyl, midCurve1Cyl, midCurve2Cyl


def matchArcLengthFractions(curveToMatch, curveToModify):
    """
    Modifies a curve to match arclength fractions from a another curve
    Note: a consistent coordinate system is required!
    For example, Cartesian or m'-theta.
    """
    # 1. Get arclength fractions of points in curveToMatch
    matchS = np.zeros(curveToMatch.shape[0])
    matchDiffs = np.diff(curveToMatch, axis=0)
    mSloc = np.linalg.norm(matchDiffs, axis=1)
    matchS[1:] = np.cumsum(mSloc)  # cumulative arclength
    arcLengthFractionsm = matchS / matchS.max()
    arcLengthFractionsm[-1] = 1
    
    # 2. Get arclenth fractions of current points in curveToModify
    curveS = np.zeros(curveToModify.shape[0])
    curveDiffs = np.diff(curveToModify, axis=0)
    cSloc = np.linalg.norm(curveDiffs, axis=1)
    curveS[1:] = np.cumsum(cSloc)  # cumulative arclength
    arcLengthFractionsc = curveS / curveS.max()
    arcLengthFractionsc[-1] = 1

    # 3. Interpolate new curve points (linear)
    curveInterp = interp1d(arcLengthFractionsc, curveToModify, axis=0)
    modifiedCurve = curveInterp(arcLengthFractionsm)

    return modifiedCurve


def matchArcLengthFractions2(distToMatch, curveToModify):
    """
    Modifies a curve to match arclength fractions from an
    input distribution
    Note: a consistent coordinate system is required!
    For example, Cartesian or m'-theta.
    """
    # 1. Get arclength fractions of points in curveToMatch
    matchS = np.zeros(distToMatch.shape[0])
    matchDiffs = np.diff(distToMatch, axis=0)
    mSloc = np.linalg.norm(matchDiffs, axis=0)
    matchS[1:] = np.cumsum(mSloc)  # cumulative arclength
    arcLengthFractionsm = matchS / matchS.max()
    arcLengthFractionsm[-1] = 1
    
    # 2. Get arclenth fractions of current points in curveToModify
    curveS = np.zeros(curveToModify.shape[0])
    curveDiffs = np.diff(curveToModify, axis=0)
    cSloc = np.linalg.norm(curveDiffs, axis=1)
    curveS[1:] = np.cumsum(cSloc)  # cumulative arclength
    arcLengthFractionsc = curveS / curveS.max()
    arcLengthFractionsc[-1] = 1

    # 3. Interpolate new curve points (linear)
    curveInterp = interp1d(arcLengthFractionsc, curveToModify, axis=0)
    modifiedCurve = curveInterp(arcLengthFractionsm)

    return modifiedCurve


def fillInOutHubCas(blade1UpExtCyl, blade2UpExtCyl, blade1DnExtCyl, blade2DnExtCyl, offset1UpCyl, offset2UpCyl, offset1DnCyl, offset2DnCyl, passageRes):
    """
    Uses offset and extension data to create arrays with interior nodes for
    the inlet, outlet, and central sections of the hub and casing
    """
    # Using transfinite interpolation (3D version)
    # tf.transfinite3D(lower, upper, left, right)

    # Inlet is bounded by:
    # blade1UpExtCyl[:, 0, :]
    # blade2UpExtCyl[:, 0, :]
    # and constant-radius curves connecting
    # (hub): blade1UpExtCyl[0, 0, :] and blade2UpExtCyl[0, 0, :]
    # (cas): blade1UpExtCyl[-1, 0, :] and blade2UpExtCyl[-1, 0, :]
    # TF interpolation should be done in cylindrical coordinates
    inletHubThetas = np.linspace(blade1UpExtCyl[0, 0, 0], blade2UpExtCyl[0, 0, 0], passageRes)
    inletHubPtsCyl = np.zeros((passageRes, 3))
    inletHubPtsCyl[:, 1:3] = blade1UpExtCyl[0, 0, 1:3]
    inletHubPtsCyl[:, 0] = inletHubThetas
    
    inletCasThetas = np.linspace(blade1UpExtCyl[-1, 0, 0], blade2UpExtCyl[-1, 0, 0], passageRes)
    inletCasPtsCyl = np.zeros((passageRes, 3))
    inletCasPtsCyl[:, 1:3] = blade1UpExtCyl[-1, 0, 1:3]
    inletCasPtsCyl[:, 0] = inletCasThetas

    inletPtsCylNodes = tf.transfinite3D(inletHubPtsCyl, inletCasPtsCyl, blade1UpExtCyl[:, 0, :], blade2UpExtCyl[:, 0, :])
    inletPtsCyl = inletPtsCylNodes.reshape(passageRes, int(inletPtsCylNodes.shape[0]/passageRes), 3)
    inletPtsCyl = np.swapaxes(inletPtsCyl, 0, 1)

    # Outlet is bounded by:
    # blade1DnExtCyl[:, -1, :]
    # blade2DnExtCyl[:, -1, :]
    # and constant-radius curves connecting
    # (hub): blade1DnExtCyl[0, -1, :] and blade2DnExtCyl[0, -1, :]
    # (cas): blade1DnExtCyl[-1, -1, :] and blade2DnExtCyl[-1, -1, :]
    # TF interpolation should be done in cylindrical coordinates
    outletHubThetas = np.linspace(blade1DnExtCyl[0, -1, 0], blade2DnExtCyl[0, -1, 0], passageRes)
    outletHubPtsCyl = np.zeros((passageRes, 3))
    outletHubPtsCyl[:, 1:3] = blade1DnExtCyl[0, -1, 1:3]
    outletHubPtsCyl[:, 0] = outletHubThetas
    
    outletCasThetas = np.linspace(blade1DnExtCyl[-1, -1, 0], blade2DnExtCyl[-1, -1, 0], passageRes)
    outletCasPtsCyl = np.zeros((passageRes, 3))
    outletCasPtsCyl[:, 1:3] = blade1DnExtCyl[-1, -1, 1:3]
    outletCasPtsCyl[:, 0] = outletCasThetas

    outletPtsCylNodes = tf.transfinite3D(outletHubPtsCyl, outletCasPtsCyl, blade1DnExtCyl[:, -1, :], blade2DnExtCyl[:, -1, :])
    outletPtsCyl = outletPtsCylNodes.reshape(passageRes, int(outletPtsCylNodes.shape[0]/passageRes), 3)
    outletPtsCyl = np.swapaxes(outletPtsCyl, 0, 1)

    # Central hub is bounded by:
    # inletHubPtsCyl
    # outletHubPtsCyl
    # blade1UpExtCyl[0, :, :], offset1UpCyl[0, 1:, :], offset1DnCyl[0, 1:, :], blade1DnExtCyl[0, 1:, :]
    # blade2UpExtCyl[0, :, :], offset2UpCyl[0, 1:, :], offset2DnCyl[0, 1:, :], blade2DnExtCyl[0, 1:, :]
    hub1 = np.concatenate((blade1UpExtCyl[0, :, :], offset1UpCyl[0, 1:, :], offset1DnCyl[0, 1:, :], blade1DnExtCyl[0, 1:, :]), axis=0)
    hub2 = np.concatenate((blade2UpExtCyl[0, :, :], offset2UpCyl[0, 1:, :], offset2DnCyl[0, 1:, :], blade2DnExtCyl[0, 1:, :]), axis=0)
    hubPtsCylNodes = tf.transfinite3D(inletHubPtsCyl, outletHubPtsCyl, hub1, hub2)
    hubPtsCyl = hubPtsCylNodes.reshape(passageRes, int(hubPtsCylNodes.shape[0]/passageRes), 3)
    hubPtsCyl = np.swapaxes(hubPtsCyl, 0, 1)

    # Central cas is bounded by:
    # inletCasbPtsCyl
    # outletCasPtsCyl
    # blade1UpExtCyl[-1, :, :], offset1UpCyl[-1, 1:, :], offset1DnCyl[-1, 1:, :], blade1DnExtCyl[-1, 1:, :]
    # blade2UpExtCyl[-1, :, :], offset2UpCyl[-1, 1:, :], offset2DnCyl[-1, 1:, :], blade2DnExtCyl[-1, 1:, :]
    cas1 = np.concatenate((blade1UpExtCyl[-1, :, :], offset1UpCyl[-1, 1:, :], offset1DnCyl[-1, 1:, :], blade1DnExtCyl[-1, 1:, :]), axis=0)
    cas2 = np.concatenate((blade2UpExtCyl[-1, :, :], offset2UpCyl[-1, 1:, :], offset2DnCyl[-1, 1:, :], blade2DnExtCyl[-1, 1:, :]), axis=0)
    casPtsCylNodes = tf.transfinite3D(inletCasPtsCyl, outletCasPtsCyl, cas1, cas2)
    casPtsCyl = casPtsCylNodes.reshape(passageRes, int(casPtsCylNodes.shape[0]/passageRes), 3)
    casPtsCyl = np.swapaxes(casPtsCyl, 0, 1)

    return inletPtsCyl, outletPtsCyl, hubPtsCyl, casPtsCyl


def fillBladeToOffset(crossPassageUpCyl, crossPassageDnCyl, lowThetaCyl, highThetaCyl):
    """
    Creates interior nodes for surfaces between blades and offsets on
    hub and casing. Takes 4 arguments, the bounding surfaces.
    Outputs 2 arrays of hub and casing interior nodes.
    """
    # Using transfinite interpolation (3D version)
    # tf.transfinite3D(lower, upper, left, right)
    crossPassagePts = crossPassageUpCyl.shape[1]
    alongPassagePts = lowThetaCyl.shape[1]
    # hub[i=0] and cas[i=-1] bounded by:
    # crossPassageUpCyl[i, :, :]
    # crossPassageDnCyl[i, :, :]
    # lowThetaCyl[i, :, :]
    # highThetaCyl[i, :, :]

    hubPtsCylNodes = tf.transfinite3D(crossPassageUpCyl[0, :, :], crossPassageDnCyl[0, :, :], lowThetaCyl[0, :, :], highThetaCyl[0, :, :])
    hubPtsCyl = hubPtsCylNodes.reshape(crossPassagePts, alongPassagePts, 3)
    hubPtsCyl = np.swapaxes(hubPtsCyl, 0, 1)

    casPtsCylNodes = tf.transfinite3D(crossPassageUpCyl[-1, :, :], crossPassageDnCyl[-1, :, :], lowThetaCyl[-1, :, :], highThetaCyl[-1, :, :])
    casPtsCyl = casPtsCylNodes.reshape(crossPassagePts, alongPassagePts, 3)
    casPtsCyl = np.swapaxes(casPtsCyl, 0, 1)

    bob = alice

    return hubPtsCyl, casPtsCyl


def combineArrays(*args):
    """
    Combine all Cartesian surface arrays into X/Y/Zvalues
    which the STL generator function expects
    """
    arrList = list(args)
    arrNum = len(arrList)
    Xvalues = []
    Yvalues = []
    Zvalues = []
    for i in range(arrNum):
        Xvalues.append(arrList[i][:, :, 0])
        Yvalues.append(arrList[i][:, :, 1])
        Zvalues.append(arrList[i][:, :, 2])

    return Xvalues, Yvalues, Zvalues


def cylToCart(arrCyl):
    """
    converts input array in cylindrical coordinates (last dimension)
    to Cartesian. Assumes (theta, r, z) for cylindrical data.
    """
    th = arrCyl[:, :, 0]
    r = arrCyl[:, :, 1]
    z = arrCyl[:, :, 2]
    x = r * np.cos(th)
    y = r * np.sin(th)
    return np.stack((x, y, z), axis=2)


def cutArcLenMaps(arr, lower_bound=0.0, upper_bound=1.0):
    firstRow = arr[0:1]
    lastRow = arr[-1:]
    middleRows = arr[1:-1]
    mask = (middleRows[:, 0] >= lower_bound) & (middleRows[:, 0] <= upper_bound)
    filteredMiddle = middleRows[mask]
    return np.concatenate((firstRow, filteredMiddle, lastRow))


def angles_between_points(p1, p2, p3):
    """
    Calculates the angles (in radians) subtended by p1 and p2
    and p3 and p2.
    The angles are measured counter-clockwise the +x direction.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate the angle of the two vectors relative to the x-axis
    ang1 = np.arctan2(y1 - y2, x1 - x2)
    ang2 = np.arctan2(y3 - y2, x3 - x2)

    return np.array([ang1, ang2])


def getBezierControlPoint(p1, p2, m1, m2):
    # Get coordinates of point pc, which is where the tangents
    # (slopes m1 and m2) from p1 and p2 intersect

    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]

    xc = ((m2*x2 - m1*x1) - (y2 - y1)) / (m2-m1)
    yc = y1 + m1*(xc - x1)
    
    return np.array([xc, yc])


def quadratic_bezier_curve(p0, p1, p2, num_points=100):
    """
    Generates points for a quadratic Bezier curve.
    https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Quadratic_B%C3%A9zier_curves

    Args:
        p0 (np.ndarray): The start control point [x, y].
        p1 (np.ndarray): The control point [x, y].
        p2 (np.ndarray): The end control point [x, y].
        num_points (int): The number of points to generate along the curve.

    Returns:
        np.ndarray: An array of (x, y) coordinates representing the curve.

    This function was generated by Google AI search result, but has been
    verified to be correct by Jeff Defoe (25 November 2025).
    https://share.google/aimode/Eui0rcQXIrYyZCmnp

    result seems mostly to be based on:
    https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy#:~:text=Building%20upon%20the%20answers%20from,Curve%20Fitting%22.%20%22%22%22
    """
    t = np.linspace(0, 1, num_points)
    curve_points = (1 - t)**2 * p0[:, np.newaxis] + \
                   2 * (1 - t) * t * p1[:, np.newaxis] + \
                   t**2 * p2[:, np.newaxis]
    return curve_points.T


# BL thickness calculator
def calcBLdelta(rho, U, L, mu):
    """ Calculate BL thickness using flat plate approximation """
    # Assumes turbulent flow
    Re = rho*U*L/mu
    delta = 0.37 * L / (Re**(1/5))
    return delta


# First cell thickness calculator
def calcFirstCellSize(rho, U, L, mu, yplus):
    """ Calculates first BL cell size using standard turbulent BL profile """
    Re = rho*U*L/mu
    Cf = 0.026 / (Re**(1/7))
    Uf = (Cf*(U**2)*0.5)**0.5
    dy1 = yplus * (mu/rho) / Uf
    return dy1


def blockMeshGradDescriptorBuilder(divisions, varName):
    """
    Takes in an array of block edge divisions, and build a string block
    descriptor for drop-in use in blockMesh that uses a number of sections
    = number of cells (len(divisions)-1) to exactly enforce the desired
    distribution.

    Also takes as input a string (varName).
    """
    # example: varName (ds[0] 1 1) (ds[1] 1 1);
    ds = np.diff(divisions)  # fraction of edge length for each cell/section
    descriptorString = varName
    for i in range(len(ds)):
        descriptorString += f" ({ds[i]} 1 1)"
    descriptorString += ";"
    return descriptorString


def pointFracs(nDiv, s, f, g):
    """
    Calculates an array of length n+1 with fractional
    arc lengths for cell points based on input parameters:
    n = number of cells
    s = array with fractional edge length divisions
    f = array with fraction of cells divisions
    g = array with grading ratio for each division
    """
    # basically this should do the same thing BlockMesh is
    # doing internally to divide points along edges
    # translation to Python by Jeff Defoe
    # from code in src/blockMesh/blockEdges/lineDivide/lineDivide.C
    # in OpenFOAM v2406

    sections = len(s)
    # print(f'{sections} sections')
    divisions = np.zeros(nDiv + 1)
    divisions[-1] = 1.0

    secStart = divisions[0]
    secnStart = 1

    # Calculate distribution of divisions to be independent
    # of the order of the sections
    secnDivs = np.zeros(sections, dtype=int)
    sumSecnDivs = 0
    secnMaxDivs = 0

    # print('Figuring out how many cells (divisions) go in each section...')
    for sectioni in range(sections):
        # print(f'Checking section {sectioni}...')
        nDivFrac = f[sectioni]
        # print(f'Fraction of cells in this section = {nDivFrac}')
        secnDivs[sectioni] = int(nDivFrac*nDiv + 0.5)
        # print(f'Number of divisions in this section = {secnDivs[sectioni]}')
        sumSecnDivs += secnDivs[sectioni]

        # Find the section with the largest number of divisions
        if (nDivFrac > f[secnMaxDivs]):
            secnMaxDivs = sectioni

    # Adjust the number of divisions on the section with the largest
    # number of that the total is nDiv
    if (sumSecnDivs != nDiv):
        # print('Adjusting section with most cells to total number is right...')
        secnDivs[secnMaxDivs] += (nDiv - sumSecnDivs)

    # print('Now looping through sections to set up division values...')
    for sectioni in range(sections):
        # print(f'Working on section {sectioni}...')
        blockFrac = s[sectioni]
        # print(f'Fraction of edge length = {blockFrac}')
        expRatio = g[sectioni]
        # print(f'Grading ratio = {expRatio}')

        secnDiv = secnDivs[sectioni]
        secnEnd = secnStart + secnDiv

        # Calculate the spacing
        if expRatio == 1:
            for i in range(secnStart, secnEnd):
                divisions[i] = secStart + blockFrac*(i - secnStart + 1)/secnDiv
        else:
            # Calculate the geometric expansion factor from the expansion ratio
            expFact = expRatio**(1/(secnDiv-1))
            # print(f'Expansion ratio = {expFact}')

            for i in range(secnStart, secnEnd):
                divisions[i] = secStart + blockFrac*(1.0 - expFact**(i - secnStart + 1)) / (1.0 - expFact**secnDiv)
                # print(f'Division (fraction edge length) {i} = {divisions[i]}')

        secStart = divisions[secnEnd - 1]
        secnStart = secnEnd

    return divisions


def calcArcLength(coords):
    """
    Calculate the arclength of a curve defined by a set of 
    points (2D or 3D) in coords

    'coords' should have each row represent coordinates of a point.
    For example: coords=np.array([x1, y1, z1], [x2, y2, z2], ...])

    Returns the cumulative arclengths.
    """
    if coords.shape[1]<=3 and coords.shape[1]>=2:
        coordsLoc = coords
    elif coords.shape[1]<2:
        print('Error, data is not 2D or 3D')
        return None
    else:
        print('Transposing input data...')
        coordsLoc = coords.T
    dataDim = min(coordsLoc.shape)
    print(f'Calculating arclength for {dataDim}-D curve...')
    diffs = np.diff(coordsLoc, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    return np.cumsum(ds)


def getNumBLCells(dx1, dxnp1, delta):
    """ Solution for number of cells in a BL """
    # by Jeff Defoe, modified from work by Ella Samms
    # based on math worked out by Tony Woo
    # Inputs: dx1, dxnp1, del
    # dx1 = first cell size
    # dxnp1 = cell size immediately outside BL
    # delta = BL thickness
    #
    # Idea: dxnp1 should have same expansion ratio
    # relative to dxn as the interior BL cells
    # initisl r estimate
    rInitial = dxnp1/delta - dx1/delta + 1
    # get n, make sure it's an integer
    n = round(np.log(dxnp1/dx1)/np.log(rInitial))
    # update r to be consistent with n:
    r = fsolve(lambda r: dx1*r**n-delta*r+delta-dx1, rInitial)
    r = r[0]
    return n, r


def splitsCalc(n, r):
    """ Finds splits for refineWallLayer to split a single cell into n BL cells"""
    splits = np.zeros((n-1,))
    A = (1-r)/(1-r**n)
    for i in range(1, n):
        S = 0
        for j in range(1, i):
            S = S + A*r**(n-j)
        splits[i-1] = 1 - A*r**(n-i)/(1-S)
    return splits


def bladeGradingFunction(vars, dxn, dx1p, L, rstart):
    """ system of equations for solving over-blade block axial grading and cell count """
    gstart, gend, n = vars
    fstart = (np.log(gstart)/np.log(rstart) + 1)/n  # eq 1
    rend = gend**(1/(n*(1-fstart)-1))  # eq 2
    dx1 = dx1p*L
    dxnfstart = gstart*dx1  # eq 3
    dxnfstartp1 = dxn/gend  # eq 4
    eqA = -0.25*L/dx1 + ((1-rstart**(n*fstart))/(1-rstart))  # eq 6
    eqB = -dxnfstartp1/dxnfstart + 0.5*(rstart+rend)  # eq 5
    eqC = -0.75*L/dxnfstartp1 + ((1-rend**(n*(1-fstart)))/(1-rend))  # eq 7
    return np.array([eqA, eqB, eqC])


def bladeGradingFunctionSpecific(vars, n, dxn, dx1, L):
    """ system of equations for solving blade edge axial grading """
    fstart, gstart, gend = vars
    rstart = gstart**(1/(fstart*n-1))
    rend = gend**(1/(n*(1-fstart)-1))
    eqA = -0.5*(rstart + rend) + (dxn/dx1)*(1/(rend**(n*(1-fstart)-1)))*(1/(rstart**(n*fstart-1)))  # eq A
    eq6 = -(1-rstart**(n*fstart))/(1-rstart) + (0.25*L/dx1)  # eq 6
    eqB = -(0.75*L/dxn)*rend**(n*(1-fstart)-1) + (1-rend**(n*(1-fstart)))/(1-rend)  # eq B
    return np.array([eqA, eq6, eqB])


def offsetGradingFunction(vars, n, dxn, L, dx1p, xpsplit):
    """ system of equations for solving offset block axial grading """
    gstart, gend, fstart, rstart, rend = vars
    dx1 = dx1p*L
    dxnfstart = gstart*dx1  # eq 3
    dxnfstartp1 = dxn/gend  # eq 4
    eq1 = -gstart + rstart**(n*fstart-1)  # eq 1
    eq2 = -gend + rend**(n*(1-fstart)-1)  # eq 2
    eq5 = -(dxnfstartp1/dxnfstart) + 0.5*(rstart + rend)  # eq 5
    eq6 = -(1-rstart**(n*fstart))/(1-rstart) + (xpsplit*L/dx1)  # eq 6
    eq7 = -(1-rend**(n*(1-fstart)))/(1-rend) + ((1-xpsplit)*L/dxnfstartp1)  # eq 7
    return np.array([eq1, eq2, eq5, eq6, eq7])
    

def outerGradingCaseChooser(L, C, dx1, rfar, dxMid, gin, nin, fin):
    """ Determine how the problem needs to be solved based on length """
    print(f'Inputs: L={L}, C={C}, dx1={dx1}, rfar={rfar}, dxMid={dxMid}, gin={gin}, nin={nin}, fin={fin}')
    gclo = dxMid/dx1
    print(f'gclo={gclo}')
    if L > 0.5*C:
        # both grading regions active
        print('L>0.5*C, solving for fclo, gfar, and n...')
        fcloGuess = min(0.3*(0.01/(dx1/C)), 0.8)
        nGuess = nin*L/C
        dxnGuess = 2*dxMid
        dxMidp1Guess = dxMid+dx1
        gfarGuess = rfar**((1-fcloGuess)*nGuess-1)
        rcloGuess = gclo**(1/(fcloGuess*nGuess-1))
        print(f'Initial guess: fclo={fcloGuess}, gfar={gfarGuess}, rclo={rcloGuess}, n={nGuess}, dxn={dxnGuess}, dxMidp1={dxMidp1Guess}')
        root = fsolve(outerAxialGradingFunction, x0=[fcloGuess, gfarGuess, rcloGuess, nGuess, dxnGuess, dxMidp1Guess], args=(dx1, dxMid, rfar, C, L, gclo))
        fclo, gfar, rclo, n, dxn, dxMidp1 = root
        n = np.round(n)  # ensure it's an integer
    else:
        # only close grading region active
        print('L<=0.5*C, directly solving for n and then truncating as needed...')
        gfar = 1.0  # doesn't matter, but have to return something
        fclo = 1.0  # because all the cells are in the "close" region
        n = np.round((np.log(gclo) + np.log(rclo))/(fclo*np.log(rclo)))
        if L < 0.5*C:
            # have to truncate
            # treat gclo, rclo, and n as aspirational
            # figure out the actual values
            # thing to keep constant is 1st cell size dx1
            # and expansion ratio rclo
            print('Truncating to get new n and gclo...')
            n = np.round(fsolve(lambda n: -(L/dx1) + ((1-rclo**n)/(1-rclo)), n))
            gclo = rclo**(n-1)
    n = int(n)
    print(f'Final calculated values: gclo={gclo}, rclo={rclo}, fclo={fclo}, gfar={gfar}, n={n}')
    return gclo, rclo, fclo, gfar, n


def outerGradingCaseChooserSpecific(L, C, dx1, dxMid, n, rcloGuess, fcloGuess):
    """ Determine axial grading parameters for edges once # of cells (n) known """
    gclo = dxMid/dx1
    # Need to make the below part specific based on L and C
    if L > 0.5*C:
        # Simultaneous solution of equations 1 and 6
        root = fsolve(outerAxialGradingFunctionSpecificrclofclo, x0=[rcloGuess, fcloGuess], args=(gclo, n, C, dx1))
        rclo, fclo = root
        # Solution for rfar from combination of equations 5 and 7
        rfar = fsolve(lambda rfar: -((L-0.5*C)/(dxMid*0.5)) + (rclo+rfar)*((1-rfar**(n*(1-fclo)))/(1-rfar)), rclo)
        # Solution for gfar from equation 2
        gfar = rfar**(n*(1-fclo)-1)
        gfar = gfar[0]
    else:
        # only close grading region active
        print('L<=0.5*C, truncating if needed to get new gclo...')
        gfar = 1.0  # doesn't matter, but have to return something
        fclo = 1.0  # because all the cells are in the "close" region
        if L < 0.5*C:
            # thing to keep constant is 1st cell size dx1
            print('Truncating to get new gclo...')
            # guess for rclo: what it would be when L=0.5C
            rclo_full = gclo**(1/(n-1))
            # use sum of geometric series to get new rclo
            rclo = fsolve(lambda rclo: -(L/dx1)*(1-rclo**n)/(1-rclo), rclo_full)
            # get new gclo
            gclo = rclo**(n-1)
            gclo = gclo[0]
    print(f'gclo={gclo}, fclo={fclo}, gfar={gfar}')
    return gclo, fclo, gfar


def getTanGradingAtInletOutlet(halfPitch, gTan, rTan, nhalf, angleLim, L):
    """ Calculate grading ratio at inlet/outlet to not exceed minimum angle contraction """
    # angleLim in degrees
    dtMid = halfPitch*gTan*(1-rTan)/(1-rTan**(nhalf/2))
    print(f'Midpassage tangential cell size = {dtMid}')
    mindt = dtMid - L*np.tan(np.radians(angleLim))
    print(f'Minimum tangential cell size at domain edge to stay within contraction limit = {mindt}')
    # Check this result. If > 0, then we have to do more checks.
    # However, if this is <=0, we're done -- simply set grading
    # ratio to 1.0.
    if mindt <= 0:
        print('Since minimum size is not > 0, just setting grading ratio to 1.0.')
        g = 1.0
    else:
        print('Minimum size > 0. Check what grading ratio is needed.')
        # get associated expansion ratio
        rTanIO = fsolve(tanExpRatioSolver, x0=1.1, args=(mindt, halfPitch, nhalf))
        # get grading ratio; if <1 set to 1
        g = min(rTanIO**(nhalf-1),1.0)
        print(f'Grading ratio = {g}')
    return g

def outerAxialGradingFunction(vars, dx1, dxMid, rfar, C, L, gclo):
    """ 6x6 equation solver for outer block axial grading """
    fclo, gfar, rclo, n, dxn, dxMidp1 = vars
    eq1 = -gclo + rclo**(n*fclo-1)
    eq2 = -gfar + rfar**(n*(1-fclo)-1)
    eq4 = -gfar + dxn/dxMidp1
    eq5 = -(dxMidp1/dxMid) + 0.5*(rclo+rfar)
    eq6 = -0.5*C/dx1 + (1-rclo**(n*fclo))/(1-rclo)
    eq7 = -(L-0.5*C)/dxMidp1 + (1-rfar**(n*(1-fclo)))/(1-rfar)
    return np.array([eq1, eq2, eq4, eq5, eq6, eq7])


def outerAxialGradingFunctionSpecificrclofclo(vars, gclo, n, C, dx1):
    """ 2x2 for rclo and fclo """
    rclo, fclo = vars
    eq1 = -gclo + rclo**(n*fclo-1)
    eq6 = -0.5*C/dx1 + (1-rclo**(n*fclo))/(1-rclo)
    return np.array([eq1, eq6])


def tanExpRatioSolver(vars, dt, s, n):
    """ Solves equation for tangential exp. ratio at I/O"""
    r = vars
    return (-dt + s*(r**(n-1))*(1-r)/(1-r**n))


def dy_dx(x, M, A):
    """ Slope equation that determines the circumferential shift for both inlet and outlet. """
    # It factors in the radius change and reduces the exaggeration of the lean and twist of the blade. 
    terms = 2 * (np.arctan(x / M) - A) * (1 / (1 + (x / M)**2)) * (1 / M)
    return np.sum(terms)


def unitVector(file, p1, p2, p3):
    """ Compute the unit normal vector to a triangular facet """
    # VECTORS TANGENT TO FACET
    vector1 = p3 - p2
    vector2 = p3 - p1
    
    normalVec = np.cross(vector1, vector2)
    magnitude = np.linalg.norm(normalVec)
    unitVec = normalVec / magnitude
    file.write(f'facet normal {unitVec[0]} {unitVec[1]} {unitVec[2]} \n'
               f'outer loop \n'
               f'vertex {p1[0]} {p1[1]} {p1[2]} \n'
               f'vertex {p2[0]} {p2[1]} {p2[2]} \n'
               f'vertex {p3[0]} {p3[1]} {p3[2]} \n'
               f'endloop \n'
               f'endfacet \n')
    return


#%% Tony Woo Function definitions, updated Oct 2025 by Jeff Defoe
def format_coord(name, arr):
    return f"{name} ({arr[0]} {arr[1]} {arr[2]}) ;\n"


def remove_duplicates(points, tol=1e-10):
    if len(points) <= 1:
        return points
    keep = [0]  # Always keep first point
    # Modified by Jeff Defoe to operate on Z coordinate data
    # this was being used on cylindrical coordinate data, and
    # the way of calculating distance was thus wrong
    # using normal points is better
    for i in range(1, len(points)):
        dist = np.abs(points[i, 2] - points[keep[-1], 2])
        if dist > tol:
            keep.append(i)
    return points[keep]


def densifyCurve(points, n_points_new, distribution='both'):
    if n_points_new <= len(points):
        return points[:n_points_new]
    points = np.atleast_2d(points)
    if points.shape[0] == 1:
        points = points.T
    # Step 1: Create parametric representation using cumulative distance
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    # Step 2: Fit cubic splines for each dimension
    n_dims = points.shape[1]
    splines = []
    for dim in range(n_dims):
        # splines.append(CubicSpline(cumulative_length, points[:, dim]))
        splines.append(interp1d(cumulative_length, points[:, dim]))
    # Step 3: Create parameter values with specified distribution
    total_length = cumulative_length[-1]
    if distribution == 'both':
        # Cosine clustering at both ends (like airfoil distribution)
        theta = np.linspace(0, np.pi, n_points_new)
        # Map from [0, π] to [0, 1], then to [0, total_length]
        xi = (1 - np.cos(theta)) / 2  # Maps to [0, 1]
        new_params = xi * total_length
    elif distribution == 'TE':
        # Clustering at start (trailing edge)
        theta = np.linspace(0, np.pi/2, n_points_new)
        xi = np.sin(theta)  # Maps to [0, 1] with clustering at start
        new_params = xi * total_length
    elif distribution == 'LE':
        # Clustering at end (leading edge)
        theta = np.linspace(0, np.pi/2, n_points_new)
        xi = 1 - np.cos(theta)  # Maps to [0, 1] with clustering at end
        new_params = xi * total_length
    elif distribution == 'uniform':
        # Uniform distribution (original behavior)
        new_params = np.linspace(0, total_length, n_points_new)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    # Step 4: Evaluate the splines at all parameter values
    new_coords = []
    for dim in range(n_dims):
        new_coords.append(splines[dim](new_params))
    return np.column_stack(new_coords)


def angle_bisector(p1, p2, p3, length=1.0):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    # Vectors from vertex (p2) to the other two points
    v1 = p1 - p2
    v2 = p3 - p2
    # Normalize to unit vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    # Bisector direction is the sum of the two unit vectors
    bisector = v1_norm + v2_norm
    # Normalize the bisector
    bisector_direction = bisector / np.linalg.norm(bisector)
    # Endpoint of bisector at specified length
    bisector_end = p2 + length * bisector_direction
    return bisector_direction, bisector_end


def angle_bisector_line(p1, p2, p3, extend=1.0):
    _, bisector_end = angle_bisector(p1, p2, p3, length=extend)
    return p2, bisector_end


def compute_bisector_slope(p1, p2, p3):
    bisector_dir, _ = angle_bisector(p1, p2, p3)
    if abs(bisector_dir[0]) < 1e-10:
        return float('inf')  # Vertical line
    slope = bisector_dir[1] / bisector_dir[0]
    return slope


def insertPoint(curve_points, new_point, closed_loop=False):
    curve = np.array(curve_points)
    new_pt = np.array(new_point)
    # Find the closest segment to insert the point
    min_dist = np.inf
    best_index = 0
    # Determine number of segments to check
    n_segments = len(curve) if closed_loop else len(curve) - 1
    for i in range(n_segments):
        p1 = curve[i]
        p2 = curve[(i + 1) % len(curve)]  # Wrap around for closed loops
        # Find closest point on segment to new_point
        closest_pt, dist = point_to_segment_distance(new_pt, p1, p2)
        if dist < min_dist:
            min_dist = dist
            best_index = i + 1  # Insert after point i
    # Insert the point
    updated_curve = np.insert(curve, best_index, new_pt, axis=0)
    return updated_curve#, best_index


def insertPoints_batch(curve_points, new_points, closed_loop=False):
    curve = np.array(curve_points)
    new_pts = np.array(new_points)
    # Ensure 2D array
    if new_pts.ndim == 1:
        new_pts = new_pts.reshape(1, -1)
    # Find insertion indices for all points
    insertion_data = []
    n_segments = len(curve) if closed_loop else len(curve) - 1
    for new_pt in new_pts:
        min_dist = np.inf
        best_index = 0
        for i in range(n_segments):
            p1 = curve[i]
            p2 = curve[(i + 1) % len(curve)]
            closest_pt, dist = point_to_segment_distance(new_pt, p1, p2)
            if dist < min_dist:
                min_dist = dist
                best_index = i + 1
        insertion_data.append((best_index, new_pt))
    # Sort by index (descending) so we insert from back to front
    # This prevents index shifting from affecting later insertions
    insertion_data.sort(key=lambda x: x[0], reverse=True)
    # Insert all points
    for idx, pt in insertion_data:
        curve = np.insert(curve, idx, pt, axis=0)
    return curve


def point_to_segment_distance(point, seg_start, seg_end):
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    # Vector from seg_start to seg_end
    segment_vec = seg_end - seg_start
    segment_length_sq = np.dot(segment_vec, segment_vec)
    if segment_length_sq == 0:
        # Degenerate segment (point)
        return seg_start, np.linalg.norm(point - seg_start)
    # Project point onto the line containing the segment
    # t is the parameter: closest_point = seg_start + t * segment_vec
    t = np.dot(point - seg_start, segment_vec) / segment_length_sq
    # Clamp t to [0, 1] to stay on the segment
    t = np.clip(t, 0, 1)
    # Find the closest point on the segment
    closest_point = seg_start + t * segment_vec
    dist = np.linalg.norm(point - closest_point)
    return closest_point, dist


def slopeAndMidPts(points):
    #This fuction computes the perpendicular slopes and midPoints 
    p1 = points[0]
    p2 = points[1]
    if (p2[0] - p1[0]) > 1e-10:
        slope = (p2[1] - p1[1])/(p2[0] - p1[0])
    elif np.isinf(p2[1] - p1[1])/(p2[0] - p1[0]):
        slope = 0
    else:
        slope = np.inf
    midpoint = (p1 + p2) / 2
    slopes = -1/slope
    return slopes, midpoint


def slopeAndPt(points):
    points = np.asarray(points)
    n = len(points)
    tangent_slopes = np.zeros(n)
    perp_slopes = np.zeros(n)
    # Compute tangent slopes at each point
    for i in range(n):
        if i == 0:
            # First point: forward difference
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif i == n - 1:
            # Last point: backward difference
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
        else:
            # Interior points: central difference
            dx = points[i + 1][0] - points[i - 1][0]
            dy = points[i + 1][1] - points[i - 1][1]
        # Compute tangent slope
        if abs(dx) < 1e-10:  # Vertical tangent
            tangent_slopes[i] = np.inf
            perp_slopes[i] = 0  # Perpendicular is horizontal
        elif abs(dy) < 1e-10:  # Horizontal tangent
            tangent_slopes[i] = 0
            perp_slopes[i] = np.inf  # Perpendicular is vertical
        else:
            tangent_slopes[i] = dy / dx
            perp_slopes[i] = -1 / tangent_slopes[i]
    return perp_slopes, points


def arcLength(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    # Compute differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    # Arc length is sum of distances between consecutive points
    segments = np.sqrt(dx**2 + dy**2)
    # totalLength = np.sum(segments)
    return segments


def pointAtDist(points, slopes, dist, side):
    x = points[0]
    y = points[1]
    n = len(points)
    newPoints = np.zeros((n))
    if np.isinf(slopes):
        newPoints[0] = x
        newPoints[1] = y + dist
    theta = np.arctan(slopes)
    if side =='lo':
        if slopes < 0:
             newPoints[0] = x - dist * np.cos(theta)
             newPoints[1] = y - dist * np.sin(theta) 
        else:
             newPoints[0] = x + dist * np.cos(theta)
             newPoints[1] = y + dist * np.sin(theta)   
    elif side == 'hi':
        if slopes < 0:
             newPoints[0] = x + dist * np.cos(theta)
             newPoints[1] = y + dist * np.sin(theta) 
        else:
             newPoints[0] = x - dist * np.cos(theta)
             newPoints[1] = y - dist * np.sin(theta)               
    return newPoints


def pointAtDistLoop(points, slopes, dist, side='PS'):
    x = points[:,0]
    y = points[:,1]
    n = len(points)
    newPoints = np.zeros((n,2))
    for i in range(n):
        if np.isinf(slopes[i]):
            newPoints[i,0] = x[i]
            newPoints[i,1] = y[i] + dist[i]
        theta = np.arctan(slopes[i])
        if side =='PS':
            if slopes[i] < 0:
                 newPoints[i,0] = x[i] - dist[i] * np.cos(theta)
                 newPoints[i,1] = y[i] - dist[i] * np.sin(theta) 
            else:
                 newPoints[i,0] = x[i] + dist[i] * np.cos(theta)
                 newPoints[i,1] = y[i] + dist[i] * np.sin(theta)   
        elif side == 'SS':
            if slopes[i] < 0:
                 newPoints[i,0] = x[i] + dist[i] * np.cos(theta)
                 newPoints[i,1] = y[i] + dist[i] * np.sin(theta) 
            else:
                 newPoints[i,0] = x[i] - dist[i] * np.cos(theta)
                 newPoints[i,1] = y[i] - dist[i] * np.sin(theta)               
    return newPoints


def slopeAndMidPtsLoop(points):
    #This fuction computes the perpendicular slopes and midPoints 
    n = len(points)-1
    slopes = np.zeros(n)
    midpoints = np.zeros((n, 2))
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        if (p2[0] - p1[0]) > 1e-10:
            slope = (p2[1] - p1[1])/(p2[0] - p1[0])
        elif np.isinf(p2[1] - p1[1])/(p2[0] - p1[0]):
            slope = 0
        else:
            slope = np.inf
        midpoint = (p1 + p2) / 2
        slopes[i] = -1/slope
        midpoints[i] = midpoint
    return slopes, midpoints
def rotate_point(x, y, theta, inverse=False):
    if inverse:
        theta = -theta
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return R @ np.array([x, y]) 

def createOffsetCurve(bladePr, offsetPt, interX, dist, center, LEslope, TEslope, side):
    Np = 21  # JD: what is this? I think this is to sample points every 5% chord
    sampledBlade = np.linspace(bladePr[1][0], bladePr[-2][0], Np)
    sampledBlade = sampledBlade[2:-2]
    sampledBlade = np.delete(sampledBlade, int(np.floor(Np/2)-2)) #Delete the midpoint to replace with the already defined mid point   
    distFunc = interp1d([sampledBlade[0],  interX[0], sampledBlade[-1]],[dist[0], dist[1], dist[2]])
    totalDist = distFunc(sampledBlade)
    offset = np.zeros((len(sampledBlade), 2))
    for j in range(len(sampledBlade)):
        dMprime = np.abs(bladePr[:,0] - sampledBlade[j])
        minIdx = np.argmin(dMprime)
        sampledPtLow = bladePr[minIdx-1,:]
        sampledPtHigh = bladePr[minIdx+1,:]
        sampledPt = np.vstack((sampledPtLow,sampledPtHigh))
        slopes, midPts = slopeAndMidPts(sampledPt)
        offset[j,:] =  pointAtDist(midPts, slopes, totalDist[j], side=side)
    if side=='lo':
        offsetMidPt = offsetPt[1]
    elif side=='hi':
        offsetMidPt = offsetPt[2]
    fullOffset = np.vstack((offsetPt[0], offset[0:int(np.floor(Np/2))-2], offsetMidPt, offset[int(np.floor(Np/2))-2:], offsetPt[3]))
    rotFullOffset = fullOffset

    # This finds the slope at the 3rd poinst -- this is what the end curve
    # is to be tangent to
    thirdPtSlopeLE = mf.Slope(rotFullOffset[1][0], rotFullOffset[1][1], rotFullOffset[2][0], rotFullOffset[2][1])
    thirdPtSlopeTE = mf.Slope(rotFullOffset[-2][0], rotFullOffset[-2][1], rotFullOffset[-3][0], rotFullOffset[-3][1])

    # define points and slopes for Bezier curves
    p0LE = rotFullOffset[0]
    p2LE = rotFullOffset[1]
    slope0LE = LEslope

    slope2LE = thirdPtSlopeLE
    # print(f'LE slope joining to curve: {slope2LE}')
    offsetLEp1 = getBezierControlPoint(p0LE, p2LE, slope0LE, slope2LE)
    # print(f'LE control point: {offsetLEp1}')
    curveLE = quadratic_bezier_curve(p0=p0LE, p1=offsetLEp1, p2=p2LE)

    p0TE = rotFullOffset[-1]
    p2TE = rotFullOffset[-2]
    slope0TE = TEslope
    slope2TE = thirdPtSlopeTE
    # print(f'TE slope joining to curve: {slope2TE}')
    offsetTEp1 = getBezierControlPoint(p0TE, p2TE, slope0TE, slope2TE)
    # print(f'TE control point: {offsetTEp1}')
    curveTE = quadratic_bezier_curve(p0=p0TE, p1=offsetTEp1, p2=p2TE)

    # assemble overall curve
    curve = np.concatenate((curveLE, rotFullOffset[2:-2], curveTE[::-1]))

    # The curve at this point has dense points along the Bezier curves at the
    # end, but is very coarse with a point only every 5% chord or so in the
    # central section.
    #
    # Need to resample this curve to get a consistent fraction of arclength
    # map compared to the input data (bladePr).

    # 1. Get arclength fractions of points in bladePr
    bladeS = np.zeros(bladePr.shape[0])
    bladeDiffs = np.diff(bladePr, axis=0)
    bSloc = np.linalg.norm(bladeDiffs, axis=1)
    bladeS[1:] = np.cumsum(bSloc)  # cumulative arclength
    arcLengthFractionsb = bladeS / bladeS.max()
    arcLengthFractionsb[-1] = 1
    
    # 2. Get arclenth fractions of current points in curve
    curveS = np.zeros(curve.shape[0])
    curveDiffs = np.diff(curve, axis=0)
    cSloc = np.linalg.norm(curveDiffs, axis=1)
    curveS[1:] = np.cumsum(cSloc)  # cumulative arclength
    arcLengthFractionsc = curveS / curveS.max()
    arcLengthFractionsc[-1] = 1

    # 3. Interpolate new curve points (linear)
    curveInterp = interp1d(arcLengthFractionsc, curve, axis=0)
    curveNew = curveInterp(arcLengthFractionsb)

    return curveNew

def arcLenFrac(lengths):
    totalLength = np.sum(lengths)
    cumLengths = np.cumsum(lengths)
    lenFrac = cumLengths / totalLength
    lenFrac = np.concatenate([[0], lenFrac])
    lenFrac[-1] = 1
    return lenFrac
    
def offsetCurve(x, y, d, smooth=True, s=0, periodic=True):
    x = np.asarray(x)
    y = np.asarray(y)
    # For periodic curves, remove duplicate last point if present
    if periodic and np.allclose([x[0], y[0]], [x[-1], y[-1]]):
        x = x[:-1]
        y = y[:-1]
    if smooth:
        # Use spline representation for smooth derivatives
        tck, u = splprep([x, y], s=s, per=periodic)
        # Evaluate derivatives
        dx_du, dy_du = splev(u, tck, der=1)
    else:
        # Use finite differences
        if periodic:
            # Periodic boundary conditions for gradients
            dx_du = np.zeros_like(x)
            dy_du = np.zeros_like(y)
            # Interior points
            dx_du[1:-1] = (x[2:] - x[:-2]) / 2
            dy_du[1:-1] = (y[2:] - y[:-2]) / 2
            # Periodic endpoints
            dx_du[0] = (x[1] - x[-1]) / 2
            dy_du[0] = (y[1] - y[-1]) / 2
            dx_du[-1] = (x[0] - x[-2]) / 2
            dy_du[-1] = (y[0] - y[-2]) / 2
        else:
            dx_du = np.gradient(x)
            dy_du = np.gradient(y)
    # Calculate tangent magnitude
    tangent_mag = np.sqrt(dx_du**2 + dy_du**2)
    # Normalize tangent vector
    tx = dx_du / tangent_mag
    ty = dy_du / tangent_mag
    # Normal vector (perpendicular to tangent, rotated 90° CCW)
    nx = -ty
    ny = tx
    # Offset points along normal
    x_offset = x + d * nx
    y_offset = y + d * ny
    # Close the curve if periodic
    if periodic:
        x_offset = np.append(x_offset, x_offset[0])
        y_offset = np.append(y_offset, y_offset[0])
    return x_offset, y_offset

def cosineSpace( N, a=0.0, b=1.0):
    beta = np.linspace(0, np.pi, N)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(beta)
    return np.flip(x)

def arcLengthIndex(x, y, percent):
    x = np.asarray(x)
    y = np.asarray(y)
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.hstack(([0], np.cumsum(ds)))  # cumulative arc length
    totalLength = s[-1]
    target_s_low = percent[0] * totalLength
    target_s_high = (1-percent[1]) * totalLength
    # indices = np.argmin((s >= target_s_low).all() and (s <= target_s_high).all())
    indices = np.hstack(([0], np.where((s >= target_s_low) & (s <= target_s_high))[0], [len(x)-1]))
    indices = np.unique(indices)
    # indices = [np.argmin(np.abs(s - ts)) for ts in target_s]
    return s, indices

def nonDimFracChooser(x, percent):
    x = np.asarray(x) 
    # indices = np.argmin((s >= target_s_low).all() and (s <= target_s_high).all())
    indices = np.hstack(([0], np.where((x >= percent[0]) & (x <= (1-percent[1])))[0], [len(x)-1]))
    indices = np.unique(indices)
    # indices = [np.argmin(np.abs(s - ts)) for ts in target_s]
    return indices

def curveFrac(bladePt, offsetCurve, projectOffset, percent):
    arcLenPt,_ = arcLengthIndex(bladePt[:,0], bladePt[:,1],percent)
    lowBFrac = arcLenPt/arcLenPt[-1]
    lowIdx = nonDimFracChooser(lowBFrac, percent)
    curveLen,_ = arcLengthIndex(offsetCurve[:,0], offsetCurve[:,1], percent)
    curveLenFunc = CubicSpline(offsetCurve[:,0], curveLen)
    arcLenOff = curveLenFunc(projectOffset[:,0])
    lowOFrac = arcLenOff/ arcLenOff[-1]
    lowHub1 = np.array([lowBFrac[lowIdx], lowOFrac[lowIdx]])
    return lowHub1


#%% Defining the STLs
def createSTLs(Xvalues, Yvalues, Zvalues, filePath, passageNum):
    """ Create and write an STL file based on input Cartesian coordinates """
    filenames = ['blade1LowTheta','offset1LowTheta','blade1HighTheta','offset1HighTheta','blade2LowTheta','offset2LowTheta','blade2HighTheta','offset2HighTheta'
                 ,'LE','TE','midChord', 'hub', 'casing', 'inlet', 'outlet', 'inletLow', 'inletHigh', 'outletLow', 'outletHigh', 'highUpHub', 'highDwHub',
                 'lowUpHub', 'lowDwHub', 'highUpCas', 'highDwCas', 'lowUpCas', 'lowDwCas', 'highOBUp', 'lowOBUp', 'highOBDw', 'lowOBDw', 'midChordLow', 'midChordHigh']

    pNs = str(passageNum)

    dirPath = os.path.join(filePath, 'passage'+pNs)
    os.makedirs(dirPath, exist_ok=True)

    # Note: this creates STLs in Cartesian coordinates -- good for visual checks
    for qq in range(len(Xvalues)):
        filename = filePath + 'passage' + pNs + '/' + str(filenames[qq]) + '.stl'
        rows = Zvalues[qq].shape[0]  ## Use Z because it is the axis of rotation
        columns = Zvalues[qq].shape[1]
        X = Xvalues[qq]
        Y = Yvalues[qq]
        Z = Zvalues[qq]
        # Yvalues[qq] *= scale
        # Xvalues[qq] *= scale
        # X = np.arctan2(Yvalues[qq], Xvalues[qq])  # theta coordinate (radians)
        # Y = Zvalues[qq]  # axial coordinate
        # Z = np.sqrt(Xvalues[qq]**2+Yvalues[qq]**2)  # radial coordinate
        
        numFacets = 0
        
        file = open(filename, 'w')
        file.write('solid \n')
        
        for i in range(rows - 1):
            for j in range(columns - 1):
                #FACET A VERTICES            
                p1 = np.asarray([X[i, j], Y[i, j], Z[i, j]])
                p2 = np.asarray([X[i, j+1], Y[i, j+1], Z[i, j+1]])
                p3 = np.asarray([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
                
                unitVector(file, p1, p2, p3)
                
                p1 = np.asarray([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
                p2 = np.asarray([X[i+1, j], Y[i+1, j], Z[i+1, j]])
                p3 = np.asarray([X[i, j], Y[i, j], Z[i, j]]) 
                
                unitVector(file, p1, p2, p3)
                
                numFacets += 2
        
        file.write('endsolid')
        file.close()
        
        print(f'Number of Facets = {numFacets} \n')


#%% Write out input file 
def calcAndWritePassageParameters(scale, Xvalues, Yvalues, Zvalues, nrad, delHub, delCas, delBla, dy1Hub, dy1Cas, dy1Bla, gRad, gTan, dax1primeLE, rLE, dax1primeTE, rTE, rUpFar, rDnFar, dataPath, passageNum, additionalTangentialRefine, additionalAxialRefine, highHub1, lowHub1, highCas1, lowCas1, highHub2, lowHub2, highCas2, lowCas2):
    """ compute and write all passage-specific parameters to file
    to be parsed by a Bash script to modify dictionaries prior to
    generating the mesh using blockMesh (OpenFOAM tool)"""

    # This combines stuff I wrote and stuff Tony wrote in a single function.

    # Updated vertex writing from Tony/Adekola

    # First, Tony Woo part -- vertex extraction

    # notation: hub|cas -- hub or casing
    #           p|n     -- high or low theta
    #           I K L M/P T U O -- "axial" locations: inlet, offset u/s end,
    #                      LE, midchord onblade/at offset, TE, offset d/s end,
    # First index is which surface
    # Second index is from hub to casing
    # Third index is from upstream to downstream

    hubIp = np.array([Xvalues[16][0][0], Yvalues[16][0][0], Zvalues[16][0][0]])
    casIp = np.array([Xvalues[16][-1][0], Yvalues[16][-1][0], Zvalues[16][-1][0]])
    hubKp = np.array([Xvalues[3][0][0], Yvalues[3][0][0], Zvalues[3][0][0]])
    casKp = np.array([Xvalues[3][-1][0], Yvalues[3][-1][0], Zvalues[3][-1][0]])
    hubLp = np.array([Xvalues[2][0][0], Yvalues[2][0][0], Zvalues[2][0][0]])
    casLp = np.array([Xvalues[2][-1][0], Yvalues[2][-1][0], Zvalues[2][-1][0]])
    hubMp = np.array([Xvalues[2][0][-1], Yvalues[2][0][-1], Zvalues[2][0][-1]])
    casMp = np.array([Xvalues[2][-1][-1], Yvalues[2][-1][-1], Zvalues[2][-1][-1]])
    hubPp = np.array([Xvalues[3][0][-1], Yvalues[3][0][-1], Zvalues[3][0][-1]])
    casPp = np.array([Xvalues[3][-1][-1], Yvalues[3][-1][-1], Zvalues[3][-1][-1]])
    hubTp = np.array([Xvalues[6][0][-1], Yvalues[6][0][-1], Zvalues[6][0][-1]])
    casTp = np.array([Xvalues[6][-1][-1], Yvalues[6][-1][-1], Zvalues[6][-1][-1]])
    hubUp = np.array([Xvalues[7][0][-1], Yvalues[7][0][-1], Zvalues[7][0][-1]])
    casUp = np.array([Xvalues[7][-1][-1], Yvalues[7][-1][-1], Zvalues[7][-1][-1]])
    hubOp = np.array([Xvalues[18][0][-1], Yvalues[18][0][-1], Zvalues[18][0][-1]])
    casOp = np.array([Xvalues[18][-1][-1], Yvalues[18][-1][-1], Zvalues[18][-1][-1]])

    hubIn = np.array([Xvalues[15][0][0], Yvalues[15][0][0], Zvalues[15][0][0]])
    casIn = np.array([Xvalues[15][-1][0], Yvalues[15][-1][0], Zvalues[15][-1][0]])
    hubKn = np.array([Xvalues[1][0][0], Yvalues[1][0][0], Zvalues[1][0][0]])
    casKn = np.array([Xvalues[1][-1][0], Yvalues[1][-1][0], Zvalues[1][-1][0]])
    hubLn = np.array([Xvalues[0][0][0], Yvalues[0][0][0], Zvalues[0][0][0]])
    casLn = np.array([Xvalues[0][-1][0], Yvalues[0][-1][0], Zvalues[0][-1][0]])
    hubMn = np.array([Xvalues[0][0][-1], Yvalues[0][0][-1], Zvalues[0][0][-1]])
    casMn = np.array([Xvalues[0][-1][-1], Yvalues[0][-1][-1], Zvalues[0][-1][-1]])
    hubPn = np.array([Xvalues[1][0][-1], Yvalues[1][0][-1], Zvalues[1][0][-1]])
    casPn = np.array([Xvalues[1][-1][-1], Yvalues[1][-1][-1], Zvalues[1][-1][-1]])
    hubTn = np.array([Xvalues[4][0][-1], Yvalues[4][0][-1], Zvalues[4][0][-1]])
    casTn = np.array([Xvalues[4][-1][-1], Yvalues[4][-1][-1], Zvalues[4][-1][-1]])
    hubUn = np.array([Xvalues[5][0][-1], Yvalues[5][0][-1], Zvalues[5][0][-1]])
    casUn = np.array([Xvalues[5][-1][-1], Yvalues[5][-1][-1], Zvalues[5][-1][-1]])
    hubOn = np.array([Xvalues[17][0][-1], Yvalues[17][0][-1], Zvalues[17][0][-1]])
    casOn = np.array([Xvalues[17][-1][-1], Yvalues[17][-1][-1], Zvalues[17][-1][-1]])

    Rvalues = [np.empty((0,)) for _ in range(len(Xvalues))]
    thetavalues = [np.empty((0,)) for _ in range(len(Xvalues))]
    for i in range(len(Xvalues)):
        Rvalues[i] = np.sqrt(Xvalues[i]**2 + Yvalues[i]**2)
        thetavalues[i] = np.arctan2(Yvalues[i], Xvalues[i])

    hubIpCyl = np.array([Rvalues[16][0][0], thetavalues[16][0][0], Zvalues[16][0][0]])
    casIpCyl = np.array([Rvalues[16][-1][0], thetavalues[16][-1][0], Zvalues[16][-1][0]])
    hubKpCyl = np.array([Rvalues[3][0][0], thetavalues[3][0][0], Zvalues[3][0][0]])
    casKpCyl = np.array([Rvalues[3][-1][0], thetavalues[3][-1][0], Zvalues[3][-1][0]])
    hubLpCyl = np.array([Rvalues[2][0][0], thetavalues[2][0][0], Zvalues[2][0][0]])
    casLpCyl = np.array([Rvalues[2][-1][0], thetavalues[2][-1][0], Zvalues[2][-1][0]])
    hubMpCyl = np.array([Rvalues[2][0][-1], thetavalues[2][0][-1], Zvalues[2][0][-1]])
    casMpCyl = np.array([Rvalues[2][-1][-1], thetavalues[2][-1][-1], Zvalues[2][-1][-1]])
    hubPpCyl = np.array([Rvalues[3][0][-1], thetavalues[3][0][-1], Zvalues[3][0][-1]])
    casPpCyl = np.array([Rvalues[3][-1][-1], thetavalues[3][-1][-1], Zvalues[3][-1][-1]])
    hubTpCyl = np.array([Rvalues[6][0][-1], thetavalues[6][0][-1], Zvalues[6][0][-1]])
    casTpCyl = np.array([Rvalues[6][-1][-1], thetavalues[6][-1][-1], Zvalues[6][-1][-1]])
    hubUpCyl = np.array([Rvalues[7][0][-1], thetavalues[7][0][-1], Zvalues[7][0][-1]])
    casUpCyl = np.array([Rvalues[7][-1][-1], thetavalues[7][-1][-1], Zvalues[7][-1][-1]])
    hubOpCyl = np.array([Rvalues[18][0][-1], thetavalues[18][0][-1], Zvalues[18][0][-1]])
    casOpCyl = np.array([Rvalues[18][-1][-1], thetavalues[18][-1][-1], Zvalues[18][-1][-1]])

    hubInCyl = np.array([Rvalues[15][0][0], thetavalues[15][0][0], Zvalues[15][0][0]])
    casInCyl = np.array([Rvalues[15][-1][0], thetavalues[15][-1][0], Zvalues[15][-1][0]])
    hubKnCyl = np.array([Rvalues[1][0][0], thetavalues[1][0][0], Zvalues[1][0][0]])
    casKnCyl = np.array([Rvalues[1][-1][0], thetavalues[1][-1][0], Zvalues[1][-1][0]])
    hubLnCyl = np.array([Rvalues[0][0][0], thetavalues[0][0][0], Zvalues[0][0][0]])
    casLnCyl = np.array([Rvalues[0][-1][0], thetavalues[0][-1][0], Zvalues[0][-1][0]])
    hubMnCyl = np.array([Rvalues[0][0][-1], thetavalues[0][0][-1], Zvalues[0][0][-1]])
    casMnCyl = np.array([Rvalues[0][-1][-1], thetavalues[0][-1][-1], Zvalues[0][-1][-1]])
    hubPnCyl = np.array([Rvalues[1][0][-1], thetavalues[1][0][-1], Zvalues[1][0][-1]])
    casPnCyl = np.array([Rvalues[1][-1][-1], thetavalues[1][-1][-1], Zvalues[1][-1][-1]])
    hubTnCyl = np.array([Rvalues[4][0][-1], thetavalues[4][0][-1], Zvalues[4][0][-1]])
    casTnCyl = np.array([Rvalues[4][-1][-1], thetavalues[4][-1][-1], Zvalues[4][-1][-1]])
    hubUnCyl = np.array([Rvalues[5][0][-1], thetavalues[5][0][-1], Zvalues[5][0][-1]])
    casUnCyl = np.array([Rvalues[5][-1][-1], thetavalues[5][-1][-1], Zvalues[5][-1][-1]])
    hubOnCyl = np.array([Rvalues[17][0][-1], thetavalues[17][0][-1], Zvalues[17][0][-1]])
    casOnCyl = np.array([Rvalues[17][-1][-1], thetavalues[17][-1][-1], Zvalues[17][-1][-1]])

    # Second, get all the geometric and grid parameters calculated
    # This section and called functions all written by Jeff Defoe

    # Radial grading-related parameters
    nrad = round(nrad/2) * 2  # makes sure nRad is an even number
    # Spans
    spanI = 0.5*(np.linalg.norm(casIp-hubIp) + np.linalg.norm(casIn-hubIn))
    spanL = 0.5*(np.linalg.norm(casLp-hubLp) + np.linalg.norm(casLn-hubLn))
    spanM = 0.5*(np.linalg.norm(casMp-hubMp) + np.linalg.norm(casMn-hubMn))
    spanT = 0.5*(np.linalg.norm(casTp-hubTp) + np.linalg.norm(casTn-hubTn))
    spanO = 0.5*(np.linalg.norm(casOp-hubOp) + np.linalg.norm(casOn-hubOn))

    # estimate midspan / midpassage radial cell size
    # get pitches at midpassage
    pitchH = np.linalg.norm(hubMp-hubMn)
    pitchC = np.linalg.norm(casMp-casMn)
    # lengths, number of cells, grading --> cell sizes in middle
    # account for endwall BLs but not blade ones, since M blade points
    # are already pushed in
    rLen = (spanM - delHub - delCas) / 2
    rCells = nrad / 2
    rExpRatio = gRad ** (1 / (rCells - 1))
    drOuter = rLen / ((1 - rExpRatio**rCells) / (1 - rExpRatio))
    drMiddle = drOuter * rExpRatio ** (rCells - 1)  # This is the approximate radial cell size midpassage
    # Calculate hub/casing BL number of cells, expansion ratio
    print('Calculating number of BL cells for hub/casing...')
    nBLcellsHub, rBLcellsHub = getNumBLCells(dy1Hub, drOuter, delHub)
    nBLcellsCas, rBLcellsCas = getNumBLCells(dy1Cas, drOuter, delCas)
    print(f'Hub: {nBLcellsHub} with r={rBLcellsHub}')
    print(f'Casing: {nBLcellsCas} with r={rBLcellsCas}')
    # Determine splits for refineWallLayer
    splitsHub = splitsCalc(nBLcellsHub, rBLcellsHub)
    splitsCas = splitsCalc(nBLcellsCas, rBLcellsCas)
    print('refineWallLayer splits for hub:')
    print(splitsHub)
    print('refineWallLayer splits for casing:')
    print(splitsCas)

    # Now for tangential grading:
    dtMiddle = drMiddle/additionalTangentialRefine  # Target AR = 1 for the middle of the passage with additional refinement = 1
    # now need to get tCells (start work on tangential grading)
    tLen = 0.5*(pitchH + pitchC) / 2
    tCells = np.round(fsolve(lambda n: -gTan*tLen/dtMiddle + (1-gTan**(n/(n-1)))/(1-gTan**(1/(n-1))), 2))
    dtOuter = dtMiddle / gTan
    tExpRatio = gTan ** (1 / (tCells -1))
    # tangential points is double this as it covers full width
    ntan = int(tCells[0] * 2)
    # Calculate blade BL number of cells, expansion ratio
    print('Calculating number of BL cells for blade...')
    nBLcellsBlade, rBLcellsBlade = getNumBLCells(dy1Bla, dtOuter, delBla)
    print(f'Blade: {nBLcellsBlade} with r={rBLcellsBlade}')
    # Determine splits for refineWallLayer
    splitsBlade = splitsCalc(nBLcellsBlade, rBLcellsBlade)
    print('refineWallLayer splits for blade:')
    print(splitsBlade)
    
    # Axial grading parameters
    # for over-blade blocks, use same AR = 1 approach at mid-passage with additional refinement = 1
    # to define cell size
    dxMiddle = drMiddle/additionalAxialRefine
    # chord at hub and casing for the pos and neg side blades
    cHP = np.linalg.norm(hubTp-hubLp)
    cHN = np.linalg.norm(hubTn-hubLn)
    cCP = np.linalg.norm(casTp-casLp)
    cCN = np.linalg.norm(casTn-casLn)
    # axial length of over-blade blocks
    xLen2 = 0.5*0.25*(cHP + cHN + cCP + cCN)
    xLen3 = xLen2  # both are the same by definition since midchord divides them
    # axial grading ratios and number of cells for over-blade blocks
    print('Solving for number of cells for first between-blades block...')
    root = fsolve(bladeGradingFunction, x0=[2.0*(0.01/dax1primeLE), 1.5, nrad], args=(dxMiddle, dax1primeLE, xLen2, rLE))
    gLE, gM1, nax2 = root
    nax2 = int(np.round(nax2))
    fLE = (np.log(gLE)/np.log(rLE) + 1)/nax2
    print(f'nax2={nax2}')
    print('Solving for number of cells for second between-blades block...')
    root = fsolve(bladeGradingFunction, x0=[2.0*(0.01/dax1primeTE), 1.5, nrad], args=(dxMiddle, dax1primeTE, xLen3, rTE))
    gTE, gM2, nax3 = root
    nax3 = int(np.round(nax3))
    fTE = (np.log(gTE)/np.log(rTE) + 1)/nax3
    gTE = 1/gTE  # invert results since this block "goes the other way"
    gM2 = 1/gM2  # invert results since this block "goes the other way"
    print(f'nax3={nax3}')

    # Re-solve on each edge now that nax2, nax3 are known
    # get arc lengths
    SbladeUpHP = calcArcLength(np.array([Xvalues[2][0][:], Yvalues[2][0][:], Zvalues[2][0][:]]))[-1]
    SbladeUpHN = calcArcLength(np.array([Xvalues[0][0][:], Yvalues[0][0][:], Zvalues[0][0][:]]))[-1]
    SbladeUpCP = calcArcLength(np.array([Xvalues[2][-1][:], Yvalues[2][-1][:], Zvalues[2][-1][:]]))[-1]
    SbladeUpCN = calcArcLength(np.array([Xvalues[0][-1][:], Yvalues[0][-1][:], Zvalues[0][-1][:]]))[-1]
    SbladeDnHP = calcArcLength(np.array([Xvalues[6][0][:], Yvalues[6][0][:], Zvalues[6][0][:]]))[-1]
    SbladeDnHN = calcArcLength(np.array([Xvalues[4][0][:], Yvalues[4][0][:], Zvalues[4][0][:]]))[-1]
    SbladeDnCP = calcArcLength(np.array([Xvalues[6][-1][:], Yvalues[6][-1][:], Zvalues[6][-1][:]]))[-1]
    SbladeDnCN = calcArcLength(np.array([Xvalues[4][-1][:], Yvalues[4][-1][:], Zvalues[4][-1][:]]))[-1]

    # solve for grading parameters
    print('Determining axial grading parameters for edges of between-blade blocks...')
    print('block 2, hub, positive theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fLE, gLE, gM1], args=(nax2, dxMiddle, dax1primeLE*0.5*cHP, SbladeUpHP))
    fLEHP, gLEHP, gM1HP = root
    print('block 2, hub, negative theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fLE, gLE, gM1], args=(nax2, dxMiddle, dax1primeLE*0.5*cHN, SbladeUpHN))
    fLEHN, gLEHN, gM1HN = root
    print('block 2, casing, positive theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fLE, gLE, gM1], args=(nax2, dxMiddle, dax1primeLE*0.5*cCP, SbladeUpCP))
    fLECP, gLECP, gM1CP = root
    print('block 2, casing, negative theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fLE, gLE, gM1], args=(nax2, dxMiddle, dax1primeLE*0.5*cCN, SbladeUpCN))
    fLECN, gLECN, gM1CN = root
    print('block 3, hub, positive theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fTE, 1/gTE, 1/gM2], args=(nax3, dxMiddle, dax1primeTE*0.5*cHP, SbladeDnHP))
    fTEHP, gTEHP, gM2HP = root
    gTEHP = 1/gTEHP
    gM2HP = 1/gM2HP
    print('block 3, hub, negative theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fTE, 1/gTE, 1/gM2], args=(nax3, dxMiddle, dax1primeTE*0.5*cHN, SbladeDnHN))
    fTEHN, gTEHN, gM2HN = root
    gTEHN = 1/gTEHN
    gM2HN = 1/gM2HN
    print('block 3, casing, positive theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fTE, 1/gTE, 1/gM2], args=(nax3, dxMiddle, dax1primeTE*0.5*cCP, SbladeDnCP))
    fTECP, gTECP, gM2CP = root
    gTECP = 1/gTECP
    gM2CP = 1/gM2CP
    print('block 3, casing, negative theta:')
    root = fsolve(bladeGradingFunctionSpecific, x0=[fTE, 1/gTE, 1/gM2], args=(nax3, dxMiddle, dax1primeTE*0.5*cCN, SbladeDnCN))
    fTECN, gTECN, gM2CN = root
    gTECN = 1/gTECN
    gM2CN = 1/gM2CN    

    # calculate fraction of arclength splits for all cell divisions on blades
    pointFracsBlade2HP = pointFracs(nax2, np.array([0.25, 0.75]), np.array([fLEHP, 1-fLEHP]), np.array([gLEHP, gM1HP]))
    pointFracsBlade2HN = pointFracs(nax2, np.array([0.25, 0.75]), np.array([fLEHN, 1-fLEHN]), np.array([gLEHN, gM1HN]))
    pointFracsBlade2CP = pointFracs(nax2, np.array([0.25, 0.75]), np.array([fLECP, 1-fLECP]), np.array([gLECP, gM1CP]))
    pointFracsBlade2CN = pointFracs(nax2, np.array([0.25, 0.75]), np.array([fLECN, 1-fLECN]), np.array([gLECN, gM1CN]))

    pointFracsBlade3HP = pointFracs(nax3, np.array([0.75, 0.25]), np.array([1-fTEHP, fTEHP]), np.array([gM2HP, gTEHP]))
    pointFracsBlade3HN = pointFracs(nax3, np.array([0.75, 0.25]), np.array([1-fTEHN, fTEHN]), np.array([gM2HN, gTEHN]))
    pointFracsBlade3CP = pointFracs(nax3, np.array([0.75, 0.25]), np.array([1-fTECP, fTECP]), np.array([gM2CP, gTECP]))
    pointFracsBlade3CN = pointFracs(nax3, np.array([0.75, 0.25]), np.array([1-fTECN, fTECN]), np.array([gM2CN, gTECN]))

    # axial grading ratios and cell split for offsets

    BtoOarcLengthMap2HP = highHub1
    BtoOarcLengthMap2HN = lowHub1
    BtoOarcLengthMap2CP = highCas1
    BtoOarcLengthMap2CN = lowCas1

    BtoOarcLengthMap3HP = highHub2
    BtoOarcLengthMap3HN = lowHub2
    BtoOarcLengthMap3CP = highCas2
    BtoOarcLengthMap3CN = lowCas2

    # Interpolate mapped edge fractions
    pointFracsOffset2HP = np.interp(pointFracsBlade2HP, BtoOarcLengthMap2HP[:,0], BtoOarcLengthMap2HP[:,1])
    pointFracsOffset2HN = np.interp(pointFracsBlade2HN, BtoOarcLengthMap2HN[:,0], BtoOarcLengthMap2HN[:,1])
    pointFracsOffset2CP = np.interp(pointFracsBlade2CP, BtoOarcLengthMap2CP[:,0], BtoOarcLengthMap2CP[:,1])
    pointFracsOffset2CN = np.interp(pointFracsBlade2CN, BtoOarcLengthMap2CN[:,0], BtoOarcLengthMap2CN[:,1])

    pointFracsOffset3HP = np.interp(pointFracsBlade3HP, BtoOarcLengthMap3HP[:,0], BtoOarcLengthMap3HP[:,1])
    pointFracsOffset3HN = np.interp(pointFracsBlade3HN, BtoOarcLengthMap3HN[:,0], BtoOarcLengthMap3HN[:,1])
    pointFracsOffset3CP = np.interp(pointFracsBlade3CP, BtoOarcLengthMap3CP[:,0], BtoOarcLengthMap3CP[:,1])
    pointFracsOffset3CN = np.interp(pointFracsBlade3CN, BtoOarcLengthMap3CN[:,0], BtoOarcLengthMap3CN[:,1])

    # Define grading parameters for offsets
    print('Setting offset axial grading...')
    axgrading2HPoffset = blockMeshGradDescriptorBuilder(pointFracsOffset2HP, 'axgrading2HPoffset')
    axgrading2HNoffset = blockMeshGradDescriptorBuilder(pointFracsOffset2HN, 'axgrading2HNoffset')
    axgrading2CPoffset = blockMeshGradDescriptorBuilder(pointFracsOffset2CP, 'axgrading2CPoffset')
    axgrading2CNoffset = blockMeshGradDescriptorBuilder(pointFracsOffset2CN, 'axgrading2CNoffset')
    axgrading3HPoffset = blockMeshGradDescriptorBuilder(pointFracsOffset3HP, 'axgrading3HPoffset')
    axgrading3HNoffset = blockMeshGradDescriptorBuilder(pointFracsOffset3HN, 'axgrading3HNoffset')
    axgrading3CPoffset = blockMeshGradDescriptorBuilder(pointFracsOffset3CP, 'axgrading3CPoffset')
    axgrading3CNoffset = blockMeshGradDescriptorBuilder(pointFracsOffset3CN, 'axgrading3CNoffset')

    #ipdb.set_trace()

    # for up- and down-stream blocks, bring other known parameters
    # into play to determine correct number of cells
    # Just 2 parameters in play: mean expansion ratio of cells
    # further than 1/2 chord away from blade row (rUpFar, rDnFar)
    # block 1
    L1HP = np.linalg.norm(hubLp-hubIp)
    L1HN = np.linalg.norm(hubLn-hubIn)
    L1CP = np.linalg.norm(casLp-casIp)
    L1CN = np.linalg.norm(casLn-casIn)
    # block 4
    L4HP = np.linalg.norm(hubOp-hubTp)
    L4HN = np.linalg.norm(hubOn-hubTn)
    L4CP = np.linalg.norm(casOp-casTp)
    L4CN = np.linalg.norm(casOn-casTn)
    # interior first cell sizes
    dxAtBlade1 = dax1primeLE*xLen2
    dxAtBlade4 = dax1primeTE*xLen3
    # Need to first evaluate this approach with blended values
    # of L and C to solve for n (number of cells)
    # Upstream:
    L1avg = 0.25*(L1HP+L1HN+L1CP+L1CN)
    cavg = 0.25*(cHP+cHN+cCP+cCN)
    if rUpFar == 1.0:  # Prevent divide-by-zero errors
        rUpFar = 1.01
    print('Determining number of axial cells for upstream block...')
    g1clo, r1clo, f1clo, g1far, nax1 = outerGradingCaseChooser(L1avg, cavg, 0.5*(dxAtBlade1+dtOuter), rUpFar, dxMiddle, 1/gLE, nax2, fLE)
    # convert results to be appropriate for the upstream block
    g1clo = 1/g1clo
    r1clo = 1/r1clo
    g1far = 1/g1far
    rUpFar = 1/rUpFar
    nax1 = int(nax1)
    # Downstream:
    L4avg = 0.25*(L4HP+L4HN+L4CP+L4CN)
    if rDnFar == 1.0:  # Prevent divide-by-zero errors
        rDnFar = 1.01
    print('Determining number of axial cells for downstream block...')
    g4clo, r4clo, f4clo, g4far, nax4 = outerGradingCaseChooser(L4avg, cavg, 0.5*(dxAtBlade4+dtOuter), rDnFar, dxMiddle, gTE, nax3, fTE)
    nax4 = int(nax4)
    # Then recompute g/f/r values for each edge
    print('Determining axial grading parameters for edges of upstream and downstream blocks...')
    print('block 1, hub, positive theta:')
    g1HPclo, f1HPclo, g1HPfar = outerGradingCaseChooserSpecific(L1HP, cHP, 0.5*(dxAtBlade1+dtOuter), dxMiddle, nax1, 1/r1clo, f1clo)
    g1HPclo = 1/g1HPclo
    g1HPfar = 1/g1HPfar
    print('block 1, hub, negative theta:')
    g1HNclo, f1HNclo, g1HNfar = outerGradingCaseChooserSpecific(L1HN, cHN, 0.5*(dxAtBlade1+dtOuter), dxMiddle, nax1, 1/r1clo, f1clo)
    g1HNclo = 1/g1HNclo
    g1HNfar = 1/g1HNfar
    print('block 1, casing, positive theta:')
    g1CPclo, f1CPclo, g1CPfar = outerGradingCaseChooserSpecific(L1CP, cCP, 0.5*(dxAtBlade1+dtOuter), dxMiddle, nax1, 1/r1clo, f1clo)
    g1CPclo = 1/g1CPclo
    g1CPfar = 1/g1CPfar
    print('block 1, casing, negative theta:')
    g1CNclo, f1CNclo, g1CNfar = outerGradingCaseChooserSpecific(L1CN, cCN, 0.5*(dxAtBlade1+dtOuter), dxMiddle, nax1, 1/r1clo, f1clo)
    g1CNclo = 1/g1CNclo
    g1CNfar = 1/g1CNfar
    print('block 4, hub, positive theta:')
    g4HPclo, f4HPclo, g4HPfar = outerGradingCaseChooserSpecific(L4HP, cHP, 0.5*(dxAtBlade4+dtOuter), dxMiddle, nax4, r4clo, f4clo)
    print('block 4, hub, negative theta:')
    g4HNclo, f4HNclo, g4HNfar = outerGradingCaseChooserSpecific(L4HN, cHN, 0.5*(dxAtBlade4+dtOuter), dxMiddle, nax4, r4clo, f4clo)
    print('block 4, casing, positive theta:')
    g4CPclo, f4CPclo, g4CPfar = outerGradingCaseChooserSpecific(L4CP, cCP, 0.5*(dxAtBlade4+dtOuter), dxMiddle, nax4, r4clo, f4clo)
    print('block 4, casing, negative theta:')
    g4CNclo, f4CNclo, g4CNfar = outerGradingCaseChooserSpecific(L4CN, cCN, 0.5*(dxAtBlade4+dtOuter), dxMiddle, nax4, r4clo, f4clo)
    # Tangential grading at inlet/outlet
    # Adjustable back-end parameter. Contraction rate of axial cell
    # lines moving away from blade LE/TE towards domain inlet/outlet
    # (max allowed). Used below to determine grading ratio at inlet/
    # outlet.
    angleLim = 20  # degrees
    print(f'Calculating inlet/outlet tangential grading based on limiting cell line contraction angle of {angleLim} degrees...')
    # At hub:
    gTanIhub = getTanGradingAtInletOutlet(0.5*pitchH, gTan, tExpRatio, ntan/2, angleLim, 0.5*(L1HP+L1HN))
    gTanOhub = getTanGradingAtInletOutlet(0.5*pitchH, gTan, tExpRatio, ntan/2, angleLim, 0.5*(L4HP+L4HN))
    # At casing:
    gTanIcas = getTanGradingAtInletOutlet(0.5*pitchC, gTan, tExpRatio, ntan/2, angleLim, 0.5*(L1CP+L1CN))
    gTanOcas = getTanGradingAtInletOutlet(0.5*pitchC, gTan, tExpRatio, ntan/2, angleLim, 0.5*(L4CP+L4CN))

    # Produce output file
    # set file name
    dirPath = os.path.join(dataPath, 'passage'+str(passageNum))
    os.makedirs(dirPath, exist_ok=True)
    scriptFile = os.path.join(dataPath, 'passage'+str(passageNum), 'passageParameters')
    # open file
    paramFile = open(scriptFile, 'w')
    # write scale parameter
    #  paramFile.write(f'scale  ({scale} 1 {scale});  \n')  # Cyl. coords ver.
    paramFile.write(f'scale  {scale};  \n')              # Cart. coords ver.
    # write geometric and grid parameters
    paramFile.write('nrad   {}; \n'.format(nrad))
    paramFile.write('delHub   {}; \n'.format(delHub))
    paramFile.write('delCas   {}; \n'.format(delCas))
    paramFile.write('bladeBLthickness   {}; \n'.format(delBla))
    paramFile.write('dely1Blade   {}; \n'.format(dy1Bla))
    paramFile.write('dy1Hub   {}; \n'.format(dy1Hub))
    paramFile.write('dy1Cas   {}; \n'.format(dy1Cas))
    paramFile.write('gRad   {}; \n'.format(gRad))
    paramFile.write('gTan   {}; \n'.format(gTan))

    paramFile.write('ntan   {}; \n'.format(ntan))
    paramFile.write('nax1   {}; \n'.format(nax1))
    paramFile.write('nax2   {}; \n'.format(nax2))
    paramFile.write('nax3   {}; \n'.format(nax3))
    paramFile.write('nax4   {}; \n'.format(nax4))

    paramFile.write('spanI   {}; \n'.format(spanI))
    paramFile.write('spanL   {}; \n'.format(spanL))
    paramFile.write('spanM   {}; \n'.format(spanM))
    paramFile.write('spanT   {}; \n'.format(spanT))
    paramFile.write('spanO   {}; \n'.format(spanO))

    paramFile.write('gTanIhub   {}; \n'.format(gTanIhub))
    paramFile.write('gTanOhub   {}; \n'.format(gTanOhub))
    paramFile.write('gTanIcas   {}; \n'.format(gTanIcas))
    paramFile.write('gTanOcas   {}; \n'.format(gTanOcas))

    paramFile.write('fLEHP   {}; \n'.format(fLEHP))
    paramFile.write('gLEHP   {}; \n'.format(gLEHP))
    paramFile.write('gM1HP   {}; \n'.format(gM1HP))
    paramFile.write('fTEHP   {}; \n'.format(fTEHP))
    paramFile.write('gTEHP   {}; \n'.format(gTEHP))
    paramFile.write('gM2HP   {}; \n'.format(gM2HP))

    paramFile.write('fLEHN   {}; \n'.format(fLEHN))
    paramFile.write('gLEHN   {}; \n'.format(gLEHN))
    paramFile.write('gM1HN   {}; \n'.format(gM1HN))
    paramFile.write('fTEHN   {}; \n'.format(fTEHN))
    paramFile.write('gTEHN   {}; \n'.format(gTEHN))
    paramFile.write('gM2HN   {}; \n'.format(gM2HN))

    paramFile.write('fLECP   {}; \n'.format(fLECP))
    paramFile.write('gLECP   {}; \n'.format(gLECP))
    paramFile.write('gM1CP   {}; \n'.format(gM1CP))
    paramFile.write('fTECP   {}; \n'.format(fTECP))
    paramFile.write('gTECP   {}; \n'.format(gTECP))
    paramFile.write('gM2CP   {}; \n'.format(gM2CP))

    paramFile.write('fLECN   {}; \n'.format(fLECN))
    paramFile.write('gLECN   {}; \n'.format(gLECN))
    paramFile.write('gM1CN   {}; \n'.format(gM1CN))
    paramFile.write('fTECN   {}; \n'.format(fTECN))
    paramFile.write('gTECN   {}; \n'.format(gTECN))
    paramFile.write('gM2CN   {}; \n'.format(gM2CN))

    paramFile.write('cHP   {}; \n'.format(cHP))
    paramFile.write('cHN   {}; \n'.format(cHN))
    paramFile.write('cCP   {}; \n'.format(cCP))
    paramFile.write('cCN   {}; \n'.format(cCN))

    paramFile.write('L1HP   {}; \n'.format(L1HP))
    paramFile.write('L1HN   {}; \n'.format(L1HN))
    paramFile.write('L1CP   {}; \n'.format(L1CP))
    paramFile.write('L1CN   {}; \n'.format(L1CN))
    paramFile.write('L4HP   {}; \n'.format(L4HP))
    paramFile.write('L4HN   {}; \n'.format(L4HN))
    paramFile.write('L4CP   {}; \n'.format(L4CP))
    paramFile.write('L4CN   {}; \n'.format(L4CN))

    paramFile.write('g1HPfar   {}; \n'.format(g1HPfar))
    paramFile.write('g1HNfar   {}; \n'.format(g1HNfar))
    paramFile.write('g1CPfar   {}; \n'.format(g1CPfar))
    paramFile.write('g1CNfar   {}; \n'.format(g1CNfar))
    paramFile.write('g1HPclo   {}; \n'.format(g1HPclo))
    paramFile.write('g1HNclo   {}; \n'.format(g1HNclo))
    paramFile.write('g1CPclo   {}; \n'.format(g1CPclo))
    paramFile.write('g1CNclo   {}; \n'.format(g1CNclo))
    paramFile.write('f1HPclo   {}; \n'.format(f1HPclo))
    paramFile.write('f1HNclo   {}; \n'.format(f1HNclo))
    paramFile.write('f1CPclo   {}; \n'.format(f1CPclo))
    paramFile.write('f1CNclo   {}; \n'.format(f1CNclo))

    paramFile.write('g4HPfar   {}; \n'.format(g4HPfar))
    paramFile.write('g4HNfar   {}; \n'.format(g4HNfar))
    paramFile.write('g4CPfar   {}; \n'.format(g4CPfar))
    paramFile.write('g4CNfar   {}; \n'.format(g4CNfar))
    paramFile.write('g4HPclo   {}; \n'.format(g4HPclo))
    paramFile.write('g4HNclo   {}; \n'.format(g4HNclo))
    paramFile.write('g4CPclo   {}; \n'.format(g4CPclo))
    paramFile.write('g4CNclo   {}; \n'.format(g4CNclo))
    paramFile.write('f4HPclo   {}; \n'.format(f4HPclo))
    paramFile.write('f4HNclo   {}; \n'.format(f4HNclo))
    paramFile.write('f4CPclo   {}; \n'.format(f4CPclo))
    paramFile.write('f4CNclo   {}; \n'.format(f4CNclo))

    paramFile.write(axgrading2HPoffset + '\n')
    paramFile.write(axgrading2HNoffset + '\n')
    paramFile.write(axgrading2CPoffset + '\n')
    paramFile.write(axgrading2CNoffset + '\n')

    paramFile.write(axgrading3HPoffset + '\n')
    paramFile.write(axgrading3HNoffset + '\n')
    paramFile.write(axgrading3CPoffset + '\n')
    paramFile.write(axgrading3CNoffset + '\n')

    # BL parameters
    paramFile.write('nBLcellsHub   {}; \n'.format(nBLcellsHub))
    paramFile.write('nBLcellsCas   {}; \n'.format(nBLcellsCas))
    paramFile.write('nBLcellsBlade   {}; \n'.format(nBLcellsBlade))

    # Write vertex coordinates
    
    paramFile.write(format_coord('hubIp', hubIp))
    paramFile.write(format_coord('casIp', casIp))
    paramFile.write(format_coord('hubKp', hubKp))
    paramFile.write(format_coord('casKp', casKp))
    paramFile.write(format_coord('hubLp', hubLp))
    paramFile.write(format_coord('casLp', casLp))
    paramFile.write(format_coord('hubMp', hubMp))
    paramFile.write(format_coord('casMp', casMp))
    paramFile.write(format_coord('hubPp', hubPp))
    paramFile.write(format_coord('casPp', casPp))
    paramFile.write(format_coord('hubTp', hubTp))
    paramFile.write(format_coord('casTp', casTp))
    paramFile.write(format_coord('hubUp', hubUp))
    paramFile.write(format_coord('casUp', casUp))
    paramFile.write(format_coord('hubOp', hubOp))
    paramFile.write(format_coord('casOp', casOp))
    paramFile.write(format_coord('hubIn', hubIn))
    paramFile.write(format_coord('casIn', casIn))
    paramFile.write(format_coord('hubKn', hubKn))
    paramFile.write(format_coord('casKn', casKn))
    paramFile.write(format_coord('hubLn', hubLn))
    paramFile.write(format_coord('casLn', casLn))
    paramFile.write(format_coord('hubMn', hubMn))
    paramFile.write(format_coord('casMn', casMn))
    paramFile.write(format_coord('hubPn', hubPn))
    paramFile.write(format_coord('casPn', casPn))
    paramFile.write(format_coord('hubTn', hubTn))
    paramFile.write(format_coord('casTn', casTn))
    paramFile.write(format_coord('hubUn', hubUn))
    paramFile.write(format_coord('casUn', casUn))
    paramFile.write(format_coord('hubOn', hubOn))
    paramFile.write(format_coord('casOn', casOn))
    """
    paramFile.write(format_coord('hubIp', hubIpCyl))
    paramFile.write(format_coord('casIp', casIpCyl))
    paramFile.write(format_coord('hubKp', hubKpCyl))
    paramFile.write(format_coord('casKp', casKpCyl))
    paramFile.write(format_coord('hubLp', hubLpCyl))
    paramFile.write(format_coord('casLp', casLpCyl))
    paramFile.write(format_coord('hubMp', hubMpCyl))
    paramFile.write(format_coord('casMp', casMpCyl))
    paramFile.write(format_coord('hubPp', hubPpCyl))
    paramFile.write(format_coord('casPp', casPpCyl))
    paramFile.write(format_coord('hubTp', hubTpCyl))
    paramFile.write(format_coord('casTp', casTpCyl))
    paramFile.write(format_coord('hubUp', hubUpCyl))
    paramFile.write(format_coord('casUp', casUpCyl))
    paramFile.write(format_coord('hubOp', hubOpCyl))
    paramFile.write(format_coord('casOp', casOpCyl))
    paramFile.write(format_coord('hubIn', hubInCyl))
    paramFile.write(format_coord('casIn', casInCyl))
    paramFile.write(format_coord('hubKn', hubKnCyl))
    paramFile.write(format_coord('casKn', casKnCyl))
    paramFile.write(format_coord('hubLn', hubLnCyl))
    paramFile.write(format_coord('casLn', casLnCyl))
    paramFile.write(format_coord('hubMn', hubMnCyl))
    paramFile.write(format_coord('casMn', casMnCyl))
    paramFile.write(format_coord('hubPn', hubPnCyl))
    paramFile.write(format_coord('casPn', casPnCyl))
    paramFile.write(format_coord('hubTn', hubTnCyl))
    paramFile.write(format_coord('casTn', casTnCyl))
    paramFile.write(format_coord('hubUn', hubUnCyl))
    paramFile.write(format_coord('casUn', casUnCyl))
    paramFile.write(format_coord('hubOn', hubOnCyl))
    paramFile.write(format_coord('casOn', casOnCyl))
    """
    # close file
    paramFile.close()


def main() -> int:
    """ All the main blocks of the code get executed here """

    """ INPUTS: """
    # By Jeff Defoe -- more general input code
    dataPath = '../inputData/'
    hubFileName = 'sHub.curve'
    casFileName = 'sCas.curve'
    bladeCurveFile = 'statorBlade_originalECL5_fromAdekola.curve' # 'sBlade_fromAdekola.curve' # 'statorBlade1_original.curve'
    outputPath = '../outputData/'
    Nb = 31  # number of blades in row
    periodic = 1  # mode selection
    
    # Grid generation tuning parameters
    percentVal = 0.04  # fraction of arclength of blades where we cut off to avoid odd cell sizes in blade-to-offset BL blocks
    percentValNonCutLE = 0.02  # fraction of arclength of blades where we cut off to avoid higghly non-orthogonal cells in blade-to-offset BL blocks (LE)
    percentValNonCutTE = 0.00  # fraction of arclength of blades where we cut off to avoid higghly non-orthogonal cells in blade-to-offset BL blocks (TE)
    # Values to constrain cross-passage and offset curves to avoid problems:
    angConstraintCurves = 10  # deg
    angConstraintOffsets = 1  # deg

    # Geometry definition inputs:
    scale = 0.001  # relationship of input data to metres.
        # For example, if input data in mm, scale = 0.001

    # Grid generation inputs
    nrad = 40  # Number of radial points outside endwall BLs
    # optional BL definition parameters
    rhoref = 1.2  # base SI units (kg/m**3)
    Uref = 100.0  # base SI units (m/s)
    LrefHub = 374.0  # input length units (cannot be calculated because it depends on components outside domain)
    LrefCas = 374.0  # input length units (cannot be calculated because it depends on components outside domain)
    LrefBla = 75.0  # input length units (JD: this should be calculated = mean chord)
    muref = 1.8e-5  # base SI units (kg/(m*s))
    yPlusHub = 5
    yPlusCas = 5
    yPlusBla = 5
    # have option to calculate BL parameters based on above, or just directly
    # provide BL thickness and first cell size (input units)
    delHub = calcBLdelta(rhoref,Uref,LrefHub*scale,muref)/scale  # or just set a value (in input units)
    delCas = calcBLdelta(rhoref,Uref,LrefCas*scale,muref)/scale  # or just set a value (in input units)
    delBla = calcBLdelta(rhoref,Uref,LrefBla*scale,muref)/scale  # or just set a value (in input units)
    dy1Hub = calcFirstCellSize(rhoref,Uref,LrefHub*scale,muref,yPlusHub)/scale  # or just set a value (in input units)
    dy1Cas = calcFirstCellSize(rhoref,Uref,LrefCas*scale,muref,yPlusCas)/scale  # or just set a value (in input units)
    dy1Bla = calcFirstCellSize(rhoref,Uref,LrefBla*scale,muref,yPlusBla)/scale  # or just set a value (in input units)
    # Radial grading parameter
        # value = ratio of cell size at midspan to cell size adjacent endwall BLs
    gRad = 2
    # Tangential grading parameters
        # value = ratio of cell size at midpassage to cell size adjacent blade BLs
    gTan = 2  #2  # note: 4 gives a reasonable-looking grid
    additionalTangentialRefine = 8  # 1 = no extra refinement. This is a factor on the midpassage cell size.
    # Axial clustering parameters
    # Leading/trailing edge clustering parameters
    dax1primeLE = 0.003  # This is for about half the blade, so 0.01 means 0.5% chord
    rLE = 1.2  # expansion ratio for clustering of cells near the LE of the blades
    dax1primeTE = 0.002  # This is for about half the blade, so 0.01 means 0.5% chord
    rTE = 1.2  # expansion ratio for clustering of cells near the LE of the blades
    # Up/downstream expansion ratios of cells further than 1/2 chord away from blades
    additionalAxialRefine = 2  # 1 = no extra refinement. This is a factor on the midpassage cell size.
    rUpFar = 1.1  # used such that values > 1 mean cells grow as we get further from blades
    rDnFar = 1.1  # used such that values > 1 mean cells grow as we get further from blades
    """ END INPUTS """
    
    # STL definition inputs, typically do not need to be modified:
    res = 30  # upstream and downstream extention resolution 
    passageRes = 360  # Resolution of points for a single passage 
    bladeRes = 400  # Increase resolution of underlying blade data 
    mul = 5  # Additional factor of extra points when refining extensions

    """
    Description of input data format:
    - basically consistent with NAX/Turbogrid format (.curve files)
    - all blades in a row must be specified by the same number of sections
    - hub/casing specified as 2 or 3 column segs of space- or comma-separated numbers, if 2 col,
        interpret as R and Z; if 3 col, interpret as X, Y, Z and compute R as sqrt(X**2 + Y**2)
    - blades: one file per blade. 3 col space- or comma-separated format, XYZ coordinates. A blank line
        separates each profile/section. Optionally have comment lines which start with "#" which
        are ignored. Each profile must be a closed loop of points, i.e. the first and last point
        must be identical. Doesn't matter if this is the LE or TE. The other end must also be
        repeated.
    """

    # Determine operation mode -- single passage or periodic sector
    if periodic == 1:
        passages = 1
        print('Operating in periodic blade row mode. Producing grid inputs for a single passage.')
    elif periodic == 0:
        # For periodic sector, get number of passages in sector
        passages = Nb
        print(f'Operating in aperiodic blade row mode. Producing grid inputs for {Nb} passages.')

    # Load hub/casing data
    with open(dataPath + hubFileName, 'rb') as f:
        clean_lines = (line.replace(b' ',b',') for line in f)
        hub = np.genfromtxt(clean_lines, delimiter=',', comments='#')
    with open(dataPath + casFileName, 'rb') as f:
        clean_lines = (line.replace(b' ',b',') for line in f)
        cas = np.genfromtxt(clean_lines, delimiter=',', comments='#')
    # check for format and adjust as necessary
    if hub.shape[1]==3:
        hub[:,0] = np.sqrt(hub[:,0]**2 + hub[:,1]**2)
        hub = np.delete(hub, 1, axis=1)
    if cas.shape[1]==3:
        cas[:,0] = np.sqrt(cas[:,0]**2 + cas[:,1]**2)
        cas = np.delete(cas, 1, axis=1)
    # now the hub and cas arrays are in R,Z

    # For periodic sector, loop over passages; make a set of outputs for each
    for a in range(passages):
        # Start individual passage work
        print(f'Doing main operations for passage {a}')
        blade1num = a
        blade2num = a + 1
        # ensure if we're on the last blade that the next blade is the first one
        if blade2num >= (passages - 1):
            blade2num = 0
        # Load in blade data (2 blades per passage)
        if periodic == 1:
            curveSuffix = ''
            blade1numStr = ''
            blade2numStr = ''
        else:
            curveSuffix = str(a)
            blade1numStr = str(blade1num)
            blade2numStr = str(blade2num)
        commandGetSections = ["grep Profile " + dataPath + bladeCurveFile + curveSuffix + " | wc -l"]
        cmdResult = subprocess.run(commandGetSections, capture_output=True, text=True, shell=True)
        nSections = int(cmdResult.stdout)
        commandGetPointsPerSection = ["sed '1,/Profile/d;/Profile/,$d' " + dataPath + bladeCurveFile + curveSuffix + " | sed '/^[[:space:]]*$/d' | wc -l"]
        cmdResult = subprocess.run(commandGetPointsPerSection, capture_output=True, text=True, shell=True)
        ptsPerSection = int(cmdResult.stdout)
        N = int(ptsPerSection/2)
        Nr = int(2*N)  #number of points on blade profiles increased 
        with open(dataPath + bladeCurveFile + blade1numStr, 'rb') as f:
            clean_lines = (line.replace(b',',b' ') for line in f)
            blade1 = np.genfromtxt(clean_lines, comments='#')
        blade1 = blade1.reshape((ptsPerSection, nSections, 3), order='F')
        if periodic == 0:
            with open(dataPath + bladeCurveFile + blade2numStr, 'rb') as f:
                clean_lines = (line.replace(b',',b' ') for line in f)
                blade2 = np.genfromtxt(clean_lines, comments='#')
            blade2 = blade2.reshape((ptsPerSection, nSections, 3), order='F')
        elif periodic == 1:
            pitchAngle = 2*np.pi/Nb
            RotMat = np.array([[np.cos(pitchAngle), -np.sin(pitchAngle), 0],
                               [np.sin(pitchAngle),  np.cos(pitchAngle), 0],
                               [                 0,                   0, 1]])
            blade2 = np.zeros(blade1.shape)
            for i, j in np.ndindex(blade2.shape[:2]):
                blade2[i, j, :] = RotMat @ blade1[i, j, :]

        # Convert data from Cartesian to cylindrical
        blade1Cyl = CartToCyl(blade1)
        blade2Cyl = CartToCyl(blade2)

        # Check that first and last blade profiles lie on or extend beyond hub/casing, adjust as needed
        blade1Cyl, blade2Cyl, nSections = trimProfilesToGasPath(blade1Cyl, blade2Cyl, hub, cas, res)
        # Now have the proper sections, proceed
        # Split blade curves at farthest-forward ("LE") and farthest-backward ("TE") points - split blade sides too
        blade1LECyl, blade1TECyl, blade1pCyl, blade1nCyl = getLETEandSplit(blade1Cyl, Nr)
        blade2LECyl, blade2TECyl, blade2pCyl, blade2nCyl = getLETEandSplit(blade2Cyl, Nr)
        # Get meridional curves on interior sections from LE/TE to inlet/outlet
        meridCurve = getMeridCurve(blade1LECyl, blade1TECyl, blade2LECyl, blade2TECyl, hub, cas, res)
        
        # Find offset end vertices on each section, update blade sides to include midpoints
        blade1OffsetVerticesCyl, blade1pCyl, blade1nCyl, mid1P, mid1N = getOffsetVertices(blade1pCyl,
                                                          blade1nCyl,
                                                          meridCurve,
                                                          blade1LECyl,
                                                          blade1TECyl,
                                                          delBla)
        blade2OffsetVerticesCyl, blade2pCyl, blade2nCyl, mid2P, mid2N = getOffsetVertices(blade2pCyl,
                                                          blade2nCyl,
                                                          meridCurve,
                                                          blade2LECyl,
                                                          blade2TECyl,
                                                          delBla)
        # Move to m'-theta coordinates
        # The *Mprime variables are (theta, R, Z, m')
        # the *mpt variables are (m', theta)
        (blade1PMprime, blade1NMprime, blade2PMprime, blade2NMprime,
         upstreamMprime, downstreamMprime, offsetUpstreamMprime,
         offsetDownstreamMprime, offsetVertex1Mprime, offsetVertex2Mprime,
         blade1pmpt, blade1nmpt, blade2pmpt, blade2nmpt, upstreammpt,
         downstreammpt, offsetVertices1mpt, offsetVertices2mpt,
         LE1mpt, LE2mpt, TE1mpt, TE2mpt) = cylToMPT(blade1pCyl,
                                                    blade1nCyl,
                                                    blade2pCyl,
                                                    blade2nCyl,
                                                    blade1OffsetVerticesCyl,
                                                    blade2OffsetVerticesCyl,
                                                    blade1LECyl,
                                                    blade1TECyl,
                                                    blade2LECyl,
                                                    blade2TECyl,
                                                    meridCurve,
                                                    bladeRes,
                                                    res)

        # Get initial estimates of directions for upstream/downstream extensions for each section
        #angleLE1, angleTE1 = getInitExtAngles(blade1LECyl, blade1TECyl, blade1pCyl, blade1nCyl)
        #angleLE2, angleTE2 = getInitExtAngles(blade2LECyl, blade2TECyl, blade2pCyl, blade2nCyl)
        angleLE1, angleTE1 = getInitExtAngles(LE1mpt, TE1mpt, blade1pmpt, blade1nmpt)
        angleLE2, angleTE2 = getInitExtAngles(LE2mpt, TE2mpt, blade2pmpt, blade2nmpt)

        # Determine optimal extension directions, fully define the extensions
        blade1UpExtmpt, blade2UpExtmpt, blade1DnExtmpt, blade2DnExtmpt = defineExt(blade1pmpt, blade1nmpt, blade2pmpt, blade2nmpt, offsetVertices1mpt, offsetVertices2mpt, LE1mpt, LE2mpt, TE1mpt, TE2mpt, upstreamMprime, downstreamMprime, offsetUpstreamMprime, offsetDownstreamMprime, angleLE1, angleTE1, angleLE2, angleTE2, res)
        # Set up cross-passage and offset curves + do arclength mapping
        crossPassageUpmpt, crossPassageDnmpt, blade1offsetmpt, blade2offsetmpt, blade1hubUpArclenmap, blade1hubDnArclenmap, blade1casUpArclenmap, blade1casDnArclenmap, blade2hubUpArclenmap, blade2hubDnArclenmap, blade2casUpArclenmap, blade2casDnArclenmap = getCurvesAndMaps(offsetVertices1mpt, offsetVertices2mpt,
                                      LE1mpt, LE2mpt, TE1mpt, TE2mpt,
                                      blade1pmpt, blade1nmpt, blade2pmpt, blade2nmpt,
                                      blade1UpExtmpt, blade1DnExtmpt, blade2UpExtmpt, blade2DnExtmpt,
                                      angConstraintCurves, angConstraintOffsets, bladeRes, passageRes,
                                      percentVal, percentValNonCutLE, percentValNonCutTE)

        # Convert everything back to cylindrical coordinates (theta, R, Z)
        crossPassageUpCyl = mptToCyl(crossPassageUpmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        crossPassageDnCyl = mptToCyl(crossPassageDnmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        blade1UpExtCyl = mptToCyl(blade1UpExtmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        blade1DnExtCyl = mptToCyl(blade1DnExtmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        blade2UpExtCyl = mptToCyl(blade2UpExtmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        blade2DnExtCyl = mptToCyl(blade2DnExtmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        offset1Cyl = mptToCyl(blade1offsetmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        offset2Cyl = mptToCyl(blade2offsetmpt,
                                     upstreamMprime, blade1PMprime,
                                     downstreamMprime, hub, cas)
        # The above curves have the following property: on the extensions,
        # the point before the endpoint that is coincident to a blade LE/TE
        # is the same as the offset ends. So, moving forward, we'd like to
        # not only trim the extensions to the correct z coordinates at the
        # inlet/outlet, but also cut them to stop at the offset ends, and
        # define new curves for the offsets with some desired number of points.
        # These curves are straight lines in cylindrical coordinates.

        # Create blade-to-offset curves
        bladeToOffsetRes = int(100)  #int(res*0.5)
        blade1toOffsetUpCyl = bladeToOffset(pt1=blade1UpExtCyl[:, -2, :],
                                            pt2=blade1UpExtCyl[:, -1, :],
                                            res=bladeToOffsetRes,
                                            hub=hub,
                                            cas=cas)
        blade2toOffsetUpCyl = bladeToOffset(pt1=blade2UpExtCyl[:, -2, :],
                                            pt2=blade2UpExtCyl[:, -1, :],
                                            res=bladeToOffsetRes,
                                            hub=hub,
                                            cas=cas)
        blade1toOffsetDnCyl = bladeToOffset(pt1=blade1DnExtCyl[:, 0, :],
                                            pt2=blade1DnExtCyl[:, 1, :],
                                            res=bladeToOffsetRes,
                                            hub=hub,
                                            cas=cas)
        blade2toOffsetDnCyl = bladeToOffset(pt1=blade2DnExtCyl[:, 0, :],
                                            pt2=blade2DnExtCyl[:, 1, :],
                                            res=bladeToOffsetRes,
                                            hub=hub,
                                            cas=cas)
        # Trim extensions to desired geometry and increase streamwise resolution
        blade1UpExtCyl = trimAndRefineExt(blade1UpExtCyl, res, mul, endZ=meridCurve[:, 0, 1], keep='hi', hub=hub, cas=cas)
        blade1DnExtCyl = trimAndRefineExt(blade1DnExtCyl, res, mul, endZ=meridCurve[:, -1, 1], keep='lo', hub=hub, cas=cas)
        blade2UpExtCyl = trimAndRefineExt(blade2UpExtCyl, res, mul, endZ=meridCurve[:, 0, 1], keep='hi', hub=hub, cas=cas)
        blade2DnExtCyl = trimAndRefineExt(blade2DnExtCyl, res, mul, endZ=meridCurve[:, -1, 1], keep='lo', hub=hub, cas=cas)
        
        # Split blade and offset curves to upstream and downstream halves
        blade1UpCyl, blade1DnCyl, blade2UpCyl, blade2DnCyl, offset1UpCyl, offset1DnCyl, offset2UpCyl, offset2DnCyl, midCurveMidCyl, midCurve2Cyl, midCurve1Cyl = splitBladesAndOffsets(blade1PMprime[:, :, 0:3],
                                                                 blade2NMprime[:, :, 0:3],
                                                                 offset1Cyl,
                                                                 offset2Cyl,
                                                                 blade1OffsetVerticesCyl,
                                                                 blade2OffsetVerticesCyl,
                                                                 mid1P,
                                                                 mid2N,
                                                                 bladeToOffsetRes,
                                                                 passageRes)

        # Correct the blade-to-offset surfaces to guarantee their boundary uses
        # exactly the points from the curves they touch -- specifically at
        # the blade
        blade1toOffsetUpCyl[:, -1, :] = blade1UpCyl[:,  0, :]
        blade2toOffsetUpCyl[:, -1, :] = blade2UpCyl[:,  0, :]
        blade1toOffsetDnCyl[:,  0, :] = blade1DnCyl[:, -1, :]
        blade2toOffsetDnCyl[:,  0, :] = blade2DnCyl[:, -1, :]

        # Define interior nodes for inlet, outlet, hub, casing
        inletPtsCyl,outletPtsCyl, hubPtsCyl, casPtsCyl = fillInOutHubCas(blade1UpExtCyl, blade2UpExtCyl, blade1DnExtCyl, blade2DnExtCyl, offset1UpCyl, offset2UpCyl, offset1DnCyl, offset2DnCyl, passageRes)
        #blade1UpHubPtsCyl, blade1UpCasPtsCyl = fillBladeToOffset(blade1toOffsetUpCyl,
        #                                                         midCurve1Cyl,
        #                                                         blade1UpCyl,
        #                                                         offset1UpCyl)
        blade1DnHubPtsCyl, blade1DnCasPtsCyl = fillBladeToOffset(midCurve1Cyl,
                                                                 blade1toOffsetDnCyl,
                                                                 blade1DnCyl,
                                                                 offset1DnCyl)
        blade2UpHubPtsCyl, blade2UpCasPtsCyl = fillBladeToOffset(blade2toOffsetUpCyl,
                                                                 midCurve2Cyl,
                                                                 offset2UpCyl,
                                                                 blade2UpCyl)
        blade2DnHubPtsCyl, blade2DnCasPtsCyl = fillBladeToOffset(midCurve2Cyl,
                                                                 blade2toOffsetDnCyl,
                                                                 offset2DnCyl,
                                                                 blade2DnCyl)

        # JD: UP TO HERE - plot to check if problem fixed
        plt.plot(blade1DnHubPtsCyl[:, :, 0]*blade1DnHubPtsCyl[:, :, 1], blade1DnHubPtsCyl[:, :, 2], 'ro')
        plt.plot(blade1DnCyl[0, :, 0]*blade1DnCyl[0, :, 1], blade1DnCyl[0, :, 2], 'b-')
        plt.axis('equal')
        plt.show()

        bob = alice
                                                                 
        # Convert everything back to Cartesian coordinates
        blade1UpCart = cylToCart(blade1UpCyl)
        offset1UpCart = cylToCart(offset1UpCyl)
        blade2UpCart = cylToCart(blade2UpCyl)
        offset2UpCart = cylToCart(offset2UpCyl)
        blade1DnCart = cylToCart(blade1DnCyl)
        offset1DnCart = cylToCart(offset1DnCyl)
        blade2DnCart = cylToCart(blade2DnCyl)
        offset2DnCart = cylToCart(offset2DnCyl)
        crossPassageUpCart = cylToCart(crossPassageUpCyl)
        crossPassageDnCart = cylToCart(crossPassageDnCyl)
        midCurveMidCart = cylToCart(midCurveMidCyl)
        blade1UpExtCart = cylToCart(blade1UpExtCyl)
        blade2UpExtCart = cylToCart(blade2UpExtCyl)
        blade1DnExtCart = cylToCart(blade1DnExtCyl)
        blade2DnExtCart = cylToCart(blade2DnExtCyl)
        blade1toOffsetUpCart = cylToCart(blade1toOffsetUpCyl)
        blade2toOffsetUpCart = cylToCart(blade2toOffsetUpCyl)
        blade1toOffsetDnCart = cylToCart(blade1toOffsetDnCyl)
        blade2toOffsetDnCart = cylToCart(blade2toOffsetDnCyl)
        midCurve1Cart = cylToCart(midCurve1Cyl)
        midCurve2Cart = cylToCart(midCurve2Cyl)
        inletCart = cylToCart(inletPtsCyl)
        outletCart = cylToCart(outletPtsCyl)
        hubCart = cylToCart(hubPtsCyl)
        casCart = cylToCart(casPtsCyl)
        blade1UpHubCart = cylToCart(blade1UpHubPtsCyl)
        blade1DnHubCart = cylToCart(blade1DnHubPtsCyl)
        blade1UpCasCart = cylToCart(blade1UpCasPtsCyl)
        blade1DnCasCart = cylToCart(blade1DnCasPtsCyl)
        blade2UpHubCart = cylToCart(blade2UpHubPtsCyl)
        blade2DnHubCart = cylToCart(blade2DnHubPtsCyl)
        blade2UpCasCart = cylToCart(blade2UpCasPtsCyl)
        blade2DnCasCart = cylToCart(blade2DnCasPtsCyl)
        # Fill final X, Y, Z arrays
        Xvalues, Yvalues, Zvalues = combineArrays(blade1UpCart,
                                                  offset1UpCart,
                                                  blade2UpCart,
                                                  offset2UpCart,
                                                  blade1DnCart,
                                                  offset1DnCart,
                                                  blade2DnCart,
                                                  offset2DnCart,
                                                  crossPassageUpCart,
                                                  crossPassageDnCart,
                                                  midCurveMidCart,
                                                  hubCart,
                                                  casCart,
                                                  inletCart,
                                                  outletCart,
                                                  blade1UpExtCart,
                                                  blade2UpExtCart,
                                                  blade1DnExtCart,
                                                  blade2DnExtCart,
                                                  blade2UpHubCart,
                                                  blade2DnHubCart,
                                                  blade1UpHubCart,
                                                  blade1DnHubCart,
                                                  blade2UpCasCart,
                                                  blade2DnCasCart,
                                                  blade1UpCasCart,
                                                  blade1DnCasCart,
                                                  blade2toOffsetUpCart,
                                                  blade1toOffsetUpCart,
                                                  blade2toOffsetDnCart,
                                                  blade1toOffsetDnCart,
                                                  midCurve1Cart,
                                                  midCurve2Cart)

        # Define/write STLs
        print('Writing STL files for passage {}'.format(a))
        createSTLs(Xvalues, Yvalues, Zvalues, outputPath, a)
        # Calculate grid/grading parameters and write passageParameters file
        print('Computing and writing parameters for passage {}'.format(a))
        calcAndWritePassageParameters(scale, Xvalues, Yvalues, Zvalues, nrad, delHub, delCas, delBla, dy1Hub, dy1Cas, dy1Bla, gRad, gTan, dax1primeLE, rLE, dax1primeTE, rTE, rUpFar, rDnFar, outputPath, a, additionalTangentialRefine, additionalAxialRefine, blade2hubUpArclenmap, blade1hubUpArclenmap, blade2casUpArclenmap, blade1casUpArclenmap, blade2hubDnArclenmap, blade1hubDnArclenmap, blade2casDnArclenmap, blade1casDnArclenmap)

        bob = alice
        
    return 0


# This goes at the very end of the file:
if __name__ == "__main__":
    sys.exit(main())  # execute the main function when script is run directly
