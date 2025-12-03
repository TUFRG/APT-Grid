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
Active development ongoing as of November 2025
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
#import ipdb

# functions:

def dy_dx(x, M, A):
    """ Slope equation that determines the circumferential shift for both inlet and outlet. """
    #$$ It factors in the radius change and reduces the exaggeration of the lean and twist of the blade. 
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


def processData(a, dataPath, nSections, N, res, passageRes, bladeRes, periodic):
    """ Load and process blade and data for the ath passage """
    filePath = dataPath +  '../constant/geometry{}.'.format(a)
    if not os.path.exists(filePath):
        os.makedirs(filePath, exist_ok=True)
        #%% Initialize variables to hold blade data
    blade1PS = np.zeros([nSections,N,3]) ## First blade pressure side
    blade1SS = np.zeros([nSections,N,3]) ## First blade sunction side
    blade2PS = np.zeros([nSections,N,3]) ## Second blade pressure side
    blade2SS = np.zeros([nSections,N,3]) ## Second blade sunction side
    #%% Load the blade data
    for b in range(nSections):
        blade1PS[b,:] = np.loadtxt(dataPath + '/blade{}/PS{}.txt'.format(a,b), delimiter=',')
        blade2PS[b,:] = np.loadtxt(dataPath + '/blade{}/PS{}.txt'.format(a+1,b), delimiter=',')
        blade1SS[b,:] = np.loadtxt(dataPath + '/blade{}/SS{}.txt'.format(a,b), delimiter=',')
        blade2SS[b,:] = np.loadtxt(dataPath + '/blade{}/SS{}.txt'.format(a+1,b), delimiter=',')

    hub = np.loadtxt(dataPath + '/hub.txt', delimiter=',')
    casing = np.loadtxt(dataPath + '/casing.txt', delimiter=',')
    print('Finished loading blade and annulus data')

    print('Basic processing of geometry underway...')
    #%% converting to cylinderical coordinate
    blade1PSCyl = np.zeros([nSections,N,3]) # (theta, r, z)
    blade1SSCyl = np.zeros([nSections,N,3])
    blade2PSCyl = np.zeros([nSections,N,3])
    blade2SSCyl = np.zeros([nSections,N,3])
    allLEBlade1 = np.zeros([nSections,3])
    allLEBlade2 = np.zeros([nSections,3])
    allTEBlade1 = np.zeros([nSections,3])
    allTEBlade2 = np.zeros([nSections,3])
    blade1Cyl = np.zeros([nSections,N*2-2,3])
    blade2Cyl = np.zeros([nSections,N*2-2,3])
    blade12D = np.zeros([nSections,N*2-2,2])
    blade22D = np.zeros([nSections,N*2-2,2])
    for c in range(nSections):
        # Change from cartesian to cyclindrical
        blade1PSCyl[c,:,:] = np.array(mf.cart2pol(blade1PS[c,:,0],blade1PS[c,:,1],blade1PS[c,:,2])).T 
        blade1SSCyl[c,:,:] = np.array(mf.cart2pol(blade1SS[c,:,0],blade1SS[c,:,1],blade1SS[c,:,2])).T
        blade2PSCyl[c,:,:] = np.array(mf.cart2pol(blade2PS[c,:,0],blade2PS[c,:,1],blade2PS[c,:,2])).T
        blade2SSCyl[c,:,:] = np.array(mf.cart2pol(blade2SS[c,:,0],blade2SS[c,:,1],blade2SS[c,:,2])).T     
        allLEBlade1[c] = blade1PSCyl[c][0]
        allLEBlade2[c] = blade2PSCyl[c][0]
        allTEBlade1[c] = blade1PSCyl[c][-1]
        allTEBlade2[c] = blade2PSCyl[c][-1]
        blade1Cyl[c,:,:] = np.vstack((blade1PSCyl[c,:,:], blade1SSCyl[c,:,:][1:-1][::-1]))
        blade2Cyl[c,:,:] = np.vstack((blade2PSCyl[c,:,:], blade2SSCyl[c,:,:][1:-1][::-1]))
        blade12D[c,:,:] = np.array([blade1Cyl[c,:,2], blade1Cyl[c,:,1]]).T
        blade22D[c,:,:] = np.array([blade2Cyl[c,:,2], blade2Cyl[c,:,1]]).T

    #%%
    # Scale the LE and TE points to the inlet and outlet
    inletZ = np.linspace(hub[0][2],casing[0][2],res)  #The inlet is taken as the first point on hub and casing
    inletFunc = interp1d([hub[0][2],casing[0][2]], [hub[0][0],casing[0][0]])
    inletR = inletFunc(inletZ)
    inlet = np.column_stack((inletZ, inletR))
    outletRth = np.linspace(hub[::-1][0][0],casing[::-1][0][0],res)
    outletFunc = interp1d([hub[::-1][0][0],casing[::-1][0][0]], [hub[::-1][0][2],casing[::-1][0][2]])
    outletZ = outletFunc(outletRth)
    outlet = np.column_stack((outletZ, outletRth))
    scaledInlet = mf.scale(min(allLEBlade1[:,1]), max(allLEBlade1[:,1]), hub[0][0], casing[0][0],allLEBlade1[:,1])
    scaledOutlet = mf.scale(min(allTEBlade1[:,1]), max(allTEBlade1[:,1]), hub[::-1][0][0], casing[::-1][0][0],allTEBlade1[:,1])
    inletFunc = interp1d(inlet[:,1], inlet[:,0])
    scaledInletZ = inletFunc(scaledInlet) #You might ask why are you scaling the inlet here. Bascially, what I have done is to map the LE and TE of the blade surface to the inlet and outlet
    outletFunc = interp1d(outlet[:,1], outlet[:,0])
    scaledOutletZ = outletFunc(scaledOutlet)
    newInlet = np.column_stack((scaledInletZ, scaledInlet))
    newOutlet = np.column_stack((scaledOutletZ, scaledOutlet))
    hubLEIdx = np.searchsorted(hub[:,2], allLEBlade1[0][2])
    hubTEIdx = np.searchsorted(hub[:,2], allTEBlade1[0][2])
    casLEIdx = np.searchsorted(casing[:,2], allLEBlade1[nSections-1][2])
    casTEIdx = np.searchsorted(casing[:,2], allTEBlade1[nSections-1][2])
    #%% Now I need to ensure that the blade profiles protudes beyond the hub and casing. This is important to get a profile that lie exactly on hub and casing 
    hub2D = hub[:,[2,0]]
    cas2D = casing[:,[2,0]]
    LE2D = np.column_stack((allLEBlade1[:,2], allLEBlade1[:,1]))
    TE2D = np.column_stack((allTEBlade1[:,2], allTEBlade1[:,1]))

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
    
    LE2D2 = np.column_stack((allLEBlade2[:,2], allLEBlade2[:,1]))
    TE2D2 = np.column_stack((allTEBlade2[:,2], allTEBlade2[:,1]))

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

    extnLE2D2 = np.vstack((leHubExtn2, LE2D, leCasExtn))
    extnTE2D2 = np.vstack((teHubExtn2, TE2D, teCasExtn))
    hubLE2 = mf.TwoLinesIntersect(hub2D, extnLE2D2)
    casLE2 = mf.TwoLinesIntersect(cas2D, extnLE2D2)
    hubTE2 = mf.TwoLinesIntersect(hub2D, extnTE2D2)
    casTE2 = mf.TwoLinesIntersect(cas2D, extnTE2D2)

    spanLE = mf.compute_span_fractions(LE2D, hubLE, casLE)
    spanTE = mf.compute_span_fractions(TE2D, hubTE, casTE)

    spanLE2 = mf.compute_span_fractions(LE2D2, hubLE2, casLE2)
    spanTE2 = mf.compute_span_fractions(TE2D2, hubTE2, casTE2)

    extInlet = mf.map_span_fractions_to_line(spanLE, newInlet[0], newInlet[-1])
    extOutlet = mf.map_span_fractions_to_line(spanTE, newOutlet[0], newOutlet[-1])

    extInlet2 = mf.map_span_fractions_to_line(spanLE2, newInlet[0], newInlet[-1])
    extOutlet2 = mf.map_span_fractions_to_line(spanTE2, newOutlet[0], newOutlet[-1])

    hubBladeRegion = np.vstack((hubLE, hub2D[hubLEIdx:hubTEIdx],hubTE)) #This is the portion of the hub that intersects with the blade
    hubRegionFunc = interp1d(hubBladeRegion[:,0], hubBladeRegion[:,1])
    hubRegionZ = np.linspace(hubBladeRegion[0,0], hubBladeRegion[-1,0], N*2-2) #Sampled same number of points on the hub as the blade
    hubRegionPts = np.array([hubRegionZ, hubRegionFunc(hubRegionZ)]).T
    casBladeRegion = np.vstack((casLE, cas2D[casLEIdx:casTEIdx],casTE))
    casRegionFunc = interp1d(casBladeRegion[:,0], casBladeRegion[:,1])
    casRegionZ = np.linspace(casBladeRegion[0,0], casBladeRegion[-1,0], N*2-2)
    casRegionPts = np.array([casRegionZ, casRegionFunc(casRegionZ)]).T

    hubBladeRegion2 = np.vstack((hubLE2, hub2D[hubLEIdx:hubTEIdx],hubTE2)) #This is the portion of the hub that intersects with the blade
    hubRegionFunc2 = interp1d(hubBladeRegion2[:,0], hubBladeRegion2[:,1])
    hubRegionZ2 = np.linspace(hubBladeRegion2[0,0], hubBladeRegion2[-1,0], N*2-2) #Sampled same number of points on the hub as the blade
    hubRegionPts2 = np.array([hubRegionZ2, hubRegionFunc2(hubRegionZ)]).T
    casBladeRegion2 = np.vstack((casLE2, cas2D[casLEIdx:casTEIdx],casTE2))
    casRegionFunc2 = interp1d(casBladeRegion2[:,0], casBladeRegion2[:,1])
    casRegionZ2 = np.linspace(casBladeRegion2[0,0], casBladeRegion2[-1,0], N*2-2)
    casRegionPts2 = np.array([casRegionZ2, casRegionFunc(casRegionZ2)]).T
    #%% Now I will try to obtain the profile that lies exactly on the blade surface. To do this I will scale the last blade profile that is closest to both the hub and casing. 
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
    while hubIdx < nSections and not hubIntersect: #
        blade1PSurface = np.column_stack((blade1PSCyl[hubIdx,:,2], blade1PSCyl[hubIdx,:,1])) # 
        blade1SSurface = np.column_stack((blade1SSCyl[hubIdx,:,2], blade1SSCyl[hubIdx,:,1])) #  
        blade1Surface = np.vstack((blade1PSurface, blade1SSurface)) 
        if not mf.TwoLinesIntersect(hubBladeRegion, blade1Surface): 
            hubIntersect = True
        else:
            hubIdx += 1
    if hubIntersect:   
        hubSpanFrac = mf.compute_span_fractions(blade12D[hubIdx-1], extnLE2D[hubIdx], extnTE2D[hubIdx])
        newHubBladeRegion = mf.rearrange_curve_by_arc_length(hubRegionPts, hubSpanFrac)
      
    else:
        None

    while casIdx-2 < nSections and not casIntersect: #The reson for subtracting 2 here is because I added extensions at both ends
        blade1PSurface = np.column_stack((blade1PSCyl[casIdx-2,:,2], blade1PSCyl[casIdx-2,:,1]))
        blade1SSurface = np.column_stack((blade1SSCyl[casIdx-2,:,2], blade1SSCyl[casIdx-2,:,1]))
        blade1Surface = np.vstack((blade1PSurface, blade1SSurface))
        if not mf.TwoLinesIntersect(casBladeRegion, blade1Surface):
            casIntersect = True
        else:
            casIdx -= 1
        print(casIntersect)
    if casIntersect:
        casSpanFrac = mf.compute_span_fractions(blade12D[nSections-1], extnLE2D[casIdx], extnTE2D[casIdx])
        newCasBladeRegion = mf.rearrange_curve_by_arc_length(casRegionPts, casSpanFrac)
    else:
        None


    ## For Blade 2
    while hubIdx2 < nSections and not hubIntersect2: #
        blade2PSurface = np.column_stack((blade2PSCyl[hubIdx2,:,2], blade2PSCyl[hubIdx2,:,1])) # 
        blade2SSurface = np.column_stack((blade2SSCyl[hubIdx2,:,2], blade2SSCyl[hubIdx2,:,1])) #  
        blade2Surface = np.vstack((blade2PSurface, blade2SSurface)) 
        if not mf.TwoLinesIntersect(hubBladeRegion2, blade2Surface):
            hubIntersect2 = True
        else:
            hubIdx2 += 1
    if hubIntersect2:   
        hubSpanFrac2 = mf.compute_span_fractions(blade22D[hubIdx2-1], extnLE2D2[hubIdx], extnTE2D2[hubIdx2])
        newHubBladeRegion2 = mf.rearrange_curve_by_arc_length(hubRegionPts2, hubSpanFrac2)
      
    else:
        None


    while casIdx2-2 < nSections and not casIntersect2:
        blade2PSurface = np.column_stack((blade2PSCyl[casIdx2-2,:,2], blade2PSCyl[casIdx2-2,:,1]))
        blade2SSurface = np.column_stack((blade2SSCyl[casIdx2-2,:,2], blade2SSCyl[casIdx2-2,:,1]))
        blade2Surface = np.vstack((blade2PSurface, blade2SSurface))
        if not mf.TwoLinesIntersect(casBladeRegion2, blade2Surface):
            casIntersect2 = True
        else:
            casIdx2 -= 1
    if casIntersect2:
        casSpanFrac2 = mf.compute_span_fractions(blade22D[nSections-1], extnLE2D2[casIdx2], extnTE2D2[casIdx2])
        newCasBladeRegion2 = mf.rearrange_curve_by_arc_length(casRegionPts2, casSpanFrac2)
    else:
        None

    if False:
        plt.plot(hub2D[:,0], hub2D[:,1], 'k')
        plt.plot(cas2D[:,0], cas2D[:,1], 'k')
        plt.plot(LE2D[:,0], LE2D[:,1], 'r')
        plt.plot(TE2D[:,0], TE2D[:,1], 'r')
        plt.plot(newHubBladeRegion2[:,0], newHubBladeRegion2[:,1], 'b.')
        plt.plot(newCasBladeRegion2[:,0], newCasBladeRegion2[:,1], 'b.')
        plt.plot(blade22D[hubIdx-1,:,0], blade22D[hubIdx-1,:,1], 'g.')
        plt.plot(blade22D[casIdx-2,:,0], blade22D[casIdx-2,:,1], 'g.')
        plt.plot(blade12D[:,:,0], blade12D[:,:,1], 'g.')
        plt.plot(blade12D[:,:,0], blade12D[:,:,1], 'g.')
        plt.plot(extnLE2D[hubIdx][0], extnLE2D[hubIdx][1], 'k.')
        plt.plot(extnTE2D[hubIdx][0], extnTE2D[hubIdx][1], 'k.')
        plt.plot(extnLE2D[casIdx-1][0], extnLE2D[casIdx-1][1], 'k.')
        plt.plot(extnTE2D[casIdx-1][0], extnTE2D[casIdx-1][1], 'k.')
        plt.plot(casLE[0], casLE[1], 'b.')
        plt.plot(casTE[0], casTE[1], 'b.')

    #%% Now get the profile on the hub and casing 
    hubBladeProfile = np.zeros([N*2-2,3])
    hubBladeProfile[:,0] = blade1Cyl[hubIdx-1,:,0]
    hubBladeProfile[:,[2,1]] = newHubBladeRegion

    casBladeProfile = np.zeros([N*2-2,3])
    casBladeProfile[:,0] = blade1Cyl[casIdx-2,:,0]
    casBladeProfile[:,[2,1]] = newCasBladeRegion

    hubBladeProfile2 = np.zeros([N*2-2,3])
    hubBladeProfile2[:,0] = blade2Cyl[hubIdx2-1,:,0]
    hubBladeProfile2[:,[2,1]] = newHubBladeRegion2

    casBladeProfile2 = np.zeros([N*2-2,3])
    casBladeProfile2[:,0] = blade2Cyl[casIdx2-2,:,0]
    casBladeProfile2[:,[2,1]] = newCasBladeRegion2

    hubPS1 = np.zeros([N,3])
    hubSS1 = np.zeros([N,3])
    casPS1 = np.zeros([N,3])
    casSS1 = np.zeros([N,3])

    hubPS2 = np.zeros([N,3])
    hubSS2 = np.zeros([N,3])
    casPS2 = np.zeros([N,3])
    casSS2 = np.zeros([N,3])

    idxHubLE = int(np.where(hubBladeProfile[:,2]==hubLE[0])[0][0])
    idxHubTE = int(np.where(np.isclose(hubBladeProfile[:,2], hubTE[0]))[0][0])
    idxCasLE = int(np.where(casBladeProfile[:,2]==casLE[0])[0][0])
    idxCasTE = int(np.where(np.isclose(casBladeProfile[:,2], casTE[0]))[0][0])

    #Splitting the blade to pressure and pressure side
    hubPS1 = hubBladeProfile[idxHubLE:idxHubTE+1]
    hubSS1 = np.vstack((hubBladeProfile[idxHubTE:len(hubBladeProfile)], hubBladeProfile[idxHubLE]))[::-1]
    casPS1 = casBladeProfile[idxCasLE:idxCasTE+1]
    casSS1 = np.vstack((casBladeProfile[idxCasTE:len(casBladeProfile)], casBladeProfile[idxCasLE]))[::-1]

    hubPS2 = hubBladeProfile2[idxHubLE:idxHubTE+1]
    hubSS2 = np.vstack((hubBladeProfile2[idxHubTE:len(hubBladeProfile2)], hubBladeProfile2[idxHubLE]))[::-1]
    casPS2 = casBladeProfile2[idxCasLE:idxCasTE+1]
    casSS2 = np.vstack((casBladeProfile2[idxCasTE:len(casBladeProfile2)], casBladeProfile2[idxCasLE]))[::-1]

    #Stacking blade for new Blade definition 
    if casIdx-2 > nSections:
        newNsection = nSections - hubIdx + 3
    else:
        newNsection = nSections - hubIdx + (nSections - casIdx-2) + 4
    newBlade1PSCyl = np.zeros([newNsection,N,3]) # (theta, r, z)
    newBlade1SSCyl = np.zeros([newNsection,N,3])
    newBlade2PSCyl = np.zeros([newNsection,N,3]) # (theta, r, z)
    newBlade2SSCyl = np.zeros([newNsection,N,3])
    newAllLEBlade1 = np.zeros([newNsection,3])
    newAllLEBlade2 = np.zeros([newNsection,3])
    newAllTEBlade1 = np.zeros([newNsection,3])
    newAllTEBlade2 = np.zeros([newNsection,3])
    for f in range(newNsection):
        if f == 0:
            newBlade1PSCyl[f,:,:] = hubPS1
            newBlade1SSCyl[f,:,:] = hubSS1
            newBlade2PSCyl[f,:,:] = hubPS2
            newBlade2SSCyl[f,:,:] = hubSS2
            newAllLEBlade1[f] = newBlade1PSCyl[f][0]
            newAllLEBlade2[f] = newBlade2PSCyl[f][0]
            newAllTEBlade1[f] = newBlade1PSCyl[f][-1]
            newAllTEBlade2[f] = newBlade2PSCyl[f][-1]        
        elif f == newNsection - 1:
            newBlade1PSCyl[f,:,:] = casPS1
            newBlade1SSCyl[f,:,:] = casSS1
            newBlade2PSCyl[f,:,:] = casPS2
            newBlade2SSCyl[f,:,:] = casSS2
            newAllLEBlade1[f] = newBlade1PSCyl[f][0]
            newAllLEBlade2[f] = newBlade2PSCyl[f][0]
            newAllTEBlade1[f] = newBlade1PSCyl[f][-1]
            newAllTEBlade2[f] = newBlade2PSCyl[f][-1]         
        else:
            newBlade1PSCyl[f,:,:] = blade1PSCyl[f+hubIdx-2,:,:]
            newBlade1SSCyl[f,:,:] = blade1SSCyl[f+hubIdx-2,:,:]
            newBlade2PSCyl[f,:,:] = blade2PSCyl[f+hubIdx-2,:,:]
            newBlade2SSCyl[f,:,:] = blade2SSCyl[f+hubIdx-2,:,:]
            newAllLEBlade1[f] = newBlade1PSCyl[f][0]
            newAllLEBlade2[f] = newBlade2PSCyl[f][0]
            newAllTEBlade1[f] = newBlade1PSCyl[f][-1]
            newAllTEBlade2[f] = newBlade2PSCyl[f][-1]

    # At this point, the tangent to the blade surface at the LE/TE is calculated.
    '''
    It should be noted that here, this angle calculation is being done in the z-rTheta coordinate system. The implication of this is that angles are not preserved. Thus, there is a slight 
    difference in the calculated angle for blade1 and blade2 even in a periodic case. To solve this problem I equate the angle for blade1 and blade2 for periodic cases and then I did not
    for aperiodic case. This is not a case of whether the angle calculcated is wrong is just an implicit problem with z-rTheta coordinate system
    '''
    # (JD: this sounds sketchy. Is there a more robust fix?)
    angleLE1 = np.zeros(newNsection)
    angleLE2 = np.zeros(newNsection)
    angleTE1 = np.zeros(newNsection)
    angleTE2 = np.zeros(newNsection)
     
    for cd in range(newNsection):
        blade1 = np.concatenate((newBlade1PSCyl[cd], newBlade1SSCyl[cd][1:][::-1]))
        indx1LE = 0
        indx1TE = int(0.5*len(blade1))
        blade2 = np.concatenate((newBlade2PSCyl[cd], newBlade2SSCyl[cd][1:][::-1]))
        indx2LE = 0
        indx2TE = int(0.5*len(blade2))
        if blade1[indx1LE+1][2] == blade1[indx1LE-1][2]:
            slopeLE1 = ((blade1[indx1LE+1][0]*blade1[indx1LE+1][1])-(blade1[indx1LE][0]*blade1[indx1LE][1]))/((blade1[indx1LE+1][2]-blade1[indx1LE][2]))
            angleLE1[cd] = np.arctan(slopeLE1) 
            slopeLE2 = ((blade2[indx2LE][0]*blade2[indx2LE][1])-(blade2[indx2LE-1][0]*blade2[indx2LE-1][1]))/((blade2[indx2LE][2]-blade2[indx2LE-1][2]))
            if periodic == 1:
                angleLE2[cd] = angleLE1[cd]
            else:
                angleLE2[cd] = np.arctan(slopeLE2) 
        else:
            slopeLE1 = ((blade1[indx1LE+1][0]*blade1[indx1LE+1][1])-(blade1[indx1LE-1][0]*blade1[indx1LE-1][1]))/((blade1[indx1LE+1][2]-blade1[indx1LE-1][2]))
            angleLE1[cd] = np.arctan(slopeLE1) 
            slopeLE2 = ((blade2[indx2LE+1][0]*blade2[indx2LE+1][1])-(blade2[indx2LE-1][0]*blade2[indx2LE-1][1]))/((blade2[indx2LE+1][2]-blade2[indx2LE-1][2]))
            if periodic == 1:
                angleLE2[cd] = angleLE1[cd]
            else:
                angleLE2[cd] = np.arctan(slopeLE2) 
        if blade1[indx1TE+1][2] == blade1[indx1TE-1][2]:
            slopeTE1 = ((blade1[indx1TE+1][0]*blade1[indx1TE+1][1])-(blade1[indx1TE][0]*blade1[indx1TE][1]))/((blade1[indx1TE+1][2]-blade1[indx1TE][2]))
            angleTE1[cd] = np.arctan(slopeTE1)
            slopeTE2 = ((blade2[indx2TE][0]*blade2[indx2TE][1])-(blade2[indx2TE-1][0]*blade2[indx2TE-1][1]))/((blade2[indx2TE][2]-blade2[indx2TE-1][2]))
            if periodic == 1:
                angleTE2[cd] = angleTE1[cd]
            else:
                angleTE2[cd] = np.arctan(slopeTE2) 
        else:        
            slopeTE1 = ((blade1[indx1TE+1][0]*blade1[indx1TE+1][1])-(blade1[indx1TE-1][0]*blade1[indx1TE-1][1]))/((blade1[indx1TE+1][2]-blade1[indx1TE-1][2]))
            angleTE1[cd] = np.arctan(slopeTE1)
            slopeTE2 = ((blade2[indx2TE+1][0]*blade2[indx2TE+1][1])-(blade2[indx2TE-1][0]*blade2[indx2TE-1][1]))/((blade2[indx2TE+1][2]-blade2[indx2TE-1][2]))
            if periodic == 1:
                angleTE2[cd] = angleTE1[cd]
            else:
                angleTE2[cd] = np.arctan(slopeTE2)


    #%%   Change the coordinate system to Mprime 
    '''
    To do this, I will connect the LE to points clustered at midsection of the PS and SS and TE. Basically I cut the blade surface at 25% chord and 75% chord to move to Mprime
    This is done because to move to Mprime coordinate system, I will either have to change the LE and TE of the blade surface of cut it to ensure the furtherest points on the 
    surface are at the LE and TE
    '''    
    tempBlade1PS = np.zeros([newNsection, bladeRes+2, 3]) #This is a temporary surface define soley for the purpose of moving to mprime theta
    tempBlade1SS = np.zeros([newNsection, bladeRes+2, 3])
    tempBlade2PS = np.zeros([newNsection, bladeRes+2, 3])
    tempBlade2SS = np.zeros([newNsection, bladeRes+2, 3])
    blade1PSMprime = np.zeros([newNsection,bladeRes+2,4]) # (theta, r, z, mprime)
    blade1SSMprime = np.zeros([newNsection,bladeRes+2,4])
    blade2PSMprime = np.zeros([newNsection,bladeRes+2,4])
    blade2SSMprime = np.zeros([newNsection,bladeRes+2,4])
    for d in range(newNsection):
        splitBefore = abs(0.25 * (newAllTEBlade1[d][2] - newAllLEBlade1[d][2])) + newAllLEBlade1[d][2] #Take a split of the surface of blade1 25% upstream of the blade
        splitAfter =  abs(0.75 * (newAllTEBlade1[d][2] - newAllLEBlade1[d][2])) + newAllLEBlade1[d][2] #Same thing, note that this is done on the basis of axial chord
        idx1PS1 = np.searchsorted(newBlade1PSCyl[d][:,2], splitBefore) #index before
        idx2PS1 = np.searchsorted(newBlade1PSCyl[d][:,2], splitAfter) #index after
        tempPS1 = newBlade1PSCyl[d][idx1PS1:idx2PS1] # Now this is the midsection of the pressure side surface 
        tempPZ1 = np.linspace(tempPS1[0][2], tempPS1[::-1][0][2], bladeRes) #Fit in 1000 points so as to improve the resolution of the surface 
        rPS1Func = interp1d(tempPS1[:,2], tempPS1[:,1])
        rPS1 = rPS1Func(tempPZ1)
        thPS1Func = interp1d(tempPS1[:,2], tempPS1[:,0])
        thPS1 = thPS1Func(tempPZ1)
        midPS1 = np.column_stack((thPS1, rPS1, tempPZ1))
        tempBlade1PS[d] = np.vstack((newAllLEBlade1[d], midPS1, newAllTEBlade1[d]))
        
        idx1SS1 = np.searchsorted(newBlade1SSCyl[d][:,2], splitBefore) #index before
        idx2SS1 = np.searchsorted(newBlade1SSCyl[d][:,2], splitAfter) #index after    
        tempSS1 = newBlade1SSCyl[d][idx1SS1:idx2SS1]
        tempSZ1 = np.linspace(tempSS1[0][2], tempSS1[::-1][0][2], bladeRes)
        rSS1Func = interp1d(tempSS1[:,2], tempSS1[:,1])
        rSS1 = rSS1Func(tempSZ1)
        thSS1Func = interp1d(tempSS1[:,2], tempSS1[:,0])
        thSS1 = thSS1Func(tempSZ1)
        midSS1 = np.column_stack((thSS1, rSS1, tempSZ1))
        tempBlade1SS[d] = np.vstack((newAllLEBlade1[d], midSS1, newAllTEBlade1[d])) 
        
        idx1PS2 = np.searchsorted(newBlade2PSCyl[d][:,2], splitBefore) #index before
        idx2PS2 = np.searchsorted(newBlade2PSCyl[d][:,2], splitAfter) #index after
        tempPS2 = newBlade2PSCyl[d][idx1PS2:idx2PS2] # Now this is the midsection of the pressure side surface 
        tempPZ2 = np.linspace(tempPS2[0][2], tempPS2[::-1][0][2], bladeRes) #Fit in 1000 points so as to improve the resolution of the surface 
        rPS2Func = interp1d(tempPS2[:,2], tempPS2[:,1])
        rPS2 = rPS2Func(tempPZ2)
        thPS2Func = interp1d(tempPS2[:,2], tempPS2[:,0])
        thPS2 = thPS2Func(tempPZ2)
        midPS2 = np.column_stack((thPS2, rPS2, tempPZ2))
        tempBlade2PS[d] = np.vstack((newAllLEBlade2[d], midPS2, newAllTEBlade2[d]))
        idx1SS2 = np.searchsorted(newBlade2SSCyl[d][:,2], splitBefore) #index before
        idx2SS2 = np.searchsorted(newBlade2SSCyl[d][:,2], splitAfter) #index after    
        tempSS2 = newBlade2SSCyl[d][idx1SS2:idx2SS2]
        tempSZ2 = np.linspace(tempSS2[0][2], tempSS2[::-1][0][2], bladeRes)
        rSS2Func = interp1d(tempSS2[:,2], tempSS2[:,1])
        rSS2 = rSS2Func(tempSZ2)
        thSS2Func = interp1d(tempSS2[:,2], tempSS2[:,0])
        thSS2 = thSS2Func(tempSZ2)
        midSS2 = np.column_stack((thSS2, rSS2, tempSZ2))
        tempBlade2SS[d] = np.vstack((newAllLEBlade2[d], midSS2, newAllTEBlade2[d]))   
        blade1PSMprime[d,:,0:3] = tempBlade1PS[d,:,:]
        blade1SSMprime[d,:,0:3] = tempBlade1SS[d,:,:]
        blade2PSMprime[d,:,0:3] = tempBlade2PS[d,:,:]
        blade2SSMprime[d,:,0:3] = tempBlade2SS[d,:,:]
        index = 0
        for e in range(bladeRes+2):   
            if index == 0:
                blade1PSMprime[d,e,3] = 0 #initialize 0 m' at the first profile point
                blade1SSMprime[d,e,3] = 0 #initialize 0 m' at the first profile point
                blade2PSMprime[d,e,3] = 0 #initialize 0 m' at the first profile point
                blade2SSMprime[d,e,3] = 0 #initialize 0 m' at the first profile point        
            else:
                mPrev1P = blade1PSMprime[d,:,3][e-1]
                rPrev1P = blade1PSMprime[d,:,1][e-1]
                rCurr1P = blade1PSMprime[d,:,1][e]
                zPrev1P = blade1PSMprime[d,:,2][e-1]
                zCurr1P = blade1PSMprime[d,:,2][e]
                blade1PSMprime[d,e,3] = mPrev1P + ((2/(rCurr1P + rPrev1P)) * np.sqrt((rCurr1P-rPrev1P)**2 + (zCurr1P-zPrev1P)**2))
                mPrev1S = blade1SSMprime[d,:,3][e-1]
                rPrev1S = blade1SSMprime[d,:,1][e-1]
                rCurr1S = blade1SSMprime[d,:,1][e]
                zPrev1S = blade1SSMprime[d,:,2][e-1]
                zCurr1S = blade1SSMprime[d,:,2][e]
                blade1SSMprime[d,e,3] = mPrev1S + ((2/(rCurr1S + rPrev1S)) * np.sqrt((rCurr1S-rPrev1S)**2 + (zCurr1S-zPrev1S)**2))
                mPrev2P = blade2PSMprime[d,:,3][e-1]
                rPrev2P = blade2PSMprime[d,:,1][e-1]
                rCurr2P = blade2PSMprime[d,:,1][e]
                zPrev2P = blade2PSMprime[d,:,2][e-1]
                zCurr2P = blade2PSMprime[d,:,2][e]
                blade2PSMprime[d,e,3] = mPrev2P + ((2/(rCurr2P + rPrev2P)) * np.sqrt((rCurr2P-rPrev2P)**2 + (zCurr2P-zPrev2P)**2))
                mPrev2S = blade2SSMprime[d,:,3][e-1]
                rPrev2S = blade2SSMprime[d,:,1][e-1]
                rCurr2S = blade2SSMprime[d,:,1][e]
                zPrev2S = blade2SSMprime[d,:,2][e-1]
                zCurr2S = blade2SSMprime[d,:,2][e]
                blade2SSMprime[d,e,3] = mPrev2S + ((2/(rCurr2S + rPrev2S)) * np.sqrt((rCurr2S-rPrev2S)**2 + (zCurr2S-zPrev2S)**2))           
            index +=1 

    # Generate new Inlet and Outlet Data. THis is done based on reduced blade profiles and the new surfaces that lies on the hub and casing
    # Also, another thing i did here is the new implementation to ensure that deltaM/deltaZ is relatively constant across the profile. So I checked for the furtherst points on the hub and casing 
    #and made them equal extent.
    scaledNewInlet = mf.scale(min(newAllLEBlade1[:,1]), max(newAllLEBlade1[:,1]), hub[0][0], casing[0][0],newAllLEBlade1[:,1])
    scaledNewOutlet = mf.scale(min(newAllTEBlade1[:,1]), max(newAllTEBlade1[:,1]), hub[::-1][0][0], casing[::-1][0][0],newAllTEBlade1[:,1])
    inletFunc = interp1d(inlet[:,1], inlet[:,0])
    scaledNewInletZ = inletFunc(scaledNewInlet) #You might ask why are you scaling the inlet here. Bascially, what I have done is to map the LE and TE of the blade surface to the inlet and outlet
    outletFunc = interp1d(outlet[:,1], outlet[:,0])
    scaledNewOutletZ = outletFunc(scaledNewOutlet)
    adjInlet = np.column_stack((scaledNewInletZ, scaledNewInlet))
    adjOutlet = np.column_stack((scaledNewOutletZ, scaledNewOutlet))
    tempHub = np.zeros([len(hub), 3])
    tempCas = np.zeros([len(casing), 3])
    if min(hub[:,2]) == min(casing[:,2]):
        tempHub = hub
        tempCas = casing
    elif min(hub[:,2]) < min(casing[:,2]):
        tempHub[:,:] = hub[:,:]
        tempCas[:,:] = casing[:,:]
        tempCas[0] = np.array([casing[0][0], 0, min(hub[:,2])])
    elif min(hub[:,2]) > min(casing[:,2]):
        tempHub[:,:] = hub[:,:]
        tempHub[0] = np.array([casing[0][0], 0, min(casing[:,2])])
        tempCas[:,:] = casing[:,:]
        
    if max(hub[:,2])  == max(casing[:,2]):
        tempHub[:,:] = tempHub[:,:]
        tempCas[:,:] = tempCas[:,:]   
    elif max(hub[:,2]) < max(casing[:,2]):
        tempHub[-1] = np.array([hub[-1][0], 0, max(casing[:,2])])
    elif max(hub[:,2]) > max(casing[:,2]):
        tempCas[-1] = np.array([casing[-1][0], 0, max(hub[:,2])])

    tempInletR = np.linspace(tempHub[0][0], tempCas[0][0],res)
    tempInletFunc = interp1d([tempHub[0][0], tempCas[0][0]], [tempHub[0][2], tempCas[0][2]])
    tempInletZ = tempInletFunc(tempInletR)
    inletTemp = np.column_stack((tempInletZ, tempInletR))
    tempOutletRth = np.linspace(tempHub[::-1][0][0], tempCas[::-1][0][0],res)
    tempOutletFunc = interp1d([tempHub[::-1][0][0], tempCas[::-1][0][0]], [tempHub[::-1][0][2], tempCas[::-1][0][2]])
    tempOutletZ = tempOutletFunc(tempOutletRth)
    outletTemp = np.column_stack((tempOutletZ, tempOutletRth))
    scaledTempNewInlet = mf.scale(min(newAllLEBlade1[:,1]), max(newAllLEBlade1[:,1]), tempHub[0][0], tempCas[0][0],newAllLEBlade1[:,1])
    scaledTempNewOutlet = mf.scale(min(newAllTEBlade1[:,1]), max(newAllTEBlade1[:,1]), tempHub[::-1][0][0], tempCas[::-1][0][0],newAllTEBlade1[:,1])
    scaledInletFunc = interp1d(inletTemp[:,1], inletTemp[:,0])
    scaledOutletFunc = interp1d(outletTemp[:,1], outletTemp[:,0])
    scaledTempNewInletZ = scaledInletFunc(scaledTempNewInlet) #You might ask why are you scaling the inlet here. Bascially, what I have done is to map the LE and TE of the blade surface to the inlet and outlet
    scaledTempNewOutletZ = scaledOutletFunc(scaledTempNewOutlet)
    tempInlet = np.column_stack((scaledTempNewInletZ, scaledTempNewInlet))
    tempOutlet = np.column_stack((scaledTempNewOutletZ, scaledTempNewOutlet))
    hubLEIdx = np.searchsorted(tempHub[:,2], newAllLEBlade1[0][2])
    hubTEIdx = np.searchsorted(tempHub[:,2], newAllTEBlade1[0][2])
    casLEIdx = np.searchsorted(tempCas[:,2], newAllLEBlade1[newNsection-1][2])
    casTEIdx = np.searchsorted(tempCas[:,2], newAllTEBlade1[newNsection-1][2])

    newLE2D = np.column_stack((newAllLEBlade1[:,2], newAllLEBlade1[:,1]))
    newTE2D = np.column_stack((newAllTEBlade1[:,2], newAllTEBlade1[:,1]))

    newLE2D2 = np.column_stack((newAllLEBlade2[:,2], newAllLEBlade2[:,1]))
    newTE2D2 = np.column_stack((newAllTEBlade2[:,2], newAllTEBlade2[:,1]))

    print('Basic processing complete.')
    print('Determining points on upstream/downstream extensions...')

    """
    Here I used transfinite interpolation to define the grid for the upstream and downstream extension. 
    """
    upHub = np.vstack((tempHub[0:hubLEIdx][:,[2,0]],hubLE))
    upHFunc = interp1d(upHub[:,0], upHub[:,1])
    upHub = np.column_stack((np.linspace(upHub[0][0], upHub[::-1][0][0],res), upHFunc(np.linspace(upHub[0][0], upHub[::-1][0][0],res))))
    dwHub = np.vstack((hubTE, tempHub[hubTEIdx:len(tempHub)][:,[2,0]]))
    dwHFunc = interp1d(dwHub[:,0], dwHub[:,1])
    dwHub = np.column_stack((np.linspace(dwHub[0][0], dwHub[::-1][0][0],res), dwHFunc(np.linspace(dwHub[0][0], dwHub[::-1][0][0],res))))
    upCas = np.vstack((tempCas[0:casLEIdx][:,[2,0]],casLE))
    upCFunc = interp1d(upCas[:,0], upCas[:,1])
    upCas = np.column_stack((np.linspace(upCas[0][0], upCas[::-1][0][0],res), upCFunc(np.linspace(upCas[0][0], upCas[::-1][0][0],res))))
    dwCas = np.vstack((casTE, tempCas[casTEIdx:len(tempCas)][:,[2,0]]))
    dwCFunc = interp1d(dwCas[:,0], dwCas[:,1])
    dwCas = np.column_stack((np.linspace(dwCas[0][0], dwCas[::-1][0][0],res), dwCFunc(np.linspace(dwCas[0][0], dwCas[::-1][0][0],res))))

    upNodes = tf.transfinite(tempInlet, newLE2D, upHub, upCas)
    dwNodes = tf.transfinite( newTE2D,tempOutlet, dwHub, dwCas)

    upstreamMprime = np.zeros([newNsection,res,4]) #This contains the hub and casing 
    dwstreamMprime = np.zeros([newNsection,res,4]) 
    for ef in range(newNsection):
        upstreamMprime[ef,:,0] = np.zeros(res)
        dwstreamMprime[ef,:,0] = np.zeros(res)
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
                upstreamMprime[ef,:,3] = blade1PSMprime[ef][0][3]#Initializing with the first value on the blade in mprime theta
                dwstreamMprime[ef,:,3] = blade1PSMprime[ef][::-1][0][3] #Initializing with the last value on the blade in mprime theta
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
    #%% Moving to 2D Mprime theta
    blade1PSCyl2D = np.zeros([newNsection,bladeRes+2,2]) #mprime theta
    blade1SSCyl2D = np.zeros([newNsection,bladeRes+2,2])
    blade2PSCyl2D = np.zeros([newNsection,bladeRes+2,2])
    blade2SSCyl2D = np.zeros([newNsection,bladeRes+2,2])
    allLEBlade12D = np.zeros([newNsection,2])
    allLEBlade22D = np.zeros([newNsection,2])
    allTEBlade12D = np.zeros([newNsection,2])
    allTEBlade22D = np.zeros([newNsection,2])
    upstreamCyl2D = np.zeros([newNsection,res,2])
    dwstreamCyl2D = np.zeros([newNsection,res,2])
    rotAngle = 0
    midLE = np.zeros([newNsection,2])
    for g in range(newNsection):
        midLE[g] = [0.5*(blade1PSMprime[g][0][3]+blade2PSMprime[g][0][3]), 0.5*(blade1PSMprime[g][0][0]+blade2PSMprime[g][0][0])]
        blade1PSCyl2D[g,:,:] = np.column_stack((blade1PSMprime[g,:,3], blade1PSMprime[g,:,0]))
        blade1SSCyl2D[g,:,:] = np.column_stack((blade1SSMprime[g,:,3], blade1SSMprime[g,:,0]))
        blade2PSCyl2D[g,:,:] = np.column_stack((blade2PSMprime[g,:,3], blade2PSMprime[g,:,0]))
        blade2SSCyl2D[g,:,:] = np.column_stack((blade2SSMprime[g,:,3], blade2SSMprime[g,:,0]))
        upstreamCyl2D[g,:,:] = np.column_stack((upstreamMprime[g,:,3], upstreamMprime[g,:,0]))
        dwstreamCyl2D[g,:,:] = np.column_stack((dwstreamMprime[g,:,3], dwstreamMprime[g,:,0]))
        allLEBlade12D[g] = blade1PSCyl2D[g][0]
        allLEBlade22D[g] = blade2SSCyl2D[g][0]
        allTEBlade12D[g] = blade1PSCyl2D[g][::-1][0]
        allTEBlade22D[g] = blade2SSCyl2D[g][::-1][0]
    
    #%% Determine midchord location on the blade
    midchordPS1 = np.zeros([newNsection,2])
    midchordSS1 = np.zeros([newNsection,2])
    midchordPS2 = np.zeros([newNsection,2])
    midchordSS2 = np.zeros([newNsection,2])
    for h in range(newNsection):
        chordLine1 = np.vstack((allLEBlade12D[h], allTEBlade12D[h]))
        slopeChordline1 = mf.Slope(chordLine1[0][0], chordLine1[0][1],chordLine1[1][0],chordLine1[1][1])
        center1 = mf.MidPts(chordLine1)
        perpLine1 = np.column_stack(([(np.linspace(allLEBlade12D[h][0], allTEBlade12D[h][0])), (-1/slopeChordline1)*(np.linspace(allLEBlade12D[h][0], allTEBlade12D[h][0]) - center1[0])+center1[1]]))
        midchordPS1[h] = mf.TwoLinesIntersect(perpLine1, blade1PSCyl2D[h])
        midchordSS1[h] = mf.TwoLinesIntersect(perpLine1, blade1SSCyl2D[h])
        chordLine2 = np.vstack((allLEBlade22D[h], allTEBlade22D[h]))
        slopeChordline2 = mf.Slope(chordLine2[0][0], chordLine2[0][1],chordLine2[1][0],chordLine2[1][1])
        center2 = mf.MidPts(chordLine2)
        perpLine2 = np.column_stack(([(np.linspace(allLEBlade22D[h][0], allTEBlade22D[h][0])), (-1/slopeChordline2)*(np.linspace(allLEBlade22D[h][0], allTEBlade22D[h][0]) - center2[0])+center2[1]]))
        midchordPS2[h] = mf.TwoLinesIntersect(perpLine2, blade2PSCyl2D[h])
        midchordSS2[h] = mf.TwoLinesIntersect(perpLine2, blade2SSCyl2D[h])

    print('Defining leading and trailing edge curves...')

    #%% Now define the ellipse 
    nl = 2
    precision = 4
    #This is where the Ellipse is defined using the General equation of an ellipse.
    LECurveRot = np.zeros([newNsection,passageRes,2]) #Section of the ellipse that defines the LE curve
    TECurveRot = np.zeros([newNsection,passageRes,2]) #section of the ellipse that defines the TE curve
    midCurveRot = np.zeros([newNsection,passageRes,2])  #The curve that joins the already defined mid point
    tangentLine1LERot = np.zeros([newNsection,nl,2]) #line tangent to blade1 at the LE
    tangentLine2LERot = np.zeros([newNsection,nl,2])  #line tangent to blade2 at the LE
    tangentLine1TERot = np.zeros([newNsection,nl,2]) #line tangent to blade1 at the TE
    tangentLine2TERot = np.zeros([newNsection,nl,2])  #line tangent to blade2 at the TE
    bisectorLE1Rot = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at LE for blade1
    bisectorLE2Rot = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at LE for blade2
    bisectorTE1Rot = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at TE for blade1
    bisectorTE2Rot = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at TE for blade2
    upDirRot = np.zeros([newNsection,nl,2]) #line that determines the direction of upstream extension
    dwDirRot = np.zeros([newNsection,nl,2]) #line that determines the direction of downstream extension
    flatUpstream1 = np.zeros([newNsection,nl,2]) # the Horizontal line that defines the bisector of the ellipse
    flatUpstream2 = np.zeros([newNsection,nl,2]) # the Horizontal line that defines the bisector of the ellipse
    flatDwstream1 = np.zeros([newNsection,nl,2]) # the Horizontal line that defines the bisector of the ellipse
    flatDwstream2 = np.zeros([newNsection,nl,2]) # the Horizontal line that defines the bisector of the ellipse
    upstreamCamber1 = np.zeros([newNsection,res,2]) # upstream line for blade1
    upstreamCamber2 = np.zeros([newNsection,res,2]) # upstream line for blade1
    downstreamCamber1 = np.zeros([newNsection,res,2]) # downstream line for blade1
    downstreamCamber2 = np.zeros([newNsection,res,2]) # downstream line for blade1
    upstreamExtnCamber1 = np.zeros([newNsection,nl,2]) #Default upstream line extension 
    upstreamExtnCamber2 = np.zeros([newNsection,nl,2]) #Default upstream line extension 
    downstreamExtnCamber1 = np.zeros([newNsection,nl,2]) #Default downsteam line extension 
    downstreamExtnCamber2 = np.zeros([newNsection,nl,2]) #Default downsteam line extension 
    upstreamExtnCamber1Adj = np.zeros([newNsection,nl,2])
    upstreamExtnCamber2Adj = np.zeros([newNsection,nl,2])
    downstreamExtnCamber1Adj = np.zeros([newNsection,nl,2])
    downstreamExtnCamber2Adj = np.zeros([newNsection,nl,2])
    blade1LEAngle = np.zeros([newNsection])
    blade1TEAngle = np.zeros([newNsection])
    blade2LEAngle = np.zeros([newNsection])
    blade2TEAngle = np.zeros([newNsection])
    for m in range(newNsection):
        lePtCam1 = allLEBlade12D[m] #Leading edge point on blade1
        tanLine1LEInterX = lePtCam1[1] - np.tan(angleLE1[m])*lePtCam1[0] 
        if angleLE1[m] > 0:
            tanLine1LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*np.tan(angleLE1[m]) + tanLine1LEInterX], lePtCam1)) #This is the tangent to the blade surface at LE
        else:
            tanLine1LE = np.vstack(([-upstreamMprime[m,0,3], -upstreamMprime[m,0,3]*np.tan(angleLE1[m]) + tanLine1LEInterX], lePtCam1))
        angle1Deviation = np.deg2rad(mf.find_angle(tanLine1LE[0], lePtCam1, [lePtCam1[0], lePtCam1[1]-res])) #This determines how much the tangent deviates from an imaginary vertical axis
        upstreamExtn1Camber = mf.vectorRotP(tanLine1LE[:,0],tanLine1LE[:,1],lePtCam1,np.deg2rad(90)) #This perpendicular line defines the default upstream direction 
        upStrmSlope1 = mf.Slope(upstreamExtn1Camber[0][0],upstreamExtn1Camber[1][0], upstreamExtn1Camber[0][1],upstreamExtn1Camber[1][1])
        upStrm1InterX = upstreamExtn1Camber[1][1] - upStrmSlope1*upstreamExtn1Camber[0][1]    
        upstreamExtnCamber1[m] = np.vstack((np.column_stack((upstreamMprime[m,0,3],upStrmSlope1*upstreamMprime[m,0,3]+upStrm1InterX)), np.column_stack((upstreamExtn1Camber[0][1], upstreamExtn1Camber[1][1]))))
        upstreamDir1 = mf.vectorRotP(upstreamExtnCamber1[m][:,0],upstreamExtnCamber1[m][:,1],lePtCam1,-1*angle1Deviation) #This defines the horizontal line that defines the bisector
        rotAngleLE1 = mf.find_angle([upstreamDir1[0][0], upstreamDir1[1][0]], lePtCam1, tanLine1LE[0])
        rotLeCam1 = mf.vectorRotP(tanLine1LE[:,0],tanLine1LE[:,1],lePtCam1,np.deg2rad(0.5*rotAngleLE1))
        leSlope1 = mf.Slope(rotLeCam1[0][0],rotLeCam1[1][0],rotLeCam1[0][::-1][0],rotLeCam1[1][::-1][0])
        bisector1InterX = rotLeCam1[1][1] - leSlope1*rotLeCam1[0][1]
        mLE1, thLE1 = upstreamMprime[m,0,3], upstreamMprime[m,0,3]*leSlope1 + bisector1InterX
        tangentLine1LERot[m] = tanLine1LE 
        bisectorLE1Rot[m] = np.vstack((np.column_stack((upstreamMprime[m,0,3],leSlope1*upstreamMprime[m,0,3]+bisector1InterX)) ,np.column_stack((rotLeCam1[0][1], rotLeCam1[1][1]))))
        flatUpstream1[m] = np.vstack((np.column_stack((upstreamMprime[m,0,3],upstreamDir1[1][0])), np.column_stack((upstreamDir1[0][1],upstreamDir1[1][1]))))
        upstreamExtnCamber1Adj[m] = upstreamExtnCamber1[m]
        lePtCam2 = allLEBlade22D[m]
        tanLine2LEInterX = lePtCam2[1] - np.tan(angleLE2[m])*lePtCam2[0]
        if angleLE2[m] > 0:
            tanLine2LE = np.vstack(([-upstreamMprime[m,0,3], -upstreamMprime[m,0,3]*np.tan(angleLE2[m]) + tanLine2LEInterX], lePtCam2))
        else:
            tanLine2LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*np.tan(angleLE2[m]) + tanLine2LEInterX], lePtCam2))
        angle2Deviation = np.deg2rad(mf.find_angle(tanLine2LE[0], lePtCam2, [lePtCam2[0], lePtCam2[1]+res]))
        upstreamExtn2Camber = mf.vectorRotP(tanLine2LE[:,0],tanLine2LE[:,1],lePtCam2,np.deg2rad(-90))
        upStrmSlope2 = mf.Slope(upstreamExtn2Camber[0][0],upstreamExtn2Camber[1][0], upstreamExtn2Camber[0][1],upstreamExtn2Camber[1][1])
        upStrm2InterX = upstreamExtn2Camber[1][1] - upStrmSlope2*upstreamExtn2Camber[0][1]
        upstreamExtnCamber2[m] = np.vstack((np.column_stack((upstreamMprime[m,0,3],upStrmSlope2*upstreamMprime[m,0,3]+upStrm2InterX)), np.column_stack((upstreamExtn2Camber[0][1], upstreamExtn2Camber[1][1]))))
        upstreamDir2 = mf.vectorRotP(upstreamExtnCamber2[m][:,0],upstreamExtnCamber2[m][:,1],lePtCam2,-1*angle2Deviation)
        rotAngleLE2 = mf.find_angle([upstreamDir2[0][0], upstreamDir2[1][0]], lePtCam2, tanLine2LE[0])
        rotLeCam2 = mf.vectorRotP(tanLine2LE[:,0],tanLine2LE[:,1],lePtCam2,np.deg2rad(-0.5*rotAngleLE2))
        leSlope2 = mf.Slope(rotLeCam2[0][0],rotLeCam2[1][0],rotLeCam2[0][::-1][0],rotLeCam2[1][::-1][0])
        bisector2InterX = rotLeCam2[1][1] - leSlope2*rotLeCam2[0][1]
        mLE2, thLE2 = upstreamMprime[m,0,3], upstreamMprime[m,0,3]*leSlope2+bisector2InterX
        tangentLine2LERot[m] = tanLine2LE
        bisectorLE2Rot[m] = np.vstack((np.column_stack((upstreamMprime[m,0,3],leSlope2*upstreamMprime[m,0,3]+bisector2InterX)), np.column_stack((rotLeCam2[0][1], rotLeCam2[1][1]))))
        flatUpstream2[m] = np.vstack((np.column_stack((upstreamMprime[m,0,3],upstreamDir2[1][0])), np.column_stack((upstreamDir2[0][1],upstreamDir2[1][1]))))
        upstreamExtnCamber2Adj[m] = upstreamExtnCamber2[m]
        upInterX = mf.TwoLinesIntersect(np.column_stack((rotLeCam1[0],rotLeCam1[1])) , np.column_stack((rotLeCam2[0], rotLeCam2[1])))
        upAngle = mf.find_angle(np.array([rotLeCam1[0][0], rotLeCam1[1][0]]), upInterX, np.array([rotLeCam2[0][0], rotLeCam2[1][0]]))
        upDir = np.vstack((np.array([rotLeCam1[0][0], rotLeCam1[1][0]]), upInterX))  
        rotUpLine = mf.vectorRotP(upDir[:,0],upDir[:,1],upInterX,np.deg2rad(0.5*upAngle))
        upSlope = mf.Slope(rotUpLine[0][0],rotUpLine[1][0],rotUpLine[0][::-1][0],rotUpLine[1][::-1][0])
        upInterX1 = lePtCam1[1] - upSlope*lePtCam1[0]
        mUp1 = np.linspace(upstreamMprime[m,0,3], upstreamDir1[0][-1], res)
        thUp1 = mUp1*upSlope + upInterX1
        upstreamCamber1[m] = np.column_stack((mUp1, thUp1))   
        upInterX2 = lePtCam2[1] - upSlope*lePtCam2[0]
        mUp2 = np.linspace(upstreamMprime[m,0,3], upstreamDir2[0][-1], res)
        thUp2 = mUp2*upSlope + upInterX2
        upstreamCamber2[m] = np.column_stack((mUp2, thUp2))
        upDirRot[m] = np.vstack(([upstreamMprime[m,0,3],upstreamMprime[m,0,3]*upSlope + (upInterX[1] - upSlope*upInterX[0])],upInterX))
        blade1LEAngle[m] = mf.find_angle(upstreamExtnCamber1[m][0],  upstreamExtnCamber1[m][1], np.array([upstreamExtnCamber1[m][1][0]-res, upstreamExtnCamber1[m][1][1]]))
        blade2LEAngle[m] = mf.find_angle(upstreamExtnCamber2[m][0],  upstreamExtnCamber2[m][1], np.array([upstreamExtnCamber2[m][1][0]-res, upstreamExtnCamber2[m][1][1]]))
        
        leMidPt = mf.MidPts(np.vstack((lePtCam1, lePtCam2)))
        leInitialGuess = mf.MidPts(np.vstack((upInterX, leMidPt)))
        # leInitialGuess = [leInitialGuess[0]+0.05, leInitialGuess[1]]
        # Defining the parameters that defines the ellipse at LE
        lePoints = [np.round(lePtCam1,precision), np.round(lePtCam2,precision)] # leading edge points for both blades
        leSlopes = [(np.round(lePtCam1,precision),np.round(leSlope1,precision)), (np.round(lePtCam2,precision),np.round(leSlope2,precision))] #The points the ellipse passes through and the slopes at that point 
        leEllipse = mf.find_optimal_ellipse_LE(lePoints, leSlopes, leInitialGuess) # Ellipse of the general form Ax**2 + Bxy + Cy**2 + Dx+ Ey +F = 0
        Ale,Ble,Cle,Dle,Ele,Fle = leEllipse
        leEllipseCenter = [(2*Cle*Dle - Ble*Ele) / (Ble**2 - 4*Ale*Cle), (2*Ale*Ele - Ble*Dle) / (Ble**2 - 4*Ale*Cle)] #Center of the ellipse 
        leTheta = 0.5 * np.arctan2(-Ble, Cle - Ale) #Angle at which the ellipse is rotated from its major axis to horizontal axis of the coordinate system
        leExpr1 = 2 * (Ale*Ele**2 + Cle*Dle**2 - Ble*Dle*Ele + (Ble**2 - 4*Ale*Cle)*Fle)

        leExpr2 = np.sqrt((Ale - Cle)**2 + Ble**2)
        aLE = -np.sqrt(leExpr1 * ((Ale + Cle) + leExpr2)) / (Ble**2 - 4*Ale*Cle) #major axis at LE
        bLE = -np.sqrt(leExpr1 * ((Ale + Cle) - leExpr2)) / (Ble**2 - 4*Ale*Cle) #minor axis at LE
        shiftedLE1 = [(lePtCam1[0] - leEllipseCenter[0]), (lePtCam1[1] - leEllipseCenter[1])] #The first point is shifted by subtracting from the center to move to the origin
        shiftedLE2 = [(lePtCam2[0] - leEllipseCenter[0]), (lePtCam2[1] - leEllipseCenter[1])] # same thing is done for the second point 
        rotLE1 = mf.rotate_point(shiftedLE1[0], shiftedLE1[1], leTheta, inverse=True) # The first point is then rotated counterclockwise sort of like rotating back to the origin 
        rotLE2 = mf.rotate_point(shiftedLE2[0], shiftedLE2[1], leTheta, inverse=True) # same thing is done for the second point 
        lePointT1 = np.arctan2(rotLE1[1]/bLE, rotLE1[0]/aLE) # A parameter t is defined to determine the start and end angle of the segment of the ellipse needed here
        lePointT2 = np.arctan2(rotLE2[1]/bLE, rotLE2[0]/aLE) # The vertical and horizontal axis are normalized by the major and minor axis because unlike a circle an ellipse is streched
        if abs(lePointT1 - lePointT2) <= 2*np.pi -  abs(lePointT1 - lePointT2): # in a weird way. So there is a need to 'un-scale' the ellipse for accurate start and end angle
            tPointsLE = np.linspace(lePointT1, lePointT2, passageRes) # This is the short path. kind of like the right way
        else:
            if lePointT1 < lePointT2: 
                lePointT2 = lePointT2 - 2*np.pi # Go in the clockwise direction 
            else:
                lePointT2 = lePointT2 + 2*np.pi # Go in the anti-clockwise direction 
            tPointsLE = np.linspace(lePointT1, lePointT2, passageRes)
        rotLEz = aLE * np.cos(tPointsLE)
        rotLErth = bLE * np.sin(tPointsLE) 
        shiftedLE = mf.rotate_point(rotLEz, rotLErth, leTheta, inverse=False) #The ellipse is rotated from alligning with the horizontal and vertical axis of the coordinate system
        zLeRot, rthLeRot = [(shiftedLE[0] + leEllipseCenter[0]), (shiftedLE[1] + leEllipseCenter[1])] ##Now the ellipse is shifted from the orgin 
        LECurveRotCorrect = np.column_stack((zLeRot, rthLeRot)) #Finally, the correct Ellipse. My supervisor is a Gem!
        LECurveRot[m] = np.concatenate(([lePtCam1], LECurveRotCorrect[1:-1], [lePtCam2])) #This ensures that it starts and ends at the exact line free of any approximation 
         
        #This is for the trailing edge curve
        tePtCam1 = allTEBlade12D[m]
        tanLine1TEInterX = tePtCam1[1] - np.tan(angleTE1[m])*tePtCam1[0]
        if angleTE1[m] > 0:
            tanLine1TE = np.vstack(([-dwstreamMprime[m,-1,3], -dwstreamMprime[m,-1,3]*np.tan(angleTE1[m]) + tanLine1TEInterX], tePtCam1))
        else:
            tanLine1TE = np.vstack(([dwstreamMprime[m,-1,3], dwstreamMprime[m,-1,3]*np.tan(angleTE1[m]) + tanLine1TEInterX], tePtCam1))
        angle1Deviation = mf.find_angle(tanLine1TE[0], tePtCam1, [tePtCam1[0], tePtCam1[1]-res])
        downstreamExtn1Camber = mf.vectorRotP(tanLine1TE[:,0],tanLine1TE[:,1] ,tePtCam1,np.deg2rad(-90))
        dwStrmSlope1 = mf.Slope(downstreamExtn1Camber[0][0],downstreamExtn1Camber[1][0], downstreamExtn1Camber[0][1],downstreamExtn1Camber[1][1])
        dwStrm1InterX = downstreamExtn1Camber[1][1] - dwStrmSlope1*downstreamExtn1Camber[0][1]
        downstreamExtnCamber1[m] = np.vstack((np.column_stack((dwstreamMprime[m,-1,3],dwStrmSlope1*dwstreamMprime[m,-1,3]+dwStrm1InterX)), np.column_stack((downstreamExtn1Camber[0][1], downstreamExtn1Camber[1][1]))))
        dwstreamDir1 = mf.vectorRotP(downstreamExtnCamber1[m][:,0],downstreamExtnCamber1[m][:,1] ,tePtCam1,np.deg2rad(1*angle1Deviation))
        rotAngTE1 = mf.find_angle([dwstreamDir1[0][0], dwstreamDir1[1][0]], tePtCam1, tanLine1TE[0])
        rotTeCam1 = mf.vectorRotP(tanLine1TE[:,0],tanLine1TE[:,1],tePtCam1,np.deg2rad(-0.5*rotAngTE1))
        teSlope1 = mf.Slope(rotTeCam1[0][0],rotTeCam1[1][0],rotTeCam1[0][::-1][0],rotTeCam1[1][::-1][0])
        bisector1InterXTE = rotTeCam1[1][1] - teSlope1*rotTeCam1[0][1]
        mTE1, thTE1 = dwstreamMprime[m][::-1][0][3], dwstreamMprime[m][::-1][0][3]*teSlope1+bisector1InterXTE
        tangentLine1TERot[m] = tanLine1TE
        ##START HERE. I NEED TO DO SAME THING I DID FOR THE UPSTREAM
        bisectorTE1Rot[m] = np.vstack((np.column_stack((rotTeCam1[0][1], rotTeCam1[1][1])), np.column_stack((dwstreamMprime[m,-1,3],teSlope1*dwstreamMprime[m,-1,3]+bisector1InterXTE))))
        flatDwstream1[m] = np.vstack((np.column_stack((dwstreamDir1[0][1],dwstreamDir1[1][1])), np.column_stack((dwstreamMprime[m,-1,3],dwstreamDir1[1][0]))))
        downstreamExtnCamber1Adj[m] = downstreamExtnCamber1[m]
           
        tePtCam2 = allTEBlade22D[m]
        tanLine2TEInterX = tePtCam2[1] - np.tan(angleTE2[m])*tePtCam2[0]
        # tanLine2TE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*np.tan(angleTE2[m]) + tanLine2TEInterX], tePtCam2))
        if angleTE2[m] > 0:
            tanLine2TE = np.vstack(([dwstreamMprime[m,-1,3], dwstreamMprime[m,-1,3]*np.tan(angleTE2[m]) + tanLine2TEInterX], tePtCam2))
        else:
            tanLine2TE = np.vstack(([-dwstreamMprime[m,-1,3], -dwstreamMprime[m,-1,3]*np.tan(angleTE2[m]) + tanLine2TEInterX], tePtCam2))
        angle2Deviation = mf.find_angle(tanLine2TE[0], tePtCam2, [tePtCam2[0], tePtCam2[1]+res])
        downstreamExtn2Camber = mf.vectorRotP(tanLine2TE[:,0], tanLine2TE[:,1],tePtCam2,np.deg2rad(90))
        dwStrmSlope2 = mf.Slope(downstreamExtn2Camber[0][0],downstreamExtn2Camber[1][0], downstreamExtn2Camber[0][1],downstreamExtn2Camber[1][1])
        dwStrm2InterX = downstreamExtn2Camber[1][1] - dwStrmSlope2*downstreamExtn2Camber[0][1]
        downstreamExtnCamber2[m] = np.vstack((np.column_stack((dwstreamMprime[m,-1,3],dwStrmSlope2*dwstreamMprime[m,-1,3] + dwStrm2InterX)), np.column_stack((downstreamExtn2Camber[0][1], downstreamExtn2Camber[1][1]))))
        # downstreamExtnCamber2[m] = np.column_stack((downstreamExtn2Camber[0], downstreamExtn2Camber[1]))
        dwstreamDir2 = mf.vectorRotP(downstreamExtnCamber2[m][:,0],downstreamExtnCamber2[m][:,1],tePtCam2,np.deg2rad(1*angle2Deviation))
        rotAngTE2 = mf.find_angle([dwstreamDir2[0][0], dwstreamDir2[1][0]], tePtCam2, tanLine2TE[0])
        rotTeCam2 = mf.vectorRotP(tanLine2TE[:,0],tanLine2TE[:,1],tePtCam2,np.deg2rad(0.5*rotAngTE2))
        teSlope2 = mf.Slope(rotTeCam2[0][0],rotTeCam2[1][0],rotTeCam2[0][::-1][0],rotTeCam2[1][::-1][0])
        bisector2InterXTE = rotTeCam2[1][1] - teSlope2 *rotTeCam2[0][1]
        mTE2, thTE2 = dwstreamMprime[m][::-1][0][3], dwstreamMprime[m][::-1][0][3]*leSlope2 + bisector2InterXTE
        tangentLine2TERot[m] = tanLine2TE   
        bisectorTE2Rot[m] = np.vstack((np.column_stack((rotTeCam2[0][1], rotTeCam2[1][1])), np.column_stack((dwstreamMprime[m,-1,3],teSlope2*dwstreamMprime[m,-1,3]+bisector2InterXTE))))
        flatDwstream2[m] = np.vstack((np.column_stack((dwstreamDir2[0][1],dwstreamDir2[1][1])), np.column_stack((dwstreamMprime[m,-1,3],dwstreamDir2[1][0]))))
        downstreamExtnCamber2Adj[m] = downstreamExtnCamber2[m]
        dwInterX = mf.TwoLinesIntersect(np.column_stack((rotTeCam1[0],rotTeCam1[1])) , np.column_stack((rotTeCam2[0], rotTeCam2[1])))
        dwAngle = mf.find_angle(np.array([rotTeCam1[0][0], rotTeCam1[1][0]]), dwInterX, np.array([rotTeCam2[0][0], rotTeCam2[1][0]]))
        dwHorAngle = mf.find_angle(np.array([rotTeCam2[0][0], rotTeCam2[1][0]]), dwInterX, np.array([dwInterX[0]+res, dwInterX[1]]))
        dwLine = np.vstack((np.array([rotTeCam1[0][0], rotTeCam1[1][0]]), dwInterX))
        rotDwLine = mf.vectorRotP(dwLine[:,0],dwLine[:,1],dwInterX,np.deg2rad(-0.5*dwAngle))
        dwSlope = mf.Slope(rotDwLine[0][0],rotDwLine[1][0],rotDwLine[0][::-1][0],rotDwLine[1][::-1][0])
        dwInterX1 = tePtCam1[1] - dwSlope*tePtCam1[0]
        mDw1 = np.linspace(dwstreamDir1[0][-1],dwstreamMprime[m,-1,3], res)
        thDw1 = mDw1*dwSlope + dwInterX1
        downstreamCamber1[m] = np.column_stack((mDw1, thDw1))   
        dwInterX2 = tePtCam2[1] - dwSlope*tePtCam2[0]
        mDw2 = np.linspace(dwstreamDir2[0][-1], dwstreamMprime[m,-1,3],res)
        thDw2 = mDw2*dwSlope + dwInterX2
        downstreamCamber2[m] = np.column_stack((mDw2, thDw2))
        dwDirRot[m] = np.vstack((dwInterX, [dwstreamMprime[m,-1,3],dwstreamMprime[m,-1,3]*dwSlope+(dwInterX[1] - dwInterX[0]*dwSlope)]))
        blade1TEAngle[m] = mf.find_angle(downstreamExtnCamber1[m][0],  downstreamExtnCamber1[m][1], np.array([downstreamExtnCamber1[m][1][0]+res, downstreamExtnCamber1[m][1][1]]))
        blade2TEAngle[m] = mf.find_angle(downstreamExtnCamber2[m][0],  downstreamExtnCamber2[m][1], np.array([downstreamExtnCamber2[m][1][0]+res, downstreamExtnCamber2[m][1][1]]))
        
        teMidPt = mf.MidPts(np.vstack((tePtCam1, tePtCam2)))
        teInitialGuess = mf.MidPts(np.vstack((dwInterX, teMidPt)))
        # Defining the parameters that defines the ellipse at LE
        tePoints = [np.round(tePtCam1,precision), np.round(tePtCam2,precision)] # leading edge points for both blades
        teSlopes = [(np.round(tePtCam1,precision),np.round(teSlope1,precision)), (np.round(tePtCam2,precision),np.round(teSlope2,precision))] #The points the ellipse passes through and the slopes at that point 
        teEllipse = mf.find_optimal_ellipse_LE(tePoints, teSlopes, teInitialGuess) # Ellipse of the general form Ax**2 + Bxy + Cy**2 + Dx+ Ey +F = 0
        Ate,Bte,Cte,Dte,Ete,Fte = teEllipse
        teEllipseCenter = [(2*Cte*Dte - Bte*Ete) / (Bte**2 - 4*Ate*Cte), (2*Ate*Ete - Bte*Dte) / (Bte**2 - 4*Ate*Cte)] #Center of the ellipse 
        teTheta = 0.5 * np.arctan2(-Bte, Cte - Ate) #Angle at which the ellipse is rotated from its major axis to horizontal axis of the coordinate system
        teExpr1 = 2 * (Ate*Ete**2 + Cte*Dte**2 - Bte*Dte*Ete + (Bte**2 - 4*Ate*Cte)*Fte)
        teExpr2 = np.sqrt((Ate - Cte)**2 + Bte**2)
     
        aTE = -np.sqrt(teExpr1 * ((Ate + Cte) + teExpr2)) / (Bte**2 - 4*Ate*Cte) #major axis at LE
        bTE = -np.sqrt(teExpr1 * ((Ate + Cte) - teExpr2)) / (Bte**2 - 4*Ate*Cte) #minor axis at LE
        shiftedTE1 = [(tePtCam1[0] - teEllipseCenter[0]), (tePtCam1[1] - teEllipseCenter[1])] #The first point is shifted by subtracting from the center to move to the origin
        shiftedTE2 = [(tePtCam2[0] - teEllipseCenter[0]), (tePtCam2[1] - teEllipseCenter[1])] # same thing is done for the second point 
        rotTE1 = mf.rotate_point(shiftedTE1[0], shiftedTE1[1], teTheta, inverse=True) # The first point is then rotated counterclockwise sort of like rotating back to the origin 
        rotTE2 = mf.rotate_point(shiftedTE2[0], shiftedTE2[1], teTheta, inverse=True) # same thing is done for the second point 
        tePointT1 = np.arctan2(rotTE1[1]/bTE, rotTE1[0]/aTE) # A parameter t is defined to determine the start and end angle of the segment of the ellipse needed here
        tePointT2 = np.arctan2(rotTE2[1]/bTE, rotTE2[0]/aTE) # The vertical and horizontal axis are normalized by the major and minor axis because unlike a circle an ellipse is streched
        if abs(tePointT1 - tePointT2) <= 2*np.pi -  abs(tePointT1 - tePointT2): # in a weird way. So there is a need to 'un-scale' the ellipse for accurate start and end angle
            tPointsTE = np.linspace(tePointT1, tePointT2, passageRes) # This is the short path. kind of like the right way
        else:
            if tePointT1 < tePointT2: 
                tePointT2 = tePointT2 - 2*np.pi # Go in the clockwise direction 
            else:
                tePointT2 = tePointT2 + 2*np.pi # Go in the anti-clockwise direction 
            tPointsTE = np.linspace(tePointT1, tePointT2, passageRes)
        rotTEz = aTE * np.cos(tPointsTE)
        rotTErth = bTE * np.sin(tPointsTE) 
        shiftedTE = mf.rotate_point(rotTEz, rotTErth, teTheta, inverse=False) #The ellipse is rotated from alligning with the horizontal and vertical axis of the coordinate system
        zTeRot, rthTeRot = [(shiftedTE[0] + teEllipseCenter[0]), (shiftedTE[1] + teEllipseCenter[1])] ##Now the ellipse is shifted from the orgin 
        TECurveRotCorrect = np.column_stack((zTeRot, rthTeRot)) #Finally, the correct Ellipse. My supervisor is a Gem!
        TECurveRot[m] = np.concatenate(([tePtCam1], TECurveRotCorrect[1:-1], [tePtCam2])) #This ensures that it starts and ends at the exact line free of any approximation 
        midCurveRot[m] = np.linspace(midchordSS1[m], midchordPS2[m],passageRes)

    #%% Solving the differential equation
    deltaMprimeLE1 = np.zeros(newNsection)
    alphaLE1 = np.zeros(newNsection)
    deltaMprimeLE2 = np.zeros(newNsection)
    alphaLE2 = np.zeros(newNsection)
    deltaMprimeTE1 = np.zeros(newNsection)
    alphaTE1 = np.zeros(newNsection)
    deltaMprimeTE2 = np.zeros(newNsection)
    alphaTE2 = np.zeros(newNsection)
    for n in range(newNsection):
        deltaMprimeLE1[n] = upstreamExtnCamber1[n,0,0] - upstreamExtnCamber1[n,1,0]
        alphaLE1[n] = np.deg2rad(mf.find_angle(upstreamExtnCamber1[n,0], upstreamExtnCamber1[n,1], np.array([upstreamExtnCamber1[n,1,0]-res, upstreamExtnCamber1[n,1,1]])))
        deltaMprimeLE2[n] = upstreamExtnCamber2[n,0,0] - upstreamExtnCamber2[n,1,0]
        alphaLE2[n] = np.deg2rad(mf.find_angle(upstreamExtnCamber2[n,0], upstreamExtnCamber2[n,1], np.array([upstreamExtnCamber2[n,1,0]-res, upstreamExtnCamber2[n,1,1]])))
        
        deltaMprimeTE1[n] = downstreamExtnCamber1[n,0,0] - downstreamExtnCamber1[n,1,0]
        alphaTE1[n] = np.deg2rad(mf.find_angle(downstreamExtnCamber1[n,0], downstreamExtnCamber1[n,1], np.array([downstreamExtnCamber1[n,1,0]+res, downstreamExtnCamber1[n,1,1]])))
        deltaMprimeTE2[n] = downstreamExtnCamber2[n,0,0] - downstreamExtnCamber2[n,1,0]
        alphaTE2[n] = np.deg2rad(mf.find_angle(downstreamExtnCamber2[n,0], downstreamExtnCamber2[n,1], np.array([downstreamExtnCamber2[n,1,0]+res, downstreamExtnCamber2[n,1,1]])))

    deltaThetaLE1 = fsolve(dy_dx, x0=0, args=(deltaMprimeLE1, alphaLE1))
    deltaThetaLE2 = fsolve(dy_dx, x0=0, args=(deltaMprimeLE2, alphaLE2))
    deltaThetaTE1 = fsolve(dy_dx, x0=0, args=(deltaMprimeTE1, alphaTE1))
    deltaThetaTE2 = fsolve(dy_dx, x0=0, args=(deltaMprimeTE2, alphaTE2))
    #%%  
    thetaInlet1 = upstreamExtnCamber1[:,1,1] - deltaThetaLE1
    thetaInlet2 = upstreamExtnCamber2[:,1,1] - deltaThetaLE2
    thetaOutlet1 = downstreamExtnCamber1[:,1,1] + deltaThetaTE1
    thetaOutlet2 = downstreamExtnCamber2[:,1,1] + deltaThetaTE2
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

    print('Now determining extension directions...')
    
    #%% Now I have to convert back to Cylindrical coordinate 
    #Basically I combined the upstream and bladesection and downstream meridional profiles together. 
    #First I have to get off mprime theta to z theta
    LECurvePol = np.zeros([newNsection,passageRes,2]) # z, theta
    TECurvePol = np.zeros([newNsection,passageRes,2])
    midCurvePol = np.zeros([newNsection,passageRes,2])
    upstreamCurve1Pol = np.zeros([newNsection,res,2])
    upstreamCurve2Pol = np.zeros([newNsection,res,2])
    downstreamCurve1Pol = np.zeros([newNsection,res,2])
    downstreamCurve2Pol = np.zeros([newNsection,res,2])
    tangentLine1LEPol = np.zeros([newNsection,nl,2]) #line tangent to blade1 at the LE
    tangentLine2LEPol = np.zeros([newNsection,nl,2])  #line tangent to blade2 at the LE
    tangentLine1TEPol = np.zeros([newNsection,nl,2]) #line tangent to blade1 at the TE
    tangentLine2TEPol = np.zeros([newNsection,nl,2])  #line tangent to blade2 at the TE
    bisectorLE1Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at LE for blade1
    bisectorLE2Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at LE for blade2
    bisectorTE1Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at TE for blade1
    bisectorTE2Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at TE for blade2
    upDirPol = np.zeros([newNsection,nl,2]) #line that determines the direction of upstream extension
    dwDirPol = np.zeros([newNsection,nl,2]) #line that determines the direction of downstream extension
    flatUpstream1Pol = np.zeros([newNsection,nl,2]) #
    flatUpstream2Pol = np.zeros([newNsection,nl,2])
    flatDwstream1Pol = np.zeros([newNsection,nl,2])
    flatDwstream2Pol = np.zeros([newNsection,nl,2])
    upstreamExtnCamber1Pol = np.zeros([newNsection,nl,2])
    upstreamExtnCamber2Pol = np.zeros([newNsection,nl,2])
    downstreamExtnCamber1Pol = np.zeros([newNsection,nl,2])
    downstreamExtnCamber2Pol = np.zeros([newNsection,nl,2])
    for p in range(newNsection):
        combinedMprime = np.concatenate((upstreamMprime[p][:-1], blade1PSMprime[p], dwstreamMprime[p][1:])) 
        func = interp1d((combinedMprime[:,3]), combinedMprime[:,2], fill_value='extrapolate')
        LECurvePol[p,:,0] = func(LECurveRot[p,:,0]) 
        LECurvePol[p,:,1] = LECurveRot[p,:,1]
        TECurvePol[p,:,0] = func(TECurveRot[p,:,0]) 
        TECurvePol[p,:,1] = TECurveRot[p,:,1]  
        midCurvePol[p,:,0] = func(midCurveRot[p,:,0])
        midCurvePol[p,:,1] = midCurveRot[p,:,1]
        upstreamCurve1Pol[p,:,0] = func(upstreamCamber1[p,:,0])
        upstreamCurve1Pol[p,:,1] = upstreamCamber1[p,:,1]
        upstreamCurve2Pol[p,:,0] = func(upstreamCamber2[p,:,0])
        upstreamCurve2Pol[p,:,1] = upstreamCamber2[p,:,1]
        
        downstreamCurve1Pol[p,:,0] = func(downstreamCamber1[p,:,0])  
        downstreamCurve1Pol[p,:,1] = downstreamCamber1[p,:,1]
        downstreamCurve2Pol[p,:,0] = func(downstreamCamber2[p,:,0]) 
        downstreamCurve2Pol[p,:,1] = downstreamCamber2[p,:,1]
        
        tangentLine1LEPol[p,:,0] = func(tangentLine1LERot[p,:,0])
        tangentLine1LEPol[p,:,1] = tangentLine1LERot[p,:,1]
        tangentLine2LEPol[p,:,0] = func(tangentLine2LERot[p,:,0])
        tangentLine2LEPol[p,:,1] = tangentLine2LERot[p,:,1]  
        tangentLine1TEPol[p,:,0] = func(tangentLine1TERot[p,:,0])
        tangentLine1TEPol[p,:,1] = tangentLine1TERot[p,:,1]
        tangentLine2TEPol[p,:,0] = func(tangentLine2TERot[p,:,0])
        tangentLine2TEPol[p,:,1] = tangentLine2TERot[p,:,1]
        bisectorLE1Pol[p,:,0] = func(bisectorLE1Rot[p,:,0])
        bisectorLE1Pol[p,:,1] = bisectorLE1Rot[p,:,1]
        bisectorLE2Pol[p,:,0] = func(bisectorLE2Rot[p,:,0])
        bisectorLE2Pol[p,:,1] = bisectorLE2Rot[p,:,1]
        bisectorTE1Pol[p,:,0] = func(bisectorTE1Rot[p,:,0])
        bisectorTE1Pol[p,:,1] = bisectorTE1Rot[p,:,1]
        bisectorTE2Pol[p,:,0] = func(bisectorTE2Rot[p,:,0])
        bisectorTE2Pol[p,:,1] = bisectorTE2Rot[p,:,1]
        flatUpstream1Pol[p,:,0] = func(flatUpstream1[p,:,0])
        flatUpstream1Pol[p,:,1] = flatUpstream1[p,:,1]
        flatUpstream2Pol[p,:,0] = func(flatUpstream2[p,:,0])
        flatUpstream2Pol[p,:,1] = flatUpstream2[p,:,1]   
        flatDwstream1Pol[p,:,0] = func(flatDwstream1[p,:,0])
        flatDwstream1Pol[p,:,1] = flatDwstream1[p,:,1]
        flatDwstream2Pol[p,:,0] = func(flatDwstream2[p,:,0])
        flatDwstream2Pol[p,:,1] = flatDwstream2[p,:,1]
        upstreamExtnCamber1Pol[p,:,0] = func(upstreamExtnCamber1[p,:,0])
        upstreamExtnCamber1Pol[p,:,1] = upstreamExtnCamber1[p,:,1]
        upstreamExtnCamber2Pol[p,:,0] = func(upstreamExtnCamber2[p,:,0])
        upstreamExtnCamber2Pol[p,:,1] = upstreamExtnCamber2[p,:,1]
        downstreamExtnCamber1Pol[p,:,0] = func(downstreamExtnCamber1[p,:,0])
        downstreamExtnCamber1Pol[p,:,1] = downstreamExtnCamber1[p,:,1]
        downstreamExtnCamber2Pol[p,:,0] = func(downstreamExtnCamber2[p,:,0])
        downstreamExtnCamber2Pol[p,:,1] = downstreamExtnCamber2[p,:,1]
        upDirPol[p,:,0] = func(upDirRot[p,:,0])
        upDirPol[p,:,1] = upDirRot[p,:,1]
        dwDirPol[p,:,0] = func(dwDirRot[p,:,0])
        dwDirPol[p,:,1] = dwDirRot[p,:,1]

    #%% Here I trimmed the surfaces both upstream and downstream to match the correct geometry. Remember that i extended either the hub and casing to get same extent.
    trimedUpstreamCurve1Pol = np.zeros([newNsection,res,2])
    trimedUpstreamCurve2Pol = np.zeros([newNsection,res,2])
    trimedDownstreamCurve1Pol = np.zeros([newNsection,res,2])
    trimedDownstreamCurve2Pol = np.zeros([newNsection,res,2])
    for pp in range(newNsection):
        inFunc1 = interp1d(upstreamCurve1Pol[pp,:,0], upstreamCurve1Pol[pp,:,1])
        up1Z = np.linspace(adjInlet[pp,0], upstreamCurve1Pol[pp,-1,0], res)
        up1Th = inFunc1(up1Z)
        trimedUpstreamCurve1Pol[pp] = np.column_stack((up1Z, up1Th))
        inFunc2 = interp1d(upstreamCurve2Pol[pp,:,0], upstreamCurve2Pol[pp,:,1])
        up2Z = np.linspace(adjInlet[pp,0], upstreamCurve2Pol[pp,-1,0], res)
        up2Th = inFunc2(up2Z)
        trimedUpstreamCurve2Pol[pp] = np.column_stack((up2Z, up2Th))   
     
        outFunc1 = interp1d(downstreamCurve1Pol[pp,:,0], downstreamCurve1Pol[pp,:,1])
        dw1Z = np.linspace(downstreamCurve1Pol[pp,0,0], adjOutlet[pp,0], res)
        dw1Th = outFunc1(dw1Z)
        trimedDownstreamCurve1Pol[pp] = np.column_stack((dw1Z, dw1Th)) 
        outFunc2 = interp1d(downstreamCurve2Pol[pp,:,0], downstreamCurve2Pol[pp,:,1])
        dw2Z = np.linspace(downstreamCurve2Pol[pp,0,0], adjOutlet[pp,0], res)
        dw2Th = outFunc2(dw2Z)
        trimedDownstreamCurve2Pol[pp] = np.column_stack((dw2Z, dw2Th))

    #%% Take everything back to Cylindrical Coordinate 
    LECurveCyl = np.zeros([newNsection,passageRes,3]) # theta, r, z
    TECurveCyl = np.zeros([newNsection,passageRes,3])
    midCurveCyl =  np.zeros([newNsection,passageRes,3])
    upCurve1Cyl = np.zeros([newNsection,res,3]) 
    dwCurve1Cyl = np.zeros([newNsection,res,3]) 
    upCurve2Cyl = np.zeros([newNsection,res,3]) 
    dwCurve2Cyl = np.zeros([newNsection,res,3]) 

    tangentLine1LECyl = np.zeros([newNsection,nl,3]) #line tangent to blade1 at the LE
    tangentLine2LECyl = np.zeros([newNsection,nl,3])  #line tangent to blade2 at the LE
    tangentLine1TECyl = np.zeros([newNsection,nl,3]) #line tangent to blade1 at the TE
    tangentLine2TECyl = np.zeros([newNsection,nl,3])  #line tangent to blade2 at the TE
    bisectorLE1Cyl = np.zeros([newNsection,nl,3]) #line that bisect angle between tangent line and horizontal line at LE for blade1
    bisectorLE2Cyl = np.zeros([newNsection,nl,3]) #line that bisect angle between tangent line and horizontal line at LE for blade2
    bisectorTE1Cyl = np.zeros([newNsection,nl,3]) #line that bisect angle between tangent line and horizontal line at TE for blade1
    bisectorTE2Cyl = np.zeros([newNsection,nl,3]) #line that bisect angle between tangent line and horizontal line at TE for blade2
    upDirCyl = np.zeros([newNsection,nl,3]) #line that determines the direction of upstream extension
    dwDirCyl = np.zeros([newNsection,nl,3]) #line that determines the direction of downstream extension
    flatUpstream1Cyl = np.zeros([newNsection,nl,3]) #
    flatUpstream2Cyl = np.zeros([newNsection,nl,3])
    flatDwstream1Cyl = np.zeros([newNsection,nl,3])
    flatDwstream2Cyl = np.zeros([newNsection,nl,3])
    upstreamExtnCamber1Cyl = np.zeros([newNsection,nl,3])
    upstreamExtnCamber2Cyl = np.zeros([newNsection,nl,3])
    downstreamExtnCamber1Cyl = np.zeros([newNsection,nl,3])
    downstreamExtnCamber2Cyl = np.zeros([newNsection,nl,3])
    for q in range(newNsection):

        combinedRadius = np.concatenate((upstreamMprime[q][:-1], blade1PSMprime[q], dwstreamMprime[q][1:])) 
        funcR = interp1d(combinedRadius[:,2], combinedRadius[:,1])
        LECurveCyl[q,:,2] = LECurvePol[q,:,0]
        LECurveCyl[q,:,1] = funcR(LECurveCyl[q,:,2])
        LECurveCyl[q,:,0] = LECurvePol[q,:,1]
        TECurveCyl[q,:,2] = TECurvePol[q,:,0]
        TECurveCyl[q,:,1] = funcR(TECurveCyl[q,:,2])
        TECurveCyl[q,:,0] = TECurvePol[q,:,1]
        midCurveCyl[q,:,2] = midCurvePol[q,:,0]
        midFunc = interp1d(newBlade1SSCyl[q][:,2],newBlade1SSCyl[q][:,1])
        midCurveCyl[q,:,1] = midFunc(midCurveCyl[q,:,2])
        midCurveCyl[q,:,0] = midCurvePol[q,:,1]
        upDwMprime = np.vstack((upstreamMprime[q], dwstreamMprime[q]))
        upDwFunc = interp1d(upDwMprime[:,2], upDwMprime[:,1], fill_value='extrapolate') #The only reason I am extrapolating here is because I have this additional lines that are not part of the 
        upCurve1Cyl[q,:,0] = trimedUpstreamCurve1Pol[q,:,1] #domain but needed for visualization
        upCurve1Cyl[q,:,1] = upDwFunc(trimedUpstreamCurve1Pol[q,:,0])
        upCurve1Cyl[q,:,2] = trimedUpstreamCurve1Pol[q,:,0]
        upCurve2Cyl[q,:,0] = trimedUpstreamCurve2Pol[q,:,1]
        upCurve2Cyl[q,:,1] = upDwFunc(trimedUpstreamCurve2Pol[q,:,0])
        upCurve2Cyl[q,:,2] = trimedUpstreamCurve2Pol[q,:,0]
        
        dwCurve1Cyl[q,:,0] = trimedDownstreamCurve1Pol[q,:,1]
        dwCurve1Cyl[q,:,1] = upDwFunc(trimedDownstreamCurve1Pol[q,:,0])
        dwCurve1Cyl[q,:,2] = trimedDownstreamCurve1Pol[q,:,0]
        dwCurve2Cyl[q,:,0] = trimedDownstreamCurve2Pol[q,:,1]
        dwCurve2Cyl[q,:,1] = upDwFunc(trimedDownstreamCurve2Pol[q,:,0])
        dwCurve2Cyl[q,:,2] = trimedDownstreamCurve2Pol[q,:,0]
        
        tangentLine1LECyl[q,:,0] = tangentLine1LEPol[q,:,1]
        tangentLine1LECyl[q,:,1] = upDwFunc(tangentLine1LEPol[q,:,0])
        tangentLine1LECyl[q,:,2] = tangentLine1LEPol[q,:,0]
        tangentLine2LECyl[q,:,0] = tangentLine2LEPol[q,:,1]
        tangentLine2LECyl[q,:,1] = upDwFunc(tangentLine2LEPol[q,:,0])
        tangentLine2LECyl[q,:,2] = tangentLine2LEPol[q,:,0]  
        tangentLine1TECyl[q,:,0] = tangentLine1TEPol[q,:,1]
        tangentLine1TECyl[q,:,1] = upDwFunc(tangentLine1TEPol[q,:,0])
        tangentLine1TECyl[q,:,2] = tangentLine1TEPol[q,:,0]
        tangentLine2TECyl[q,:,0] = tangentLine2TEPol[q,:,1]
        tangentLine2TECyl[q,:,1] = upDwFunc(tangentLine2TEPol[q,:,0])
        tangentLine2TECyl[q,:,2] = tangentLine2TEPol[q,:,0]  
        bisectorLE1Cyl[q,:,0] = bisectorLE1Pol[q,:,1]
        bisectorLE1Cyl[q,:,1] = upDwFunc(bisectorLE1Pol[q,:,0])
        bisectorLE1Cyl[q,:,2] = bisectorLE1Pol[q,:,0]
        bisectorLE2Cyl[q,:,0] = bisectorLE2Pol[q,:,1]
        bisectorLE2Cyl[q,:,1] = upDwFunc(bisectorLE2Pol[q,:,0])
        bisectorLE2Cyl[q,:,2] = bisectorLE2Pol[q,:,0]
        bisectorTE1Cyl[q,:,0] = bisectorTE1Pol[q,:,1]
        bisectorTE1Cyl[q,:,1] = upDwFunc(bisectorTE1Pol[q,:,0])
        bisectorTE1Cyl[q,:,2] = bisectorTE1Pol[q,:,0]
        bisectorTE2Cyl[q,:,0] = bisectorTE2Pol[q,:,1]
        bisectorTE2Cyl[q,:,1] = upDwFunc(bisectorTE2Pol[q,:,0])
        bisectorTE2Cyl[q,:,2] = bisectorTE2Pol[q,:,0]    
        flatUpstream1Cyl[q,:,0] = flatUpstream1Pol[q,:,1]
        flatUpstream1Cyl[q,:,1] = upDwFunc(flatUpstream1Pol[q,:,0])
        flatUpstream1Cyl[q,:,2] = flatUpstream1Pol[q,:,0]
        flatUpstream2Cyl[q,:,0] = flatUpstream2Pol[q,:,1]
        flatUpstream2Cyl[q,:,1] = upDwFunc(flatUpstream2Pol[q,:,0])
        flatUpstream2Cyl[q,:,2] = flatUpstream2Pol[q,:,0]  
        flatDwstream1Cyl[q,:,0] = flatDwstream1Pol[q,:,1]
        flatDwstream1Cyl[q,:,1] = upDwFunc(flatDwstream1Pol[q,:,0])
        flatDwstream1Cyl[q,:,2] = flatDwstream1Pol[q,:,0]
        flatDwstream2Cyl[q,:,0] = flatDwstream2Pol[q,:,1]
        flatDwstream2Cyl[q,:,1] = upDwFunc(flatDwstream2Pol[q,:,0])
        flatDwstream2Cyl[q,:,2] = flatDwstream2Pol[q,:,0]  
        upstreamExtnCamber1Cyl[q,:,0] = upstreamExtnCamber1Pol[q,:,1]
        upstreamExtnCamber1Cyl[q,:,1] = upDwFunc(upstreamExtnCamber1Pol[q,:,0])
        upstreamExtnCamber1Cyl[q,:,2] = upstreamExtnCamber1Pol[q,:,0]
        upstreamExtnCamber2Cyl[q,:,0] = upstreamExtnCamber2Pol[q,:,1]
        upstreamExtnCamber2Cyl[q,:,1] = upDwFunc(upstreamExtnCamber2Pol[q,:,0])
        upstreamExtnCamber2Cyl[q,:,2] = upstreamExtnCamber2Pol[q,:,0]   
        downstreamExtnCamber1Cyl[q,:,0] = downstreamExtnCamber1Pol[q,:,1]
        downstreamExtnCamber1Cyl[q,:,1] = upDwFunc(downstreamExtnCamber1Pol[q,:,0])
        downstreamExtnCamber1Cyl[q,:,2] = downstreamExtnCamber1Pol[q,:,0]
        downstreamExtnCamber2Cyl[q,:,0] = downstreamExtnCamber2Pol[q,:,1]
        downstreamExtnCamber2Cyl[q,:,1] = upDwFunc(downstreamExtnCamber2Pol[q,:,0])
        downstreamExtnCamber2Cyl[q,:,2] = downstreamExtnCamber2Pol[q,:,0]    
        upDirCyl[q,:,0] = upDirPol[q,:,1]
        upDirCyl[q,:,1] = upDwFunc(upDirPol[q,:,0])
        upDirCyl[q,:,2] = upDirPol[q,:,0]
        dwDirCyl[q,:,0] = dwDirPol[q,:,1]
        dwDirCyl[q,:,1] = upDwFunc(dwDirPol[q,:,0])
        dwDirCyl[q,:,2] = dwDirPol[q,:,0]    

    print('Collecting final data arrays...')

    hubLEInlet1 = np.column_stack((upCurve1Cyl[0][0,0]*upCurve1Cyl[0][0,1], upCurve1Cyl[0][0,2]))
    hubLEInlet2 = np.column_stack((upCurve2Cyl[0][0,0]*upCurve2Cyl[0][0,1], upCurve2Cyl[0][0,2]))
    casLEInlet1 = np.column_stack((upCurve1Cyl[-1][0,0]*upCurve1Cyl[-1][0,1], upCurve1Cyl[-1][0,2]))
    casLEInlet2 = np.column_stack((upCurve2Cyl[-1][0,0]*upCurve2Cyl[-1][0,1], upCurve2Cyl[-1][0,2]))
    hubTEInlet1 = np.column_stack((dwCurve1Cyl[0][-1,0]*dwCurve1Cyl[0][-1,1], dwCurve1Cyl[0][-1,2]))
    hubTEInlet2 = np.column_stack((dwCurve2Cyl[0][-1,0]*dwCurve2Cyl[0][-1,1], dwCurve2Cyl[0][-1,2]))
    casTEInlet1 = np.column_stack((dwCurve1Cyl[-1][-1,0]*dwCurve1Cyl[-1][-1,1], dwCurve1Cyl[-1][-1,2]))
    casTEInlet2 = np.column_stack((dwCurve2Cyl[-1][-1,0]*dwCurve2Cyl[-1][-1,1], dwCurve2Cyl[-1][-1,2]))

    hubLETheta = np.linspace(hubLEInlet1[0], hubLEInlet2[0], passageRes)
    hubTETheta = np.linspace(hubTEInlet1[0], hubTEInlet2[0], passageRes)
    casLETheta = np.linspace(casLEInlet1[0], casLEInlet2[0], passageRes)
    casTETheta = np.linspace(casTEInlet1[0], casTEInlet2[0], passageRes)
    # %% Here I made a split of he domain. The reason for doing this is to make sure that the midPoint shares a unique point 
    #Break the bladeSurface into two separated by the midpoint
    leSection1 = np.zeros([newNsection, bladeRes, 2])
    leSection2 = np.zeros([newNsection, bladeRes, 2])
    teSection1 = np.zeros([newNsection, bladeRes, 2])
    teSection2 = np.zeros([newNsection, bladeRes, 2])
    leSectionB1 = np.zeros([newNsection, bladeRes, 3])
    leSectionB2 = np.zeros([newNsection, bladeRes, 3])
    teSectionB1 = np.zeros([newNsection, bladeRes, 3])
    teSectionB2 = np.zeros([newNsection, bladeRes, 3])
    inletLowTheta = np.zeros([newNsection,res,2])
    inletHighTheta = np.zeros([newNsection,res,2])
    outletLowTheta = np.zeros([newNsection,res,2])
    outletHighTheta = np.zeros([newNsection,res,2])
    LECurveZRTH = np.zeros([newNsection,passageRes,2])
    TECurveZRTH = np.zeros([newNsection,passageRes,2])
    MidCurveZRTH = np.zeros([newNsection,passageRes,2])
    for r in range(newNsection):
        mid1Idx = np.searchsorted(newBlade1SSCyl[r,:,2], midCurveCyl[r][0][2])
        mid2Idx = np.searchsorted(newBlade2PSCyl[r,:,2], midCurveCyl[r][-1][2])
        leCurve1 = np.vstack((newBlade1SSCyl[r,0:mid1Idx], midCurveCyl[r][0]))
        teCurve1 = np.vstack((midCurveCyl[r][0], newBlade1SSCyl[r,mid1Idx:N]))
        leSectionA = mf.resample_curve_preserve_density(leCurve1, bladeRes)
        leSection1[r,:,:] = np.column_stack((leSectionA[:,1]*leSectionA[:,0], leSectionA[:,2])) 
        leSectionB1[r,:,:] = mf.resample_curve_preserve_density(leCurve1, bladeRes)
        teSectionA = mf.resample_curve_preserve_density(teCurve1, bladeRes) 
        teSection1[r,:,:] = np.column_stack((teSectionA[:,1]*teSectionA[:,0], teSectionA[:,2])) 
        teSectionB1[r,:,:] = mf.resample_curve_preserve_density(teCurve1, bladeRes) 
        leCurve2 = np.vstack((newBlade2PSCyl[r,0:mid2Idx], midCurveCyl[r][-1]))
        teCurve2 = np.vstack((midCurveCyl[r][-1], newBlade2PSCyl[r,mid2Idx:N]))
        leSectionB = mf.resample_curve_preserve_density(leCurve2, bladeRes)
        leSection2[r,:,:] = np.column_stack((leSectionB[:,1]*leSectionB[:,0], leSectionB[:,2])) 
        leSectionB2[r,:,:] = mf.resample_curve_preserve_density(leCurve2, bladeRes)
        teSectionB = mf.resample_curve_preserve_density(teCurve2, bladeRes) 
        teSection2[r,:,:] = np.column_stack((teSectionB[:,1]*teSectionB[:,0], teSectionB[:,2])) 
        teSectionB2[r,:,:] = mf.resample_curve_preserve_density(teCurve2, bladeRes)
        inletLowTheta[r,:,:] = np.column_stack((upCurve1Cyl[r,:,1]*upCurve1Cyl[r,:,0], upCurve1Cyl[r,:,2]))
        inletHighTheta[r,:,:] = np.column_stack((upCurve2Cyl[r,:,1]*upCurve2Cyl[r,:,0], upCurve2Cyl[r,:,2]))
        outletLowTheta[r,:,:] = np.column_stack((dwCurve1Cyl[r,:,1]*dwCurve1Cyl[r,:,0], dwCurve1Cyl[r,:,2]))
        outletHighTheta[r,:,:] = np.column_stack((dwCurve2Cyl[r,:,1]*dwCurve2Cyl[r,:,0], dwCurve2Cyl[r,:,2]))   
        LECurveZRTH[r,:,:] = np.column_stack((LECurveCyl[r,:,1]*LECurveCyl[r,:,0], LECurveCyl[r,:,2]))  
        TECurveZRTH[r,:,:] = np.column_stack((TECurveCyl[r,:,1]*TECurveCyl[r,:,0], TECurveCyl[r,:,2])) 
        MidCurveZRTH[r,:,:] = np.column_stack((midCurveCyl[r,:,1]*midCurveCyl[r,:,0], midCurveCyl[r,:,2])) 

    inletHubLETheta = np.linspace(inletLowTheta[0][0][[0,1]], inletHighTheta[0][0][[0,1]], passageRes)
    inletHubTETheta = np.linspace(inletLowTheta[0][::-1][0][[0,1]], inletHighTheta[0][::-1][0][[0,1]], passageRes)
    inletCasLETheta = np.linspace(inletLowTheta[newNsection-1][0][[0,1]], inletHighTheta[newNsection-1][0][[0,1]], passageRes)
    inletCasTETheta = np.linspace(inletLowTheta[newNsection-1][::-1][0][[0,1]], inletHighTheta[newNsection-1][::-1][0][[0,1]], passageRes)

    outletHubLETheta = np.linspace(outletLowTheta[0][-1][[0,1]], outletHighTheta[0][-1][[0,1]], passageRes)
    outletHubTETheta = np.linspace(outletLowTheta[0][::-1][-1][[0,1]], outletHighTheta[0][::-1][-1][[0,1]], passageRes)
    outletCasLETheta = np.linspace(outletLowTheta[newNsection-1][-1][[0,1]], outletHighTheta[newNsection-1][-1][[0,1]], passageRes)
    outletCasTETheta = np.linspace(outletLowTheta[newNsection-1][::-1][-1][[0,1]], outletHighTheta[newNsection-1][::-1][-1][[0,1]], passageRes)

    #%% Now I need to get the inner points for the inlet outlet hub and casing 
    #Using transfinite interpolation 
    '''
     transfinite(lower, upper, left, right)
    '''
    inletHubNodes = tf.transfinite(inletHubLETheta, LECurveZRTH[0][:,[0,1]], inletLowTheta[0][:,[0,1]], inletHighTheta[0][:,[0,1]])
    inletCasNodes = tf.transfinite(inletCasLETheta, LECurveZRTH[-1][:,[0,1]], inletLowTheta[-1][:,[0,1]], inletHighTheta[-1][:,[0,1]])
    leHubNodes = tf.transfinite(LECurveZRTH[0][:,[0,1]], MidCurveZRTH[0][:,[0,1]], leSection1[0][:,[0,1]], leSection2[0][:,[0,1]])
    teHubNodes = tf.transfinite(MidCurveZRTH[0][:,[0,1]], TECurveZRTH[0][:,[0,1]], teSection1[0][:,[0,1]], teSection2[0][:,[0,1]])
    leCasNodes = tf.transfinite(LECurveZRTH[-1][:,[0,1]], midCurveCyl[-1][:,[0,1]], leSection1[-1][:,[0,1]], leSection2[-1][:,[0,1]])
    teCasNodes = tf.transfinite(MidCurveZRTH[-1][:,[0,1]], TECurveZRTH[-1][:,[0,1]], teSection1[-1][:,[0,1]], teSection2[-1][:,[0,1]])
    outletHubNodes = tf.transfinite(TECurveZRTH[0][:,[0,1]], outletHubLETheta, outletLowTheta[0][:,[0,1]], outletHighTheta[0][:,[0,1]])
    outletCasNodes = tf.transfinite(TECurveZRTH[-1][:,[0,1]], outletCasLETheta, outletLowTheta[-1][:,[0,1]], outletHighTheta[-1][:,[0,1]])
    inNodes = tf.transfinite(hubLETheta, casLETheta, inletLowTheta[:,0][:,[0,1]], inletHighTheta[:,0][:,[0,1]])
    outNodes = tf.transfinite(hubTETheta, casTETheta, outletLowTheta[:,-1][:,[0,1]], outletHighTheta[:,-1][:,[0,1]])

    #%%
    #Now split the nodes to get the profile at constant radius
    inletHubProfiles = np.zeros([passageRes,res,2])
    inletCasProfiles = np.zeros([passageRes,res,2])
    outletHubProfiles = np.zeros([passageRes,res,2])
    outletCasProfiles = np.zeros([passageRes,res,2])
    leHubProfiles = np.zeros([passageRes,bladeRes,2])
    leCasProfiles = np.zeros([passageRes,bladeRes,2])
    teHubProfiles = np.zeros([passageRes,bladeRes,2])
    teCasProfiles = np.zeros([passageRes,bladeRes,2])
    allInProfiles = np.zeros([passageRes,newNsection,2])
    allOutProfiles = np.zeros([passageRes,newNsection,2])
    for t in range(passageRes):
        tempInletHub = np.zeros([res,2])
        tempInletCas = np.zeros([res,2])
        tempOutletHub = np.zeros([res,2])
        tempOutletCas = np.zeros([res,2])
        tempLeHubNodes = np.zeros([bladeRes,2])
        tempLeCasNodes = np.zeros([bladeRes,2])
        tempTeHubNodes = np.zeros([bladeRes,2])
        tempTeCasNodes = np.zeros([bladeRes,2])
        tempIn = np.zeros([newNsection,2])
        tempOut = np.zeros([newNsection,2])
        for u in range(res):
            tempInletHub[u] = inletHubNodes[t*(res) + u]
            tempInletCas[u] = inletCasNodes[t*(res) + u]
            tempOutletHub[u] = outletHubNodes[t*(res) + u]
            tempOutletCas[u] = outletCasNodes[t*(res) + u]
            if u < newNsection:
                tempIn[u] = inNodes[t*newNsection + u]
                tempOut[u] = outNodes[t*newNsection + u]
        for v in range(bladeRes):
            tempLeHubNodes[v] = leHubNodes[t*(bladeRes) + v]
            tempLeCasNodes[v] = leCasNodes[t*(bladeRes) + v]
            tempTeHubNodes[v] = teHubNodes[t*(bladeRes) + v]
            tempTeCasNodes[v] = teCasNodes[t*(bladeRes) + v]
        allInProfiles[t] = tempIn
        allOutProfiles[t] = tempOut
        inletHubProfiles[t] = tempInletHub
        inletCasProfiles[t] =  tempInletCas
        outletHubProfiles[t] = tempOutletHub
        outletCasProfiles[t] = tempOutletCas
        leHubProfiles[t] = tempLeHubNodes
        leCasProfiles[t] = tempLeCasNodes
        teHubProfiles[t] = tempTeHubNodes
        teCasProfiles[t] = tempTeCasNodes

    #%%
    #Take everything back to Cartesian Coordinate 
    inletHubCyl = np.zeros([passageRes,res,3])
    inletCasCyl = np.zeros([passageRes,res,3])
    leHubCyl = np.zeros([passageRes,bladeRes,3])
    leCasCyl = np.zeros([passageRes,bladeRes,3])
    teHubCyl = np.zeros([passageRes,bladeRes,3])
    teCasCyl = np.zeros([passageRes,bladeRes,3])
    outletHubCyl = np.zeros([passageRes,res,3])
    outletCasCyl = np.zeros([passageRes,res,3])
    inCyl = np.zeros([passageRes, newNsection, 3])
    outCyl = np.zeros([passageRes, newNsection, 3])

    for u in range(passageRes):
        rInletHubFunc = interp1d(np.round(upstreamMprime[0,:,2], precision),np.round(upstreamMprime[0,:,1], precision), bounds_error=True)
        rInletHub = rInletHubFunc(np.round(inletHubProfiles[u][:,1],precision))
        inletHubCyl[u] = np.column_stack((inletHubProfiles[u][:,0]/rInletHub, rInletHub,inletHubProfiles[u][:,1]))
        rInletCasFunc = interp1d(np.round(upstreamMprime[-1,:,2], precision),np.round(upstreamMprime[-1,:,1], precision), bounds_error=True)
        rInletCas = rInletCasFunc(np.round(inletCasProfiles[u][:,1], precision))
        inletCasCyl[u] = np.column_stack((inletCasProfiles[u][:,0]/rInletCas, rInletCas,inletCasProfiles[u][:,1]))
        rLeHubCylFunc = interp1d(hub[:,2], hub[:,0], bounds_error=True)
        rLeHubCyl = rLeHubCylFunc(np.round(leHubProfiles[u,:,1], precision))
        leHubCyl[u] = np.column_stack((leHubProfiles[u,:,0]/rLeHubCyl,  rLeHubCyl, leHubProfiles[u,:,1]))
        rLeCasCylFunc = interp1d(casing[:,2], casing[:,0], bounds_error=True)
        rLeCasCyl = rLeCasCylFunc(np.round(leCasProfiles[u,:,1], precision))
        leCasCyl[u] = np.column_stack((leCasProfiles[u,:,0]/rLeCasCyl ,  rLeCasCyl, leCasProfiles[u,:,1]))
        rTeHubCyl =  rLeHubCylFunc(np.round(teHubProfiles[u,:,1], precision))
        teHubCyl[u] = np.column_stack((teHubProfiles[u,:,0]/rTeHubCyl ,  rTeHubCyl, teHubProfiles[u,:,1]))
        rTeCasCyl = rLeCasCylFunc(np.round(teCasProfiles[u,:,1], precision))
        teCasCyl[u] = np.column_stack((teCasProfiles[u,:,0]/rTeCasCyl,  rTeCasCyl,  teCasProfiles[u,:,1]))
        rOutletHubFunc = interp1d(np.round(dwstreamMprime[0,:,2], precision), np.round(dwstreamMprime[0,:,1], precision), bounds_error=True)
        rOutletHub = rOutletHubFunc(np.round(outletHubProfiles[u][:,1], precision))
        outletHubCyl[u] = np.column_stack((outletHubProfiles[u][:,0]/rOutletHub, rOutletHub, outletHubProfiles[u][:,1]))
        rOutletCasFunc = interp1d(np.round(dwstreamMprime[-1,:,2], precision), np.round(dwstreamMprime[-1,:,1], precision), bounds_error=True)
        rOutletCas = rOutletCasFunc(np.round(outletCasProfiles[u][:,1], precision))
        outletCasCyl[u] = np.column_stack((outletCasProfiles[u][:,0]/rOutletCas, rOutletCas ,outletCasProfiles[u][:,1]))
        rInFunc = interp1d(np.round(allInProfiles[u][:,0],10), allInProfiles[u][:,1])
        rIn = upCurve1Cyl[:,0,1]
        inCyl[u] = np.column_stack((allInProfiles[u][:,0]/rIn, rIn, allInProfiles[u][:,1]))
        rOutFunc = interp1d(np.round(allOutProfiles[u][:,0],10), allOutProfiles[u][:,1])
        rOut = dwCurve1Cyl[:,-1,1]
        outCyl[u] = np.column_stack((allOutProfiles[u][:,0]/rOut, rOut, allOutProfiles[u][:,1]))   
        
    inletHubCart = np.zeros([passageRes,res,3])
    inletCasCart = np.zeros([passageRes,res,3])
    leHubCart = np.zeros([passageRes,bladeRes,3])
    leCasCart = np.zeros([passageRes,bladeRes,3])
    teHubCart = np.zeros([passageRes,bladeRes,3])
    teCasCart = np.zeros([passageRes,bladeRes,3])
    outletHubCart = np.zeros([passageRes,res,3])
    outletCasCart = np.zeros([passageRes,res,3])
    inCart = np.zeros([passageRes, newNsection, 3])
    outCart = np.zeros([passageRes, newNsection, 3])

    inletLowThetaCart = np.zeros([newNsection, res, 3])
    inletHighThetaCart = np.zeros([newNsection, res, 3])
    leLowThetaCart = np.zeros([newNsection, bladeRes, 3])
    leHighThetaCart = np.zeros([newNsection, bladeRes, 3])
    teLowThetaCart = np.zeros([newNsection, bladeRes, 3])
    teHighThetaCart = np.zeros([newNsection, bladeRes, 3])
    outletLowThetaCart = np.zeros([newNsection, res, 3])
    outletHighThetaCart = np.zeros([newNsection, res, 3])

    LECart =  np.zeros([newNsection, passageRes, 3])
    TECart = np.zeros([newNsection, passageRes, 3])
    midCart = np.zeros([newNsection, passageRes, 3])

    for m in range(passageRes):
        inletHubCart[m] = np.array(mf.pol2cart(inletHubCyl[m][:,0], inletHubCyl[m][:,1], inletHubCyl[m][:,2])).T
        inletCasCart[m] = np.array(mf.pol2cart(inletCasCyl[m][:,0], inletCasCyl[m][:,1], inletCasCyl[m][:,2])).T
        leHubCart[m] = np.array(mf.pol2cart(leHubCyl[m][:,0], leHubCyl[m][:,1], leHubCyl[m][:,2])).T
        leCasCart[m] = np.array(mf.pol2cart(leCasCyl[m][:,0], leCasCyl[m][:,1], leCasCyl[m][:,2])).T
        teHubCart[m] = np.array(mf.pol2cart(teHubCyl[m][:,0], teHubCyl[m][:,1], teHubCyl[m][:,2])).T
        teCasCart[m] = np.array(mf.pol2cart(teCasCyl[m][:,0], teCasCyl[m][:,1], teCasCyl[m][:,2])).T 
        outletHubCart[m] = np.array(mf.pol2cart(outletHubCyl[m][:,0], outletHubCyl[m][:,1], outletHubCyl[m][:,2])).T
        outletCasCart[m] = np.array(mf.pol2cart(outletCasCyl[m][:,0], outletCasCyl[m][:,1], outletCasCyl[m][:,2])).T    
        inCart[m] = np.array(mf.pol2cart(inCyl[m][:,0], inCyl[m][:,1], inCyl[m][:,2])).T
        outCart[m] = np.array(mf.pol2cart(outCyl[m][:,0], outCyl[m][:,1], outCyl[m][:,2])).T
        if m < newNsection:   
            inletLowThetaCart[m] = np.array(mf.pol2cart(upCurve1Cyl[m][:,0], upCurve1Cyl[m][:,1], upCurve1Cyl[m][:,2])).T
            inletHighThetaCart[m] = np.array(mf.pol2cart(upCurve2Cyl[m][:,0], upCurve2Cyl[m][:,1], upCurve2Cyl[m][:,2])).T
            leLowThetaCart[m] = np.array(mf.pol2cart(leSectionB1[m][:,0], leSectionB1[m][:,1], leSectionB1[m][:,2])).T
            leHighThetaCart[m] = np.array(mf.pol2cart(leSectionB2[m][:,0], leSectionB2[m][:,1], leSectionB2[m][:,2])).T
            teLowThetaCart[m] = np.array(mf.pol2cart(teSectionB1[m][:,0], teSectionB1[m][:,1], teSectionB1[m][:,2])).T
            teHighThetaCart[m] = np.array(mf.pol2cart(teSectionB2[m][:,0], teSectionB2[m][:,1], teSectionB2[m][:,2])).T
            outletLowThetaCart[m] = np.array(mf.pol2cart(dwCurve1Cyl[m][:,0], dwCurve1Cyl[m][:,1], dwCurve1Cyl[m][:,2])).T
            outletHighThetaCart[m] = np.array(mf.pol2cart(dwCurve2Cyl[m][:,0], dwCurve2Cyl[m][:,1], dwCurve2Cyl[m][:,2])).T    
            LECart[m] = np.array(mf.pol2cart(LECurveCyl[m][:,0], LECurveCyl[m][:,1], LECurveCyl[m][:,2])).T  
            TECart[m] = np.array(mf.pol2cart(TECurveCyl[m][:,0], TECurveCyl[m][:,1], TECurveCyl[m][:,2])).T 
            midCart[m] = np.array(mf.pol2cart(midCurveCyl[m][:,0], midCurveCyl[m][:,1], midCurveCyl[m][:,2])).T 
            

    lowThetaCart = np.zeros([newNsection, res*2 + bladeRes*2 -3, 3])
    highThetaCart = np.zeros([newNsection, res*2 + bladeRes*2 -3, 3])
    hubCart = np.zeros([passageRes, res*2 + bladeRes*2 - 3, 3])
    casCart =  np.zeros([passageRes, res*2 + bladeRes*2 - 3, 3])
    for x in range(passageRes):
        hubCart[x] = np.vstack((inletHubCart[x], leHubCart[x][1:-1], teHubCart[x][:-1], outletHubCart[x]))
        casCart[x] = np.vstack((inletCasCart[x], leCasCart[x][1:-1], teCasCart[x][:-1], outletCasCart[x]))
        if x < newNsection:
            lowThetaCart[x] = np.vstack((inletLowThetaCart[x], leLowThetaCart[x][1:-1], teLowThetaCart[x][:-1], outletLowThetaCart[x]))
            highThetaCart[x] = np.vstack((inletHighThetaCart[x], leHighThetaCart[x][1:-1], teHighThetaCart[x][:-1], outletHighThetaCart[x]))
    #%%
    Xp = np.zeros([newNsection, res*2 + bladeRes*2 -3])
    Yp = np.zeros([newNsection, res*2 + bladeRes*2 -3])
    Zp = np.zeros([newNsection, res*2 + bladeRes*2 -3])
    Xs = np.zeros([newNsection, res*2 + bladeRes*2 -3])
    Ys = np.zeros([newNsection, res*2 + bladeRes*2 -3])
    Zs = np.zeros([newNsection, res*2 + bladeRes*2 -3])
    Xle = np.zeros([newNsection, passageRes])
    Yle = np.zeros([newNsection, passageRes])
    Zle = np.zeros([newNsection, passageRes])
    Xte = np.zeros([newNsection, passageRes])
    Yte = np.zeros([newNsection, passageRes])
    Zte = np.zeros([newNsection, passageRes])
    Xmid = np.zeros([newNsection, passageRes])
    Ymid = np.zeros([newNsection, passageRes])
    Zmid = np.zeros([newNsection, passageRes])

    Xh = np.zeros([passageRes, res*2 + bladeRes*2 -3])
    Yh = np.zeros([passageRes, res*2 + bladeRes*2 -3])
    Zh = np.zeros([passageRes, res*2 + bladeRes*2 -3])
    Xc = np.zeros([passageRes, res*2 + bladeRes*2 -3])
    Yc = np.zeros([passageRes, res*2 + bladeRes*2 -3])
    Zc = np.zeros([passageRes, res*2 + bladeRes*2 -3])
    Xi = np.zeros([passageRes,newNsection])
    Yi = np.zeros([passageRes,newNsection])
    Zi = np.zeros([passageRes,newNsection])
    Xo = np.zeros([passageRes,newNsection])
    Yo = np.zeros([passageRes,newNsection])
    Zo = np.zeros([passageRes,newNsection])
    for p in range(newNsection):
        for q in range(res*2 + bladeRes*2 -3):
            Xp[p,q] = lowThetaCart[p][q,0]
            Yp[p,q] = lowThetaCart[p][q,1]
            Zp[p,q] = lowThetaCart[p][q,2]
            Xs[p,q] = highThetaCart[p][q,0]
            Ys[p,q] = highThetaCart[p][q,1]
            Zs[p,q] = highThetaCart[p][q,2]          
        for r in range(passageRes):
            Xle[p,r] = LECart[p][r,0]
            Yle[p,r] = LECart[p][r,1]
            Zle[p,r] = LECart[p][r,2]
            Xte[p,r] = TECart[p][r,0]
            Yte[p,r] = TECart[p][r,1]
            Zte[p,r] = TECart[p][r,2]   
            Xmid[p,r] = midCart[p][r,0]
            Ymid[p,r] = midCart[p][r,1]
            Zmid[p,r] = midCart[p][r,2]
    for s in range(passageRes):
        for t in range(res*2 + bladeRes*2 -3):
            Xh[s,t] = hubCart[s][t,0]
            Yh[s,t] = hubCart[s][t,1]
            Zh[s,t] = hubCart[s][t,2]
            Xc[s,t] = casCart[s][t,0]
            Yc[s,t] = casCart[s][t,1]
            Zc[s,t] = casCart[s][t,2]
            if t < newNsection:
                Xi[s,t] = inCart[s][t,0]
                Yi[s,t] = inCart[s][t,1]
                Zi[s,t] = inCart[s][t,2]
                Xo[s,t] = outCart[s][t,0]
                Yo[s,t] = outCart[s][t,1]
                Zo[s,t] = outCart[s][t,2]

    # JD: The rest of this function has been updated to reflect offset surface inclusion.

    #  [offset low theta upstream, offset high theta upstream , blade low theta upstream, blade high theta upstream, ... same sequence for downstream ... LE, TE, midchord, upstream side low theta, upstream side high theta, downstream side low theta, downstream side high theta, hub, casing, inlet, outlet]
    Xvalues = [Xb1L, Xo1L, Xb1H, Xo1H, Xb2L, Xo2L, Xb2H, Xo2H, Xle, Xte, Xmid, Xh, Xc, Xi, Xo, XinL, XinH, XoutL, XoutH]
    Yvalues = [Yb1L, Yo1L, Yb1H, Yo1H, Yb2L, Yo2L, Yb2H, Yo2H, Yle, Yte, Ymid, Yh, Yc, Yi, Yo, YinL, YinH, YoutL, YoutH]
    Zvalues = [Zb1L, Zo1L, Zb1H, Zo1H, Zb2L, Zo2L, Zb2H, Zo2H, Zle, Zte, Zmid, Zh, Zc, Zi, Zo, ZinL, ZinH, ZoutL, ZoutH]

    print('Completed processing for this passage.')

    return Xvalues, Yvalues, Zvalues, filePath, LECurveRot, TECurveRot, midCurveRot, blade1SSCyl2D, blade2PSCyl2D, tangentLine1LERot, tangentLine1TERot, tangentLine2LERot, tangentLine2TERot, upstreamMprime, blade1SSMprime, blade2PSMprime, dwstreamMprime, inletHubCart, inletCasCart, outletHubCart, outletCasCart


#%% Defining the STLs
def createSTLs(Xvalues, Yvalues, Zvalues, filePath):
    """ Create and write an STL file based on input Cartesian coordinates """
    filenames = ['bladeLowTheta1','offsetLowTheta1','bladeHighTheta1','offsetHighTheta1','bladeLowTheta2','offsetLowTheta2','bladeHighTheta2','offsetHighTheta2','LE','TE','midChord', 'hub', 'casing', 'inlet', 'outlet', 'inletLow', 'inletHigh', 'outletLow', 'outletHigh']

    for qq in range(len(Xvalues)):
        print('Creating STL for suface: {}'.format(filenames[qq]))
        filename = filePath + '/{}.stl'.format(filenames[qq])
        rows = Zvalues[qq].shape[0]  ## Use Z because it is the axis of rotation
        columns = Zvalues[qq].shape[1]
        X = Xvalues[qq]
        Y = Yvalues[qq]
        Z = Zvalues[qq]
        
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


#%% Tony Woo Function definitions, updated Oct 2025 by Jeff Defoe
#  (removed ones not needed)

def format_coord(name, arr):
    return f"{name} ({arr[0]} {arr[1]} {arr[2]}) ;\n"


#%% Write out input file 
def calcAndWritePassageParameters(scale, Xvalues, Yvalues, Zvalues, nrad, delHub, delCas, delBla, dy1Hub, dy1Cas, dy1Bla, gRad, gTan, dax1primeLE, rLE, dax1primeTE, rTE, rUpFar, rDnFar, dataPath, additionalTangentialRefine, additionalAxialRefine):
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

    # NOTE hack for now: read in files Adekola's code produces
    BtoOarcLengthMap2HP = np.loadtxt('highHub1.txt', delimiter=',')
    BtoOarcLengthMap2HN = np.loadtxt('lowHub1.txt', delimiter=',')
    BtoOarcLengthMap2CP = np.loadtxt('highCas1.txt', delimiter=',')
    BtoOarcLengthMap2CN = np.loadtxt('lowCas1.txt', delimiter=',')

    BtoOarcLengthMap3HP = np.loadtxt('highHub2.txt', delimiter=',')
    BtoOarcLengthMap3HN = np.loadtxt('lowHub2.txt', delimiter=',')
    BtoOarcLengthMap3CP = np.loadtxt('highCas2.txt', delimiter=',')
    BtoOarcLengthMap3CN = np.loadtxt('lowCas2.txt', delimiter=',')

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
    scriptFile = os.path.join(dataPath, 'passageParameters')
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

# L4avg, cavg, 0.5*(dxAtBlade4+dtOuter), rDnFar, dxMiddle, gTE, nax3, fTE
# root = fsolve(outerAxialGradingFunction, x0=[0.6*(0.01/(dx1/C)), 1.2*rfar, 0.8*gclo, nin*L/C, 2*dxMid, dxMid+dx1], args=(dx1, dxMid, rfar, C, L, gclo))
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


def main() -> int:
    """ All the main blocks of the code get executed here """

    """ INPUTS: """
    # Geometry definition inputs:
    scale = 0.001  # relationship of input data to metres.
        # For example, if input data in mm, scale = 0.001
    Nb = 31  # number of blades in row
    nSections = 23  # number of airfoil sections used to define blades
    periodic = 1  # periodic (1) or aperiodic (0) blade row
    dataPath = './ECL5_original/periodic'  # location of input files
    N = 101  # Number of points on each side of each section (PS/SS)
        # (JD: both the nSections and N really should be automatically determined)

    # Grid generation inputs
    nrad = 40  # Number of radial points outside endwall BLs    
    # optional BL definition parameters
    rhoref = 1.2  # base SI units (kg/m**3)
    Uref = 100.0  # base SI units (m/s)
    LrefHub = 400.0  # input length units
    LrefCas = 400.0  # input length units
    LrefBla = 75.0  # input length units (JD: this should be calculated = mean chord)
    muref = 1.8e-5  # base SI units (kg/(m*s))
    yPlusHub = 2
    yPlusCas = 2
    yPlusBla = 2
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
    res = 30  # Upstream and downstream extention resolution
    passageRes = 359  # Resolution (JD: what direction?) of points for a passage
    bladeRes = 500  # Set resolution of underlying blade data

    if periodic == 1:
        passages = 1
        print('Operating in periodic blade row mode.')
    elif periodic == 0:
        passages = Nb
        print('Operating in aperiodic blade row mode.')

    for a in range(passages):
        print('Doing main operations for passage {}'.format(a))
        # 1. Do data processing to compute the arrays used to create STLs
        Xvalues, Yvalues, Zvalues, filePath, LECurveRot, TECurveRot, midCurveRot, blade1SSCyl2D, blade2PSCyl2D, tangentLine1LERot, tangentLine1TERot, tangentLine2LERot, tangentLine2TERot, upstreamMprime, blade1SSMprime, blade2PSMprime, dwstreamMprime, inletHubCart, inletCasCart, outletHubCart, outletCasCart = processData(passages, dataPath, nSections, N, res, passageRes, bladeRes, periodic)
        # 2. Create STLs
        print('Writing STL files for passage {}'.format(a))
        #createSTLs(Xvalues, Yvalues, Zvalues, filePath)
        # 3. Do calculations and write out parameters for grid generation
        print('Computing and writing parameters for passage {}'.format(a))
        calcAndWritePassageParameters(scale, Xvalues, Yvalues, Zvalues, nrad, delHub, delCas, delBla, dy1Hub, dy1Cas, dy1Bla, gRad, gTan, dax1primeLE, rLE, dax1primeTE, rTE, rUpFar, rDnFar, dataPath, additionalTangentialRefine, additionalAxialRefine)
        
    return 0


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


# This goes at the very end of the file:
if __name__ == "__main__":
    sys.exit(main())  # execute the main function when script is run directly
