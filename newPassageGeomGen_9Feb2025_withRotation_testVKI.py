#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Updated 17 Janjary 2025 to correctly implement curvy LE surface
Remaining issue on LE end: extensions aren't going along the
meridional sections, nor are their lengths correct.


Update 19 Nov 2024: add curvy LE surface

Update early Oct 2024: change LE, midchord, and TE surface
definitions to be based around the further-forward/backward
axial points, not some kind of "smart" approximation of the
camber line.

@author: Adekola Adeyemi and Jeff Defoe, updates by Justin Smart and Mohamad Hamad
"""

# Import packages
import numpy as np
import scipy.interpolate as spinterp
from scipy.interpolate import interp1d, CubicSpline
import model_function as mf  # custom library of functions called
import TransfiniteInterpolation as tf  # function for transfinite interpolation in here
import sys
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import scipy.optimize as opt

plt.close()

#pathToSectionFiles = os.path.join('VKIturbine', 'bladeSectionsTEfixed')  # relative path, or can be absolute
pathToSectionFiles = os.path.join('VKIturbine', 'bladeSections')  # relative path, or can be absolute
pathToMeridGeomFiles = os.path.join('VKIturbine', 'gasPath')  # relative path, or can be absolute
nSections = len([f for f in os.listdir(pathToSectionFiles) if f.startswith('EX_OGV_SuctionSide_') and f.endswith('.dat')]) #automate nsection files to look through folder to find how much files are present, rather then having the value hardcoded
dataScale = 0.001  # scale relative to base SI units of input data, e.g. 0.001 for mm
pathForSTLs = os.path.join('constant', 'geometry')  # relative path, or can be absolute
scriptFile = 'system/passageGeometry'  # filename for script to modify blockMeshDict
Nb = 29  # number of blades in the blade row
# Resolution of extensions of camber line upstream and downsteam
# (generally don't need to change):
N = 30
# Number of points to discretize in circumferential direction
thetaPts = 359  # must yield smaller triangles than desired grid cells in CFD

# Import data from files
bladeP = []
bladeS = []
LE = []
TE = []

# Load blade sections from files (this replaces the old code by making the functionalty the same just much cleaner)
for a in range(nSections):
    #format the section index as a two-digit string
    filename_suffix = f'{a:02d}'  #This formats 'a' as a two-digit number with leading zeros
    #here load the pressure and suction side data
    fileP = np.loadtxt(f'{pathToSectionFiles}/EX_OGV_SuctionSide_{filename_suffix}.dat')
    fileS = np.loadtxt(f'{pathToSectionFiles}/EX_OGV_PressureSide_{filename_suffix}.dat')
    bladeP.append(fileP) #append fileP and fileS into the correct list
    bladeS.append(fileS)
    # Extract leading edge and trailing edge points
    LE.append([fileP[0, 2], np.sqrt(fileP[0, 0]**2 + fileP[0, 1]**2)])   # Leading edge (z, r)
    TE.append([fileP[-1, 2], np.sqrt(fileP[-1, 0]**2 + fileP[-1, 1]**2)]) # Trailing edge (z, r)

# Convert LE and TE to NumPy arrays
LE = np.array(LE)
TE = np.array(TE)

# Testing
bladePzrq = np.zeros((nSections, 2, bladeP[0].shape[0]))
bladeSzrq = np.zeros((nSections, 2, bladeS[0].shape[0]))

#for a in range(nSections):
#    # Combine PS and SS
#    sectionPts = np.concatenate((bladeP[a], bladeS[a]))
#    # Find LE (minimum z)
#    LEindex = sectionPts[:, 2].argmin()
#    # Overwrite LE for this section
#    LE[a] = [sectionPts[LEindex, 2], np.sqrt(sectionPts[LEindex, 0]**2 + sectionPts[LEindex, 1]**2) ]
#    # Find TE (maximum z)
#    TEindex = sectionPts[:,2].argmax()
#    # Overwrite TE for this section
#    TE[a] = [sectionPts[TEindex, 2], np.sqrt(sectionPts[TEindex, 0]**2 + sectionPts[TEindex, 1]**2) ]

    # Now we have (z,r) coordinates for the LE and TE that are correct.
    # Next, need to re-divide the PS and SS based on these points.
    # Is the LE on the SS or PS (in current data)? Fix data definitions.
#    if np.equal(bladeS[a], sectionPts[LEindex, :]).prod(axis=1).max():
#        LEloc = np.where(np.equal(bladeS[a], sectionPts[LEindex, :]).prod(axis=1)==1)
#        bladeSnewThisSection = bladeS[a][int(LEloc[0]):, :]
#        SpointsToMovetoPS = bladeS[a][0:int(LEloc[0])+1:, :]  # includes LE again
#        bladePnewThisSection = np.concatenate((SpointsToMovetoPS[::-1, :], bladeP[a]))
#    elif np.equal(bladeP[a], sectionPts[LEindex, :]).prod(axis=1).max():
#        LEloc = np.where(np.equal(bladeP[a], sectionPts[LEindex, :]).prod(axis=1)==1)
#        bladePnewThisSection = bladeP[a][int(LEloc[0]):, :]  # remove points
#        PpointsToMovetoSS = bladeP[a][0:int(LEloc[0])+1:, :]  # includes LE again
#        bladeSnewThisSection = np.concatenate((PpointsToMovetoSS[::-1, :], bladeS[a]))
    # overwrite data, this works since bladeP/S are lists and each
    # element can be a differently-sized array
#    bladeP[a] = bladePnewThisSection
#    bladeS[a] = bladeSnewThisSection
#    # Is the TE on the SS or PS (in current data)? Fix data definitions.
#    if np.equal(bladeS[a], sectionPts[TEindex, :]).prod(axis=1).max():
#        TEloc = np.where(np.equal(bladeS[a], sectionPts[TEindex, :]).prod(axis=1)==1)
#        bladeSnewThisSection = bladeS[a][::int(TEloc[0]), :]
#        SpointsToMovetoPS = bladeS[a][int(TEloc[0])+1::, :]  # includes TE again
#        bladePnewThisSection = np.concatenate((SpointsToMovetoPS[::-1, :], bladeP[a]))
#    elif np.equal(bladeP[a], sectionPts[TEindex, :]).prod(axis=1).max():
#        TEloc = np.where(np.equal(bladeP[a], sectionPts[TEindex, :]).prod(axis=1)==1)
#        bladePnewThisSection = bladeP[a][::int(TEloc[0]), :]  # remove points
#        PpointsToMovetoSS = bladeP[a][int(TEloc[0])+1:, :]  # includes LE again
#        bladeSnewThisSection = np.concatenate((PpointsToMovetoSS[::-1, :], bladeS[a]))
#    # overwrite data, this works since bladeP/S are lists and each
#    # element can be a differently-sized array
#    bladeP[a] = bladePnewThisSection
#    bladeS[a] = bladeSnewThisSection


# hub and casing curves (extents: desired length of domain):
hub = np.loadtxt(pathToMeridGeomFiles + '/' + 'sHub.curve')  # these are (r, theta, z)
cas = np.loadtxt(pathToMeridGeomFiles + '/' + 'sCasing.curve')  # these are (r, theta, z)

nPtsCamber = 100  # number of points to discretize camber line

allCamber = np.zeros([nSections, nPtsCamber + 2*N, 3])  # holds full camber surface, for all profiles, from inlet to outlet (Cartesian)
#allCamber = np.zeros([nSections, 2*(N+1), 3])  # holds full camber surface, for all profiles, from inlet to outlet (Cartesian)
justCamber = np.zeros([nSections, nPtsCamber, 3])  # holds camber surface from LE to TE for all profiles (Cartesian)
OUTxyz = np.zeros([nSections, 3])  # outlet curve in Cartesian coordinates
INxyz = np.zeros([nSections, 3])  # inlet curve in Cartesian coordinates
#camberTheta = np.zeros([nSections, nPtsCamber + 2*N, 1])  # theta coordinates of full camber surface (rad)
camberTheta = np.zeros([nSections, nPtsCamber + 2*N, 1])  # theta coordinates of full camber surface (rad)

# nTotal = nPtsCamber + 2*N
# camberTheta = np.zeros((nSections, nTotal, 1))
# allCamber   = np.zeros((nSections, nTotal, 3)) 


# Get intersecting points between inlet, outlet, hub, and casing
# (meridonal coordinates (y, r))
casIN = [cas[0][2], cas[0][0]]
casOUT = [cas[::-1][0][2], cas[::-1][0][0]]
hubIN = [hub[0][2], hub[0][0]]
hubOUT = [hub[::-1][0][2],hub[::-1][0][0]]

upLine = np.vstack((hubIN, casIN))  # combines inlet data
dwLine = np.vstack((hubOUT, casOUT))  # combines outlet data

# Create intermediate points along inlet and outlet between hub and casing
nIN = np.zeros([len(LE), 2])
nIN[:, 1] = mf.scale(max(LE[:, 1]), min(LE[:, 1]), casIN[1], hubIN[1], LE[:, 1])
f = interp1d(upLine[:, 1], upLine[:, 0])
nIN[:, 0] = f(nIN[:, 1])  # meridional (z,r) coordinates of inlet curve
nIN = np.flip(nIN,0)  # flip order so points are from hub to casing

nOUT = np.zeros([len(TE), 2])
nOUT[:, 1] = mf.scale(min(TE[:, 1]), max(TE[:, 1]), hubOUT[1], casOUT[1], TE[:, 1])
f = interp1d(dwLine[:, 1], dwLine[:, 0])
nOUT[:, 0] = f(nOUT[:, 1])  # meridional (z,r) coordinates of outlet curve
nOUT = np.flip(nOUT,0)  # flip order so points are from hub to casing

# Before we figure out the 3D profiles of the interior points of the camber
# line extensions, we need to use transfinite interpolation to get the
# interior points on the meridional plane.
#
# For now, use linear interpolation to create a set of points on the hub and
# casing that have the same number of points. Later, improve this using cubic
# spline-based interpolation.
#
# We also need to do this between the LE and TE for a spline that will
# connect the up- and down-stream extensions. Number of points = nPtsCamber
#
# We have the parameter N which is how many points we want along these
# extensions. Use this to uniformly interpolate between the LE z and
# the IN z, and between the TE z and the OUT z. Map onto the hub/casing
# curves to get the meridional points. Then we'll have everything we need
# to carry out the transfinite interpolation.

# Hub
i = 0  # section 0 is the hub
# Start with the upstream extension
hubNUp = np.zeros([N+1, 2])  # (z,r)
hubNUp[:, 0] = np.linspace(start=nIN[i, 0], stop=LE[i, 0], num=N+1)
hubNUp[:, 1] = np.interp(hubNUp[:, 0], hub[:, 2], hub[:, 0])
#hubNUp[N, 1] = LE[i, 1]

# Then downstream extension
hubNDn = np.zeros([N+1, 2])  # (z,r)
hubNDn[:, 0] = np.linspace(start=TE[i, 0], stop=nOUT[i, 0], num=N+1)
hubNDn[:, 1] = np.interp(hubNDn[:, 0], hub[:, 2], hub[:, 0])
#hubNDn[0, 1] = TE[i, 1]

# Casing
i = nSections-1  # last section is the casing
# Start with the upstream extension
casNUp = np.zeros([N+1, 2])  # (z,r)
casNUp[:, 0] = np.linspace(start=nIN[i, 0], stop=LE[i, 0], num=N+1)
casNUp[:, 1] = np.interp(casNUp[:, 0], cas[:, 2], cas[:, 0])
#casNUp[N, 1] = LE[i, 1]

# Then downstream extension
casNDn = np.zeros([N+1, 2])  # (z,r)
casNDn[:, 0] = np.linspace(start=TE[i, 0], stop=nOUT[i, 0], num=N+1)
casNDn[:, 1] = np.interp(casNDn[:, 0], cas[:, 2], cas[:, 0])
#casNDn[0, 1] = TE[i, 1]

# Create arrays needed for LE-to-TE dataset
i = 0
hubNIn = np.zeros([nPtsCamber, 2])  # (z,r)
hubNIn[:, 0] = np.linspace(start=LE[i, 0], stop=TE[i, 0], num=nPtsCamber)
hubNIn[:, 1] = np.interp(hubNIn[:, 0], hub[:, 2], hub[:, 0])
#hubNIn[nPtsCamber-1, 1] = TE[i, 1]

i = nSections-1
casNIn = np.zeros([nPtsCamber, 2])  # (z,r)
casNIn[:, 0] = np.linspace(start=LE[i, 0], stop=TE[i, 0], num=nPtsCamber)
casNIn[:, 1] = np.interp(casNIn[:, 0], cas[:, 2], cas[:, 0])
#casNIn[nPtsCamber-1, 1] = TE[i, 1]

# Do transfinite interpolation to get (z,r) coordinates of interior points
nodesUp = tf.transfinite(hubNUp, casNUp, nIN, LE)
nodesDn = tf.transfinite(hubNDn, casNDn, TE, nOUT)
nodesIn = tf.transfinite(hubNIn, casNIn, LE, TE)

LExyz = np.zeros((nSections,3))
TExyz = np.zeros((nSections,3))

# Create half-pitch variable for later rotating "midline" to get periodic boundary locations
halfPitch = (360/Nb) * 0.5
halfPitch_radians = np.deg2rad(halfPitch)

thetaPtsForLETEmidSurfs = int(np.round(thetaPts/2))  # Calculate the number of points for the leading and trailing edge mid-surfaces

# Initialize arrays for curvy LE, TE, (and later: midchord)
leadEdge = np.zeros([2*thetaPtsForLETEmidSurfs-1, len(LE), 3])
trailEdge = np.zeros([2*thetaPtsForLETEmidSurfs-1, len(TE), 3])

def objective_LEoffset(x):
    """
    Given an LE offset 'x', build the two splines and return
    the larger (worst) maximum curvature between them.
    """

    # 1) Build / update LEqp and LEqm arrays using offset x
    LErqoffset = x * LE_INCurve_Slope
    LErp = np.interp(LEzrq[0] + x, zNew, rNew)
    LErm = np.interp(LEzrq[0] + x, zNew, rNew)

    # Re-define LEqp, LEqm based on offset x
    LEqp = np.array([
        LEzrq,
        LEzrq + (x, LErqoffset + np.deg2rad(halfPitch)*LErp)
    ])
    LEqm = np.array([
        LEzrq + (x, LErqoffset - np.deg2rad(halfPitch)*LErm),
        LEzrq
    ])
    #print(LEqp)
    #print(LEqm)

    # 1.5) Identify a local coordinate system where the splines
    # are guaranteed to be well-defined

    # Angles of slopes (one might not need the + pi...)
    betaLE_PS = np.arctan2(bisector_vector_PS[1],bisector_vector_PS[0]) + np.pi
    betaLE_SS = np.arctan2(bisector_vector_SS[1],bisector_vector_SS[0]) + np.pi
    # Rotation matrix angles
    phiLE_PS = 0.25*np.pi + 0.5*betaLE_PS
    phiLE_SS = 0.25*np.pi + 0.5*betaLE_SS
    # Rotation matrices
    rotLE_PS = np.array([[np.cos(phiLE_PS), np.sin(phiLE_PS)], [-np.sin(phiLE_PS), np.cos(phiLE_PS)]])
    rotLE_SS = np.array([[np.cos(phiLE_SS), np.sin(phiLE_SS)], [-np.sin(phiLE_SS), np.cos(phiLE_SS)]])
    # Endpoints in rotated coordinates
    XLEqp = LEqp  #np.matmul(rotLE_PS, LEqp.T).T
    XLEqm = LEqm  #np.matmul(rotLE_SS, LEqm.T).T
    #print(XLEqp)
    #print(XLEqm)
    # Check for increasing sequence
    XLEqp1 = XLEqp[XLEqp[:,1].argsort()]
    #XLEqp1 = np.sort(XLEqp, axis=0)
    XLEqm1 = XLEqm[XLEqm[:,1].argsort()]
    #XLEqm1 = np.sort(XLEqm, axis=0)
    # check if anything happened; if so,  change order of BC slopes
    if np.array_equal(XLEqp, XLEqp1):
        bcpt = ((1, 1/bisect_slope_PS), (1, 0))
    else:
        bcpt = ((1, 0), (1, 1/bisect_slope_PS))
    if np.array_equal(XLEqm, XLEqm1):
        bcmt = ((1, 0), (1, 1/bisect_slope_SS))
    else:
        bcmt = ((1, 1/bisect_slope_SS), (1, 0))
    #bcpt = ((1, 1/bisect_slope_PS), (1, 0))
    #bcmt = ((1, 0), (1, 1/bisect_slope_SS))
    #print(XLEqp1)
    #print(XLEqm1)

    # 2) Build the two splines
#    LEqplusSpline = spinterp.CubicSpline(
#        XLEqp[:,1], XLEqp[:,0],
#        bc_type=((1, np.tan(betaLE_PS + phiLE_PS)), (1, np.tan(np.pi/2 + phiLE_PS)))
#    )
#    LEqminusSpline = spinterp.CubicSpline(
#        XLEqm[:,1], XLEqm[:,0],
#        bc_type=((1, np.tan(betaLE_SS + phiLE_SS)), (1, np.tan(np.pi/2 + phiLE_SS)))
#    )
    LEqplusSpline = spinterp.CubicSpline(
        XLEqp1[:,1], XLEqp1[:,0],
        bc_type=bcpt
    )
    LEqminusSpline = spinterp.CubicSpline(
        XLEqm1[:,1], XLEqm1[:,0],
        bc_type=bcmt
    )

    # 3) Find the maximum curvature of each spline
    #    e.g. by sampling or your max_curvature_point function:
    x_plus = np.linspace(LEqp[0, 1], LEqp[1, 1], 1000)
    x_minus = np.linspace(LEqm[0, 1], LEqm[1, 1], 1000)

    # The s-values where curvature is max:
    x_plus_star  = max_curvature_point(LEqplusSpline, x_plus)
    x_minus_star = max_curvature_point(LEqminusSpline, x_minus)

    # The actual curvature values at those points:
    maxCurv_plus  = curvature(LEqplusSpline,  x_plus_star)
    maxCurv_minus = curvature(LEqminusSpline, x_minus_star)

    # 4) The objective is the larger of the two max curvatures
    return max(maxCurv_plus, maxCurv_minus)

# objective TE offset
def objective_TEoffset(x):
    """
    Given a TE offset 'x', build the two splines and return
    the larger (worst) maximum curvature between them.
    """
    # 1) Build / update TEqp and TEqm arrays using offset x
    TErqoffset = x * TE_OUTCurve_Slope
    TErp = np.interp(TEzrq[0] + x, vNew, wNew)
    TErm = np.interp(TEzrq[0] + x, vNew, wNew)

    # Re-define TEqp, TEqm based on offset x
    TEqp = np.array([
        TEzrq,
        TEzrq + (x, TErqoffset + np.deg2rad(halfPitch)*TErp)
    ])
    TEqm = np.array([
        TEzrq + (x, TErqoffset - np.deg2rad(halfPitch)*TErm),
        TEzrq
    ])

    # Extra robustness stuff (see explanation in LE function)
    # Endpoints in rotated coordinates
    XTEqp = TEqp
    XTEqm = TEqm

    XTEqp1 = XTEqp[XTEqp[:,1].argsort()]
    XTEqm1 = XTEqm[XTEqm[:,1].argsort()]
    # check if anything happened; if so,  change order of BC slopes
    if np.array_equal(XTEqp, XTEqp1):
        bcptTE = ((1, 1/bisect_slope_SS_downstream), (1, TEperiodicSlope))
    else:
        bcptTE = ((1, TEperiodicSlope), (1, 1/bisect_slope_SS_downstream))
    if np.array_equal(XTEqm, XTEqm1):
        bcmtTE = ((1, TEperiodicSlope), (1, 1/bisect_slope_PS_downstream))
    else:
        bcmtTE = ((1, 1/bisect_slope_PS_downstream), (1, TEperiodicSlope))

    # 2) Build the two splines
    TEqplusSpline = spinterp.CubicSpline(
        XTEqp1[:,1], XTEqp1[:,0],
        bc_type=bcptTE
    )
    TEqminusSpline = spinterp.CubicSpline(
        XTEqm1[:,1], XTEqm1[:,0],
        bc_type=bcmtTE
    )

    # 3) Find the maximum curvature of each spline
    #    e.g. by sampling or your max_curvature_point function:
    x_plus = np.linspace(TEqp[0, 1], TEqp[1, 1], 1000)
    x_minus = np.linspace(TEqm[0, 1], TEqm[1, 1], 1000)

    # The s-values where curvature is max:
    x_plus_star  = max_curvature_point(TEqplusSpline, x_plus)
    x_minus_star = max_curvature_point(TEqminusSpline, x_minus)

    # The actual curvature values at those points:
    maxCurv_plus  = curvature(TEqplusSpline,  x_plus_star)
    maxCurv_minus = curvature(TEqminusSpline, x_minus_star)

    # 4) The objective is the larger of the two max curvatures
    return max(maxCurv_plus, maxCurv_minus)

# define function which determines curvature (kappa) of any given cubic spline using the formula: kappa = |f''(x)| / (1 + f'(x)^2)^(3/2)
def curvature(spline, x):
    # Calculate the first derivative of the spline
    first_derivative = spline.derivative(1)
    # Calculate the second derivative of the spline
    second_derivative = spline.derivative(2)
    # Calculate the first derivative of the spline at x
    first_derivative_at_x = first_derivative(x)
    # Calculate the second derivative of the spline at x
    second_derivative_at_x = second_derivative(x)
    # Calculate the curvature of the spline at x
    curvature = np.abs(second_derivative_at_x) / (1 + first_derivative_at_x**2)**(3/2)
    return curvature

# define a function which finds the point on a spline where the curvature is at a maximum
def max_curvature_point(spline, x_values):
    # Initialize variables to store the maximum curvature and the x value at which it occurs
    max_curvature = 0
    max_curvature_x = 0
    # Loop through the x values
    for x in x_values:
        # Calculate the curvature at the current x value
        current_curvature = curvature(spline, x)
        # Check if the current curvature is greater than the maximum curvature
        if current_curvature > max_curvature:
            # Update the maximum curvature and the x value at which it occurs
            max_curvature = current_curvature
            max_curvature_x = x
    return max_curvature_x

# (Functions below written by Justin Smart)
# Rotate line function
def rotate_line(points, pivot, angle_deg):
    """
    Rotates a set of points around a pivot point by a specified angle in degrees.

    Parameters:
    - points (np.ndarray): An Nx2 array of (x, y) coordinates representing the line to be rotated.
    - pivot (array-like): A 2-element array-like object representing the (x, y) coordinates of the pivot point.
    - angle_deg (float): The angle in degrees by which to rotate the line.

    Returns:
    - rotated_points (np.ndarray): An Nx2 array of the rotated (x, y) coordinates.
    """
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle_deg)
    # Create rotation matrix
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_angle, -sin_angle],[sin_angle,  cos_angle]])
    
    # Translate points to pivot
    translated_points = points - pivot
    
    # Rotate points
    rotated_translated_points = np.dot(translated_points, rotation_matrix.T)
    
    # Translate points back
    rotated_points = rotated_translated_points + pivot
    
    return rotated_points


# Define functions for finding intersections
def func_PS(z):
    return PS_interp(z) - (midChordR + (z - midChordZ) * perpChordSlope)


def func_SS(z):
    return SS_interp(z) - (midChordR + (z - midChordZ) * perpChordSlope)


def find_zero_crossings(z_values, func_values):
    zero_crossings = []
    for i in range(len(z_values) - 1):
        if func_values[i] * func_values[i + 1] < 0:
            zero_crossings.append((z_values[i], z_values[i + 1]))
    return zero_crossings

# Loop over each blade section to compute mid-passage curve and
# leading/trailing edge curves across the passage
for i in range(nSections):

    # Convert blade data to Cylindrical coordinates
    PScyl = mf.cart2pol(bladeP[i][:, 0], bladeP[i][:, 1], bladeP[i][:, 2])
    SScyl = mf.cart2pol(bladeS[i][:, 0], bladeS[i][:, 1], bladeS[i][:, 2])

    # Cartesian coordintes of LE and TE on this section
    LExyz[i, :] = bladeP[i][0, :]
    TExyz[i, :] = bladeP[i][-1, :]

    # Convert LE and TE data to cylindrical coordinates (theta, r, z)
    LEcyl = mf.cart2pol(bladeP[i][0, 0], bladeP[i][0, 1], bladeP[i][0, 2])
    TEcyl = mf.cart2pol(bladeP[i][-1, 0], bladeP[i][-1, 1], bladeP[i][-1, 2])

    # Convert to (z,r*theta)
    LEzrq = np.array([LEcyl[2],LEcyl[0]*LEcyl[1]])
    TEzrq = np.array([TEcyl[2],TEcyl[0]*TEcyl[1]])

    # Don't do this here any more -- wait until LE and TE extended curves are
    # defined and then join with a circular arc segment

#    # Make points between in (z,r*theta) coordinates
#    camberInBladePtsz = np.linspace(start=LEzrq[0], stop=TEzrq[0], num=nPtsCamber)
#    # To get the r*theta values, need to interpolate SS and PS data and then
#    # take average:
    PSzrq = np.array([PScyl[2],PScyl[0]*PScyl[1]])
    SSzrq = np.array([SScyl[2],SScyl[0]*SScyl[1]])

    bladePzrq[i,:,:] = PSzrq
    bladeSzrq[i,:,:] = SSzrq
    
    #    
#    camberInBladePtsrq = 0.5*( np.interp(camberInBladePtsz,PSzrq[0,:],PSzrq[1,:]) + np.interp(camberInBladePtsz,SSzrq[0,:],SSzrq[1,:]) )

    # Map onto meridional plane -- use transfinite interpolation
    # to fill in the path each section should take based on the hub and
    # casing curves and the LE and TE (similar to what's being done
    # already for the inlet and outlet extensions); then use the r coordinate
    # at each z to convert (z,r*theta) data to cylindrical (to get theta vals)

    # For now, instead, just assume a straight line
    LEr = LEcyl[1]
    TEr = TEcyl[1]
#    camberInBladePtsr = np.linspace(start=LEr, stop=TEr, num=nPtsCamber)
#    camberLinecyl = np.array([camberInBladePtsrq / camberInBladePtsr, camberInBladePtsr , camberInBladePtsz]).T

    # convert data to (z,r*theta)
    PSzrq = np.zeros([len(bladeP[i]), 2])
    PSzrq[:, 1] = PScyl[0]*(PScyl[1])
    PSzrq[:, 0] = PScyl[2]
    SSzrq = np.zeros([len(bladeS[i]), 2])
    SSzrq[:, 1] = SScyl[0]*SScyl[1]
    SSzrq[:, 0] = SScyl[2]
#    camberLinezrq = np.array([camberInBladePtsz, camberInBladePtsrq]).T

    # Take everything back to 3D (Cylindrical)

    # inlet
    zNew = nodesUp[i::nSections, 0]  # axial coordinates from transfinite interpolation
    rNew = nodesUp[i::nSections, 1]  # radial coordinates from transfinite interpolation

    # outlet
    vNew = nodesDn[i::nSections, 0]  # axial coordinates from transfinite interpolation
    wNew = nodesDn[i::nSections, 1]  # radial coordinates from transfinite interpolation

    # Compute chord line slope
    chordSlope = (TEzrq[1] - LEzrq[1]) / (TEzrq[0] - LEzrq[0])
    perpChordSlope = -1 / chordSlope

    # Compute mid-chord point
    midChordZ = (LEzrq[0] + TEzrq[0]) / 2
    midChordR = (LEzrq[1] + TEzrq[1]) / 2

    # Define the mid-blade line (make go past the blade surface just a bit)
    zLine = np.linspace(midChordZ - 10, midChordZ + 10, 101)
    rLine = midChordR + (zLine - midChordZ) * perpChordSlope

    # Plot the mid-blade line
    # plt.plot(zLine, rLine, '-g', label='Mid-blade Line')

    # Create interpolation functions for PS and SS
    PS_interp = interp1d(PSzrq[:, 0], PSzrq[:, 1], kind='linear', assume_sorted=False)
    SS_interp = interp1d(SSzrq[:, 0], SSzrq[:, 1], kind='linear', assume_sorted=False)
    # Find zero crossings for PS
    z_dense_PS = np.linspace(PSzrq[:, 0].min(), PSzrq[:, 0].max(), 1000)
    func_PS_values = func_PS(z_dense_PS)
    zero_crossings_PS = find_zero_crossings(z_dense_PS, func_PS_values)

    # Find intersection points for PS
    intersection_points_PS = []
    for interval in zero_crossings_PS:
        try:
            z_root = brentq(func_PS, interval[0], interval[1])
            r_root = PS_interp(z_root)
            intersection_points_PS.append((z_root, r_root))
        except ValueError:
            pass

    # Find zero crossings for SS
    z_dense_SS = np.linspace(SSzrq[:, 0].min(), SSzrq[:, 0].max(), 1000)
    func_SS_values = func_SS(z_dense_SS)
    zero_crossings_SS = find_zero_crossings(z_dense_SS, func_SS_values)

    # Find intersection points for SS
    intersection_points_SS = []
    for interval in zero_crossings_SS:
        try:
            z_root = brentq(func_SS, interval[0], interval[1])
            r_root = SS_interp(z_root)
            intersection_points_SS.append((z_root, r_root))
        except ValueError:
            pass

    # Create a line between intersection_pointsPS and LEzrq
    for point in intersection_points_PS:
        # plt.plot([point[0], LEzrq[0]], [point[1], LEzrq[1]], 'r')
        # Define the line
        PSslope = (point[1] - LEzrq[1]) / (point[0] - LEzrq[0])
        perpPSslope = -1 / PSslope
        PSextPt = [point[0] + 1, point[1] + perpPSslope]
        PSLine = mf.line(point, PSextPt)
        # Create a nx2 array containing x and y values for the line between the intersection point and LEzrq
        PSzLine = np.array([np.linspace(point[0], LEzrq[0], 100), np.linspace(point[1], LEzrq[1], 100)]).T 
    for point in intersection_points_SS:
        # plt.plot([point[0], LEzrq[0]], [point[1], LEzrq[1]], 'b')
        # Define the line
        SSslope = (point[1] - LEzrq[1]) / (point[0] - LEzrq[0])
        perpSSslope = -1 / SSslope
        SSextPt = [point[0] + 1, point[1] + perpSSslope]
        SSLine = mf.line(point, SSextPt)
        # Create an array for SSLine containing 100 points
        SSzLine = np.array([np.linspace(point[0], LEzrq[0], 100), np.linspace(point[1], LEzrq[1], 100)]).T

    # Extract pivot point (ensure it's a 2-element array)
    pivot_point = LEzrq[:2]  # Assuming LEzrq has at least two elements representing z and r

    # Create lines and rotate them
    # For Pressure Side Line
    for point in intersection_points_PS:
        # Plot the original line
        # plt.plot([point[0], LEzrq[0]], [point[1], LEzrq[1]], 'r', label='Original PS Line')
        
        # Create an Nx2 array containing x (z) and y (r) values for the line between the intersection point and LEzrq
        PSz_values = np.linspace(point[0], LEzrq[0], 100)
        PSr_values = np.linspace(point[1], LEzrq[1], 100)
        PSzLine = np.column_stack((PSz_values, PSr_values))
        
        # Rotate the line
        rotated_PSLine = rotate_line(PSzLine, pivot_point, 89)
        
        # Plot the rotated line
        # plt.plot(rotated_PSLine[:, 0], rotated_PSLine[:, 1], 'r', label='Rotated PS Line')

    # For Suction Side Line
    for point in intersection_points_SS:
        # Plot the original line
        # plt.plot([point[0], LEzrq[0]], [point[1], LEzrq[1]], 'b', label='Original SS Line')
        
        # Create an Nx2 array containing x (z) and y (r) values for the line between the intersection point and LEzrq
        SSz_values = np.linspace(point[0], LEzrq[0], 100)
        SSr_values = np.linspace(point[1], LEzrq[1], 100)
        SSzLine = np.column_stack((SSz_values, SSr_values))
        
        # Rotate the line
        rotated_SSLine = rotate_line(SSzLine, pivot_point, -89)
        
        # Plot the rotated line
        # plt.plot(rotated_SSLine[:, 0], rotated_SSLine[:, 1], 'b', label='Rotated SS Line')

    # For rotated_PSLine
    delta_z_rotated = rotated_PSLine[-1, 0] - rotated_PSLine[-2, 0]
    delta_rq_rotated = rotated_PSLine[-1, 1] - rotated_PSLine[-2, 1]
    direction_rotated_PSLine = np.array([delta_z_rotated, delta_rq_rotated])
    direction_rotated_PSLine = direction_rotated_PSLine / np.linalg.norm(direction_rotated_PSLine)

    # do the same for the SS
    delta_z_rotated = rotated_SSLine[-1, 0] - rotated_SSLine[-2, 0]
    delta_rq_rotated = rotated_SSLine[-1, 1] - rotated_SSLine[-2, 1]
    direction_rotated_SSLine = np.array([delta_z_rotated, delta_rq_rotated])
    direction_rotated_SSLine = direction_rotated_SSLine / np.linalg.norm(direction_rotated_SSLine)

    # Calculate the bisector vector between the two direction vectors
    bisect_direction = (direction_rotated_PSLine + direction_rotated_SSLine) / 2
    bisect_direction = bisect_direction / np.linalg.norm(bisect_direction)  # Normalize the bisector

    # Replace LE_INCurve_Slope with the slope of the bisector vector
    LE_INCurve_Slope = bisect_direction[1] / bisect_direction[0]  # Calculate the slope (dq/dz)

    # fine the TE_OUTCurve_Slope using the same method as for the LE_INCurve_Slope
    # Pivot at the trailing edge in (z, rθ):
    pivot_point_TE = TEzrq[:2]

    # find TEzLine_PS and TEzLine_SS be defining the line between the PS and SS midchord intersection and the trailing edge respectively
    TEzLine_PS = np.array([intersection_points_PS[-1], TEzrq[:2]])
    TEzLine_SS = np.array([intersection_points_SS[-1], TEzrq[:2]])

    # Suppose you have small lines near the TE on PS & SS:
    # e.g. TEzLine_PS and TEzLine_SS, each Nx2 array of (z, rθ).
    rotated_PSLine_TE = rotate_line(TEzLine_PS, pivot_point_TE, +89)
    rotated_SSLine_TE = rotate_line(TEzLine_SS, pivot_point_TE, -89)

    # For the rotated PS line near TE:
    delta_z_PS_TE = rotated_PSLine_TE[-1, 0] - rotated_PSLine_TE[-2, 0]
    delta_rq_PS_TE = rotated_PSLine_TE[-1, 1] - rotated_PSLine_TE[-2, 1]
    direction_rotated_PSLine_TE = np.array([delta_z_PS_TE, delta_rq_PS_TE])
    direction_rotated_PSLine_TE /= np.linalg.norm(direction_rotated_PSLine_TE)

    # For the rotated SS line near TE:
    delta_z_SS_TE = rotated_SSLine_TE[-1, 0] - rotated_SSLine_TE[-2, 0]
    delta_rq_SS_TE = rotated_SSLine_TE[-1, 1] - rotated_SSLine_TE[-2, 1]
    direction_rotated_SSLine_TE = np.array([delta_z_SS_TE, delta_rq_SS_TE])
    direction_rotated_SSLine_TE /= np.linalg.norm(direction_rotated_SSLine_TE)

    # Bisector vector:
    bisect_direction_TE = direction_rotated_PSLine_TE + direction_rotated_SSLine_TE
    bisect_direction_TE = bisect_direction_TE / np.linalg.norm(bisect_direction_TE)

    TE_OUTCurve_Slope = bisect_direction_TE[1] / bisect_direction_TE[0]  # dq/dz

    # Create a linear line (straight line) of the form rq = m*z + b
    b = LEzrq[1] - LE_INCurve_Slope * LEzrq[0]  # intercept

    # Compute rq values at the z points of the upstream extension
    rqNew = LE_INCurve_Slope*zNew + b

    # [JEFF DEFOE COMMENT: WHAT THE HECK DOES THIS DO?]
    index_inCurve = np.argmin(np.abs(zNew - LEzrq[0]))
    if index_inCurve > 0 and index_inCurve < len(zNew) - 1:
        delta_z_inCurve = zNew[index_inCurve + 1] - zNew[index_inCurve - 1]
        delta_rq_inCurve = rqNew[index_inCurve + 1] - rqNew[index_inCurve - 1]
    else:
        delta_z_inCurve = zNew[index_inCurve] - zNew[index_inCurve - 1]
        delta_rq_inCurve = rqNew[index_inCurve] - rqNew[index_inCurve - 1]
    
    direction_inCurve = np.array([delta_z_inCurve, delta_rq_inCurve])
    direction_inCurve = direction_inCurve / np.linalg.norm(direction_inCurve)

    # Compute bisector vector
    bisector_vector_PS = direction_inCurve + direction_rotated_PSLine
    bisector_vector_PS = bisector_vector_PS / np.linalg.norm(bisector_vector_PS)
    # Find the slope of the bisector line
    bisect_slope_PS = bisector_vector_PS[1] / bisector_vector_PS[0]

    # Compute bisector vector
    bisector_vector_SS = direction_inCurve + direction_rotated_SSLine
    bisector_vector_SS = bisector_vector_SS / np.linalg.norm(bisector_vector_SS)
    # Find the slope of the bisector line, slope should be a single value
    bisect_slope_SS = bisector_vector_SS[1] / bisector_vector_SS[0]

    # now compute the downstream extension
    # Create a linear line (straight line) of the form rq = m*z + b
    b_downstream = TEzrq[1] - TE_OUTCurve_Slope * TEzrq[0]  # intercept

    # Compute rq values at the z points of the downstream extension
    rqNew_downstream = TE_OUTCurve_Slope*vNew + b_downstream

    index_outCurve = np.argmin(np.abs(vNew - TEzrq[0]))
    if index_outCurve > 0 and index_outCurve < len(vNew) - 1:
        delta_z_outCurve = vNew[index_outCurve + 1] - vNew[index_outCurve - 1]
        delta_rq_outCurve = rqNew_downstream[index_outCurve + 1] - rqNew_downstream[index_outCurve - 1]
    else:
        delta_z_outCurve = vNew[index_outCurve] - vNew[index_outCurve - 1]
        delta_rq_outCurve = rqNew_downstream[index_outCurve] - rqNew_downstream[index_outCurve - 1]

    direction_outCurve = np.array([delta_z_outCurve, delta_rq_outCurve])
    direction_outCurve = direction_outCurve / np.linalg.norm(direction_outCurve)

    # Compute bisector vector
    bisector_vector_PS_downstream = direction_outCurve + direction_rotated_PSLine_TE
    bisector_vector_PS_downstream = bisector_vector_PS_downstream / np.linalg.norm(bisector_vector_PS_downstream)
    # Find the slope of the bisector line
    bisect_slope_PS_downstream = bisector_vector_PS_downstream[1] / bisector_vector_PS_downstream[0]

    # Compute bisector vector
    bisector_vector_SS_downstream = direction_outCurve + direction_rotated_SSLine_TE
    bisector_vector_SS_downstream = bisector_vector_SS_downstream / np.linalg.norm(bisector_vector_SS_downstream)
    # Find the slope of the bisector line
    bisect_slope_SS_downstream = bisector_vector_SS_downstream[1] / bisector_vector_SS_downstream[0]

    # make camber3DOUT
    camber3DOUT = np.array([rqNew_downstream/wNew, wNew, vNew]).T

    # cylindrical coordinates (theta, r, z) camber surfaces for extensions
    camber3DIN = np.array([rqNew/rNew, rNew, zNew]).T

    # create a cubic spline defined by the LE and TE points, as well as the LE_INCurve_Slope and TE_OUTCurve_Slope
    camber_spline = CubicSpline([LEzrq[0], TEzrq[0]], [LEzrq[1], TEzrq[1]], bc_type=((1, LE_INCurve_Slope), (1, TE_OUTCurve_Slope)))
    # this spline needs to be mapped onto some locations between LE and TE
    # Best to take the LE and TE points + the hub/casing points between,
    # and define a further transfinite interpolation -- this ought to be
    # done before this loop, like up- and down-stream ones.

    zIn = nodesIn[i::nSections, 0]  # axial coordinates from transfinite interpolation
    rIn = nodesIn[i::nSections, 1]  # radial coordinates from transfinite interpolation
    # JD question: why are these and other similar indexing operations into nodes* not of the form [i, 0]/[i, 1]?
    
    camberLinezrq = np.array([zIn, camber_spline(zIn)]).T
    camberLinecyl = np.array([camberLinezrq[:,1] / rIn, rIn, zIn]).T  # (theta, r, z)

    # combine all Camber (cylindrical coordinates)
    camber = np.concatenate((camber3DIN, camberLinecyl[1:len(camberLinecyl)-1], camber3DOUT)).reshape((-1, 3), order='F')

    # print camber
    #print(camber.shape)
    #print camberTheta
    #print(camberTheta.shape)

    # Return back to Cartesian Coordinate System
    camberTheta[i,:,:] = camber[:,0].reshape(-1,1)
    medCamber = mf.pol2cart(camber[:, 0], camber[:, 1], camber[:, 2])
    medCamber = np.array(medCamber).T
    allCamber[i, :, :] = medCamber

    # combine the blade also
    onlyCamber = mf.pol2cart(camberLinecyl[:, 0], camberLinecyl[:, 1], camberLinecyl[:, 2])
    onlyCamber = np.array(onlyCamber).T
    justCamber[i, :, :] = onlyCamber
    OUTxyz[i, :] = medCamber[::-1][0]
    INxyz[i, :] = medCamber[0]

    # make an array containing the 3D coordinates of the camber for just this section
    camberSection = allCamber[i,:,:]
    # rotate the camber section by +halfPitch and -halfPitch
    # pCamber_section = mf.vectorRot3D(camberSection[:,2], camberSection[:,1], camberSection[:,0], halfPitch_radians)
    # pCamber_section = np.array(pCamber_section)
    # pCamber_Section = (pCamber_section).T
    # nCamber_section = mf.vectorRot3D(camberSection[:,2], camberSection[:,1], camberSection[:,0], -halfPitch_radians)
    # nCamber_section = np.array(nCamber_section)
    # nCamber_Section = (nCamber_section).T
    # # convert the rotated camber sections to the z, r*theta coordinate system
    # pCamber_Section_zrtheta = np.array([pCamber_section[:,2], pCamber_section[:,0]*pCamber_section[:,1]]).T
    # nCamber_Section_zrtheta = np.array([nCamber_section[:,2], nCamber_section[:,0]*nCamber_section[:,1]]).T    

# -----------------------------------------> Check )

    perpLEslope = LE_INCurve_Slope

    # define the camber surface using a cubic spline, with the 2 defining points being the LE and TE point, as well as the LE and TE slope (LE_INCurve_Slope for the LE, find the TE slope from the TE point and the point just upstream of the TE point)
    

    # Calculate angles for leading edge bisector
    LEangle = np.arctan2( LE_INCurve_Slope,1)
    perpLEangle = np.arctan2(perpLEslope,1)  # [THIS IS GIVING AN UNDEFINED VARIABLE ERROR, NEED TO GO THROUGH AND FIGURE OUT WHAT THIS SHOULD BE].
    ggg = 0.5  # weighting average factor
    bisectLEangle = (ggg*LEangle + (1-ggg)*perpLEangle) - np.pi/2
    bisectDirLEqplus = np.tan(bisectLEangle)
    # instead of using the rotated LEzrq point, use the LE point of the pCamber_Section_zrtheta

    # Do the same for the other side of LE
    hhh = ggg  # weighting average factor
    bisectLEangle = (hhh*LEangle + (1-hhh)*perpLEangle) - np.pi/2
    bisectDirLEqminus = np.tan(bisectLEangle)

    # using the scipy optimization function, minimize_scalar, find the point on the LEqplusSpline where the curvature is at a maximum
    # Given the objective function should be J = max(max(curvature(LEqplusSpline)), max(curvature(LEqminusSpline))), using scipy's minimize_scalar function, minimize J for LEoffset >=0
    # This will give the optimal LEoffset for the given section and will be done for each section
    res = opt.minimize_scalar(objective_LEoffset,
                            bounds=(-110, -90),
                            method='bounded')
    # DEBUG NOTE: If I set the upper bound to - something, I get an error about non-increasing X in one of the splines.
    # If I hardcode the offset to -100 it gives a good result
    LEoffset = res.x
    best_score = res.fun
    print(f"Best LEoffset for section {i} is {LEoffset} with a score of {best_score}")
    LErqoffset = LEoffset * LE_INCurve_Slope
    perpLEslope = LE_INCurve_Slope
    # update the LEqp and LEqm arrays with the new LEoffset
    LErp = np.interp(LEzrq[0]+LEoffset, zNew, rNew)
    LErm = np.interp(LEzrq[0]+LEoffset, zNew, rNew)
    LEqp = np.array([LEzrq, LEzrq+(LEoffset,LErqoffset+np.deg2rad(halfPitch)*LErp)])
    LEqm = np.array([LEzrq+(LEoffset,LErqoffset-np.deg2rad(halfPitch)*LErm), LEzrq])

    XLEqp = LEqp
    XLEqm = LEqm
    #XLEqp = np.sort(LEqp, axis=0)
    #XLEqm = np.sort(LEqm, axis=0)

    XLEqp1 = XLEqp[XLEqp[:,1].argsort()]
    #XLEqp1 = np.sort(XLEqp, axis=0)
    XLEqm1 = XLEqm[XLEqm[:,1].argsort()]
    #XLEqm1 = np.sort(XLEqm, axis=0)
    # check if anything happened; if so,  change order of BC slopes
    if np.array_equal(XLEqp, XLEqp1):
        bcpt = ((1, 1/bisect_slope_PS), (1, 0))
    else:
        bcpt = ((1, 0), (1, 1/bisect_slope_PS))
    if np.array_equal(XLEqm, XLEqm1):
        bcmt = ((1, 0), (1, 1/bisect_slope_SS))
    else:
        bcmt = ((1, 1/bisect_slope_SS), (1, 0))

    LEqplusSpline = spinterp.CubicSpline(
        XLEqp1[:,1], XLEqp1[:,0],
        bc_type=bcpt
    )
    LEqminusSpline = spinterp.CubicSpline(
        XLEqm1[:,1], XLEqm1[:,0],
        bc_type=bcmt
    )

#    LEqplusSpline = spinterp.CubicSpline(XLEqp[:, 1],XLEqp[:, 0],bc_type =
#                            ((1, 1/bisect_slope_PS),(1, 0)))
#    LEqminusSpline = spinterp.CubicSpline(XLEqm[:, 1],XLEqm[:, 0],bc_type =
#                            ((1, 0),(1, 1/bisect_slope_SS))) # change to tangential direction at boundary
    
    # Now generate the curvy TE surface using the same method as the LE surface
    perpTEslope = TE_OUTCurve_Slope

    TEangle = np.arctan2(TE_OUTCurve_Slope,1)
    perpTEangle = np.arctan2(perpTEslope,1)
    ggg = 0.5  # weighting average factor
    bisectTEangle = (ggg*TEangle + (1-ggg)*perpTEangle) - np.pi/2
    bisectDirTEqplus = np.tan(bisectTEangle)
    # instead of using the rotated TEzrq point, use the TE point of the nCamber_Section_zrtheta

    # Do the same for the other side of TE
    hhh = ggg  # weighting average factor
    bisectTEangle = (hhh*TEangle + (1-hhh)*perpTEangle) - np.pi/2
    bisectDirTEqminus = np.tan(bisectTEangle)

    # using the scipy optimization function, minimize_scalar, find the point on the TEqplusSpline where the curvature is at a maximum
    # Given the objective function should be J = max(max(curvature(TEqplusSpline)), max(curvature(TEqminusSpline))), using scipy's minimize_scalar function, minimize J for TEoffset >=0
    # This will give the optimal TEoffset for the given section and will be done for each section
    TEperiodicSlope = 0.0  # 0 is default; 1 means 45 deg
    #resTE = opt.minimize_scalar(objective_TEoffset, bounds=(50, 56), method='bounded')
    TEoffsetVector = (42, 53, 66) # (46, 58, 71) <-- values that worked on periodics with TE at actual back
    TEoffset = TEoffsetVector[i]  #resTE.x
    best_scoreTE = 0  #resTE.fun
    print(f"Best TEoffset for section {i} is {TEoffset} with a score of {best_scoreTE}")
    TErqoffset = TEoffset * TE_OUTCurve_Slope
    perpTEslope = TE_OUTCurve_Slope
    # update the TEqp and TEqm arrays with the new TEoffset
    TErp = np.interp(TEzrq[0]+TEoffset, vNew, wNew)
    TErm = np.interp(TEzrq[0]+TEoffset, vNew, wNew)
    TEqp = np.array([TEzrq, TEzrq+(TEoffset,TErqoffset+np.deg2rad(halfPitch)*TErp)])
    TEqm = np.array([TEzrq+(TEoffset,TErqoffset-np.deg2rad(halfPitch)*TErm), TEzrq])

    XTEqp = TEqp
    XTEqm = TEqm

    XTEqp1 = XTEqp[XTEqp[:,1].argsort()]
    XTEqm1 = XTEqm[XTEqm[:,1].argsort()]
    # check if anything happened; if so,  change order of BC slopes
    if np.array_equal(XTEqp, XTEqp1):
        bcptTE = ((1, 1/bisect_slope_SS_downstream), (1, TEperiodicSlope))
    else:
        bcptTE = ((1, TEperiodicSlope), (1, 1/bisect_slope_SS_downstream))
    if np.array_equal(XTEqm, XTEqm1):
        bcmtTE = ((1, TEperiodicSlope), (1, 1/bisect_slope_PS_downstream))
    else:
        bcmtTE = ((1, 1/bisect_slope_PS_downstream), (1, TEperiodicSlope))

    TEqplusSpline = spinterp.CubicSpline(XTEqp1[:, 1], XTEqp1[:, 0], bc_type=bcptTE)
    TEqminusSpline = spinterp.CubicSpline(XTEqm1[:, 1], XTEqm1[:, 0], bc_type=bcmtTE)

# --------------------------------------------------------------------------------------------------------------------------> Check )

    # Fill array for LE
    rqpPts = np.linspace(LEqp[0, 1], LEqp[1, 1], thetaPtsForLETEmidSurfs)
    zpPts = LEqplusSpline(rqpPts)
    rqmPts = np.linspace(LEqm[0, 1], LEqm[1, 1], thetaPtsForLETEmidSurfs)
    zmPts = LEqminusSpline(rqmPts)

    # Fill array for TE
    rqpPtsTE = np.linspace(TEqp[0, 1], TEqp[1, 1], thetaPtsForLETEmidSurfs)
    vqpPtsTE = TEqplusSpline(rqpPtsTE)
    rqmPtsTE = np.linspace(TEqm[0, 1], TEqm[1, 1], thetaPtsForLETEmidSurfs)
    vqmPtsTE = TEqminusSpline(rqmPtsTE)

    # Interpolate r vs z from camber data using the interpolation function for LE
    rpPts = np.interp(zpPts, zNew, rNew)
    rmPts = np.interp(zmPts, zNew, rNew)

    # Interpolate r vs z from camber data using the interpolation function for TE
    vpPtsTE = np.interp(vqpPtsTE, vNew, wNew)
    vmPtsTE = np.interp(vqmPtsTE, vNew, wNew)

    # Convert to cylindrical coordinates for LE
    qpPts = rqpPts / rpPts
    qmPts = rqmPts / rmPts

    # Convert to cylindrical coordinates for TE
    vqpPtsTEa = rqpPtsTE / vpPtsTE
    vqmPtsTEa = rqmPtsTE / vmPtsTE

    # combine data for LE
    zPts = np.concatenate((zmPts, zpPts))
    rPts = np.concatenate((rmPts, rpPts))
    qPts = np.concatenate((qmPts, qpPts))

    # combine data for TE USING THE: vqpPtsTE, vqmPtsTE, vqpPtsTE, vqmPtsTE
    zPtsTE = np.concatenate((vqmPtsTE[:-1], vqpPtsTE))
    rPtsTE = np.concatenate((vmPtsTE[:-1], vpPtsTE))
    vPtsTE = np.concatenate((vqmPtsTEa[:-1], vqpPtsTEa))  # Indices added here to avoid duplication of TE point

    # remove duplicate centre point for LE
    qunique, qindices = np.unique(qPts, return_index=True)
    # handle case where "duplicate" values are not EXACTLY the same
    if len(qunique) == len(qPts):
        #print statement for debugging can remove for clarity
        print('No unique points at LE, removing points that are almost the same...')
        relTol = 0.01 * np.abs(np.diff(qunique)).mean()  # find a critical tolerance
        qBool = np.diff(qunique) < relTol  # true where the two elements are almost the same
        qindices = np.delete(qindices, qBool.nonzero())
        qunique = qunique[qindices]

    # remove duplicate centre point for LE
    runique = rPts[qindices]
    zunique = zPts[qindices]

    # remove duplicate centre point for TE
    #vunique, vindices = np.unique(vPtsTE, return_index=True)
    vunique = vPtsTE  # use manual duplication removal instead
    # handle case where "duplicate" values are not EXACTLY the same
    #if len(vunique) == len(vPtsTE):
    #    #print statement for debugging can remove for clarity
    #    print('No unique points at TE, removing points that are almost the same...')
    #    relTol = 0.01 * np.abs(np.diff(vunique)).mean()
    #    vBool = np.diff(vunique) < relTol
    #    vindices = np.delete(vindices, vBool.nonzero())
    #    vunique = vunique[vindices]

    # remove duplicate centre point for TE
    runiqueTE = rPtsTE#[vindices]
    zuniqueTE = zPtsTE#[vindices]


    LEuniquePtsxyz = np.array(mf.pol2cart(qunique, runique, zunique)).T
    leadEdge[:, i, :] = LEuniquePtsxyz

    TEuniquePtsxyz = np.array(mf.pol2cart(vunique, runiqueTE, zuniqueTE)).T
    trailEdge[:, i, :] = TEuniquePtsxyz

# Rotate camber surfaces to create periodic boundaries
pCamber = np.zeros(allCamber.shape)
nCamber = np.zeros(allCamber.shape)
mCamber = np.zeros(allCamber.shape)

for j in range(len(allCamber)):
    pCamber[j, :, :] = np.array(mf.vectorRot3D(
        allCamber[j][:, 2], allCamber[j][:, 1], allCamber[j][:, 0], np.deg2rad(halfPitch))).T
    pCamber[j][:, [0, 1, 2]] = pCamber[j][:, [2, 1, 0]]
    mCamber[j, :, :] = np.array(mf.vectorRot3D(
        allCamber[j][:, 2], allCamber[j][:, 1], allCamber[j][:, 0], np.deg2rad(0))).T
    mCamber[j][:, [0, 1, 2]] = mCamber[j][:, [2, 1, 0]]
    nCamber[j, :, :] = np.array(mf.vectorRot3D(
        allCamber[j][:, 2], allCamber[j][:, 1], allCamber[j][:, 0], -1*np.deg2rad(halfPitch))).T
    nCamber[j][:, [0, 1, 2]] = nCamber[j][:, [2, 1, 0]]

# Find in and max theta values for p/n camber
minq = np.min((np.array(mf.cart2pol(pCamber[:,:,0], pCamber[:,:,1], pCamber[:,:,2]))[0,:,:].min(), 
            np.array(mf.cart2pol(nCamber[:,:,0], nCamber[:,:,1], nCamber[:,:,2]))[0,:,:].min()))
maxq = np.max((np.array(mf.cart2pol(pCamber[:,:,0], pCamber[:,:,1], pCamber[:,:,2]))[0,:,:].max(), 
            np.array(mf.cart2pol(nCamber[:,:,0], nCamber[:,:,1], nCamber[:,:,2]))[0,:,:].max()))
offset = maxq - minq - 2*(np.pi/180)*halfPitch

# Initialize arrays for some surfaces
Angle = np.linspace(-2*halfPitch*np.pi/180-offset, 2*halfPitch*np.pi/180+offset, thetaPts)
outlet = np.zeros([thetaPts, nSections, 3])
inlet = np.zeros([thetaPts, nSections, 3])
# trailEdge = np.zeros([thetaPts, nSections, 3])
hubS = np.zeros([thetaPts, len(hub), 3])
casing = np.zeros([thetaPts, len(cas), 3])

# Fill The arrays with data
for b in range(thetaPts):
    outlet[b, :, :] = np.array(mf.vectorRot3D(
        OUTxyz[:, 2], OUTxyz[:, 1], OUTxyz[:, 0], Angle[b])).T
    inlet[b, :, :] = np.array(mf.vectorRot3D(
        INxyz[:, 2], INxyz[:, 1], INxyz[:, 0], Angle[b])).T
    # trailEdge[b, :, :] = np.array(mf.vectorRot3D(
    #     TExyz[:, 2], TExyz[:, 1], TExyz[:,0], Angle[b])).T
    hubS[b, :, :] = np.array(mf.vectorRot3D(
        hub[:, 2], hub[:, 1], hub[:, 0], Angle[b])).T
    casing[b, :, :] = np.array(mf.vectorRot3D(
        cas[:, 2], cas[:, 1], cas[:, 0], Angle[b])).T

# Get 50% axial chord surface data
halfCamber = np.zeros([nSections, 3])
for c in range(nSections):
    z_min, z_max = LE[c][0], TE[c][0]
    zhalf = 0.5*(z_min + z_max)
    yhalf = np.interp(zhalf, justCamber[c][:, 2], justCamber[c][:, 1])
    xhalf = np.interp(zhalf, justCamber[c][:, 2], justCamber[c][:, 0])
    halfCamber[c] = [xhalf, yhalf, zhalf]
midChord = np.zeros([thetaPts, nSections, 3])
for d in range(thetaPts):
    midChord[d, :, :] = np.array(mf.vectorRot3D(
        halfCamber[:, 2], halfCamber[:, 1], halfCamber[:, 0], Angle[d])).T

# ---------------------------------------------------------------------------------------> Plotting from Fridat Notes
# Convert allCamber to cylindrical coordinates
# The 'allCamber' array contains the camber lines for all sections in Cartesian coordinates (x, y, z).
# We need to convert these to cylindrical coordinates (theta, r, z) for comparison and plotting.

# Initialize arrays to hold cylindrical coordinates for 'allCamber'
theta_allCamber = np.zeros_like(allCamber[:, :, 0]) # Array to store angular coordinates (theta)
r_allCamber = np.zeros_like(allCamber[:, :, 0]) # Array to store radial distances (r)
z_allCamber = np.zeros_like(allCamber[:, :, 0]) # Array to store axial positions (z)

# Loop over each blade section and each point along the camber line
for i in range(nSections):
    for j in range(allCamber.shape[1]):
        x = allCamber[i, j, 0] # Extract x-coordinate of the point
        y = allCamber[i, j, 1] # Extract y-coordinate
        z = allCamber[i, j, 2] # Extract z-coordinate (axial position)

        # Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (theta, r, z)
        # 'mf.cart2pol' is a function that converts Cartesian to cylindrical coordinates
        theta, r, _ = mf.cart2pol(x, y, z)

        # Store the cylindrical coordinates in the initialized arrays
        theta_allCamber[i, j] = theta # Angular coordinate (theta) in radians
        r_allCamber[i, j] = r # Radial distance (r)
        z_allCamber[i, j] = z # Axial position (z) remains the same

# Rotate the camber lines by +halfPitch and -halfPitch
# The arrays 'pCamber' and 'nCamber' contain the camber lines rotated by +halfPitch and -halfPitch, respectively.
# We will extract their cylindrical coordinates for comparison.

# Initialize arrays to hold cylindrical coordinates for 'pCamber' (rotated +halfPitch)
theta_pCamber = np.zeros_like(pCamber[:, :, 0]) # Angular coordinates (theta) for pCamber
r_pCamber = np.zeros_like(pCamber[:, :, 0]) # Radial distances (r) for pCamber
z_pCamber = np.zeros_like(pCamber[:, :, 0]) # Axial positions (z) for pCamber

# Initialize arrays to hold cylindrical coordinates for 'nCamber' (rotated -halfPitch)
theta_nCamber = np.zeros_like(nCamber[:, :, 0]) # Angular coordinates (theta) for nCamber
r_nCamber = np.zeros_like(nCamber[:, :, 0]) # Radial distances (r) for nCamber
z_nCamber = np.zeros_like(nCamber[:, :, 0]) # Axial positions (z) for nCamber

# # Loop over each blade section and each point to convert 'pCamber' and 'nCamber' to cylindrical coordinates
# for i in range(nSections):
#     for j in range(pCamber.shape[1]):
#         # Process 'pCamber' (rotated by +halfPitch)
#         x = pCamber[i, j, 0] # Extract x-coordinate of the point in pCamber
#         y = pCamber[i, j, 1] # Extract y-coordinate
#         z = pCamber[i, j, 2] # Extract z-coordinate

#         # Convert to cylindrical coordinates
#         theta, r, _ = mf.cart2pol(x, y, z)

#         # Store the cylindrical coordinates
#         theta_pCamber[i, j] = theta # Angular coordinate for pCamber
#         r_pCamber[i, j] = r # Radial distance for pCamber
#         z_pCamber[i, j] = z # Axial position remains the same

#         # Process 'nCamber' (rotated by -halfPitch)
#         x = nCamber[i, j, 0] # Extract x-coordinate of the point in nCamber
#         y = nCamber[i, j, 1] # Extract y-coordinate
#         z = nCamber[i, j, 2] # Extract z-coordinate

#         # Convert to cylindrical coordinates
#         theta, r, _ = mf.cart2pol(x, y, z)

#         # Store the cylindrical coordinates
#         theta_nCamber[i, j] = theta # Angular coordinate for nCamber
#         r_nCamber[i, j] = r # Radial distance for nCamber
#         z_nCamber[i, j] = z # Axial position remains the same
    
#     # Convert pCamber and nCamber coordinates for the current section i into cylindrical coordinates
#     xp = pCamber[i, :, 0]
#     yp = pCamber[i, :, 1]
#     zp = pCamber[i, :, 2]
#     thetap, rp, _ = mf.cart2pol(xp, yp, zp)

#     xn = nCamber[i, :, 0]
#     yn = nCamber[i, :, 1]
#     zn = nCamber[i, :, 2]
#     thetan, rn, _ = mf.cart2pol(xn, yn, zn)

#     # Compute r*theta for both pCamber and nCamber lines
#     rq_p = rp * thetap
#     rq_n = rn * thetan

#     # Define the target axial location where we want to sample the rotated camber surfaces
#     z_target = LEzrq[0] + LEoffset

#     # Interpolate r*theta at z_target for pCamber (for LEqplus) and nCamber (for LEqminus)
#     rq_target_p = np.interp(z_target, zp, rq_p)
#     rq_target_n = np.interp(z_target, zn, rq_n)

#     # Previously, LEqplus and LEqminus splines used a hardcoded approach.
#     # Now we define LEqp and LEqm using the interpolated values:
#     #
#     # For LEqplusSpline:
#     # The original first point is the LE point (LEzrq).
#     # The second (upper) point is now (z_target, rq_target_p).
#     LEqp = np.array([
#         [LEzrq[0],        LEzrq[1]],       # LE point
#         [z_target,        rq_target_p]     # Point from pCamber at z_target
#     ])

#     # For LEqminusSpline:
#     # The first (lower) point is now (z_target, rq_target_n) from nCamber
#     # The second point is the LE point (LEzrq).
#     LEqm = np.array([
#         [z_target,        rq_target_n],    # Point from nCamber at z_target
#         [LEzrq[0],        LEzrq[1]]        # LE point
#     ])

#     # Now compute LErqoffset using the new LE_INCurve_Slope as before
#     LErqoffset = LEoffset * LE_INCurve_Slope

#     # For LEqplusSpline
#     x_vals_plus = LEqp[:,1] + np.array([0, LErqoffset])
#     y_vals_plus = LEqp[:,0] + np.array([0, LEoffset])

#     # Sort by x_vals_plus to ensure they are strictly increasing
#     sorted_indices_plus = np.argsort(x_vals_plus)
#     x_vals_plus_sorted = x_vals_plus[sorted_indices_plus]
#     y_vals_plus_sorted = y_vals_plus[sorted_indices_plus]

#     LEqplusSpline = spinterp.CubicSpline(x_vals_plus_sorted,
#                                         y_vals_plus_sorted,
#                                         bc_type=((1, 1/bisect_slope_PS), (1, 0)))

#     # For LEqminusSpline
#     x_vals_minus = LEqm[:,1] + np.array([LErqoffset, 0])
#     y_vals_minus = LEqm[:,0] + np.array([LEoffset, 0])

#     # Sort by x_vals_minus to ensure they are strictly increasing
#     sorted_indices_minus = np.argsort(x_vals_minus)
#     x_vals_minus_sorted = x_vals_minus[sorted_indices_minus]
#     y_vals_minus_sorted = y_vals_minus[sorted_indices_minus]

#     LEqminusSpline = spinterp.CubicSpline(x_vals_minus_sorted,
#                                         y_vals_minus_sorted,
#                                         bc_type=((1, 0), (1, 1/bisect_slope_SS)))

# Choose a blade section to plot (e.g., the middle section)
#i = 0  #nSections // 2 # Index of the section to plot. You can change this to plot other sections.
rqoffForPlot = 0
zoffForPlot = 300

# Create a new figure for plotting
plt.figure()
    
for i in range(3):

    # Plot blade
    plt.plot(bladePzrq[i,0,:]+zoffForPlot*i, bladePzrq[i,1,:]+ rqoffForPlot*i, color='purple')
    plt.plot(bladeSzrq[i,0,:]+zoffForPlot*i, bladeSzrq[i,1,:]+ rqoffForPlot*i, color='green')

    # Plot the original camber line (no rotation) in cylindrical coordinates
    plt.plot(
        z_allCamber[i, :]+zoffForPlot*i, # Axial positions (z) along the camber line
        theta_allCamber[i, :] * r_allCamber[i, :] + rqoffForPlot*i, # Circumferential positions (r * theta)
        color='blue' # Label for the original camber line
    )

    # Plot the camber line rotated by +halfPitch
    plt.plot(
        z_allCamber[i, :]+zoffForPlot*i, # Axial positions (z)
        (theta_allCamber[i, :] + halfPitch_radians) * r_allCamber[i, :] + rqoffForPlot*i, # Circumferential positions (r * theta)
        color='blue' # Label for the camber line rotated by +halfPitch
    )

    # Plot the camber line rotated by -halfPitch
    plt.plot(
        z_allCamber[i, :]+zoffForPlot*i, # Axial positions (z)
        (theta_allCamber[i, :] - halfPitch_radians) * r_allCamber[i, :] + rqoffForPlot*i, # Circumferential positions (r * theta)
        color='blue' # Label for the camber line rotated by -halfPitch
    )

    # Extract the LE data for the selected section
    LE_section = leadEdge[:, i, :]  # Shape: [number of points, 3]

    # Extract the TE data for the selected section
    TE_section = trailEdge[:, i, :]  # Shape: [number of points, 3]

    # Initialize arrays to hold cylindrical coordinates for LE_section
    theta_LE = np.zeros_like(LE_section[:, 0])
    r_LE = np.zeros_like(LE_section[:, 0])
    z_LE = np.zeros_like(LE_section[:, 0])

    # Convert LE_section to cylindrical coordinates
    for j in range(LE_section.shape[0]):
        x = LE_section[j, 0]
        y = LE_section[j, 1]
        z = LE_section[j, 2]

        theta, r, _ = mf.cart2pol(x, y, z)

        theta_LE[j] = theta
        r_LE[j] = r
        z_LE[j] = z

    # repeate for TE
    theta_TE = np.zeros_like(TE_section[:, 0])
    r_TE = np.zeros_like(TE_section[:, 0])
    z_TE = np.zeros_like(TE_section[:, 0])

    # Convert TE_section to cylindrical coordinates
    for j in range(TE_section.shape[0]):
        x = TE_section[j, 0]
        y = TE_section[j, 1]
        z = TE_section[j, 2]

        theta, r, _ = mf.cart2pol(x, y, z)

        theta_TE[j] = theta
        r_TE[j] = r
        z_TE[j] = z

    # Plot LE curve on the same figure
    plt.plot(
        z_LE +zoffForPlot*i,
        theta_LE * r_LE + rqoffForPlot*i,
        label='Leading Edge Curve',
        linestyle='-',
        color='black'
    )

    # plot the TE curve on the same figure
    plt.plot(
        z_TE +zoffForPlot*i,
        theta_TE * r_TE + rqoffForPlot*i,
        label='Trailing Edge Curve',
        linestyle='-',
        color='red'
    )

# Add labels and title to the plot
plt.xlabel('z') # Label for the x-axis (axial coordinate)
plt.ylabel('r * θ') # Label for the y-axis (circumferential position)
plt.title('Comparison of Camber Lines in Cylindrical Coordinates')  # Title of the plot

# Display the legend to differentiate between the lines
#plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# ---------------------------------------------------> Plotting from Fridat Notes --------> 3D

# Choose a blade section to plot (e.g., the middle section)
section_index = nSections // 2  # Index of the section to plot ------------> Note: maybe change the section?
# Extract the camber lines for the selected section
# Original camber line (no rotation)
x_allCamber = allCamber[section_index, :, 0]
y_allCamber = allCamber[section_index, :, 1]
z_allCamber = allCamber[section_index, :, 2]
# Rotated camber lines
# Rotated by +halfPitch
x_pCamber = pCamber[section_index, :, 0]
y_pCamber = pCamber[section_index, :, 1]
z_pCamber = pCamber[section_index, :, 2]
# Rotated by -halfPitch
x_nCamber = nCamber[section_index, :, 0]
y_nCamber = nCamber[section_index, :, 1]
z_nCamber = nCamber[section_index, :, 2]
# Create a new figure for 3D plotting
#fig = plt.figure(figsize=(12, 8))
#ax = fig.add_subplot(111, projection='3d')
# Plot the original camber line
#ax.plot3D(
#    x_allCamber,
#    y_allCamber,
#    z_allCamber,
#    label='Original Camber (No Rotation)',
#    linewidth=2,
#    color='blue'
#)
# Plot the camber line rotated by +halfPitch
#ax.plot3D(
#    x_pCamber,
#    y_pCamber,
#    z_pCamber,
#    label='Rotated +halfPitch',
#    linestyle='-',
#    color='blue'
#)
# Plot the camber line rotated by -halfPitch
#ax.plot3D(
#    x_nCamber,
#    y_nCamber,
#    z_nCamber,
#    label='Rotated -halfPitch',
#    linestyle='-',
#    color='blue'
#)
# Extract the LE data for the selected section
LE_section = leadEdge[:, section_index, :]  # Shape: [number of points, 3]
x_LE = LE_section[:, 0]
y_LE = LE_section[:, 1]
z_LE = LE_section[:, 2]

# Plot the LE curve
#ax.plot3D(
#    x_LE,
#    y_LE,
#    z_LE,
#    label='Leading Edge Curve',
#    linestyle='-',
#    color='red',
#    #make thicker line
#    linewidth=2
#
#)
# Optionally, plot the camber line without extensions (justCamber)
# Extract coordinates
x_justCamber = justCamber[section_index, :, 0]
y_justCamber = justCamber[section_index, :, 1]
z_justCamber = justCamber[section_index, :, 2]
# Plot justCamber data
#ax.plot3D(
#    x_justCamber,
#    y_justCamber,
#    z_justCamber,
#    label='Camber Line (Without Extensions)',
#    linestyle=':',
#    color='purple'
#)
# Set labels and title
#ax.set_xlabel('X Coordinate')
#ax.set_ylabel('Y Coordinate')
#ax.set_zlabel('Z Coordinate (Axial Position)')
#ax.set_title('3D Visualization of Camber Lines')
#ax.set_box_aspect([np.ptp(a) for a in [x_allCamber, y_allCamber, z_allCamber]])
#ax.legend()
#ax.grid(True)
#ax.view_init(elev=30, azim=45)  # Adjust elevation and azimuth as needed
# plt.show()

# Create surfaces used for projection of the template mesh
# Parameters are numbers of points, none of these are new,
# independent parameters
Nz = nSections
Nr = len(camber)
Nle = len(LExyz)
Nh = len(hub)
Nc = len(cas)
 
Xp = np.zeros([Nz, Nr]) # pPeriodic
Yp = np.zeros([Nz, Nr])
Zp = np.zeros([Nz, Nr])
 
Xm = np.zeros([Nz, Nr]) # mPeriodic
Ym = np.zeros([Nz, Nr])
Zm = np.zeros([Nz, Nr])
 
Xn = np.zeros([Nz, Nr]) # nPeriodic
Yn = np.zeros([Nz, Nr])
Zn = np.zeros([Nz, Nr])
 
XLE = np.zeros([thetaPts, Nle]) # Blade LE
YLE = np.zeros([thetaPts, Nle])
ZLE = np.zeros([thetaPts, Nle])
 
XTE = np.zeros([thetaPts, Nle]) # Blade TE
YTE = np.zeros([thetaPts, Nle])
ZTE = np.zeros([thetaPts, Nle])
 
Xin = np.zeros([thetaPts, Nz]) # Block inlet
Yin = np.zeros([thetaPts, Nz])
Zin = np.zeros([thetaPts, Nz])
 
Xout = np.zeros([thetaPts, Nz]) # Block outlet
Yout = np.zeros([thetaPts, Nz])
Zout = np.zeros([thetaPts, Nz])
 
Xmid = np.zeros([thetaPts, Nz]) # Mid chord
Ymid = np.zeros([thetaPts, Nz])
Zmid = np.zeros([thetaPts, Nz])
 
Xh = np.zeros([thetaPts, Nh]) # Hub
Yh = np.zeros([thetaPts, Nh])
Zh = np.zeros([thetaPts, Nh])
 
Xc = np.zeros([thetaPts, Nc]) # Casing
Yc = np.zeros([thetaPts, Nc])
Zc = np.zeros([thetaPts, Nc])
 
XBp = [] # Blade Pressure Side
YBp = []
ZBp = []

XBs = [] # Blade Sunction Side
YBs = []
ZBs = []
 
# Fill the arrays
for k in range(Nz):
    for l in range(Nr):
        Xp[k, l] = pCamber[k][l, 0]
        Yp[k, l] = pCamber[k][l, 1]
        Zp[k, l] = pCamber[k][l, 2]
        Xm[k, l] = mCamber[k][l, 0]
        Ym[k, l] = mCamber[k][l, 1]
        Zm[k, l] = mCamber[k][l, 2]
        Xn[k, l] = nCamber[k][l, 0]
        Yn[k, l] = nCamber[k][l, 1]
        Zn[k, l] = nCamber[k][l, 2]
    # Blades:
    XBp.append(bladeP[k][:, 0])
    YBp.append(bladeP[k][:, 1])
    ZBp.append(bladeP[k][:, 2])
    XBs.append(bladeS[k][:, 0])
    YBs.append(bladeS[k][:, 1])
    ZBs.append(bladeS[k][:, 2])
for c in range(thetaPts):
    for d in range(Nz):
        Xin[c, d] = inlet[c][d][2]
        Yin[c, d] = inlet[c][d][1]
        Zin[c, d] = inlet[c][d][0]
        Xout[c, d] = outlet[c][d][2]
        Yout[c, d] = outlet[c][d][1]
        Zout[c, d] = outlet[c][d][0]
        Xmid[c, d] = midChord[c][d][2]
        Ymid[c, d] = midChord[c][d][1]
        Zmid[c, d] = midChord[c][d][0]
    for e in range(Nle):
        XLE[c, e] = leadEdge[c][e][0]#[2]
        YLE[c, e] = leadEdge[c][e][1]
        ZLE[c, e] = leadEdge[c][e][2]#[0]
        XTE[c, e] = trailEdge[c][e][0]
        YTE[c, e] = trailEdge[c][e][1]
        ZTE[c, e] = trailEdge[c][e][2]
    for p in range(Nh):
        Xh[c, p] = hubS[c][p][2]
        Yh[c, p] = hubS[c][p][1]
        Zh[c, p] = hubS[c][p][0]
    for q in range(Nc):
        Xc[c, q] = casing[c][q][2]
        Yc[c, q] = casing[c][q][1]
        Zc[c, q] = casing[c][q][0]
# Set up STL filenames
filenames = ['mPeriodic','nPeriodic','pPeriodic','pBlade','sBlade','LE','TE','inlet','outlet','Casing','Hub','midChord']
Xvalues = [Xm, Xn, Xp, XBp, XBs, XLE, XTE, Xin, Xout, Xc, Xh, Xmid]
Yvalues = [Ym, Yn, Yp, YBp, YBs, YLE, YTE, Yin, Yout, Yc, Yh, Ymid]
Zvalues = [Zm, Zn, Zp, ZBp, ZBs, ZLE, ZTE, Zin, Zout, Zc, Zh, Zmid]
 
def unitVector(file, p1, p2, p3):
    vector1 = p3 - p2
    vector2 = p3 - p1
    normalVec = np.cross(vector1, vector2)
    magnitude = np.linalg.norm(normalVec)
    unitVec = normalVec / magnitude
    file.write(f'facet normal {unitVec[0]:.12e} {unitVec[1]:.12e} {unitVec[2]:.12e} \n'
               f'outer loop \n'
               f'vertex {p1[0]:.12e} {p1[1]:.12e} {p1[2]:.12e} \n'
               f'vertex {p2[0]:.12e} {p2[1]:.12e} {p2[2]:.12e} \n'
               f'vertex {p3[0]:.12e} {p3[1]:.12e} {p3[2]:.12e} \n'
               f'endloop \n'
               f'endfacet \n')
    return
# Create STLs and write them to disk
# Can possibly replace qq with a clearer name for clarity 
for qq in range(len(Xvalues)): #
 
    # Define the filename for the current STL
    filename = pathForSTLs + '/{}.stl'.format(filenames[qq])
 
    numFacets = 0
 
    file = open(filename, 'w')
    file.write('solid ' + filenames[qq]  + ' \n') # name solids so STLs could be combined if desired
 
    if not (qq == 3 or qq == 4):
        rows = Zvalues[qq].shape[0] # Use Z because it is the axis of rotation
        columns = Zvalues[qq].shape[1]
        X = Xvalues[qq]
        Y = Yvalues[qq]
        Z = Zvalues[qq]
 
        for i in range(rows - 1):
            for j in range(columns - 1):
                # Define the first triangle
                p1 = np.asarray([X[i, j], Y[i, j], Z[i, j]])
                p2 = np.asarray([X[i, j+1], Y[i, j+1], Z[i, j+1]])
                p3 = np.asarray([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
        
                # Write the triangle to the STL file
                unitVector(file, p1, p2, p3)

                # Define the second triangle
                p1 = np.asarray([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
                p2 = np.asarray([X[i+1, j], Y[i+1, j], Z[i+1, j]])
                p3 = np.asarray([X[i, j], Y[i, j], Z[i, j]])
        
                # Write the triangle to the STL file
                unitVector(file, p1, p2, p3)
        
                numFacets += 2
 
    elif (qq == 3 or qq == 4):
        for i in range(nSections-1):
            X = np.concatenate((Xvalues[qq][i], Xvalues[qq][i+1]), axis=0)
            Y = np.concatenate((Yvalues[qq][i], Yvalues[qq][i+1]), axis=0)
            Z = np.concatenate((Zvalues[qq][i], Zvalues[qq][i+1]), axis=0)
            R = np.sqrt(X ** 2 + Y ** 2)
 
            Z1 = Zvalues[qq][i]
            Z2 = Zvalues[qq][i+1]
            R1 = np.sqrt(Xvalues[qq][i] ** 2 + Yvalues[qq][i] ** 2)
            R2 = np.sqrt(Xvalues[qq][i+1] ** 2 + Yvalues[qq][i+1] ** 2)
 
            dS1 = np.linalg.norm(np.diff(np.array([Z1, R1])),axis=0)
            dS2 = np.linalg.norm(np.diff(np.array([Z2, R2])),axis=0)
 
            S1 = dS1.sum()
            S2 = dS2.sum()
 
            nS1 = np.concatenate(([0], dS1.cumsum() / S1))
            nS2 = np.concatenate(([0], dS2.cumsum() / S2))
 
            nS = np.concatenate((nS1, nS2))
 
            nR1 = np.zeros(R1.shape)
            nR2 = np.ones(R2.shape)
 
            nR = np.concatenate((nR1, nR2))
 
            DT = Delaunay(np.array([nS, nR]).T)
 
            triangles = DT.simplices.shape[0]
 
            for j in range(triangles):
                p1 = np.array([X[DT.simplices[j,0]], Y[DT.simplices[j,0]], Z[DT.simplices[j,0]]])
                p2 = np.array([X[DT.simplices[j,1]], Y[DT.simplices[j,1]], Z[DT.simplices[j,1]]])
                p3 = np.array([X[DT.simplices[j,2]], Y[DT.simplices[j,2]], Z[DT.simplices[j,2]]])
 
                unitVector(file, p1, p2, p3)
                numFacets += 1
        
    file.write('endsolid \n')
    file.close()
 
rotor = open(scriptFile, 'w')
rotor.write('scale  {};  \n'.format(dataScale))
rotor.write('rMin   {}; \n'.format(min(allCamber[0][:, 0])))
rotor.write('rMax   {}; \n'.format(max(allCamber[len(allCamber)-1][:, 0])))
rotor.write('zMin   {};  \n'.format(min(allCamber[0][:, 2])))
rotor.write('zMax   {}; \n'.format(max(allCamber[len(allCamber)-1][:, 2])))
rotor.write('qMax   {};  \n'.format(max(camberTheta.flatten())))
rotor.write('qMin   {};  \n'.format(min(camberTheta.flatten())))
rotor.write('zLE    {}; \n' .format(min(LE[:, 0])))
rotor.write('zTE    {};  \n'.format(max(TE[:, 0])))
rotor.write('zSS    {};  \n'.format(0.5*(max(TE[:, 0]) + min(LE[:, 0]))))
rotor.write('zPS    {};  \n'.format(0.5*(max(TE[:, 0]) + min(LE[:, 0]))))
rotor.close()
