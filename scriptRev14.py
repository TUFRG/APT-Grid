#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:58:35 2025

@author: Adekola Adeyemi and Jeff defoe
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import model_function as mf
import TransfiniteInterpolation as tf
import findLastQuadPointFunction as fq
from scipy.interpolate import interp1d, CubicSpline, splprep, splev
from scipy.optimize import fsolve, minimize
import importlib
import bladePassageSurfaceGenerator as bpsg

# INPUTS


IGVorECL5orECL5mod = 1 #if IGV put 0, if ECL5_orignial put 1, if ECL5_modified 2
periodicORaperiodic = 0 # 0 for periodic cases and 1 for aperiodic case
#%%
if IGVorECL5orECL5mod == 0:
    Nb = 20
    N = 121
    nSections = 21
    if periodicORaperiodic == 0:
        dataPath = './IGV/periodic/'
    else:
        dataPath = './IGV/nonPeriodic/'
elif IGVorECL5orECL5mod == 1:
    Nb = 31
    N = 101
    nSections = 23   
    if periodicORaperiodic == 0:
        dataPath = './ECL5_original/periodic'
    else:
        dataPath = './ECL5_original/nonPeriodic'
else:
    Nb = 31
    N = 121
    nSections = 21
    if periodicORaperiodic == 0:
        dataPath = './ECL5_modified/periodic'
    else:
        dataPath = './ECL5_modified/nonPeriodic'

res = 30  #upstream and downstream extention resolution 
passageRes = 360  # Resolution of points for a single passage 
bladeRes = 400  #Increase resolution of underlying blade data 
Nr = 2*N  #number of points on blade profiles increased 
delta = 2.0  #boundary layer thickness 
percentVal = 0.04  # fraction of arclength of blades where we cut off to avoid odd cell sizes in blade-to-offset BL blocks
percentValNonCut = 0.02  # fraction of arclength of blades where we cut off to avoid higghly non-orthogonal cells in blade-to-offset BL blocks

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
LrefHub = 400.0  # input length units
LrefCas = 400.0  # input length units
LrefBla = 75.0  # input length units (JD: this should be calculated = mean chord)
muref = 1.8e-5  # base SI units (kg/(m*s))
yPlusHub = 2
yPlusCas = 2
yPlusBla = 2
# have option to calculate BL parameters based on above, or just directly
# provide BL thickness and first cell size (input units)
delHub = bpsg.calcBLdelta(rhoref,Uref,LrefHub*scale,muref)/scale  # or just set a value (in input units)
delCas = bpsg.calcBLdelta(rhoref,Uref,LrefCas*scale,muref)/scale  # or just set a value (in input units)
delBla = delta  # calcBLdelta(rhoref,Uref,LrefBla*scale,muref)/scale  # or just set a value (in input units)
dy1Hub = bpsg.calcFirstCellSize(rhoref,Uref,LrefHub*scale,muref,yPlusHub)/scale  # or just set a value (in input units)
dy1Cas = bpsg.calcFirstCellSize(rhoref,Uref,LrefCas*scale,muref,yPlusCas)/scale  # or just set a value (in input units)
dy1Bla = bpsg.calcFirstCellSize(rhoref,Uref,LrefBla*scale,muref,yPlusBla)/scale  # or just set a value (in input units)
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

#%%  Load the blade data
# for a in range(1): #This is done to ensure that the entire 31 blades can be generated 
a = 0
filePath = dataPath +  '../constant/geometry{}'.format(a)
if not os.path.exists(filePath):
    os.makedirs(filePath, exist_ok=True)
blade1PS = np.zeros([nSections,N,3]) ## First blade pressure side
blade1SS = np.zeros([nSections,N,3]) ## First blade sunction side
blade2PS = np.zeros([nSections,N,3]) ## Second blade pressure side
blade2SS = np.zeros([nSections,N,3]) ## Second blade sunction side
for b in range(nSections):
    blade1PS[b,:] = np.loadtxt(dataPath + '/blade{}/PS{}.txt'.format(a,b), delimiter=',')
    blade2PS[b,:] = np.loadtxt(dataPath + '/blade{}/PS{}.txt'.format(a+1,b), delimiter=',')          
    blade1SS[b,:] = np.loadtxt(dataPath + '/blade{}/SS{}.txt'.format(a,b), delimiter=',')
    blade2SS[b,:] = np.loadtxt(dataPath + '/blade{}/SS{}.txt'.format(a+1,b), delimiter=',')    

hub = np.loadtxt(dataPath + '/hub.txt', delimiter=',')
casing = np.loadtxt(dataPath + '/casing.txt', delimiter=',')

# FUNCTION DEFINITIONS

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
    # need to get coordinates of point p1, which is where the tangents
    # (slope lines) from p0 and p2 intersect
    # c0, c1 are the "y" intercepts of each line

    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]

    xc = ((m2*x2 - m1*x1) - (y2 - y1)) / (m2-m1)
    yc = y1 + m1*(xc - x1)
    
    #c0 = p0[0] - slope0*p0[1]
    #c2 = p2[0] - slope2*p2[1]
    #x1 = (c2 - c0)/(slope0 - slope2)
    #y1 = slope0*x1 + c0
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


def remove_duplicates(points, tol=1e-10):
    if len(points) <= 1:
        return points
    keep = [0]  # Always keep first point
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[keep[-1]])
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
    print(f'LE slope joining to curve: {slope2LE}')
    offsetLEp1 = getBezierControlPoint(p0LE, p2LE, slope0LE, slope2LE)
    print(f'LE control point: {offsetLEp1}')
    curveLE = quadratic_bezier_curve(p0=p0LE, p1=offsetLEp1, p2=p2LE)

    p0TE = rotFullOffset[-1]
    p2TE = rotFullOffset[-2]
    slope0TE = TEslope
    slope2TE = thirdPtSlopeTE
    print(f'TE slope joining to curve: {slope2TE}')
    offsetTEp1 = getBezierControlPoint(p0TE, p2TE, slope0TE, slope2TE)
    print(f'TE control point: {offsetTEp1}')
    curveTE = quadratic_bezier_curve(p0=p0TE, p1=offsetTEp1, p2=p2TE)

    # assemble overall curve
    curve = np.concatenate((curveLE, rotFullOffset[2:-2], curveTE[::-1]))
    
    return curve

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

# START OF ACTUAL MAIN OPERATIONS

#%% convert read-in data to cylinderical coordinate
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
    blade1PSCyl[c,:,:] = np.array(mf.cart2pol(blade1PS[c,:,0],blade1PS[c,:,1],blade1PS[c,:,2])).T #Change from cartesian to cyclindrical
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

hubLEIdx = np.searchsorted(hub[:,2], allLEBlade1[0][2])
hubTEIdx = np.searchsorted(hub[:,2], allTEBlade1[0][2])
casLEIdx = np.searchsorted(casing[:,2], allLEBlade1[nSections-1][2])
casTEIdx = np.searchsorted(casing[:,2], allTEBlade1[nSections-1][2])

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
hubRegionPts2 = np.array([hubRegionZ2, hubRegionFunc2(hubRegionZ)]).T
casBladeRegion2 = np.vstack((casLE2, cas2D[casLEIdx:casTEIdx],casTE2))
casRegionFunc2 = interp1d(casBladeRegion2[:,0], casBladeRegion2[:,1])
casRegionZ2 = np.linspace(casBladeRegion2[0,0], casBladeRegion2[-1,0], Nr*2-2)
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
#%% Now get the profile on the hub and casing 

hubBladeProfile = np.zeros([N*2-2,3])
hubBladeProfile[:,0] = blade1Cyl[hubIdx-1,:,0]
hubBladeProfile[:,[2,1]] = newHubBladeRegion

casBladeProfile = np.zeros([N*2-2,3])
casBladeProfile[:,0] = blade1Cyl[casIdx-2,:,0]
# casBladeProfile[:,0] = interpolate_theta_parametric(blade1Cyl[casIdx-1], blade1Cyl[casIdx-2], newCasBladeRegion[:,1], newCasBladeRegion[:,0])
casBladeProfile[:,[2,1]] = newCasBladeRegion

hubBladeProfile2 = np.zeros([N*2-2,3])
hubBladeProfile2[:,0] = blade2Cyl[hubIdx2-1,:,0]
hubBladeProfile2[:,[2,1]] = newHubBladeRegion2

casBladeProfile2 = np.zeros([N*2-2,3])
casBladeProfile2[:,0] =blade2Cyl[casIdx2-2,:,0]
# casBladeProfile2[:,0] = interpolate_theta_parametric(blade2Cyl[casIdx2-1], blade2Cyl[casIdx2-2], newCasBladeRegion2[:,1], newCasBladeRegion2[:,0])
casBladeProfile2[:,[2,1]] = newCasBladeRegion2

hubPS1 = np.zeros([N,3])
hubSS1 = np.zeros([N,3])
casPS1 = np.zeros([N,3])
casSS1 = np.zeros([N,3])

hubPS2 = np.zeros([N,3])
hubSS2 = np.zeros([N,3])
casPS2 = np.zeros([N,3])
casSS2 = np.zeros([N,3])

idxHubLE = int(np.where(np.isclose(hubBladeProfile[:,2], hubLE[0]))[0][0])
idxHubTE = int(np.where(np.isclose(hubBladeProfile[:,2], hubTE[0]))[0][0])
idxCasLE = int(np.where(np.isclose(casBladeProfile[:,2], casLE[0]))[0][0])
idxCasTE = int(np.where(np.isclose(casBladeProfile[:,2], casTE[0]))[0][0])

idxHub2LE = int(np.where(np.isclose(hubBladeProfile2[:,2], hubLE[0]))[0][0])
idxHub2TE = int(np.where(np.isclose(hubBladeProfile2[:,2], hubTE[0]))[0][0])
idxCas2LE = int(np.where(np.isclose(casBladeProfile2[:,2], casLE[0]))[0][0])
idxCas2TE = int(np.where(np.isclose(casBladeProfile2[:,2], casTE[0]))[0][0])
   
#Splitting the blade to pressure and pressure side
tempHubPS1 = hubBladeProfile[idxHubLE:idxHubTE+1]
tempHubPS1 = remove_duplicates(tempHubPS1)
tempHubSS1 = np.vstack((hubBladeProfile[idxHubTE:len(hubBladeProfile)], hubBladeProfile[idxHubLE]))[::-1]
tempHubSS1 = remove_duplicates(tempHubSS1)
tempCasPS1 = casBladeProfile[idxCasLE:idxCasTE+1]
tempCasPS1 = remove_duplicates(tempCasPS1)
tempCasSS1 = np.vstack((casBladeProfile[idxCasTE:len(casBladeProfile)], casBladeProfile[idxCasLE]))[::-1]
tempCasSS1 = remove_duplicates(tempCasSS1)

tempHubPS2 = hubBladeProfile2[idxHub2LE:idxHub2TE+1]
tempHubPS2 = remove_duplicates(tempHubPS2)
tempHubSS2 = np.vstack((hubBladeProfile2[idxHub2TE:len(hubBladeProfile2)], hubBladeProfile2[idxHub2LE]))[::-1]
tempHubSS2 = remove_duplicates(tempHubSS2)
tempCasPS2 = casBladeProfile2[idxCas2LE:idxCas2TE+1]
tempCasPS2 = remove_duplicates(tempCasPS2)
tempCasSS2 = np.vstack((casBladeProfile2[idxCas2TE:len(casBladeProfile2)], casBladeProfile2[idxCas2LE]))[::-1]
tempCasSS2 = remove_duplicates(tempCasSS2)

hubPS1 = mf.densify_curve_robust(tempHubPS1, N)
hubSS1 = mf.densify_curve_robust(tempHubSS1, N)
casPS1 = mf.densify_curve_robust(tempCasPS1, N)
casSS1 = mf.densify_curve_robust(tempCasSS1, N)

hubPS2 = mf.densify_curve_robust(tempHubPS2, N)
hubSS2 = mf.densify_curve_robust(tempHubSS2, N)
casPS2 = mf.densify_curve_robust(tempCasPS2, N)
casSS2 = mf.densify_curve_robust(tempCasSS2, N)

#%%
#Stacking blade for new Blade definition 
if casIdx-2 > nSections:
    newNsection = nSections - hubIdx + 3 # Remember that i added two points for the extension both at hub and casing. So for cases where 
    # the blade did not touch the casing. so this is the number of points in original radial direction + the two new points i added (hub nad cas)
    # then minus the point already excluded at the hub - 1. For example imagine there are 7 points originally, at the hub I excluded 2 points
    # but the casing is not touching. By definition my hubIdx is 3 (not 2 because of how i defined it) so 7 + 2 - 3 + 1 = 7 - 3 + 3.
else:
    newNsection = nSections - hubIdx + (nSections - casIdx-2) + 4 #In this case the casing is touching so 

#I will compute 6 distances 
hubMidIdx = int(0.5*N)
distHubALE = mf.dist2D(blade1PSCyl[hubIdx][0,2], blade1PSCyl[hubIdx][0,1]*blade1PSCyl[hubIdx][0,0], #Check the distance of the profile before the hub proile and the profile after the hub profile from original 
                     blade1PSCyl[hubIdx-2][0,2], blade1PSCyl[hubIdx-2][0,1]*blade1PSCyl[hubIdx-2][0,0]) # blade data at the LE
distHubATE = mf.dist2D(blade1PSCyl[hubIdx][-1,2], blade1PSCyl[hubIdx][-1,1]*blade1PSCyl[hubIdx][-1,0],  #Check the distance of the profile before the hub proile and the profile after the hub profile from original 
                     blade1PSCyl[hubIdx-2][-1,2], blade1PSCyl[hubIdx-2][-1,1]*blade1PSCyl[hubIdx-2][-1,0]) # blade data at the TE
distHubAMid = mf.dist2D(blade1PSCyl[hubIdx][hubMidIdx,2], blade1PSCyl[hubIdx][hubMidIdx,1]*blade1PSCyl[hubIdx][hubMidIdx,0],  #Check the distance of the profile before the hub proile and the profile 
                     blade1PSCyl[hubIdx-2][hubMidIdx,2], blade1PSCyl[hubIdx-2][hubMidIdx,1]*blade1PSCyl[hubIdx-2][hubMidIdx,0]) #after the hub profile from original blade data at the midChord
distHubBLE = mf.dist2D(blade1PSCyl[hubIdx][0,2], blade1PSCyl[hubIdx][0,1]*blade1PSCyl[hubIdx][0,0], #Check the distance of the profile of the hub proile and the profile after the hub profile from original
                     hubPS1[0,2], hubPS1[0,1]*hubPS1[0,0]) # blade data at the LE
distHubBTE = mf.dist2D(blade1PSCyl[hubIdx][-1,2], blade1PSCyl[hubIdx][-1,1]*blade1PSCyl[hubIdx][-1,0], #Check the distance of the profile of the hub proile and the profile after the hub profile from original
                     hubPS1[-1,2], hubPS1[-1,1]*hubPS1[-1,0]) # blade data at the TE
distHubBMid = mf.dist2D(blade1PSCyl[hubIdx][hubMidIdx,2], blade1PSCyl[hubIdx][hubMidIdx,1]*blade1PSCyl[hubIdx][hubMidIdx,0], #Check the distance of the profile of the hub proile and the profile 
                     hubPS1[hubMidIdx,2], hubPS1[hubMidIdx,1]*hubPS1[hubMidIdx,0]) #after the hub profile from original after the hub profile from original blade data at the midChord

distHubA = min(distHubALE, distHubATE, distHubAMid) # The minimum distance is the closest 
distHubB = min(distHubBLE, distHubBTE, distHubBMid)

distCasALE = mf.dist2D(blade1PSCyl[casIdx-1][0,2], blade1PSCyl[casIdx-1][0,1]*blade1PSCyl[casIdx-1][0,0], 
                     blade1PSCyl[casIdx-2][0,2], blade1PSCyl[casIdx-2][0,1]*blade1PSCyl[casIdx-2][0,0])
distCasATE = mf.dist2D(blade1PSCyl[casIdx-1][-1,2], blade1PSCyl[casIdx-1][-1,1]*blade1PSCyl[casIdx-1][-1,0], 
                     blade1PSCyl[casIdx-2][-1,2], blade1PSCyl[casIdx-2][-1,1]*blade1PSCyl[casIdx-2][-1,0])
distCasAMid = mf.dist2D(blade1PSCyl[casIdx-1][hubMidIdx,2], blade1PSCyl[casIdx-1][hubMidIdx,1]*blade1PSCyl[casIdx-1][hubMidIdx,0], 
                     blade1PSCyl[casIdx-2][hubMidIdx,2], blade1PSCyl[casIdx-2][hubMidIdx,1]*blade1PSCyl[casIdx-2][hubMidIdx,0])
distCasBLE = mf.dist2D(blade1PSCyl[casIdx-1][0,2], blade1PSCyl[casIdx-1][0,1]*blade1PSCyl[casIdx-1][0,0], 
                     casPS1[0,2], casPS1[0,1]*casPS1[0,0])
distCasBTE = mf.dist2D(blade1PSCyl[casIdx-1][-1,2], blade1PSCyl[casIdx-1][-1,1]*blade1PSCyl[casIdx-1][-1,0], 
                     casPS1[-1,2], casPS1[-1,1]*casPS1[-1,0])
distCasBMid = mf.dist2D(blade1PSCyl[casIdx-1][hubMidIdx,2], blade1PSCyl[casIdx-1][hubMidIdx,1]*blade1PSCyl[casIdx-1][hubMidIdx,0], 
                     casPS1[hubMidIdx,2], casPS1[hubMidIdx,1]*casPS1[hubMidIdx,0])

distCasA = min(distCasALE, distCasATE, distCasAMid) # The minimum distance is the closest 
distCasB = min(distCasBLE, distCasBTE, distCasBMid)
if distHubB > 0.5*distHubA:
    newNsection = newNsection
else:
    newNsection = newNsection - 1
if distCasB > 0.5*distCasA:
    newNsection = newNsection - 1
else:
    newNsection = newNsection
#%%
oldBlade1PSCyl = np.zeros([newNsection,N,3]) # (theta, r, z)
oldBlade1SSCyl = np.zeros([newNsection,N,3])
oldBlade2PSCyl = np.zeros([newNsection,N,3]) # (theta, r, z)
oldBlade2SSCyl = np.zeros([newNsection,N,3])
oldAllLEBlade1 = np.zeros([newNsection,3])
oldAllLEBlade2 = np.zeros([newNsection,3])
oldAllTEBlade1 = np.zeros([newNsection,3])
oldAllTEBlade2 = np.zeros([newNsection,3])
oldBlade1Cyl = np.zeros([newNsection,N*2-1,3])
oldBlade2Cyl = np.zeros([newNsection,N*2-1,3])
for f in range(newNsection):
    if f == 0:
        oldBlade1PSCyl[f,:,:] = hubPS1
        oldBlade1SSCyl[f,:,:] = hubSS1
        oldBlade2PSCyl[f,:,:] = hubPS2
        oldBlade2SSCyl[f,:,:] = hubSS2
        oldAllLEBlade1[f] = oldBlade1PSCyl[f][0]
        oldAllLEBlade2[f] = oldBlade2PSCyl[f][0]
        oldAllTEBlade1[f] = oldBlade1PSCyl[f][-1]
        oldAllTEBlade2[f] = oldBlade2PSCyl[f][-1]        
    elif f == newNsection - 1:
        oldBlade1PSCyl[f,:,:] = casPS1
        oldBlade1SSCyl[f,:,:] = casSS1
        oldBlade2PSCyl[f,:,:] = casPS2
        oldBlade2SSCyl[f,:,:] = casSS2
        oldAllLEBlade1[f] = oldBlade1PSCyl[f][0]
        oldAllLEBlade2[f] = oldBlade2PSCyl[f][0]
        oldAllTEBlade1[f] = oldBlade1PSCyl[f][-1]
        oldAllTEBlade2[f] = oldBlade2PSCyl[f][-1]  
    elif distHubB < 0.5*distHubA:
        oldBlade1PSCyl[f,:,:] = blade1PSCyl[f+hubIdx-1,:,:]
        oldBlade1SSCyl[f,:,:] = blade1SSCyl[f+hubIdx-1,:,:]
        oldBlade2PSCyl[f,:,:] = blade2PSCyl[f+hubIdx-1,:,:]
        oldBlade2SSCyl[f,:,:] = blade2SSCyl[f+hubIdx-1,:,:]
        oldAllLEBlade1[f] = oldBlade1PSCyl[f][0]
        oldAllLEBlade2[f] = oldBlade2PSCyl[f][0]
        oldAllTEBlade1[f] = oldBlade1PSCyl[f][-1]
        oldAllTEBlade2[f] = oldBlade2PSCyl[f][-1]        
    else:
        oldBlade1PSCyl[f,:,:] = blade1PSCyl[f+hubIdx-2,:,:]
        oldBlade1SSCyl[f,:,:] = blade1SSCyl[f+hubIdx-2,:,:]
        oldBlade2PSCyl[f,:,:] = blade2PSCyl[f+hubIdx-2,:,:]
        oldBlade2SSCyl[f,:,:] = blade2SSCyl[f+hubIdx-2,:,:]
        oldAllLEBlade1[f] = oldBlade1PSCyl[f][0]
        oldAllLEBlade2[f] = oldBlade2PSCyl[f][0]
        oldAllTEBlade1[f] = oldBlade1PSCyl[f][-1]
        oldAllTEBlade2[f] = oldBlade2PSCyl[f][-1]
    oldBlade1Cyl[f] = np.vstack((oldBlade1PSCyl[f,:,:], oldBlade1SSCyl[f,:,:][:-1][::-1]))
    oldBlade2Cyl[f] = np.vstack((oldBlade2PSCyl[f,:,:], oldBlade2SSCyl[f,:,:][:-1][::-1]))



#%% modify the blade surface. Split the blade at LE and TE
"""
This is portion of the code redefine the definition of the LE and TE. It changes the LE and TE of the blade data to the furthest point on both ends for LE and TE
"""
tempBlade1PS = [None for _ in range(newNsection)]
tempBlade1SS = [None for _ in range(newNsection)]  
tempBlade2PS = [None for _ in range(newNsection)]
tempBlade2SS = [None for _ in range(newNsection)] 
newAllLEBlade1 = np.zeros([newNsection,3])
newAllLEBlade2 = np.zeros([newNsection,3])
newAllTEBlade1 = np.zeros([newNsection,3])
newAllTEBlade2 = np.zeros([newNsection,3])

for cc in range(newNsection): 
    # Find LE (minimum z)
    #This portion of the code, some were taken from Jeff's previous code 
    LEindex1 = oldBlade1Cyl[cc][:, 2].argmin()
    LEindex2 = oldBlade2Cyl[cc][:, 2].argmin()
    TEindex1 = oldBlade1Cyl[cc][:, 2].argmax()
    TEindex2 = oldBlade2Cyl[cc][:, 2].argmax()    
    newAllLEBlade1[cc] = [oldBlade1Cyl[cc][LEindex1, 0], oldBlade1Cyl[cc][LEindex1, 1], oldBlade1Cyl[cc][LEindex1, 2]]
    newAllLEBlade2[cc] = [oldBlade2Cyl[cc][LEindex2, 0], oldBlade2Cyl[cc][LEindex2, 1], oldBlade2Cyl[cc][LEindex2, 2]]
    newAllTEBlade1[cc] = [oldBlade1Cyl[cc][TEindex1, 0], oldBlade1Cyl[cc][TEindex1, 1], oldBlade1Cyl[cc][TEindex1, 2]]
    newAllTEBlade2[cc] = [oldBlade2Cyl[cc][TEindex2, 0], oldBlade2Cyl[cc][TEindex2, 1], oldBlade2Cyl[cc][TEindex2, 2]]
    if np.equal(oldBlade1SSCyl[cc], oldBlade1Cyl[cc][LEindex1, :]).prod(axis=1).max():
        LEloc = np.where(np.equal(oldBlade1SSCyl[cc], oldBlade1Cyl[cc][LEindex1, :]).prod(axis=1)==1)
        le_index = LEloc[0].item() if LEloc[0].size > 0 else 0  # Safe conversion with fallback
        tempBlade1SS[cc] = oldBlade1SSCyl[cc][le_index:, :]
        SpointsToMovetoPS = oldBlade1SSCyl[cc][0:le_index+1:, :]  # includes LE again
        tempBlade1PS[cc] = np.concatenate((SpointsToMovetoPS[::-1, :], oldBlade1PSCyl[cc]))
    elif np.equal(oldBlade1PSCyl[cc], oldBlade1Cyl[cc][LEindex1, :]).prod(axis=1).max():
        LEloc = np.where(np.equal(oldBlade1PSCyl[cc], oldBlade1Cyl[cc][LEindex1, :]).prod(axis=1)==1)
        le_index = LEloc[0].item() if LEloc[0].size > 0 else 0  # Safe conversion with fallback
        tempBlade1PS[cc] = oldBlade1PSCyl[cc][le_index:, :]  # remove points
        PpointsToMovetoSS = oldBlade1PSCyl[cc][0:le_index+1:, :]  # includes LE again
        tempBlade1SS[cc] = np.concatenate((PpointsToMovetoSS[::-1, :],oldBlade1SSCyl[cc]))
                    
    if np.equal(oldBlade2SSCyl[cc], oldBlade2Cyl[cc][LEindex2, :]).prod(axis=1).max():
        LEloc2 = np.where(np.equal(oldBlade2SSCyl[cc], oldBlade2Cyl[cc][LEindex2, :]).prod(axis=1)==1)
        le_index2 = LEloc2[0].item() if LEloc2[0].size > 0 else 0  # Safe conversion with fallback
        tempBlade2SS[cc] = oldBlade2SSCyl[cc][le_index2:, :]
        SpointsToMovetoPS2 = oldBlade2SSCyl[cc][0:le_index2+1:, :]  # includes LE again
        tempBlade2PS[cc] = np.concatenate((SpointsToMovetoPS2[::-1, :], oldBlade2PSCyl[cc]))
    elif np.equal(oldBlade2PSCyl[cc], oldBlade2Cyl[cc][LEindex2, :]).prod(axis=1).max():
        LEloc2 = np.where(np.equal(oldBlade2PSCyl[cc], oldBlade2Cyl[cc][LEindex2, :]).prod(axis=1)==1)
        le_index2 = LEloc2[0].item() if LEloc2[0].size > 0 else 0  # Safe conversion with fallback
        tempBlade2PS[cc] = oldBlade2PSCyl[cc][le_index2:, :]  # remove points
        PpointsToMovetoSS2 = oldBlade2PSCyl[cc][0:le_index2+1:, :]  # includes LE again
        tempBlade2SS[cc] = np.concatenate((PpointsToMovetoSS2[::-1, :],oldBlade2SSCyl[cc]))     
               

newBlade1PSCyl = np.zeros([newNsection,Nr,3]) # (theta, r, z)
newBlade1SSCyl = np.zeros([newNsection,Nr,3])
newBlade2PSCyl = np.zeros([newNsection,Nr,3])
newBlade2SSCyl = np.zeros([newNsection,Nr,3])
newBlade1Cyl = np.zeros([newNsection,Nr*2-1,3])
newBlade2Cyl = np.zeros([newNsection,Nr*2-1,3])
for cd in range(newNsection):     
    tempBlade1PS[cd] = remove_duplicates(tempBlade1PS[cd])
    tempBlade1SS[cd] = remove_duplicates(tempBlade1SS[cd])
    tempBlade2PS[cd] = remove_duplicates(tempBlade2PS[cd])
    tempBlade2SS[cd] = remove_duplicates(tempBlade2SS[cd])
    newBlade1PSCyl[cd,:] = mf.densify_curve_robust(tempBlade1PS[cd], Nr)
    newBlade1SSCyl[cd,:] = mf.densify_curve_robust(tempBlade1SS[cd], Nr)
    newBlade2PSCyl[cd,:] = mf.densify_curve_robust(tempBlade2PS[cd], Nr)
    newBlade2SSCyl[cd,:] = mf.densify_curve_robust(tempBlade2SS[cd], Nr)
    newBlade1Cyl[cd,:,:] = np.vstack((newBlade1PSCyl[cd,:,:], newBlade1SSCyl[cd,:,:][:-1][::-1]))
    newBlade2Cyl[cd,:,:] = np.vstack((newBlade2PSCyl[cd,:,:], newBlade2SSCyl[cd,:,:][:-1][::-1]))    

inletZ = np.linspace(hub2D[0][0],cas2D[0][0],res)  #The inlet is taken as the first point on hub and casing
inletFunc = interp1d([hub2D[0][0],cas2D[0][0]], [hub2D[0][1],cas2D[0][1]])
inletR = inletFunc(inletZ)
inlet = np.column_stack((inletZ, inletR))
scaledNewInlet = mf.scale(min(newAllLEBlade1[:,1]), max(newAllLEBlade1[:,1]), hub[0][0], casing[0][0],newAllLEBlade1[:,1])
scaledNewOutlet = mf.scale(min(newAllTEBlade1[:,1]), max(newAllTEBlade1[:,1]), hub[::-1][0][0], casing[::-1][0][0],newAllTEBlade1[:,1])
inletFunc = interp1d(inlet[:,1], inlet[:,0])
scaledNewInletZ = inletFunc(scaledNewInlet) #You might ask why are you scaling the inlet here. Bascially, what I have done is to map the LE and TE of the blade surface to the inlet and outlet
outletRth = np.linspace(hub2D[-1][1],cas2D[-1][1],res) 
outletFunc = interp1d([hub2D[-1][1],cas2D[-1][1]], [hub2D[-1][0],cas2D[-1][0]])
outletZ = outletFunc(outletRth)
outlet = np.column_stack((outletZ, outletRth))
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
tempInletFunc = interp1d([tempHub[0][1], tempCas[0][0]], [tempHub[0][2], tempCas[0][2]])
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

#Using transfinite interpolation to get the meridional curve. 
hub1 = np.delete(hub, 1, axis=1)
casing1 = np.delete(casing,1, axis=1)
meridionalNodes = tf.transfinite(tempInlet, tempOutlet, hub1[:,[1,0]], casing1[:,[1,0]])
meridCurve = meridionalNodes.reshape(newNsection, len(hub1),2)
meridCurve = meridCurve[:,:,[1,0]]
# plt.plot(meridCurve[0,:,0], meridCurve[0,:,1], 'k.')

angleLE1 = np.zeros(newNsection)
angleLE2 = np.zeros(newNsection)
angleTE1 = np.zeros(newNsection)
angleTE2 = np.zeros(newNsection)
     
for cd in range(newNsection):
    chordlengthBlade1 = mf.dist2D(newAllLEBlade1[cd][2], newAllLEBlade1[cd][1]*newAllLEBlade1[cd][0], newAllTEBlade1[cd][2], newAllTEBlade1[cd][1]*newAllLEBlade1[cd][0])
    chordlengthBlade2 = mf.dist2D(newAllLEBlade2[cd][2], newAllLEBlade2[cd][1]*newAllLEBlade2[cd][0], newAllTEBlade2[cd][2], newAllTEBlade2[cd][1]*newAllTEBlade2[cd][0])
    #add 5% of chord length to the axial distance of the LE and TE to get vertical line that intersect blade surface
    PS1func = interp1d(newBlade1PSCyl[cd,:,2], newBlade1PSCyl[cd,:,1]*newBlade1PSCyl[cd,:,0]) #Interpolation function pressure side blade1
    SS1func = interp1d(newBlade1SSCyl[cd,:,2], newBlade1SSCyl[cd,:,1]*newBlade1SSCyl[cd,:,0]) #Interpolation function sunction side blade1
    PS2func = interp1d(newBlade2PSCyl[cd,:,2], newBlade2PSCyl[cd,:,1]*newBlade2PSCyl[cd,:,0]) #Interpolation function pressure side blade2
    SS2func = interp1d(newBlade2SSCyl[cd,:,2], newBlade2SSCyl[cd,:,1]*newBlade2SSCyl[cd,:,0]) #Interpolation function sunction side blade1
    #Determine the slope at the LE and TE of blade1
    zLE1 = newAllLEBlade1[cd][2] + 0.05*chordlengthBlade1 
    rThLEPS1 = PS1func(zLE1)
    rThLESS1 = SS1func(zLE1)
    rThLE1 = 0.5*(rThLEPS1+rThLESS1)
    zTE1 = newAllTEBlade1[cd][2] - 0.05*chordlengthBlade1
    rThTEPS1 = PS1func(zTE1)
    rThTESS1 = SS1func(zTE1) 
    rThTE1 = 0.5*(rThTEPS1 + rThTESS1)
    slopeLE1 = mf.Slope(newAllLEBlade1[cd][2], newAllLEBlade1[cd][1]*newAllLEBlade1[cd][0], zLE1, rThLE1)
    slopeTE1 = mf.Slope(newAllTEBlade1[cd][2], newAllTEBlade1[cd][1]*newAllTEBlade1[cd][0], zTE1, rThTE1)
    
    zLE2 = newAllLEBlade2[cd][2] + 0.05*chordlengthBlade2
    rThLEPS2 = PS2func(zLE2)
    rThLESS2 = SS2func(zLE2)   
    rThLE2 = 0.5*(rThLEPS2 + rThLESS2)
    zTE2 = newAllTEBlade2[cd][2] - 0.05*chordlengthBlade2
    rThTEPS2 = PS2func(zTE2)
    rThTESS2 = SS2func(zTE2)  
    rThTE2 = 0.5*(rThTEPS2 + rThTESS2)
    slopeLE2 = mf.Slope(newAllLEBlade2[cd][2], newAllLEBlade2[cd][1]*newAllLEBlade2[cd][0], zLE2, rThLE2)
    slopeTE2 = mf.Slope(newAllTEBlade2[cd][2], newAllTEBlade2[cd][1]*newAllTEBlade2[cd][0], zTE2, rThTE2)    
    
    if periodicORaperiodic == 0:
        angleLE1[cd] = np.arctan(slopeLE1)
        angleLE2[cd] = angleLE1[cd]
    else:
        angleLE2[cd] = np.arctan(slopeLE2)
        
    if periodicORaperiodic == 0:
        angleTE1[cd] = np.arctan(slopeTE1)
        angleTE2[cd] = angleTE1[cd]
    else:
        angleTE2[cd] = np.arctan(slopeTE2) 

#%% Now convert the newBlades to cartesian coordinate and find the vertex of the offset points 
##Please note that PS is the high theta and SS is the low theta
offsetVertex1Cart = np.zeros([newNsection,4,3]) #LE,Mid,TE,Mid
offserVertex2Cart = np.zeros([newNsection,4,3])
offsetVertex1Cyl = np.zeros([newNsection,4,3]) #LE,Mid,TE,Mid
offsetVertex2Cyl = np.zeros([newNsection,4,3])
newBlade1PSCylM = np.zeros([newNsection,Nr+1,3]) #so i added the midPopints values here 
newBlade1SSCylM = np.zeros([newNsection,Nr+1,3])
newBlade2PSCylM = np.zeros([newNsection,Nr+1,3])
newBlade2SSCylM = np.zeros([newNsection,Nr+1,3])
mid1PS = np.zeros((newNsection, 3))
mid1SS = np.zeros((newNsection, 3))
mid2PS = np.zeros((newNsection, 3))
mid2SS = np.zeros((newNsection, 3))
for d in range(newNsection):
    newBlade1PSCart = np.array(mf.pol2cart(newBlade1PSCyl[d,:,0], newBlade1PSCyl[d,:,1], newBlade1PSCyl[d,:,2])).T
    newBlade1SSCart = np.array(mf.pol2cart(newBlade1SSCyl[d,:,0], newBlade1SSCyl[d,:,1], newBlade1SSCyl[d,:,2])).T
    newBlade2PSCart = np.array(mf.pol2cart(newBlade2PSCyl[d,:,0], newBlade2PSCyl[d,:,1], newBlade2PSCyl[d,:,2])).T
    newBlade2SSCart = np.array(mf.pol2cart(newBlade2SSCyl[d,:,0], newBlade2SSCyl[d,:,1], newBlade2SSCyl[d,:,2])).T
    midZ1Point = 0.5*(newBlade1PSCart[0,2]+ newBlade1PSCart[-1,2]) # This is the mid point of the axial chord
    midZ2Point = 0.5*(newBlade2PSCart[0,2]+ newBlade2PSCart[-1,2]) # This is the mid point of the axial chord
    xPS1Func = CubicSpline(newBlade1PSCart[:,2], newBlade1PSCart[:,0])
    yPS1Func = CubicSpline(newBlade1PSCart[:,2], newBlade1PSCart[:,1])
    xSS1Func = CubicSpline(newBlade1SSCart[:,2], newBlade1SSCart[:,0])
    ySS1Func = CubicSpline(newBlade1SSCart[:,2], newBlade1SSCart[:,1])
    leBlade1 = newBlade1PSCart[0] #This is point E
    teBlade1 = newBlade1PSCart[-1] #This is also point E
    midPSBlade1 = np.array([xPS1Func(midZ1Point), yPS1Func(midZ1Point), midZ1Point]) #This is the point C
    midPSBlade1Cyl = np.array(mf.cart2pol(midPSBlade1[0], midPSBlade1[1], midPSBlade1[2])).T
    newBlade1PSCylM[d] = insertPoint(newBlade1PSCyl[d], midPSBlade1Cyl)
    midSSBlade1 = np.array([xSS1Func(midZ1Point), ySS1Func(midZ1Point), midZ1Point]) #This is the point B
    midSSBlade1Cyl = np.array(mf.cart2pol(midSSBlade1[0], midSSBlade1[1], midSSBlade1[2])).T
    newBlade1SSCylM[d] = insertPoint(newBlade1SSCyl[d], midSSBlade1Cyl) 
    mid1PS[d] = midPSBlade1Cyl
    mid1SS[d] = midSSBlade1Cyl
    #Here i made an assumption that the offset at midpoint is at constant radius with the point close to the blade surface
    rPSBlade1 = np.sqrt(midPSBlade1[0]**2 + midPSBlade1[1]**2)
    rSSBlade1 = np.sqrt(midSSBlade1[0]**2 + midSSBlade1[1]**2)
    rBlade1 = 0.5*(rPSBlade1 + rSSBlade1)
    #Using the definition of arc length, I can determine the theta shift and I can determine the x and y value from that. Assuming the BL thickness is the arc length
    thPSBlade1 = delta/rBlade1 #s=r*theta
    thSSBlade1 = -delta/rBlade1 # Negative is added here to move in the opposite direction
    thOrigPSBlade1 = np.arctan2(midPSBlade1[1], midPSBlade1[0])
    thOrigSSBlade1 = np.arctan2(midSSBlade1[1], midSSBlade1[0])
    midPSOffset1 = np.array([rBlade1*np.cos(thPSBlade1 + thOrigPSBlade1), rBlade1*np.sin(thPSBlade1+thOrigPSBlade1), midZ1Point]) # This is the point D
    midSSOffset1 = np.array([rBlade1*np.cos(thSSBlade1 + thOrigSSBlade1), rBlade1*np.sin(thSSBlade1+thOrigSSBlade1 ), midZ1Point]) # This is the point A  
    leOffset1 = fq.getFvertex(midSSOffset1, midSSBlade1, midPSBlade1, midPSOffset1, leBlade1, meridCurve[d])  
    teOffset1 = fq.getFvertex(midSSOffset1, midSSBlade1, midPSBlade1, midPSOffset1, teBlade1, meridCurve[d])  
    offsetVertex1Cart[d,:] = np.vstack((leOffset1, midPSOffset1, teOffset1, midSSOffset1))
    leOffset1Cyl = np.array(mf.cart2pol(leOffset1[0], leOffset1[1], leOffset1[2])).T
    teOffset1Cyl = np.array(mf.cart2pol(teOffset1[0], teOffset1[1], teOffset1[2])).T
    midPSOffset1Cyl = np.array(mf.cart2pol(midPSOffset1[0], midPSOffset1[1], midPSOffset1[2])).T
    midSSOffset1Cyl = np.array(mf.cart2pol(midSSOffset1[0], midSSOffset1[1], midSSOffset1[2])).T
    offsetVertex1Cyl[d,:] = np.vstack((leOffset1Cyl, midPSOffset1Cyl, teOffset1Cyl, midSSOffset1Cyl))
    
    xPS2Func = CubicSpline(newBlade2PSCart[:,2], newBlade2PSCart[:,0])
    yPS2Func = CubicSpline(newBlade2PSCart[:,2], newBlade2PSCart[:,1])
    xSS2Func = CubicSpline(newBlade2SSCart[:,2], newBlade2SSCart[:,0])
    ySS2Func = CubicSpline(newBlade2SSCart[:,2], newBlade2SSCart[:,1])     
    leBlade2 = newBlade2PSCart[0] #This is point E
    teBlade2 = newBlade2PSCart[-1] #This is also point E    
    midPSBlade2 = np.array([xPS2Func(midZ2Point), yPS2Func(midZ2Point), midZ2Point]) #This is the point C
    midPSBlade2Cyl = np.array(mf.cart2pol(midPSBlade2[0], midPSBlade2[1], midPSBlade2[2])).T
    newBlade2PSCylM[d] = insertPoint(newBlade2PSCyl[d], midPSBlade2Cyl)    
    midSSBlade2 = np.array([xSS2Func(midZ2Point), ySS2Func(midZ2Point), midZ2Point]) #This is the point B
    midSSBlade2Cyl = np.array(mf.cart2pol(midSSBlade2[0], midSSBlade2[1], midSSBlade2[2])).T
    newBlade2SSCylM[d] = insertPoint(newBlade2SSCyl[d], midSSBlade2Cyl)  
    mid2PS[d] = midPSBlade2Cyl
    mid2SS[d] = midSSBlade2Cyl
    #Here i made an assumption that the offset at midpoint is at constant radius with the point close to the blade surface
    rPSBlade2 = np.sqrt(midPSBlade2[0]**2 + midPSBlade2[1]**2)
    rSSBlade2 = np.sqrt(midSSBlade2[0]**2 + midSSBlade2[1]**2)
    rBlade2 = 0.5*(rPSBlade2 + rSSBlade2)
    #Using the definition of arc length, I can determine the theta shift and I can determine the x and y value from that. Assuming the BL thickness is the arc length
    thPSBlade2 = delta/rBlade2
    thSSBlade2 = -delta/rBlade2
    thOrigPSBlade2 = np.arctan2(midPSBlade2[1], midPSBlade2[0])
    thOrigSSBlade2 = np.arctan2(midSSBlade2[1], midSSBlade2[0])    
    midPSOffset2 = np.array([rBlade2*np.cos(thPSBlade2 + thOrigPSBlade2), rBlade2*np.sin(thPSBlade2+thOrigPSBlade2), midZ2Point]) # This is the point D
    midSSOffset2 = np.array([rBlade2*np.cos(thSSBlade2 + thOrigSSBlade2), rBlade2*np.sin(thSSBlade2+thOrigSSBlade2 ), midZ2Point]) # This is the point A  
    leOffset2 = fq.getFvertex(midSSOffset2, midSSBlade2, midPSBlade2, midPSOffset2, leBlade2, meridCurve[d])  
    teOffset2 = fq.getFvertex(midSSOffset2, midSSBlade2, midPSBlade2, midPSOffset2, teBlade2, meridCurve[d])
    offserVertex2Cart[d,:] = np.vstack((leOffset2, midPSOffset2, teOffset2, midSSOffset2))
    leOffset2Cyl = np.array(mf.cart2pol(leOffset2[0], leOffset2[1], leOffset2[2])).T
    teOffset2Cyl = np.array(mf.cart2pol(teOffset2[0], teOffset2[1], teOffset2[2])).T
    midPSOffset2Cyl = np.array(mf.cart2pol(midPSOffset2[0], midPSOffset2[1], midPSOffset2[2])).T
    midSSOffset2Cyl = np.array(mf.cart2pol(midSSOffset2[0], midSSOffset2[1], midSSOffset2[2])).T
    offsetVertex2Cyl[d,:] = np.vstack((leOffset2Cyl, midPSOffset2Cyl, teOffset2Cyl, midSSOffset2Cyl))    

#%%  Change the coordinate system to Mprime 
blade1PSMprime = np.zeros([newNsection,bladeRes,4]) # (theta, r, z, mprime)
blade1SSMprime = np.zeros([newNsection,bladeRes,4])
blade2PSMprime = np.zeros([newNsection,bladeRes,4])
blade2SSMprime = np.zeros([newNsection,bladeRes,4])

for d in range(newNsection):
    blade1PSMprime[d,:,0:3] = mf.densify_curve_robust(newBlade1PSCylM[d], bladeRes)
    blade1SSMprime[d,:,0:3] = mf.densify_curve_robust(newBlade1SSCylM[d], bladeRes)
    blade2PSMprime[d,:,0:3] = mf.densify_curve_robust(newBlade2PSCylM[d], bladeRes)
    blade2SSMprime[d,:,0:3] = mf.densify_curve_robust(newBlade2SSCylM[d], bladeRes)  
    index = 0
    for e in range(bladeRes):   
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
        
hubLEIdx = np.searchsorted(tempHub[:,2], newAllLEBlade1[0][2])
hubTEIdx = np.searchsorted(tempHub[:,2], newAllTEBlade1[0][2])
casLEIdx = np.searchsorted(tempCas[:,2], newAllLEBlade1[newNsection-1][2])
casTEIdx = np.searchsorted(tempCas[:,2], newAllTEBlade1[newNsection-1][2])

newLE2D = np.column_stack((newAllLEBlade1[:,2], newAllLEBlade1[:,1]))
newTE2D = np.column_stack((newAllTEBlade1[:,2], newAllTEBlade1[:,1]))

newLE2D2 = np.column_stack((newAllLEBlade2[:,2], newAllLEBlade2[:,1]))
newTE2D2 = np.column_stack((newAllTEBlade2[:,2], newAllTEBlade2[:,1]))

upHub = np.vstack((tempHub[0:hubLEIdx][:,[2,0]],newLE2D[0]))#hubLE))
upHFunc = interp1d(upHub[:,0], upHub[:,1])
upHub = np.column_stack((np.linspace(upHub[0][0], upHub[::-1][0][0],res), upHFunc(np.linspace(upHub[0][0], upHub[::-1][0][0],res))))
dwHub = np.vstack((newTE2D[0], tempHub[hubTEIdx:len(tempHub)][:,[2,0]]))
dwHFunc = interp1d(dwHub[:,0], dwHub[:,1])
dwHub = np.column_stack((np.linspace(dwHub[0][0], dwHub[::-1][0][0],res), dwHFunc(np.linspace(dwHub[0][0], dwHub[::-1][0][0],res))))
upCas = np.vstack((tempCas[0:casLEIdx][:,[2,0]],newLE2D[-1]))#casLE))
upCFunc = interp1d(upCas[:,0], upCas[:,1])
upCas = np.column_stack((np.linspace(upCas[0][0], upCas[::-1][0][0],res), upCFunc(np.linspace(upCas[0][0], upCas[::-1][0][0],res))))
dwCas = np.vstack((newTE2D[-1], tempCas[casTEIdx:len(tempCas)][:,[2,0]]))
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
#%% Convert the offset vertices to mPrime 
## This is not a straight forward approach as I have to ensure that I make reference to the mprime definition of the blade data
## To do this, will use the interpolation to get the m prime value of the LE offset point then work from there 
offsetVertex1Mprime = np.zeros([newNsection, 4, 4])
offsetVertex2Mprime = np.zeros([newNsection, 4, 4])
offsetVertex1MprimeP = np.zeros([newNsection, 3, 4])
offsetVertex1MprimeS = np.zeros([newNsection, 3, 4])
offsetVertex2MprimeP = np.zeros([newNsection, 3, 4])
offsetVertex2MprimeS = np.zeros([newNsection, 3, 4])
count = 0
for g in range(newNsection):
    combinedMprime = np.concatenate((upstreamMprime[g][:-1], blade1SSMprime[g], dwstreamMprime[g][1:]))
    combinedMprime2 = np.concatenate((upstreamMprime[g][:-1], blade2PSMprime[g], dwstreamMprime[g][1:]))
    deltaFunc = CubicSpline(combinedMprime[:,2], combinedMprime[:,3])

    offsetVertex1MprimeP[g,:,0:3] = np.array([offsetVertex1Cyl[g][0],offsetVertex1Cyl[g][1],offsetVertex1Cyl[g][2]])
    offsetVertex1MprimeS[g,:,0:3] = np.array([offsetVertex1Cyl[g][0],offsetVertex1Cyl[g][3],offsetVertex1Cyl[g][2]])
    offsetVertex2MprimeP[g,:,0:3] = np.array([offsetVertex2Cyl[g][0],offsetVertex2Cyl[g][1],offsetVertex2Cyl[g][2]])
    offsetVertex2MprimeS[g,:,0:3] = np.array([offsetVertex2Cyl[g][0],offsetVertex2Cyl[g][3],offsetVertex2Cyl[g][2]])    
    for gg in range(3):
        if gg == 0:
            offsetVertex1Mprime[g,:,0:3] = offsetVertex1Cyl[g]
            offsetVertex2Mprime[g,:,0:3] = offsetVertex2Cyl[g]
            offsetVertex1Mprime[g,gg,3] = deltaFunc(offsetVertex1Cyl[g,0,2]) #to initialize the mprime value, I have to figure out where it lies on the 
            offsetVertex2Mprime[g,gg,3] = deltaFunc(offsetVertex2Cyl[g,0,2]) #meriodional curve in mprime
            offsetVertex1MprimeP[g,gg,3] = deltaFunc(offsetVertex1Cyl[g,0,2])
            offsetVertex1MprimeS[g,gg,3] = deltaFunc(offsetVertex1Cyl[g,0,2])
            offsetVertex2MprimeP[g,gg,3] = deltaFunc(offsetVertex2Cyl[g,0,2])
            offsetVertex2MprimeS[g,gg,3] = deltaFunc(offsetVertex2Cyl[g,0,2])
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
    
            mPrev1 = offsetVertex1MprimeS[g,:,3][gg-1]
            rPrev1 = offsetVertex1MprimeS[g,:,1][gg-1]
            rCurr1 = offsetVertex1MprimeS[g,:,1][gg]
            zPrev1 = offsetVertex1MprimeS[g,:,2][gg-1]
            zCurr1 = offsetVertex1MprimeS[g,:,2][gg]
            offsetVertex1MprimeS[g,gg,3] = mPrev1 + ((2/(rCurr1 + rPrev1)) * np.sqrt((rCurr1-rPrev1)**2 + (zCurr1-zPrev1)**2))
            mPrev2 = offsetVertex2MprimeS[g,:,3][gg-1]
            rPrev2 = offsetVertex2MprimeS[g,:,1][gg-1]
            rCurr2 = offsetVertex2MprimeS[g,:,1][gg]
            zPrev2 = offsetVertex2MprimeS[g,:,2][gg-1]
            zCurr2 = offsetVertex2MprimeS[g,:,2][gg]
            offsetVertex2MprimeS[g,gg,3] = mPrev2 + ((2/(rCurr2 + rPrev2)) * np.sqrt((rCurr2-rPrev2)**2 + (zCurr2-zPrev2)**2)) 
        if gg == 1:
            offsetVertex1Mprime[g,gg,:] = offsetVertex1MprimeP[g,gg,:]  
            offsetVertex2Mprime[g,gg,:] = offsetVertex2MprimeP[g,gg,:]  
        elif gg == 2:
            offsetVertex1Mprime[g,gg,:] = offsetVertex1MprimeS[g,gg-1,:]  
            offsetVertex2Mprime[g,gg,:] = offsetVertex2MprimeS[g,gg-1,:] 
        offsetVertex1Mprime[g,3,:] = offsetVertex1MprimeP[g,gg,:]  
        offsetVertex2Mprime[g,3,:] = offsetVertex2MprimeP[g,gg,:]    
        # count +=1  

#%% Now working in 2D mprime-theta
blade1PS2D = np.zeros([newNsection,bladeRes,2]) #mprime theta
blade1SS2D = np.zeros([newNsection,bladeRes,2])
blade2PS2D = np.zeros([newNsection,bladeRes,2])
blade2SS2D = np.zeros([newNsection,bladeRes,2])
offsetBlade12D = np.zeros([newNsection, 4, 2])
offsetBlade22D = np.zeros([newNsection, 4, 2])
upstream2D = np.zeros([newNsection,res,2])
dwstream2D = np.zeros([newNsection,res,2])
allLEBlade12D = np.zeros([newNsection,2])
allLEBlade22D = np.zeros([newNsection,2])
allTEBlade12D = np.zeros([newNsection,2])
allTEBlade22D = np.zeros([newNsection,2])

for h in range(newNsection):
    blade1PS2D[h,:,:] = np.column_stack((blade1PSMprime[h,:,3], blade1PSMprime[h,:,0]))
    blade1SS2D[h,:,:] = np.column_stack((blade1SSMprime[h,:,3], blade1SSMprime[h,:,0]))
    blade2PS2D[h,:,:] = np.column_stack((blade2PSMprime[h,:,3], blade2PSMprime[h,:,0]))
    blade2SS2D[h,:,:] = np.column_stack((blade2SSMprime[h,:,3], blade2SSMprime[h,:,0]))   
    upstream2D[h,:,:] = np.column_stack((upstreamMprime[h,:,3], upstreamMprime[h,:,0]))
    dwstream2D[h,:,:] = np.column_stack((dwstreamMprime[h,:,3], dwstreamMprime[h,:,0]))
    offsetBlade12D[h,:,:] = np.column_stack((offsetVertex1Mprime[h,:,3], offsetVertex1Mprime[h,:,0]))
    offsetBlade22D[h,:,:] = np.column_stack((offsetVertex2Mprime[h,:,3], offsetVertex2Mprime[h,:,0]))
    allLEBlade12D[h] = blade1PS2D[h][0]
    allLEBlade22D[h] = blade2SS2D[h][0]
    allTEBlade12D[h] = blade1PS2D[h][::-1][0]
    allTEBlade22D[h] = blade2SS2D[h][::-1][0] 


# JD: inserted code from previous "part 2" here that figures out the up/downstream extension directions

nl = 2
LECurveRot = np.zeros([newNsection,passageRes,2]) #Section of the ellipse that defines the LE curve
TECurveRot = np.zeros([newNsection,passageRes,2]) #section of the ellipse that defines the TE curve
offsetSplinedBlade12D = np.zeros([newNsection, bladeRes, 2])
offsetSplinedBlade22D = np.zeros([newNsection, bladeRes, 2])


#%% Solving the differential equation
'''
We came up with this equation that determines the circunferential shift for both inlet and outlet. It factors in the radius change and reduces the exaggeration of the lean and twist of the 
blade. 
'''
def dy_dx(x, M, A):
    terms = 2 * (np.arctan(x / M) - A) * (1 / (1 + (x / M)**2)) * (1 / M)
    return np.sum(terms)


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


midchordPS1 = np.zeros([newNsection,2])
midchordSS1 = np.zeros([newNsection,2])
midchordPS2 = np.zeros([newNsection,2])
midchordSS2 = np.zeros([newNsection,2])

for i in range(newNsection):
    blade1 = np.concatenate((blade1PS2D[i][:-1], blade1SS2D[i][::-1])) #Joing both surfaces to create a closed loop
    distLE1 = mf.dist2D(offsetBlade12D[i][0,0], offsetBlade12D[i][0,1], allLEBlade12D[i][0], allLEBlade12D[i][1]) #compute the distance of offsetLE to bladeLE
    distTE1 = mf.dist2D(offsetBlade12D[i][-1,0], offsetBlade12D[i][-1,1], allTEBlade12D[i][0], allTEBlade12D[i][1]) #compute the distance between offsetTE to bladeTE
    midLine1 = np.vstack((offsetBlade12D[i][1], offsetBlade12D[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
    ssInterX, psInterX = mf.TwoLinesIntersect(midLine1, blade1) #Find the intersections of the line on the blade curve at both pressure and suction side 
    if ssInterX[1] > psInterX[1]:
        ssInterX, psInterX = psInterX, ssInterX
    # ssInterX[0] = psInterX[0] = offsetBlade12D[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
    distSS1 = mf.dist2D(ssInterX[0], ssInterX[1], offsetBlade12D[i][2,0], offsetBlade12D[i][2,1])
    distPS1 = mf.dist2D(psInterX[0], psInterX[1], offsetBlade12D[i][1,0], offsetBlade12D[i][1,1])
    dist1 = np.array([distLE1, distSS1, distTE1])
    
    #Do the same for blade2
    blade2 = np.concatenate((blade2PS2D[i][:-1], blade2SS2D[i][::-1])) #Joing both surfaces to create a closed loop
    distLE2 = mf.dist2D(offsetBlade22D[i][0,0], offsetBlade22D[i][0,1], allLEBlade22D[i][0], allLEBlade22D[i][1]) #compute the distance of offsetLE to bladeLE
    distTE2 = mf.dist2D(offsetBlade22D[i][-1,0], offsetBlade22D[i][-1,1], allTEBlade22D[i][0], allTEBlade22D[i][1]) #compute the distance between offsetTE to bladeTE
    midLine2 = np.vstack((offsetBlade22D[i][1], offsetBlade22D[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
    ss2InterX, ps2InterX = mf.TwoLinesIntersect(midLine2, blade2) #Find the intersections of the line on the blade curve at both pressure and suction side 
    if ss2InterX[1] > ps2InterX[1]:
        ss2InterX, ps2InterX = ps2InterX, ss2InterX
    # ss2InterX[0] = ps2InterX[0] = offsetBlade22D[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
    distSS2 = mf.dist2D(ss2InterX[0], ss2InterX[1], offsetBlade22D[i][2,0], offsetBlade22D[i][2,1])
    distPS2 = mf.dist2D(ps2InterX[0], ps2InterX[1], offsetBlade22D[i][1,0], offsetBlade22D[i][1,1])
    dist2 = np.array([distLE2, distPS2, distTE2])
    
    center = mf.MidPts(np.vstack((ssInterX, ps2InterX))) #this is the mid point between the midpoints on the suction side of blade1 and pressure side of blade2
    #curve1 = createOffsetCurve(blade1SS2D[i], offsetBlade12D[i], ssInterX, dist1, center, 'SS')        
    #offsetSplinedBlade12D[i] = densifyCurve(curve1, bladeRes, 'both')
    midchordPS1[i] = psInterX
    midchordSS1[i] = ssInterX
    
    #curve2 = createOffsetCurve(blade2PS2D[i], offsetBlade22D[i], ps2InterX, dist2, center, 'PS')
    #offsetSplinedBlade22D[i] = densifyCurve(curve2, bladeRes, 'both')  
    midchordPS2[i] = ps2InterX
    midchordSS2[i] = ss2InterX   

# Loop to initialize the upstream/downstream extension directions
for m in range(newNsection):
    lePtBlade1 = allLEBlade12D[m] #LE of the high blade
    lePtBlade2 = allLEBlade22D[m] #LE of the low blade

    midPt1 = angle_bisector_line(midchordSS1[m],lePtBlade1, midchordPS1[m]) #Compute the angle between SS,LE,PS
    leLine1Slope = compute_bisector_slope(midchordSS1[m],lePtBlade1, midchordPS1[m]) #determine the slope of the bisector
    tanLine1LEInterX = lePtBlade1[1] - leLine1Slope*lePtBlade1[0]  #Trying to get the equation of the line
    tanLine1LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*leLine1Slope + tanLine1LEInterX], lePtBlade1)) 
    lePtCam1 = offsetBlade12D[m][0]
    upLine1LEInterX = lePtCam1[1] - np.tan(angleLE1[m])*lePtCam1[0] 
    upLine1LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*np.tan(angleLE1[m]) + upLine1LEInterX], lePtCam1, lePtBlade1))

    midPt2 = angle_bisector_line(midchordSS2[m],lePtBlade2, midchordPS2[m])
    leLine2Slope = compute_bisector_slope(midchordSS2[m],lePtBlade2, midchordPS2[m])
    tanLine2LEInterX = lePtBlade2[1] - leLine2Slope*lePtBlade2[0] 
    tanLine2LE = np.vstack(([upstreamMprime[m,0,3], upstreamMprime[m,0,3]*leLine2Slope + tanLine2LEInterX],lePtBlade2)) 
    lePtCam2 = offsetBlade22D[m][0]
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
    tePtBlade1 = allTEBlade12D[m]
    tePtBlade2 = allTEBlade22D[m]
    
    midPtTE1 = angle_bisector_line(midchordSS1[m],tePtBlade1, midchordPS1[m])
    teLine1Slope = compute_bisector_slope(midchordSS1[m],tePtBlade1, midchordPS1[m])
    tanLine1TEInterX = tePtBlade1[1] - teLine1Slope*tePtBlade1[0] 
    tanLine1TE = np.vstack(([dwstreamMprime[m,-1,3], dwstreamMprime[m,-1,3]*teLine1Slope + tanLine1TEInterX], tePtBlade1))
    tePtCam1 = offsetBlade12D[m][-1]
    dwLine1TEInterX = tePtCam1[1] - np.tan(angleTE1[m])*tePtCam1[0]
    dwLine1TE = np.vstack(([dwstreamMprime[m,-1,3], dwstreamMprime[m,-1,3]*np.tan(angleTE1[m]) + dwLine1TEInterX], tePtCam1, tePtBlade1))
  
    midPtTE2 = angle_bisector_line(midchordSS2[m],tePtBlade2, midchordPS2[m])
    teLine2Slope = compute_bisector_slope(midchordSS2[m],tePtBlade2, midchordPS2[m])
    tanLine2TEInterX = tePtBlade2[1] - teLine2Slope*tePtBlade2[0] 
    tanLine2TE = np.vstack(([dwstreamMprime[m,-1,3], dwstreamMprime[m,-1,3]*teLine2Slope + tanLine2TEInterX], tePtBlade2))
    tePtCam2 = offsetBlade22D[m][-1]
    dwLine2TEInterX = tePtCam2[1] - np.tan(angleTE2[m])*tePtCam2[0]
    dwLine2TE = np.vstack(([dwstreamMprime[m,-1,3], dwstreamMprime[m,-1,3]*np.tan(angleTE2[m]) + dwLine2TEInterX], tePtCam2, tePtBlade2))

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
    offsetBlade1LEpt = offsetBlade12D[m][0]
    offsetBlade2LEpt = offsetBlade22D[m][0]
    offsetBlade1TEpt = offsetBlade12D[m][-1]
    offsetBlade2TEpt = offsetBlade22D[m][-1]

    lePtBlade1 = allLEBlade12D[m]
    lePtBlade2 = allLEBlade22D[m]
    tePtBlade1 = allTEBlade12D[m]
    tePtBlade2 = allTEBlade22D[m]

    farPtBlade1LE = upstreamCamber1[m][-3]
    farPtBlade2LE = upstreamCamber2[m][-3]
    farPtBlade1TE = downstreamCamber1[m][2]
    farPtBlade2TE = downstreamCamber2[m][2]

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

    #Please note at this point that the part of the code profile that lies in the domain is the SS for blade 1 and PS for blade2.
    i = m  # this was a separate loop from here down with different indexing, so to avoid changing all the code for now I just set i=m
    blade1 = np.concatenate((blade1PS2D[i][:-1], blade1SS2D[i][::-1])) #Joing both surfaces to create a closed loop
    distLE1 = mf.dist2D(offsetBlade12D[i][0,0], offsetBlade12D[i][0,1], allLEBlade12D[i][0], allLEBlade12D[i][1]) #compute the distance of offsetLE to bladeLE
    distTE1 = mf.dist2D(offsetBlade12D[i][-1,0], offsetBlade12D[i][-1,1], allTEBlade12D[i][0], allTEBlade12D[i][1]) #compute the distance between offsetTE to bladeTE
    midLine1 = np.vstack((offsetBlade12D[i][1], offsetBlade12D[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
    ssInterX, psInterX = mf.TwoLinesIntersect(midLine1, blade1) #Find the intersections of the line on the blade curve at both pressure and suction side 
    if ssInterX[1] > psInterX[1]:
        ssInterX, psInterX = psInterX, ssInterX
    # ssInterX[0] = psInterX[0] = offsetBlade12D[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
    distSS1 = mf.dist2D(ssInterX[0], ssInterX[1], offsetBlade12D[i][2,0], offsetBlade12D[i][2,1])
    distPS1 = mf.dist2D(psInterX[0], psInterX[1], offsetBlade12D[i][1,0], offsetBlade12D[i][1,1])
    dist1 = np.array([distLE1, distSS1, distTE1])
    
    #Do the same for blade2
    blade2 = np.concatenate((blade2PS2D[i][:-1], blade2SS2D[i][::-1])) #Joing both surfaces to create a closed loop
    distLE2 = mf.dist2D(offsetBlade22D[i][0,0], offsetBlade22D[i][0,1], allLEBlade22D[i][0], allLEBlade22D[i][1]) #compute the distance of offsetLE to bladeLE
    distTE2 = mf.dist2D(offsetBlade22D[i][-1,0], offsetBlade22D[i][-1,1], allTEBlade22D[i][0], allTEBlade22D[i][1]) #compute the distance between offsetTE to bladeTE
    midLine2 = np.vstack((offsetBlade22D[i][1], offsetBlade22D[i][2])) #Connect the line joing the two points at midPoint on offset curve (curve not yet defined)
    ss2InterX, ps2InterX = mf.TwoLinesIntersect(midLine2, blade2) #Find the intersections of the line on the blade curve at both pressure and suction side 
    if ss2InterX[1] > ps2InterX[1]:
        ss2InterX, ps2InterX = ps2InterX, ss2InterX
    # ss2InterX[0] = ps2InterX[0] = offsetBlade22D[i][1,0] #I have to ensure the mprime values are exactly the same, no precision errors 
    distSS2 = mf.dist2D(ss2InterX[0], ss2InterX[1], offsetBlade22D[i][2,0], offsetBlade22D[i][2,1])
    distPS2 = mf.dist2D(ps2InterX[0], ps2InterX[1], offsetBlade22D[i][1,0], offsetBlade22D[i][1,1])
    dist2 = np.array([distLE2, distPS2, distTE2])
    
    center = mf.MidPts(np.vstack((ssInterX, ps2InterX))) #this is the mid point between the midpoints on the suction side of blade1 and pressure side of blade2

    # This now uses quadratic Bezier curves
    curve1 = createOffsetCurve(bladePr=blade1SS2D[i], offsetPt=offsetBlade12D[i], interX=ssInterX, dist=dist1, center=center, LEslope=offsetSlopeLE1, TEslope=offsetSlopeTE1, side='hi')
    
    offsetSplinedBlade12D[i] = densifyCurve(curve1, bladeRes, 'both')
    midchordPS1[i] = psInterX
    midchordSS1[i] = ssInterX

    # This now uses quadratic Bezier curves
    curve2 = createOffsetCurve(bladePr=blade2PS2D[i], offsetPt=offsetBlade22D[i], interX=ps2InterX, dist=dist2, center=center, LEslope=offsetSlopeLE2, TEslope=offsetSlopeTE2, side='lo')
    
    offsetSplinedBlade22D[i] = densifyCurve(curve2, bladeRes, 'both')  
    midchordPS2[i] = ps2InterX
    midchordSS2[i] = ss2InterX   

    blade1Func = CubicSpline(blade1SS2D[i][:,0], blade1SS2D[i][:,1]) #Interpolation function for high theta blade
    blade2Func = CubicSpline(blade2PS2D[i][:,0], blade2PS2D[i][:,1]) #Interpolation function for low theta blade
    offset1Func = CubicSpline(offsetSplinedBlade12D[i][:,0], offsetSplinedBlade12D[i][:,1])
    offset2Func = CubicSpline(offsetSplinedBlade22D[i][:,0], offsetSplinedBlade22D[i][:,1])

    # 6) get arclength mapping for hub and casing sections
    if i == 0:
        #H is high, L is low, 1 is upstream and 2 is downstream. I am trying to replace naming convention from here
        mBladeH1 = cosineSpace(bladeRes+1, blade1SS2D[i][0,0], ssInterX[0]) #upstream portion of the high theta blade
        mBladeH2 = cosineSpace(bladeRes+1, ssInterX[0], blade1SS2D[i][-1,0]) #downstream portion of the high theta blade
        mBladeL1 = cosineSpace(bladeRes+1, blade2PS2D[i][0,0], ps2InterX[0]) #upstream portion of the low theta blade
        mBladeL2 = cosineSpace(bladeRes+1, ps2InterX[0], blade2PS2D[i][-1,0]) #downstream portion of the low theta blade 
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
        # JD: Should replace the dist1[0] with dist1.max(), as currently this ASSUMES the
        # m'-theta distance at the LE is the largest, which may not always be the case!
        offsetL1Pt = pointAtDistLoop(lowHub1MidPt, lowHub1Slopes, np.full(bladeRes,1.5*dist1[0]), 'PS')[:-1] # This is offseting the point on the blade surface in the normal direction at some made up dist
        offsetL2Pt = pointAtDistLoop(lowHub2MidPt, lowHub2Slopes, np.full(bladeRes,1.5*dist1[0]), 'PS')[:-1]
        offsetH1Pt = pointAtDistLoop(highHub1MidPt, highHub1Slopes, np.full(bladeRes,1.5*dist1[0]), 'SS')[:-1]
        offsetH2Pt = pointAtDistLoop(highHub2MidPt, highHub2Slopes, np.full(bladeRes,1.5*dist1[0]), 'SS')[:-1]        
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
        highHub1OffsetPt[-1] = offsetBlade12D[i][2] # insert the mid point of the offset curve 
        highHub2OffsetPt[0] = offsetBlade12D[i][2] #insert mid point of the offset curve as the last point
        highHub2OffsetPt[-1] = curve1[-1]
        
        lowHub1OffsetPt[0] = curve2[0]
        lowHub1OffsetPt[-1] = offsetBlade22D[i][1]
        lowHub2OffsetPt[0] = offsetBlade22D[i][1]  
        lowHub2OffsetPt[-1] = curve2[-1]
        
        #Now find the angle to determine the points to ignore
        
        lowHub1OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][0], bladeL1Pt[0], bladeL1Pt[1]) #Angle at LE
        lowHub1OffsetPtAngle2 =  mf.find_angle(offsetBlade22D[i][1], bladeL1Pt[-1], bladeL1Pt[-2]) #Angle at Mid
        lowHub2OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][-1], bladeL2Pt[-1], bladeL2Pt[-2]) #Angle at TE
        lowHub2OffsetPtAngle2 =  mf.find_angle(offsetBlade22D[i][1], bladeL2Pt[0], bladeL2Pt[1]) #Angle at Mid
        
        highHub1OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][0], bladeH1Pt[0], bladeH1Pt[1]) #Angle at LE
        highHub1OffsetPtAngle2 = mf.find_angle(offsetBlade12D[i][2], bladeH1Pt[-1], bladeH1Pt[-2]) #Angle at Mid
        highHub2OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][-1], bladeH2Pt[-1], bladeH2Pt[-2]) #Angle at TE
        highHub2OffsetPtAngle2 = mf.find_angle(offsetBlade12D[i][2], bladeH2Pt[0], bladeH2Pt[1]) #Angle at Mid    
        
        #Now decide on which offset to truncate points from based on angle less than 90 degrees
        #This metric I am using will change later one but choose based on 4% for now!      
        percentLowH1 = np.array([(lowHub1OffsetPtAngle1 < 90),(lowHub1OffsetPtAngle2 < 90)])*percentVal
        lowHub1 = curveFrac(bladeL1Pt, offsetOrigL1Pt, lowHub1OffsetPt, percentLowH1).T
        lowHub1[0,1] = 0.0
        lowHub1 = cutArcLenMaps(lowHub1, lower_bound=percentValNonCut, upper_bound=1.0)
        percentLowH2 = np.array([(lowHub2OffsetPtAngle1 < 90),(lowHub2OffsetPtAngle2 < 90)])*percentVal
        lowHub2 = curveFrac(bladeL2Pt, offsetOrigL2Pt, lowHub2OffsetPt, percentLowH2).T
        lowHub2[0,1] = 0.0
        lowHub2 = cutArcLenMaps(lowHub2, lower_bound=0.0, upper_bound=1.0-percentValNonCut)
        percentHighH1 = np.array([(highHub1OffsetPtAngle1 < 90),(highHub1OffsetPtAngle2 < 90)])*percentVal
        highHub1 = curveFrac(bladeH1Pt, offsetOrigH1Pt, highHub1OffsetPt, percentHighH1).T
        highHub1[0,1] = 0.0
        highHub1 = cutArcLenMaps(highHub1, lower_bound=percentValNonCut, upper_bound=1.0)
        percentHighH2 = np.array([(highHub2OffsetPtAngle1 < 90),(highHub2OffsetPtAngle2 < 90)])*percentVal
        highHub2 = curveFrac(bladeH2Pt, offsetOrigH2Pt, highHub2OffsetPt, percentHighH2).T
        highHub2[0,1] = 0.0
        highHub2 = cutArcLenMaps(highHub2, lower_bound=0.0, upper_bound=1.0-percentValNonCut)
       
    elif i == newNsection - 1:
        mBladeH1 = cosineSpace(bladeRes+1, blade1SS2D[i][0,0], ssInterX[0]) #upstream portion of the high theta blade
        mBladeH2 = cosineSpace(bladeRes+1, ssInterX[0], blade1SS2D[i][-1,0]) #downstream portion of the high theta blade
        mBladeL1 = cosineSpace(bladeRes+1, blade2PS2D[i][0,0], ps2InterX[0]) #upstream portion of the low theta blade
        mBladeL2 = cosineSpace(bladeRes+1, ps2InterX[0], blade2PS2D[i][-1,0]) #downstream portion of the low theta blade 
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
        # JD: Should replace the dist1[0] with dist1.max(), as currently this ASSUMES the
        # m'-theta distance at the LE is the largest, which may not always be the case!
        offsetL1Pt = pointAtDistLoop(lowCas1MidPt, lowCas1Slopes, np.full(bladeRes,1.5*dist1[0]), 'PS')[:-1] # This is offseting the point on the blade surface in the normal direction at some made up dist
        offsetL2Pt = pointAtDistLoop(lowCas2MidPt, lowCas2Slopes, np.full(bladeRes,1.5*dist1[0]), 'PS')[:-1]
        offsetH1Pt = pointAtDistLoop(highCas1MidPt, highCas1Slopes, np.full(bladeRes,1.5*dist1[0]), 'SS')[:-1]
        offsetH2Pt = pointAtDistLoop(highCas2MidPt, highCas2Slopes, np.full(bladeRes,1.5*dist1[0]), 'SS')[:-1]        
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
        highCas1OffsetPt[-1] = offsetBlade12D[i][2] # insert the mid point of the offset curve 
        highCas2OffsetPt[0] = offsetBlade12D[i][2] #insert mid point of the offset curve as the last point
        highCas2OffsetPt[-1] = curve1[-1]
        
        lowCas1OffsetPt[0] = curve2[0]
        lowCas1OffsetPt[-1] = offsetBlade22D[i][1]
        lowCas2OffsetPt[0] = offsetBlade22D[i][1]  
        lowCas2OffsetPt[-1] = curve2[-1]
        
        #Now find the angle to determine the points to ignore
        
        lowCas1OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][0], bladeL1Pt[0], bladeL1Pt[1]) #Angle at LE
        lowCas1OffsetPtAngle2 =  mf.find_angle(offsetBlade22D[i][1], bladeL1Pt[-1], bladeL1Pt[-2]) #Angle at Mid
        lowCas2OffsetPtAngle1 =  mf.find_angle(offsetSplinedBlade22D[i][-1], bladeL2Pt[-1], bladeL2Pt[-2]) #Angle at TE
        lowCas2OffsetPtAngle2 =  mf.find_angle(offsetBlade22D[i][1], bladeL2Pt[0], bladeL2Pt[1]) #Angle at Mid

        highCas1OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][0], bladeH1Pt[0], bladeH1Pt[1]) #Angle at LE
        highCas1OffsetPtAngle2 = mf.find_angle(offsetBlade12D[i][2], bladeH1Pt[-1], bladeH1Pt[-2]) #Angle at Mid
        highCas2OffsetPtAngle1 = mf.find_angle(offsetSplinedBlade12D[i][-1], bladeH2Pt[-1], bladeH2Pt[-2]) #Angle at TE
        highCas2OffsetPtAngle2 = mf.find_angle(offsetBlade12D[i][2], bladeH2Pt[0], bladeH2Pt[1]) #Angle at Mid   
        
        percentLowH1 = np.array([(lowCas1OffsetPtAngle1 < 90),(lowCas1OffsetPtAngle2 < 90)])*percentVal
        lowCas1 = curveFrac(bladeL1Pt, offsetOrigL1Pt, lowCas1OffsetPt, percentLowH1).T
        lowCas1[0,1] = 0.0
        lowCas1 = cutArcLenMaps(lowCas1, lower_bound=percentValNonCut, upper_bound=1.0)
        percentLowH2 = np.array([(lowCas2OffsetPtAngle1 < 90),(lowCas2OffsetPtAngle2 < 90)])*percentVal
        lowCas2 = curveFrac(bladeL2Pt, offsetOrigL2Pt, lowCas2OffsetPt, percentLowH2).T
        lowCas2[0,1] = 0.0
        lowCas2 = cutArcLenMaps(lowCas2, lower_bound=0.0, upper_bound=1.0-percentValNonCut)
        percentHighH1 = np.array([(highCas1OffsetPtAngle1 < 90),(highCas1OffsetPtAngle2 < 90)])*percentVal
        highCas1 = curveFrac(bladeH1Pt, offsetOrigH1Pt, highCas1OffsetPt, percentHighH1).T
        highCas1[0,1] = 0.0
        highCas1 = cutArcLenMaps(highCas1, lower_bound=percentValNonCut, upper_bound=1.0)
        percentHighH2 = np.array([(highCas2OffsetPtAngle1 < 90),(highCas2OffsetPtAngle2 < 90)])*percentVal
        highCas2 = curveFrac(bladeH2Pt, offsetOrigH2Pt, highCas2OffsetPt, percentHighH2).T       
        highCas2[0,1] = 0.0 
        highCas2 = cutArcLenMaps(highCas2, lower_bound=0.0, upper_bound=1.0-percentValNonCut)

#%%
#np.savetxt(dataPath + '/lowHub1.txt', lowHub1, delimiter=',')
#np.savetxt(dataPath + '/lowHub2.txt', lowHub2, delimiter=',')
#np.savetxt(dataPath + '/lowCas1.txt', lowCas1, delimiter=',')
#np.savetxt(dataPath + '/lowCas2.txt', lowCas2, delimiter=',')

#np.savetxt(dataPath + '/highHub1.txt', highHub1, delimiter=',')
#np.savetxt(dataPath + '/highHub2.txt', highHub2, delimiter=',')
#np.savetxt(dataPath + '/highCas1.txt', highCas1, delimiter=',')
#np.savetxt(dataPath + '/highCas2.txt', highCas2, delimiter=',')

#%% Now I have to convert back to Cylindrical coordinate 
#Basically I combined the upstream and bladesection and downstream meridional profiles together. 
#First I have to get off mprime theta to z theta
LECurvePol = np.zeros([newNsection,passageRes,2]) # z, theta
TECurvePol = np.zeros([newNsection,passageRes,2])
upstreamCurve1Pol = np.zeros([newNsection,res,2])
upstreamCurve2Pol = np.zeros([newNsection,res,2])
downstreamCurve1Pol = np.zeros([newNsection,res,2])
downstreamCurve2Pol = np.zeros([newNsection,res,2])
#tangentLine1LEPol = np.zeros([newNsection,nl,2]) #line tangent to blade1 at the LE
#tangentLine2LEPol = np.zeros([newNsection,nl,2])  #line tangent to blade2 at the LE
#tangentLine1TEPol = np.zeros([newNsection,nl,2]) #line tangent to blade1 at the TE
#tangentLine2TEPol = np.zeros([newNsection,nl,2])  #line tangent to blade2 at the TE
#bisectorLE1Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at LE for blade1
#bisectorLE2Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at LE for blade2
#bisectorTE1Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at TE for blade1
#bisectorTE2Pol = np.zeros([newNsection,nl,2]) #line that bisect angle between tangent line and horizontal line at TE for blade2

upstreamExtnCamber1Pol = np.zeros([newNsection,nl+1,2]) #These are not used for anything other than visual representation
upstreamExtnCamber2Pol = np.zeros([newNsection,nl+1,2])
downstreamExtnCamber1Pol = np.zeros([newNsection,nl+1,2])
downstreamExtnCamber2Pol = np.zeros([newNsection,nl+1,2])

offsetBlade1Pol = np.zeros([newNsection,bladeRes,2]) #offset blade High
offsetBlade2Pol = np.zeros([newNsection,bladeRes,2]) #offset blade low

for p in range(newNsection):
    combinedBladeMprime = np.concatenate((upstreamMprime[p][:-1], blade1PSMprime[p], dwstreamMprime[p][1:])) 
    combinedMprime = np.concatenate((upstreamMprime[p][:-1], offsetVertex1MprimeS[p], dwstreamMprime[p][1:])) 
    func = CubicSpline((combinedMprime[:,3]), combinedMprime[:,2])#fill_value='extrapolate')
    midBladeFunc = CubicSpline((combinedBladeMprime[:,3]), combinedBladeMprime[:,2])
    LECurvePol[p,:,0] = func(LECurveRot[p,:,0]) 
    LECurvePol[p,:,1] = LECurveRot[p,:,1]
    TECurvePol[p,:,0] = func(TECurveRot[p,:,0]) 
    TECurvePol[p,:,1] = TECurveRot[p,:,1]  
    upstreamCurve1Pol[p,:,0] = func(upstreamCamber1[p,:,0])
    upstreamCurve1Pol[p,-1,0] = blade1SSMprime[p][0,2]
    upstreamCurve1Pol[p,:,1] = upstreamCamber1[p,:,1]
    upstreamCurve2Pol[p,:,0] = func(upstreamCamber2[p,:,0])
    upstreamCurve2Pol[p,-1,0] = blade2PSMprime[p][0,2]
    upstreamCurve2Pol[p,:,1] = upstreamCamber2[p,:,1]
    
    downstreamCurve1Pol[p,:,0] = func(downstreamCamber1[p,:,0])  
    downstreamCurve1Pol[p,0,0] = blade1SSMprime[p][-1,2]
    downstreamCurve1Pol[p,:,1] = downstreamCamber1[p,:,1]
    downstreamCurve2Pol[p,:,0] = func(downstreamCamber2[p,:,0]) 
    downstreamCurve2Pol[p,0,0] = blade2PSMprime[p][-1,2]
    downstreamCurve2Pol[p,:,1] = downstreamCamber2[p,:,1]
    
    #tangentLine1LEPol[p,:,0] = func(tangentLine1LE[p,:,0])
    #tangentLine1LEPol[p,:,1] = tangentLine1LE[p,:,1]
    #tangentLine2LEPol[p,:,0] = func(tangentLine2LE[p,:,0])
    #tangentLine2LEPol[p,:,1] = tangentLine2LE[p,:,1]  
    #tangentLine1TEPol[p,:,0] = func(tangentLine1TE[p,:,0])
    #tangentLine1TEPol[p,:,1] = tangentLine1TE[p,:,1]
    #tangentLine2TEPol[p,:,0] = func(tangentLine2TE[p,:,0])
    #tangentLine2TEPol[p,:,1] = tangentLine2TE[p,:,1]
    
    #bisectorLE1Pol[p,:,0] = func(bisectorLE1[p,:,0])
    #bisectorLE1Pol[p,:,1] = bisectorLE1[p,:,1]
    #bisectorLE2Pol[p,:,0] = func(bisectorLE2[p,:,0])
    #bisectorLE2Pol[p,:,1] = bisectorLE2[p,:,1]
    #bisectorTE1Pol[p,:,0] = func(bisectorTE1[p,:,0])
    #bisectorTE1Pol[p,:,1] = bisectorTE1[p,:,1]
    #bisectorTE2Pol[p,:,0] = func(bisectorTE2[p,:,0])
    #bisectorTE2Pol[p,:,1] = bisectorTE2[p,:,1]
    
    upstreamExtnCamber1Pol[p,:,0] = func(upstreamExtnCamber1[p,:,0])
    upstreamExtnCamber1Pol[p,:,1] = upstreamExtnCamber1[p,:,1]
    upstreamExtnCamber2Pol[p,:,0] = func(upstreamExtnCamber2[p,:,0])
    upstreamExtnCamber2Pol[p,:,1] = upstreamExtnCamber2[p,:,1]
    downstreamExtnCamber1Pol[p,:,0] = func(downstreamExtnCamber1[p,:,0])
    downstreamExtnCamber1Pol[p,:,1] = downstreamExtnCamber1[p,:,1]
    downstreamExtnCamber2Pol[p,:,0] = func(downstreamExtnCamber2[p,:,0])
    downstreamExtnCamber2Pol[p,:,1] = downstreamExtnCamber2[p,:,1]
    
    offset1PolFunc = CubicSpline(offsetVertex1MprimeS[p,:,3], offsetVertex1MprimeS[p,:,2])
    offsetBlade1Pol[p,:,0] = offset1PolFunc(offsetSplinedBlade12D[p,:,0])
    offsetBlade1Pol[p,:,1] = offsetSplinedBlade12D[p,:,1]
    
    offset2PolFunc = CubicSpline(offsetVertex2MprimeP[p,:,3], offsetVertex2MprimeP[p,:,2])
    offsetBlade2Pol[p,:,0] = offset2PolFunc(offsetSplinedBlade22D[p,:,0])
    offsetBlade2Pol[p,:,1] = offsetSplinedBlade22D[p,:,1] 

#%% Here I trimmed the surfaces both upstream and downstream to match the correct geometry. Remember that i extended either the hub and casing to get same extent.
mul = 5 #I tried to increase the resolution of the upstream and downstream direction 
trimmedUpstreamCurve1Pol = np.zeros([newNsection,res*mul,2])
trimmedUpstreamCurve2Pol = np.zeros([newNsection,res*mul,2])
trimmedDownstreamCurve1Pol = np.zeros([newNsection,res*mul,2])
trimmedDownstreamCurve2Pol = np.zeros([newNsection,res*mul,2])
offBladeUp1 = np.zeros([newNsection,int(res*0.5),2]) #This is the line connecting the LE of blade to LE of offset
offBladeUp2 = np.zeros([newNsection,int(res*0.5),2])
offBladeDw1 = np.zeros([newNsection,int(res*0.5),2])
offBladeDw2 = np.zeros([newNsection,int(res*0.5),2])
for pp in range(newNsection):
    inFunc1 = interp1d(upstreamCurve1Pol[pp,:,0], upstreamCurve1Pol[pp,:,1])
    # bfFunc = interp1d(upstreamCurve1Pol[pp,:,0], upstreamCurve1Pol[pp,:,1])
    up1Z = np.linspace(adjInlet[pp,0], upstreamCurve1Pol[pp,-2,0], mul*res)
    up1Th = inFunc1(up1Z)
    trimmedUpstreamCurve1Pol[pp] = np.column_stack((up1Z, up1Th))
    offU1Z = np.linspace(upstreamCurve1Pol[pp,-2,0],upstreamCurve1Pol[pp,-1,0], int(0.5*res))
    offU1Th = inFunc1(offU1Z)
    offBladeUp1[pp] = np.column_stack((offU1Z, offU1Th))
    inFunc2 = interp1d(upstreamCurve2Pol[pp,:,0], upstreamCurve2Pol[pp,:,1])
    up2Z = np.linspace(adjInlet[pp,0], upstreamCurve2Pol[pp,-2,0], mul*res)
    # up2Z = insertPoint(up2Z, upstreamCurve2Pol[pp,-2,0])
    up2Th = inFunc2(up2Z)
    trimmedUpstreamCurve2Pol[pp] = np.column_stack((up2Z, up2Th))   
    offU2Z = np.linspace(upstreamCurve2Pol[pp,-2,0],upstreamCurve2Pol[pp,-1,0], int(0.5*res))
    offU2Th = inFunc2(offU2Z)
    offBladeUp2[pp] = np.column_stack((offU2Z, offU2Th))
 
    outFunc1 = interp1d(downstreamCurve1Pol[pp,:,0], downstreamCurve1Pol[pp,:,1])
    dw1Z = np.linspace(downstreamCurve1Pol[pp,1,0], adjOutlet[pp,0], mul*res)
    # dw1Z = insertPoint(dw1Z, downstreamCurve1Pol[pp,1,0])
    dw1Th = outFunc1(dw1Z)
    trimmedDownstreamCurve1Pol[pp] = np.column_stack((dw1Z, dw1Th)) 
    offD1Z = np.linspace(downstreamCurve1Pol[pp,0,0],downstreamCurve1Pol[pp,1,0], int(0.5*res))
    offD1Th = outFunc1(offD1Z)
    offBladeDw1[pp] = np.column_stack((offD1Z, offD1Th))    
    outFunc2 = interp1d(downstreamCurve2Pol[pp,:,0], downstreamCurve2Pol[pp,:,1])
    dw2Z = np.linspace(downstreamCurve2Pol[pp,1,0], adjOutlet[pp,0], mul*res)
    # dw2Z = insertPoint(dw2Z, downstreamCurve2Pol[pp,1,0])
    dw2Th = outFunc2(dw2Z)
    trimmedDownstreamCurve2Pol[pp] = np.column_stack((dw2Z, dw2Th))
    offD2Z = np.linspace(downstreamCurve2Pol[pp,0,0],downstreamCurve2Pol[pp,1,0], int(0.5*res))
    offD2Th = outFunc2(offD2Z)
    offBladeDw2[pp] = np.column_stack((offD2Z, offD2Th))     

""" start of significant Jeff Defoe Nov 2025 edits """

# Cylindrical coordinate data is organized as (theta, r, z)
LECurveCyl = np.zeros([newNsection,passageRes,3])  # upstream cross-passage ellipse
TECurveCyl = np.zeros([newNsection,passageRes,3])  # downstream cross-passage ellipse
upCurve1Cyl = np.zeros([newNsection,res*mul,3])  # high-theta upstream extension
dwCurve1Cyl = np.zeros([newNsection,res*mul,3])  # high-theta downstream extension
upCurve2Cyl = np.zeros([newNsection,res*mul,3])  # low-theta upstream extension
dwCurve2Cyl = np.zeros([newNsection,res*mul,3])  # low-theta downstream extension

offBladeUp1Cyl = np.zeros([newNsection,int(res*0.5),3])  # high-theta between blade and upstream offset
offBladeUp2Cyl = np.zeros([newNsection,int(res*0.5),3])  # low-theta between blade and upstream offset
offBladeDw1Cyl = np.zeros([newNsection,int(res*0.5),3])  # high-theta between blade and downstream offset
offBladeDw2Cyl = np.zeros([newNsection,int(res*0.5),3])  # low-theta between blade and downstream offset

offsetBlade1Cyl = np.zeros([newNsection,bladeRes,3])  # high-theta offset
offsetBlade2Cyl = np.zeros([newNsection,bladeRes,3])  # low-theta offset

# Only used for visualization:
tangentLine1LECyl = np.zeros([newNsection,nl,3])  # line tangent to blade1 at the LE
tangentLine2LECyl = np.zeros([newNsection,nl,3])  # line tangent to blade2 at the LE
tangentLine1TECyl = np.zeros([newNsection,nl,3])  # line tangent to blade1 at the TE
tangentLine2TECyl = np.zeros([newNsection,nl,3])  # line tangent to blade2 at the TE
bisectorLE1Cyl = np.zeros([newNsection,nl,3])  # line that bisect angle between tangent line and horizontal line at LE for blade1
bisectorLE2Cyl = np.zeros([newNsection,nl,3])  # line that bisect angle between tangent line and horizontal line at LE for blade2
bisectorTE1Cyl = np.zeros([newNsection,nl,3])  # line that bisect angle between tangent line and horizontal line at TE for blade1
bisectorTE2Cyl = np.zeros([newNsection,nl,3])  # line that bisect angle between tangent line and horizontal line at TE for blade2
upstreamExtnCamber1Cyl = np.zeros([newNsection,nl+1,3])
upstreamExtnCamber2Cyl = np.zeros([newNsection,nl+1,3])
downstreamExtnCamber1Cyl = np.zeros([newNsection,nl+1,3])
downstreamExtnCamber2Cyl = np.zeros([newNsection,nl+1,3])

# Now fill these surfaces:
for q in range(newNsection):
    # Define a cubic split for m' to r conversion based on existing m'-z mapping:
    combinedRadius = np.concatenate((upstreamMprime[q][:-1], blade1SSMprime[q], dwstreamMprime[q][1:])) 

    # special treatment for hub/casing to ensure
    # points lie on hub/casing
    if q == 0:  # hub
        funcR = CubicSpline(hub2D[:,0], hub2D[:,1])
    elif q == newNsection-1:  # casing
        funcR = CubicSpline(cas2D[:,0], cas2D[:,1])
    else:
        funcR = CubicSpline(combinedRadius[:,2], combinedRadius[:,1])

    # upstream cross-passage ellipse
    LECurveCyl[q,:,2] = LECurvePol[q,:,0]
    LECurveCyl[q,:,1] = funcR(LECurveCyl[q,:,2])
    LECurveCyl[q,:,0] = LECurvePol[q,:,1]
    # downstream cross-passage ellipse
    TECurveCyl[q,:,2] = TECurvePol[q,:,0]
    TECurveCyl[q,:,1] = funcR(TECurveCyl[q,:,2])
    TECurveCyl[q,:,0] = TECurvePol[q,:,1]
    # high-theta upstream extension
    upCurve1Cyl[q,:,0] = trimmedUpstreamCurve1Pol[q,:,1]
    upCurve1Cyl[q,:,1] = funcR(trimmedUpstreamCurve1Pol[q,:,0])
    upCurve1Cyl[q,:,2] = trimmedUpstreamCurve1Pol[q,:,0]
    upCurve1Cyl[q,-1] = LECurveCyl[q,0]  # cross-pasage ellipses are indexed from high theta to low theta
    # low-theta upstream extension
    upCurve2Cyl[q,:,0] = trimmedUpstreamCurve2Pol[q,:,1]
    upCurve2Cyl[q,:,1] = funcR(trimmedUpstreamCurve2Pol[q,:,0])
    upCurve2Cyl[q,:,2] = trimmedUpstreamCurve2Pol[q,:,0]   
    upCurve2Cyl[q,-1] = LECurveCyl[q,-1]  # cross-pasage ellipses are indexed from high theta to low theta
    # high-theta downstream extension
    dwCurve1Cyl[q,:,0] = trimmedDownstreamCurve1Pol[q,:,1]
    dwCurve1Cyl[q,:,1] = funcR(trimmedDownstreamCurve1Pol[q,:,0])
    dwCurve1Cyl[q,:,2] = trimmedDownstreamCurve1Pol[q,:,0]
    dwCurve1Cyl[q,0] = TECurveCyl[q,0]  # cross-pasage ellipses are indexed from high theta to low theta
    # low-theta downstream extension
    dwCurve2Cyl[q,:,0] = trimmedDownstreamCurve2Pol[q,:,1]
    dwCurve2Cyl[q,:,1] = funcR(trimmedDownstreamCurve2Pol[q,:,0])
    dwCurve2Cyl[q,:,2] = trimmedDownstreamCurve2Pol[q,:,0]
    dwCurve2Cyl[q,0] = TECurveCyl[q,-1]  # cross-pasage ellipses are indexed from high theta to low theta
    # high-theta between blade and upstream offset
    offBladeUp1Cyl[q,:,0] = offBladeUp1[q,:,1]
    offBladeUp1Cyl[q,:,1] = funcR(offBladeUp1[q,:,0])
    offBladeUp1Cyl[q,:,2] = offBladeUp1[q,:,0]
    offBladeUp1Cyl[q,-1] = newBlade1SSCyl[q][0]  # replace last point (blade LE) to guarantee consistency
    offBladeUp1Cyl[q,0] = LECurveCyl[q,0]  # cross-pasage ellipses are indexed from high theta to low theta
    # low-theta between blade and upstream offset
    offBladeUp2Cyl[q,:,0] = offBladeUp2[q,:,1]
    offBladeUp2Cyl[q,:,1] = funcR(offBladeUp2[q,:,0])
    offBladeUp2Cyl[q,:,2] = offBladeUp2[q,:,0]
    offBladeUp2Cyl[q,-1] = newBlade2PSCyl[q][0]  # replace last point (blade LE) to guarantee consistency
    offBladeUp2Cyl[q,0] = LECurveCyl[q,-1]  # cross-pasage ellipses are indexed from high theta to low theta
    # high-theta between blade and downstream offset
    offBladeDw1Cyl[q,:,0] = offBladeDw1[q,:,1]
    offBladeDw1Cyl[q,:,1] = funcR(offBladeDw1[q,:,0])
    offBladeDw1Cyl[q,:,2] = offBladeDw1[q,:,0]
    offBladeDw1Cyl[q,0] = newBlade1SSCyl[q][-1]  # replace first point (blade TE) to guarantee consistency
    offBladeDw1Cyl[q,-1] = TECurveCyl[q,0]  # cross-pasage ellipses are indexed from high theta to low theta
    # low-theta between blade and downstream offset
    offBladeDw2Cyl[q,:,0] = offBladeDw2[q,:,1]
    offBladeDw2Cyl[q,:,1] = funcR(offBladeDw2[q,:,0])
    offBladeDw2Cyl[q,:,2] = offBladeDw2[q,:,0]
    offBladeDw2Cyl[q,0] = newBlade2PSCyl[q][-1]  # replace first point (blade TE) to guarantee consistency
    offBladeDw2Cyl[q,-1] = TECurveCyl[q,-1]  # cross-pasage ellipses are indexed from high theta to low theta

    # Just for visualization (can ignore):
    """
    tangentLine1LECyl[q,:,0] = tangentLine1LEPol[q,:,1]
    tangentLine1LECyl[q,:,1] = funcR(tangentLine1LEPol[q,:,0])
    tangentLine1LECyl[q,:,2] = tangentLine1LEPol[q,:,0]
    tangentLine2LECyl[q,:,0] = tangentLine2LEPol[q,:,1]
    tangentLine2LECyl[q,:,1] = funcR(tangentLine2LEPol[q,:,0])
    tangentLine2LECyl[q,:,2] = tangentLine2LEPol[q,:,0]  
    tangentLine1TECyl[q,:,0] = tangentLine1TEPol[q,:,1]
    tangentLine1TECyl[q,:,1] = funcR(tangentLine1TEPol[q,:,0])
    tangentLine1TECyl[q,:,2] = tangentLine1TEPol[q,:,0]
    tangentLine2TECyl[q,:,0] = tangentLine2TEPol[q,:,1]
    tangentLine2TECyl[q,:,1] = funcR(tangentLine2TEPol[q,:,0])
    tangentLine2TECyl[q,:,2] = tangentLine2TEPol[q,:,0]  
    bisectorLE1Cyl[q,:,0] = bisectorLE1Pol[q,:,1]
    bisectorLE1Cyl[q,:,1] = funcR(bisectorLE1Pol[q,:,0])
    bisectorLE1Cyl[q,:,2] = bisectorLE1Pol[q,:,0]
    bisectorLE2Cyl[q,:,0] = bisectorLE2Pol[q,:,1]
    bisectorLE2Cyl[q,:,1] = funcR(bisectorLE2Pol[q,:,0])
    bisectorLE2Cyl[q,:,2] = bisectorLE2Pol[q,:,0]
    bisectorTE1Cyl[q,:,0] = bisectorTE1Pol[q,:,1]
    bisectorTE1Cyl[q,:,1] = funcR(bisectorTE1Pol[q,:,0])
    bisectorTE1Cyl[q,:,2] = bisectorTE1Pol[q,:,0]
    bisectorTE2Cyl[q,:,0] = bisectorTE2Pol[q,:,1]
    bisectorTE2Cyl[q,:,1] = funcR(bisectorTE2Pol[q,:,0])
    bisectorTE2Cyl[q,:,2] = bisectorTE2Pol[q,:,0]    
    upstreamExtnCamber1Cyl[q,:,0] = upstreamExtnCamber1Pol[q,:,1]
    upstreamExtnCamber1Cyl[q,:,1] = funcR(upstreamExtnCamber1Pol[q,:,0])
    upstreamExtnCamber1Cyl[q,:,2] = upstreamExtnCamber1Pol[q,:,0]
    upstreamExtnCamber2Cyl[q,:,0] = upstreamExtnCamber2Pol[q,:,1]
    upstreamExtnCamber2Cyl[q,:,1] = funcR(upstreamExtnCamber2Pol[q,:,0])
    upstreamExtnCamber2Cyl[q,:,2] = upstreamExtnCamber2Pol[q,:,0]   
    downstreamExtnCamber1Cyl[q,:,0] = downstreamExtnCamber1Pol[q,:,1]
    downstreamExtnCamber1Cyl[q,:,1] = funcR(downstreamExtnCamber1Pol[q,:,0])
    downstreamExtnCamber1Cyl[q,:,2] = downstreamExtnCamber1Pol[q,:,0]
    downstreamExtnCamber2Cyl[q,:,0] = downstreamExtnCamber2Pol[q,:,1]
    downstreamExtnCamber2Cyl[q,:,1] = funcR(downstreamExtnCamber2Pol[q,:,0])
    downstreamExtnCamber2Cyl[q,:,2] = downstreamExtnCamber2Pol[q,:,0]    
    """

    # high-theta offset
    offsetBlade1Cyl[q,:,2] = offsetBlade1Pol[q,:,0]
    offsetBlade1Cyl[q,:,0] = offsetBlade1Pol[q,:,1]
    offsetBlade1Cyl[q,:,1] = funcR(offsetBlade1Pol[q,:,0])
    # low-theta offset    
    offsetBlade2Cyl[q,:,2] = offsetBlade2Pol[q,:,0]
    offsetBlade2Cyl[q,:,0] = offsetBlade2Pol[q,:,1]
    offsetBlade2Cyl[q,:,1] = funcR(offsetBlade2Pol[q,:,0])
    
# Vertices on hub and casing at domain inlet and outlet (messy naming!)
hubLEInlet1 = np.column_stack((upCurve1Cyl[0][0,0]*upCurve1Cyl[0][0,1], upCurve1Cyl[0][0,2]))
hubLEInlet2 = np.column_stack((upCurve2Cyl[0][0,0]*upCurve2Cyl[0][0,1], upCurve2Cyl[0][0,2]))
casLEInlet1 = np.column_stack((upCurve1Cyl[-1][0,0]*upCurve1Cyl[-1][0,1], upCurve1Cyl[-1][0,2]))
casLEInlet2 = np.column_stack((upCurve2Cyl[-1][0,0]*upCurve2Cyl[-1][0,1], upCurve2Cyl[-1][0,2]))
hubTEInlet1 = np.column_stack((dwCurve1Cyl[0][-1,0]*dwCurve1Cyl[0][-1,1], dwCurve1Cyl[0][-1,2]))
hubTEInlet2 = np.column_stack((dwCurve2Cyl[0][-1,0]*dwCurve2Cyl[0][-1,1], dwCurve2Cyl[0][-1,2]))
casTEInlet1 = np.column_stack((dwCurve1Cyl[-1][-1,0]*dwCurve1Cyl[-1][-1,1], dwCurve1Cyl[-1][-1,2]))
casTEInlet2 = np.column_stack((dwCurve2Cyl[-1][-1,0]*dwCurve2Cyl[-1][-1,1], dwCurve2Cyl[-1][-1,2]))
# Edges at hub and casing at domain inlet and outlet (cylindrical coordinates)
hubLETheta = np.linspace(hubLEInlet1[0], hubLEInlet2[0], passageRes)
hubTETheta = np.linspace(hubTEInlet1[0], hubTEInlet2[0], passageRes)
casLETheta = np.linspace(casLEInlet1[0], casLEInlet2[0], passageRes)
casTETheta = np.linspace(casTEInlet1[0], casTEInlet2[0], passageRes)



# plt.plot(offsetBlade2Cyl[:,:,2], offsetBlade2Cyl[:,:,1], 'k.')
# plt.plot(offsetSplinedBlade22D[10,:,0], offsetSplinedBlade22D[:,:,1], 'k.')
#%%
# Split the surface of the blade and the offset to upstream and downstream
# Remember SS1 is the highTheta and PS2 is the low theta

lowThetaB1Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of blade surface
lowThetaO1Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of offset surface
lowThetaB2Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of blade surface
lowThetaO2Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of offset surface

highThetaB1Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of blade surface
highThetaO1Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the upstream portion of offset surface
highThetaB2Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of blade surface
highThetaO2Cyl = np.zeros((newNsection, bladeRes, 3))  # This is the downstream portion of offset surface

midCurveCylMid =  np.zeros([newNsection,passageRes,3])  # midCurve in between offsets
midCurveCylLow =  np.zeros([newNsection,int(res*0.5),3])  # midCurve between low theta blade and offset
midCurveCylHigh =  np.zeros([newNsection,int(res*0.5),3])  # midCurve between offset and high theta blade

for v in range(newNsection):
    
    blade1Idx = np.argmin(np.abs(newBlade1SSCylM[v][:,2] - mid1SS[v][2]))
    highThetaB1 = newBlade1SSCylM[v][0:blade1Idx+1]
    highThetaB1Cyl[v] = densifyCurve(highThetaB1, bladeRes, 'LE')
    highThetaB2 = newBlade1SSCylM[v][blade1Idx:]
    highThetaB2Cyl[v] = densifyCurve(highThetaB2, bladeRes, 'TE') 
  
    blade2Idx = np.argmin(np.abs(newBlade2PSCylM[v][:,2] - mid2PS[v][2]))
    lowThetaB1 = newBlade2PSCylM[v][0:blade2Idx+1]
    lowThetaB1Cyl[v] = densifyCurve(lowThetaB1, bladeRes, 'LE')
    lowThetaB2 = newBlade2PSCylM[v][blade2Idx:]
    lowThetaB2Cyl[v] = densifyCurve(lowThetaB2, bladeRes, 'TE')   
    
    offsetBlade1CylM = insertPoint(offsetBlade1Cyl[v], offsetVertex1Cyl[v][3])
    offset1Idx = np.argmin(np.abs(offsetBlade1CylM[:,2] - offsetVertex1Cyl[v][3][2]))
    highThetaO1 = offsetBlade1CylM[0:offset1Idx+1]
    highThetaO1Cyl[v] = densifyCurve(highThetaO1, bladeRes, 'LE')
    highThetaO2 = offsetBlade1CylM[offset1Idx:]
    highThetaO2Cyl[v] = densifyCurve(highThetaO2, bladeRes, 'TE') 
    
    offsetBlade2CylM = insertPoint(offsetBlade2Cyl[v], offsetVertex2Cyl[v][1])
    offset2Idx = np.argmin(np.abs(offsetBlade2CylM[:,2] - offsetVertex2Cyl[v][1][2]))
    lowThetaO1 = offsetBlade2CylM[0:offset2Idx+1]
    lowThetaO1Cyl[v] = densifyCurve(lowThetaO1, bladeRes, 'LE')
    lowThetaO2 = offsetBlade2CylM[offset2Idx:]
    lowThetaO2Cyl[v] = densifyCurve(lowThetaO2, bladeRes, 'TE') 

    # "midCurve" --> just the set of four points blade-offset-offset-blade on this section
    midCurve = np.vstack((highThetaB2[0], highThetaO2[0], lowThetaO2[0], lowThetaB2[0]))
    # these arrays hold the surface points for the 3 sections of the midChord surface:
    midCurveCylMid[v][:,2] = np.linspace(midCurve[0,2], midCurve[-1,2], passageRes)
    midCurveCylLow[v][:,2] = np.linspace(midCurve[0,2], midCurve[-1,2], int(res*0.5))
    midCurveCylHigh[v][:,2] = np.linspace(midCurve[0,2], midCurve[-1,2], int(res*0.5))
    midCurveCylMid[v][:,1] = np.linspace(midCurve[0,1], midCurve[-1,1], passageRes)
    midCurveCylLow[v][:,1] = np.linspace(midCurve[0,1], midCurve[-1,1], int(res*0.5))
    midCurveCylHigh[v][:,1] = np.linspace(midCurve[0,1], midCurve[-1,1],int(res*0.5))
    midCurveCylHigh[v][:,0]= np.linspace(midCurve[:,0][0], midCurve[:,0][1], int(0.5*res))
    midCurveCylMid[v][:,0] = np.linspace(midCurve[:,0][1], midCurve[:,0][2], passageRes)
    midCurveCylLow[v][:,0] = np.linspace(midCurve[:,0][2], midCurve[:,0][3], int(0.5*res))
    # midCurveCyl[v][:,0] = np.hstack((midCurveTh1, midCurveTh2[1:-1], midCurveTh3))
    # midCurveC = densifyCurve(midCurve, passageRes)
    # midCurveCyl[v] = insertPoints_batch(midCurveC, [highThetaO2[0], lowThetaO2[0]])
#%%
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(highThetaO1Cyl[:,:,0], highThetaO1Cyl[:,:,1], highThetaO1Cyl[:,:,2], 'k.')
#ax.plot(highThetaO2Cyl[:,:,0], highThetaO2Cyl[:,:,1], highThetaO2Cyl[:,:,2], 'r.')
# ax.plot(lowThetaO1Cyl[:,:,0], lowThetaO1Cyl[:,:,1], lowThetaO1Cyl[:,:,2], 'k.')
# ax.plot(lowThetaO2Cyl[:,:,0], lowThetaO2Cyl[:,:,1], lowThetaO2Cyl[:,:,2], 'r.')
# ax.plot(offsetBlade2Cyl[:,:,0], offsetBlade2Cyl[:,:,1], offsetBlade2Cyl[:,:,2], 'b.')
# plt.plot(highThetaO2Cyl[-1,:,2], highThetaO2Cyl[-1,:,1]*highThetaO2Cyl[-1,:,0], 'k')
# plt.plot(lowThetaO2Cyl[-1,:,2], lowThetaO2Cyl[-1,:,1]*lowThetaO2Cyl[-1,:,0], 'k')
# plt.plot(highThetaB2Cyl[-1,:,2], highThetaB2Cyl[-1,:,1]*highThetaB2Cyl[-1,:,0], 'k')
# plt.plot(highThetaB2[:,2], highThetaB2[:,1]*highThetaB2[:,0], 'r')

# NOTE: issue is that 6 surfaces meet at the points the offsets hit the periodics, but the vertices
# don't seem to be shared. Specifically:
# - the surface from blade to offset and the extension periodic ARE consistent
# - the hub/casing offset section and the offset surface ARE consistent
# - the hub extension and hub between-blades surfaces ARE consistent
# But:
# - all three of these sets are not consistent with one another, that is, there are THREE
# corners vertices where there ought to be ONE.
#
# To solve: as the hub and casing surfaces seem to be the last ones to be defined, it ought to be
# easy to make them consistent with the rest
#
# Issue is, of the other two, which one is right? Check the blade-to-offset and offset surface
# definitions to see what's going on.
#
# Blade and offset surface definitions should be treated as correct. Everything else must conform to them.
# 
# Fixed with 3 changes:
# Upstream in the code, the wrong LE/TE points were being used in meridional curve definition. That has been fixed.
# Also, code has been added to make the extensions, blade-to-offset, and cross-passage ellipses explicitly consistent.
# Finally, for non-hub and casing sections, the wrong interpolator object was being used, which has been fixed.

# The hub and casing also seem OK now!

#%% Here I made a split of he domain. The reason for doing this is to make sure that the midPoint shares a unique point 

# Break the bladeSurface into two separated by the midpoint. This is important before doing the transfinite interpolation 
leSection1 = np.zeros([newNsection, bladeRes, 2]) #edge of the high upstream offset curve
leSection2 = np.zeros([newNsection, bladeRes, 2]) #edge of the low upstream offset curve
teSection1 = np.zeros([newNsection, bladeRes, 2]) #edge of the high downstream offset curve
teSection2 = np.zeros([newNsection, bladeRes, 2]) #edge of the low downstream offset curve
leBladeSection1 = np.zeros([newNsection, bladeRes, 2]) #edge of the high upstream blade curve
leBladeSection2 = np.zeros([newNsection, bladeRes, 2]) #edge of the low upstream blade curve
teBladeSection1 = np.zeros([newNsection, bladeRes, 2]) #edge of the high downstream blade curve
teBladeSection2 = np.zeros([newNsection, bladeRes, 2]) #edge of the low downstream blade curve
inletLowTheta = np.zeros([newNsection,res*mul,2]) #edge of low inlet curve
inletHighTheta = np.zeros([newNsection,res*mul,2]) #edge of high inlet curve
outletLowTheta = np.zeros([newNsection,res*mul,2]) #edge of low outlet curve
outletHighTheta = np.zeros([newNsection,res*mul,2]) #edge of high outlet curve
LECurveZRTH = np.zeros([newNsection,passageRes,2]) # LE curve joing high and low
TECurveZRTH = np.zeros([newNsection,passageRes,2]) #TE curve joing high and low
MidCurveZRTH = np.zeros([newNsection,passageRes,2]) #Mid curve joing high blade and low blade
lowMidZRTH = np.zeros([newNsection,int(0.5*res),2]) #Mid curve joining low offset and low blade
highMidZRTH = np.zeros([newNsection,int(0.5*res),2]) #Mid curve joining high offset and high blade
offBladeUp1MRTH = np.zeros([newNsection,int(res*0.5),2]) #edge of blade LE and offset LE high
offBladeUp2MRTH = np.zeros([newNsection,int(res*0.5),2]) #edge of blade LE and offset LE low
offBladeDw1MRTH = np.zeros([newNsection,int(res*0.5),2])  #edge of blade TE and offset TE high
offBladeDw2MRTH = np.zeros([newNsection,int(res*0.5),2]) #edge of blade TE and offset TE low

for r in range(newNsection):
    leSection1[r,:,:] = np.column_stack((highThetaO1Cyl[r,:,0]* highThetaO1Cyl[r,:,1], highThetaO1Cyl[r,:,2]))
    teSection1[r,:,:] = np.column_stack((highThetaO2Cyl[r,:,0]* highThetaO2Cyl[r,:,1], highThetaO2Cyl[r,:,2]))
    leSection2[r,:,:] = np.column_stack((lowThetaO1Cyl[r,:,0]*lowThetaO1Cyl[r,:,1], lowThetaO1Cyl[r,:,2]))
    teSection2[r,:,:] = np.column_stack((lowThetaO2Cyl[r,:,0]*lowThetaO2Cyl[r,:,1], lowThetaO2Cyl[r,:,2]))
    leBladeSection1[r,:,:] = np.column_stack((highThetaB1Cyl[r,:,0]* highThetaB1Cyl[r,:,1], highThetaB1Cyl[r,:,2]))
    teBladeSection1[r,:,:] = np.column_stack((highThetaB2Cyl[r,:,0]* highThetaB2Cyl[r,:,1], highThetaB2Cyl[r,:,2]))
    leBladeSection2[r,:,:] = np.column_stack((lowThetaB1Cyl[r,:,0]*lowThetaB1Cyl[r,:,1], lowThetaB1Cyl[r,:,2]))
    teBladeSection2[r,:,:] = np.column_stack((lowThetaB2Cyl[r,:,0]*lowThetaB2Cyl[r,:,1], lowThetaB2Cyl[r,:,2]))
    inletLowTheta[r,:,:] = np.column_stack((upCurve1Cyl[r,:,1]*upCurve1Cyl[r,:,0], upCurve1Cyl[r,:,2]))
    inletHighTheta[r,:,:] = np.column_stack((upCurve2Cyl[r,:,1]*upCurve2Cyl[r,:,0], upCurve2Cyl[r,:,2]))
    outletLowTheta[r,:,:] = np.column_stack((dwCurve1Cyl[r,:,1]*dwCurve1Cyl[r,:,0], dwCurve1Cyl[r,:,2]))#[::-1]
    outletHighTheta[r,:,:] = np.column_stack((dwCurve2Cyl[r,:,1]*dwCurve2Cyl[r,:,0], dwCurve2Cyl[r,:,2]))#[::-1] 
    LECurveZRTH[r,:,:] = np.column_stack((LECurveCyl[r,:,1]*LECurveCyl[r,:,0], LECurveCyl[r,:,2]))  
    TECurveZRTH[r,:,:] = np.column_stack((TECurveCyl[r,:,1]*TECurveCyl[r,:,0], TECurveCyl[r,:,2])) 
    MidCurveZRTH[r,:,:] = np.column_stack((midCurveCylMid[r,:,1]*midCurveCylMid[r,:,0], midCurveCylMid[r,:,2])) 
    lowMidZRTH[r,:,:] =  np.column_stack((midCurveCylLow[r,:,1]*midCurveCylLow[r,:,0], midCurveCylLow[r,:,2])) 
    highMidZRTH[r,:,:] =  np.column_stack((midCurveCylHigh[r,:,1]*midCurveCylHigh[r,:,0], midCurveCylHigh[r,:,2])) 
    offBladeUp1MRTH [r,:,:] = np.column_stack((offBladeUp1Cyl[r,:,1]*offBladeUp1Cyl[r,:,0], offBladeUp1Cyl[r,:,2])) 
    offBladeUp2MRTH [r,:,:] = np.column_stack((offBladeUp2Cyl[r,:,1]*offBladeUp2Cyl[r,:,0], offBladeUp2Cyl[r,:,2])) 
    offBladeDw1MRTH [r,:,:] = np.column_stack((offBladeDw1Cyl[r,:,1]*offBladeDw1Cyl[r,:,0], offBladeDw1Cyl[r,:,2])) 
    offBladeDw2MRTH [r,:,:] = np.column_stack((offBladeDw1Cyl[r,:,1]*offBladeDw1Cyl[r,:,0], offBladeDw1Cyl[r,:,2])) 
    


inletHubLETheta = np.linspace(inletLowTheta[0][-1][[0,1]], inletHighTheta[0][-1][[0,1]], passageRes) #inlet hub joining low to high
inletCasLETheta = np.linspace(inletLowTheta[newNsection-1][0][[0,1]], inletHighTheta[newNsection-1][0][[0,1]], passageRes) #inlet casing joining low to high

outletHubTETheta = np.linspace(outletLowTheta[0][::-1][-1][[0,1]], outletHighTheta[0][::-1][-1][[0,1]], passageRes) #outlet hub joining low to high
outletCasTETheta = np.linspace(outletLowTheta[newNsection-1][::-1][-1][[0,1]], outletHighTheta[newNsection-1][::-1][-1][[0,1]], passageRes) #outlet casing joining low to high


#%%
#Now I need to get the inner points for the inlet outlet hub and casing 
#Using transfinite interpolation 
'''
 transfinite(lower, upper, left, right)
'''
inletHubNodes = tf.transfinite(inletHubLETheta, LECurveZRTH[0], inletLowTheta[0], inletHighTheta[0])
inletCasNodes = tf.transfinite(inletCasLETheta, LECurveZRTH[-1], inletLowTheta[-1], inletHighTheta[-1])
leHubNodes = tf.transfinite(LECurveZRTH[0], MidCurveZRTH[0], leSection1[0], leSection2[0])
teHubNodes = tf.transfinite(MidCurveZRTH[0], TECurveZRTH[0], teSection1[0], teSection2[0])
leCasNodes = tf.transfinite(LECurveZRTH[-1], MidCurveZRTH[-1], leSection1[-1], leSection2[-1])
teCasNodes = tf.transfinite(MidCurveZRTH[-1], TECurveZRTH[-1], teSection1[-1], teSection2[-1])
outletHubNodes = tf.transfinite(TECurveZRTH[0], outletHubTETheta, outletLowTheta[0], outletHighTheta[0])
outletCasNodes = tf.transfinite(TECurveZRTH[-1], outletCasTETheta, outletLowTheta[-1], outletHighTheta[-1])
inNodes = tf.transfinite(hubLETheta, casLETheta, inletLowTheta[:,0], inletHighTheta[:,0])
outNodes = tf.transfinite(hubTETheta, casTETheta, outletLowTheta[:,-1], outletHighTheta[:,-1])

highLEHubNodes = tf.transfinite(offBladeUp1MRTH[0], highMidZRTH[0], leSection1[0], leBladeSection1[0])
highTEHubNodes = tf.transfinite(highMidZRTH[0], offBladeDw1MRTH[0], teSection1[0], teBladeSection1[0])
lowLEHubNodes = tf.transfinite(offBladeUp2MRTH[0], lowMidZRTH[0], leSection2[0], leBladeSection2[0])
lowTEHubNodes = tf.transfinite(lowMidZRTH[0], offBladeDw2MRTH[0], teSection2[0], teBladeSection2[0])
highLECasNodes = tf.transfinite(offBladeUp1MRTH[-1], highMidZRTH[-1], leSection1[-1], leBladeSection1[-1])
highTECasNodes = tf.transfinite(highMidZRTH[-1], offBladeDw1MRTH[-1], teSection1[-1], teBladeSection1[-1])
lowLECasNodes = tf.transfinite(offBladeUp2MRTH[-1], lowMidZRTH[-1], leSection2[-1], leBladeSection2[-1])
lowTECasNodes = tf.transfinite(lowMidZRTH[-1], offBladeDw2MRTH[-1], teSection2[-1], teBladeSection2[-1])
#%%
#plt.figure()
#plt.plot(inletHubNodes[:,1], inletHubNodes[:,0], 'k.')
#plt.plot(leHubNodes[:,1], leHubNodes[:,0], 'g.')
#plt.plot(teHubNodes[:,1], teHubNodes[:,0], 'r.')
#plt.plot( MidCurveZRTH[0][:,[0,1]][:,1],  MidCurveZRTH[0][:,[0,1]][:,0])
#plt.plot(highLEHubNodes[:,1], highLEHubNodes[:,0], 'b.')
#plt.plot(highTEHubNodes[:,1], highTEHubNodes[:,0], 'b.')
#plt.plot(outletHubNodes[:,1], outletHubNodes[:,0], 'k.')
#plt.plot(lowLEHubNodes[:,1], lowLEHubNodes[:,0], 'b.')
#plt.plot(lowTEHubNodes[:,1], lowTEHubNodes[:,0], 'b.')

#%%
#
inletHubProfiles = np.zeros([passageRes,res*mul,2])
inletCasProfiles = np.zeros([passageRes,res*mul,2])
outletHubProfiles = np.zeros([passageRes,res*mul,2])
outletCasProfiles = np.zeros([passageRes,res*mul,2])
leHubProfiles = np.zeros([passageRes,bladeRes,2])
leCasProfiles = np.zeros([passageRes,bladeRes,2])
teHubProfiles = np.zeros([passageRes,bladeRes,2])
teCasProfiles = np.zeros([passageRes,bladeRes,2])
allInProfiles = np.zeros([passageRes,newNsection,2])
allOutProfiles = np.zeros([passageRes,newNsection,2])
highLEHubProfiles = np.zeros([int(0.5*res),bladeRes,2])
highTEHubProfiles = np.zeros([int(0.5*res),bladeRes,2])
highLECasProfiles = np.zeros([int(0.5*res),bladeRes,2])
highTECasProfiles = np.zeros([int(0.5*res),bladeRes,2]) 
lowLEHubProfiles = np.zeros([int(0.5*res),bladeRes,2])
lowTEHubProfiles = np.zeros([int(0.5*res),bladeRes,2])
lowLECasProfiles = np.zeros([int(0.5*res),bladeRes,2])
lowTECasProfiles = np.zeros([int(0.5*res),bladeRes,2]) 
for t in range(passageRes):
    tempInletHub = np.zeros([res*mul,2])
    tempInletCas = np.zeros([res*mul,2])
    tempOutletHub = np.zeros([res*mul,2])
    tempOutletCas = np.zeros([res*mul,2])
    tempLeHubNodes = np.zeros([bladeRes,2])
    tempLeCasNodes = np.zeros([bladeRes,2])
    tempTeHubNodes = np.zeros([bladeRes,2])
    tempTeCasNodes = np.zeros([bladeRes,2])
    tempIn = np.zeros([newNsection,2])
    tempOut = np.zeros([newNsection,2])
    for u in range(res*mul):
        tempInletHub[u] = inletHubNodes[t*(res*mul) + u]
        tempInletCas[u] = inletCasNodes[t*(res*mul) + u]
        tempOutletHub[u] = outletHubNodes[t*(res*mul) + u]
        tempOutletCas[u] = outletCasNodes[t*(res*mul) + u]
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

for tt in range(int(0.5*res)):
    tempHighLEHub =  np.zeros([bladeRes,2])
    tempHighTEHub =  np.zeros([bladeRes,2])
    tempHighLECas =  np.zeros([bladeRes,2])
    tempHighTECas =  np.zeros([bladeRes,2])
    tempLowLEHub =  np.zeros([bladeRes,2])
    tempLowTEHub =  np.zeros([bladeRes,2])
    tempLowLECas =  np.zeros([bladeRes,2])
    tempLowTECas =  np.zeros([bladeRes,2])    
    for uu in range(bladeRes):
        tempHighLEHub[uu] = highLEHubNodes[tt*(bladeRes) + uu]
        tempHighTEHub[uu] = highTEHubNodes[tt*(bladeRes) + uu]
        tempHighLECas[uu] = highLECasNodes[tt*(bladeRes) + uu]
        tempHighTECas[uu] = highTECasNodes[tt*(bladeRes) + uu]
        tempLowLEHub[uu] = lowLEHubNodes[tt*(bladeRes) + uu]
        tempLowTEHub[uu] = lowTEHubNodes[tt*(bladeRes) + uu]
        tempLowLECas[uu] = lowLECasNodes[tt*(bladeRes) + uu]
        tempLowTECas[uu] = lowTECasNodes[tt*(bladeRes) + uu]    
    
    highLEHubProfiles[tt] = tempHighLEHub 
    highTEHubProfiles[tt] = tempHighTEHub
    highLECasProfiles[tt] = tempHighLECas 
    highTECasProfiles[tt] = tempHighTECas
    lowLEHubProfiles[tt] = tempLowLEHub
    lowTEHubProfiles[tt] = tempLowTEHub 
    lowLECasProfiles[tt] = tempLowLECas
    lowTECasProfiles[tt] = tempLowTECas     
        
#%%
#plt.plot(highTEHubProfiles[:,:,1], highTEHubProfiles[:,:,0], 'c.')
#%%
#Take everything back to Cartesian Coordinate 
inletHubCyl = np.zeros([passageRes,res*mul,3])
inletCasCyl = np.zeros([passageRes,res*mul,3])
leHubCyl = np.zeros([passageRes,bladeRes,3])
leCasCyl = np.zeros([passageRes,bladeRes,3])
teHubCyl = np.zeros([passageRes,bladeRes,3])
teCasCyl = np.zeros([passageRes,bladeRes,3])
outletHubCyl = np.zeros([passageRes,res*mul,3])
outletCasCyl = np.zeros([passageRes,res*mul,3])
inCyl = np.zeros([passageRes, newNsection, 3])
outCyl = np.zeros([passageRes, newNsection, 3])
highLEHubCyl = np.zeros([int(0.5*res),bladeRes,3])
highTEHubCyl = np.zeros([int(0.5*res),bladeRes,3])
highLECasCyl = np.zeros([int(0.5*res),bladeRes,3])
highTECasCyl = np.zeros([int(0.5*res),bladeRes,3]) 
lowLEHubCyl = np.zeros([int(0.5*res),bladeRes,3])
lowTEHubCyl = np.zeros([int(0.5*res),bladeRes,3])
lowLECasCyl = np.zeros([int(0.5*res),bladeRes,3])
lowTECasCyl = np.zeros([int(0.5*res),bladeRes,3]) 

hubFunc = CubicSpline(hub2D[:,0], hub2D[:,1])
casFunc = CubicSpline(cas2D[:,0], cas2D[:,1])

for u in range(passageRes):
    # hubFunc = CubicSpline(hub2D[:,0], hub2D[:,1])
    # casFunc = CubicSpline(cas2D[:,0], cas2D[:,1])
    # rInletHubFunc = CubicSpline(upstreamMprime[0,:,2], upstreamMprime[0,:,1])
    # rInletHub = rInletHubFunc(inletHubProfiles[u][:,1])
    # inletHubCyl[u] = np.column_stack((inletHubProfiles[u][:,0]/rInletHub, rInletHub,inletHubProfiles[u][:,1]))
    # rInletCasFunc = CubicSpline(upstreamMprime[-1,:,2], upstreamMprime[-1,:,1])
    # rInletCas = rInletCasFunc(inletCasProfiles[u][:,1])
    # inletCasCyl[u] = np.column_stack((inletCasProfiles[u][:,0]/rInletCas, rInletCas,inletCasProfiles[u][:,1]))
    # combinedHubRadius = np.concatenate((upstreamMprime[0][:-1], blade1PSMprime[0], dwstreamMprime[0][1:])) 
    # rHubCylFunc = CubicSpline(combinedHubRadius[:,2], combinedHubRadius[:,1])
    # rLeHubCyl = rHubCylFunc(leHubProfiles[u,:,1])
    # leHubCyl[u] = np.column_stack((leHubProfiles[u,:,0]/rLeHubCyl,  rLeHubCyl, leHubProfiles[u,:,1]))
    # combinedCasRadius = np.concatenate((upstreamMprime[-1][:-1], blade1PSMprime[-1], dwstreamMprime[-1][1:]))
    # rCasCylFunc = CubicSpline(combinedCasRadius[:,2], combinedCasRadius[:,1])
    # rLeCasCyl = rCasCylFunc(leCasProfiles[u,:,1])
    # leCasCyl[u] = np.column_stack((leCasProfiles[u,:,0]/rLeCasCyl ,  rLeCasCyl, leCasProfiles[u,:,1]))
    # rTeHubCyl =  rHubCylFunc(teHubProfiles[u,:,1])
    # teHubCyl[u] = np.column_stack((teHubProfiles[u,:,0]/rTeHubCyl ,  rTeHubCyl, teHubProfiles[u,:,1]))
    # rTeCasCyl = rCasCylFunc(teCasProfiles[u,:,1])
    # teCasCyl[u] = np.column_stack((teCasProfiles[u,:,0]/rTeCasCyl,  rTeCasCyl,  teCasProfiles[u,:,1]))
    # rOutletHubFunc = CubicSpline(dwstreamMprime[0,:,2], dwstreamMprime[0,:,1])
    # rOutletHub = rOutletHubFunc(outletHubProfiles[u][:,1])
    # outletHubCyl[u] = np.column_stack((outletHubProfiles[u][:,0]/rOutletHub, rOutletHub, outletHubProfiles[u][:,1]))
    # rOutletCasFunc = CubicSpline(dwstreamMprime[-1,:,2], dwstreamMprime[-1,:,1])
    # rOutletCas = rOutletCasFunc(outletCasProfiles[u][:,1])
    # outletCasCyl[u] = np.column_stack((outletCasProfiles[u][:,0]/rOutletCas, rOutletCas ,outletCasProfiles[u][:,1]))
    # rInFunc = interp1d(np.round(allInProfiles[u][:,0],10), allInProfiles[u][:,1])
    # rIn = upCurve1Cyl[:,0,1]
    # inCyl[u] = np.column_stack((allInProfiles[u][:,0]/rIn, rIn, allInProfiles[u][:,1]))
    # rOutFunc = interp1d(np.round(allOutProfiles[u][:,0],10), allOutProfiles[u][:,1])
    # rOut = dwCurve1Cyl[:,-1,1]
    # outCyl[u] = np.column_stack((allOutProfiles[u][:,0]/rOut, rOut, allOutProfiles[u][:,1]))  
    # if u < int(0.5*res):
    #     rHighLEHub = rHubCylFunc(highLEHubProfiles[u,:,1])
    #     highLEHubCyl[u] = np.column_stack((highLEHubProfiles[u,:,0]/rHighLEHub, rHighLEHub, highLEHubProfiles[u,:,1]))
    #     rHighTEHub = rHubCylFunc(highTEHubProfiles[u,:,1])
    #     highTEHubCyl[u] = np.column_stack((highTEHubProfiles[u,:,0]/rHighTEHub, rHighTEHub, highTEHubProfiles[u,:,1]))
    #     rLowLEHub = rHubCylFunc(lowLEHubProfiles[u,:,1])
    #     lowLEHubCyl[u] = np.column_stack((lowLEHubProfiles[u,:,0]/rLowLEHub, rLowLEHub, lowLEHubProfiles[u,:,1]))
    #     rLowTEHub = rHubCylFunc(lowTEHubProfiles[u,:,1])
    #     lowTEHubCyl[u] = np.column_stack((lowTEHubProfiles[u,:,0]/rLowTEHub, rLowTEHub, lowTEHubProfiles[u,:,1]))
        
   
    #     rHighLECas = casFunc(highLECasProfiles[u,:,1])
    #     highLECasCyl[u] = np.column_stack((highLECasProfiles[u,:,0]/rHighLECas, rHighLECas, highLECasProfiles[u,:,1]))
    #     rHighTECas = casFunc(highLECasProfiles[u,:,1])
    #     highTECasCyl[u] = np.column_stack((highTECasProfiles[u,:,0]/rHighTECas, rHighTECas, highTECasProfiles[u,:,1]))
    #     rLowLECas = casFunc(lowLECasProfiles[u,:,1])
    #     lowLECasCyl[u] = np.column_stack((lowLECasProfiles[u,:,0]/rLowLECas, rLowLECas, lowLECasProfiles[u,:,1]))
    #     rLowTECas = casFunc(lowTECasProfiles[u,:,1])
    #     lowTECasCyl[u] = np.column_stack((lowTECasProfiles[u,:,0]/rLowTECas, rLowTECas, lowTECasProfiles[u,:,1]))    
        
        
    rInletHub = hubFunc(inletHubProfiles[u][:,1])
    inletHubCyl[u] = np.column_stack((inletHubProfiles[u][:,0]/rInletHub, rInletHub,inletHubProfiles[u][:,1]))
    rInletCas = casFunc(inletCasProfiles[u][:,1])
    inletCasCyl[u] = np.column_stack((inletCasProfiles[u][:,0]/rInletCas, rInletCas,inletCasProfiles[u][:,1]))
    rLeHubCyl = hubFunc(leHubProfiles[u,:,1])
    leHubCyl[u] = np.column_stack((leHubProfiles[u,:,0]/rLeHubCyl,  rLeHubCyl, leHubProfiles[u,:,1]))
    rLeCasCyl = casFunc(leCasProfiles[u,:,1])
    leCasCyl[u] = np.column_stack((leCasProfiles[u,:,0]/rLeCasCyl ,  rLeCasCyl, leCasProfiles[u,:,1]))
    rTeHubCyl =  hubFunc(teHubProfiles[u,:,1])
    teHubCyl[u] = np.column_stack((teHubProfiles[u,:,0]/rTeHubCyl ,  rTeHubCyl, teHubProfiles[u,:,1]))
    rTeCasCyl = casFunc(teCasProfiles[u,:,1])
    teCasCyl[u] = np.column_stack((teCasProfiles[u,:,0]/rTeCasCyl,  rTeCasCyl,  teCasProfiles[u,:,1]))
    rOutletHub = hubFunc(outletHubProfiles[u][:,1])
    outletHubCyl[u] = np.column_stack((outletHubProfiles[u][:,0]/rOutletHub, rOutletHub, outletHubProfiles[u][:,1]))
    rOutletCas = casFunc(outletCasProfiles[u][:,1])
    outletCasCyl[u] = np.column_stack((outletCasProfiles[u][:,0]/rOutletCas, rOutletCas ,outletCasProfiles[u][:,1]))
    rInFunc = interp1d(allInProfiles[u][:,0], allInProfiles[u][:,1])
    rIn = upCurve1Cyl[:,0,1]
    inCyl[u] = np.column_stack((allInProfiles[u][:,0]/rIn, rIn, allInProfiles[u][:,1]))
    rOutFunc = interp1d(np.round(allOutProfiles[u][:,0],10), allOutProfiles[u][:,1])
    rOut = dwCurve1Cyl[:,-1,1]
    outCyl[u] = np.column_stack((allOutProfiles[u][:,0]/rOut, rOut, allOutProfiles[u][:,1]))  
    if u < int(0.5*res):
        rHighLEHub = hubFunc(highLEHubProfiles[u,:,1])
        highLEHubCyl[u] = np.column_stack((highLEHubProfiles[u,:,0]/rHighLEHub, rHighLEHub, highLEHubProfiles[u,:,1]))
        rHighTEHub = hubFunc(highTEHubProfiles[u,:,1])
        highTEHubCyl[u] = np.column_stack((highTEHubProfiles[u,:,0]/rHighTEHub, rHighTEHub, highTEHubProfiles[u,:,1]))
        rLowLEHub = hubFunc(lowLEHubProfiles[u,:,1])
        lowLEHubCyl[u] = np.column_stack((lowLEHubProfiles[u,:,0]/rLowLEHub, rLowLEHub, lowLEHubProfiles[u,:,1]))
        rLowTEHub = hubFunc(lowTEHubProfiles[u,:,1])
        lowTEHubCyl[u] = np.column_stack((lowTEHubProfiles[u,:,0]/rLowTEHub, rLowTEHub, lowTEHubProfiles[u,:,1]))
        
   
        rHighLECas = casFunc(highLECasProfiles[u,:,1])
        highLECasCyl[u] = np.column_stack((highLECasProfiles[u,:,0]/rHighLECas, rHighLECas, highLECasProfiles[u,:,1]))
        rHighTECas = casFunc(highTECasProfiles[u,:,1])
        highTECasCyl[u] = np.column_stack((highTECasProfiles[u,:,0]/rHighTECas, rHighTECas, highTECasProfiles[u,:,1]))
        rLowLECas = casFunc(lowLECasProfiles[u,:,1])
        lowLECasCyl[u] = np.column_stack((lowLECasProfiles[u,:,0]/rLowLECas, rLowLECas, lowLECasProfiles[u,:,1]))
        rLowTECas = casFunc(lowTECasProfiles[u,:,1])
        lowTECasCyl[u] = np.column_stack((lowTECasProfiles[u,:,0]/rLowTECas, rLowTECas, lowTECasProfiles[u,:,1]))           
        
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(leHubCyl[:,:,0], leHubCyl[:,:,1], leHubCyl[:,:,2], 'k.')
# ax.plot(outletHubCyl[:,:,0], outletHubCyl[:,:,1], outletHubCyl[:,:,2], 'r.')
# ax.plot(inletHubCyl[:,:,0], inletHubCyl[:,:,1], inletHubCyl[:,:,2], 'r.') 
#plt.plot(highLEHubCyl[:,:,2], highLEHubCyl[:,:,0]*highLEHubCyl[:,:,1], 'k.')
#plt.plot(highTEHubCyl[:,:,2], highTEHubCyl[:,:,0]*highTEHubCyl[:,:,1], 'm.')
#plt.plot(outletHubCyl[0,:,2], outletHubCyl[0,:,1]* outletHubCyl[0,:,0], 'r.')
#%%
inletHubCart = np.zeros([passageRes,res*mul,3])
inletCasCart = np.zeros([passageRes,res*mul,3])
leHubCart = np.zeros([passageRes,bladeRes,3])
leCasCart = np.zeros([passageRes,bladeRes,3])
teHubCart = np.zeros([passageRes,bladeRes,3])
teCasCart = np.zeros([passageRes,bladeRes,3])
outletHubCart = np.zeros([passageRes,res*mul,3])
outletCasCart = np.zeros([passageRes,res*mul,3])
inCart = np.zeros([passageRes, newNsection, 3])
outCart = np.zeros([passageRes, newNsection, 3])

inletLowThetaCart = np.zeros([newNsection, res*mul, 3])
inletHighThetaCart = np.zeros([newNsection, res*mul, 3])
outletLowThetaCart = np.zeros([newNsection, res*mul, 3])
outletHighThetaCart = np.zeros([newNsection, res*mul, 3])

LECart =  np.zeros([newNsection, passageRes, 3])
TECart = np.zeros([newNsection, passageRes, 3])
midCart = np.zeros([newNsection, passageRes, 3])
bladeLowCart = np.zeros([newNsection, bladeRes, 3])
bladeHighCart = np.zeros([newNsection, bladeRes, 3])
offsetLowCart = np.zeros([newNsection, bladeRes, 3])
offsetHighCart = np.zeros([newNsection, bladeRes, 3])

lowThetaB1Cart = np.zeros((newNsection, bladeRes, 3)) #This is the upstream portion of blade surface
lowThetaO1Cart = np.zeros((newNsection, bladeRes, 3)) #This is the upstream portion of offset surface
lowThetaB2Cart = np.zeros((newNsection, bladeRes, 3)) #This is the downstream portion of blade surface
lowThetaO2Cart = np.zeros((newNsection, bladeRes, 3)) #This is the downstream portion of offset surface

highThetaB1Cart = np.zeros((newNsection, bladeRes, 3)) #This is the upstream portion of blade surface
highThetaO1Cart = np.zeros((newNsection, bladeRes, 3)) #This is the upstream portion of offset surface
highThetaB2Cart = np.zeros((newNsection, bladeRes, 3)) #This is the downstream portion of blade surface
highThetaO2Cart = np.zeros((newNsection, bladeRes, 3)) #This is the downstream portion of offset surface

offBladeUp1Cart = np.zeros([newNsection,int(res*0.5),3])
offBladeUp2Cart = np.zeros([newNsection,int(res*0.5),3])
offBladeDw1Cart = np.zeros([newNsection,int(res*0.5),3])
offBladeDw2Cart = np.zeros([newNsection,int(res*0.5),3])

midLowCart = np.zeros([newNsection,int(res*0.5),3])
midHighCart = np.zeros([newNsection,int(res*0.5),3])

highLEHubCart = np.zeros([int(0.5*res),bladeRes,3])
highTEHubCart = np.zeros([int(0.5*res),bladeRes,3])
highLECasCart = np.zeros([int(0.5*res),bladeRes,3])
highTECasCart = np.zeros([int(0.5*res),bladeRes,3]) 
lowLEHubCart = np.zeros([int(0.5*res),bladeRes,3])
lowTEHubCart = np.zeros([int(0.5*res),bladeRes,3])
lowLECasCart = np.zeros([int(0.5*res),bladeRes,3])
lowTECasCart = np.zeros([int(0.5*res),bladeRes,3]) 

#Remember PS is high theta SS is low theta. Blade1 is high theta and Blade2 is low theta
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
        inletLowThetaCart[m] = np.array(mf.pol2cart(upCurve2Cyl[m][:,0], upCurve2Cyl[m][:,1], upCurve2Cyl[m][:,2])).T
        inletHighThetaCart[m] = np.array(mf.pol2cart(upCurve1Cyl[m][:,0], upCurve1Cyl[m][:,1], upCurve1Cyl[m][:,2])).T
        outletLowThetaCart[m] = np.array(mf.pol2cart(dwCurve2Cyl[m][:,0], dwCurve2Cyl[m][:,1], dwCurve2Cyl[m][:,2])).T
        outletHighThetaCart[m] = np.array(mf.pol2cart(dwCurve1Cyl[m][:,0], dwCurve1Cyl[m][:,1], dwCurve1Cyl[m][:,2])).T    
        LECart[m] = np.array(mf.pol2cart(LECurveCyl[m][:,0], LECurveCyl[m][:,1], LECurveCyl[m][:,2])).T  
        TECart[m] = np.array(mf.pol2cart(TECurveCyl[m][:,0], TECurveCyl[m][:,1], TECurveCyl[m][:,2])).T 
        midCart[m] = np.array(mf.pol2cart(midCurveCylMid[m][:,0], midCurveCylMid[m][:,1], midCurveCylMid[m][:,2])).T 
        midLowCart[m]  = np.array(mf.pol2cart(midCurveCylLow[m][:,0], midCurveCylLow[m][:,1], midCurveCylLow[m][:,2])).T 
        midHighCart[m] = np.array(mf.pol2cart(midCurveCylHigh[m][:,0], midCurveCylHigh[m][:,1], midCurveCylHigh[m][:,2])).T 
        
        lowThetaB1Cart[m] = np.array(mf.pol2cart(lowThetaB1Cyl[m][:,0], lowThetaB1Cyl[m][:,1], lowThetaB1Cyl[m][:,2])).T
        lowThetaO1Cart[m] = np.array(mf.pol2cart(lowThetaO1Cyl[m][:,0], lowThetaO1Cyl[m][:,1], lowThetaO1Cyl[m][:,2])).T
        lowThetaB2Cart[m] = np.array(mf.pol2cart(lowThetaB2Cyl[m][:,0], lowThetaB2Cyl[m][:,1], lowThetaB2Cyl[m][:,2])).T
        lowThetaO2Cart[m] = np.array(mf.pol2cart(lowThetaO2Cyl[m][:,0], lowThetaO2Cyl[m][:,1], lowThetaO2Cyl[m][:,2])).T
        
        highThetaB1Cart[m] = np.array(mf.pol2cart(highThetaB1Cyl[m][:,0], highThetaB1Cyl[m][:,1], highThetaB1Cyl[m][:,2])).T
        highThetaO1Cart[m] = np.array(mf.pol2cart(highThetaO1Cyl[m][:,0], highThetaO1Cyl[m][:,1], highThetaO1Cyl[m][:,2])).T
        highThetaB2Cart[m] = np.array(mf.pol2cart(highThetaB2Cyl[m][:,0], highThetaB2Cyl[m][:,1], highThetaB2Cyl[m][:,2])).T
        highThetaO2Cart[m] = np.array(mf.pol2cart(highThetaO2Cyl[m][:,0], highThetaO2Cyl[m][:,1], highThetaO2Cyl[m][:,2])).T  
        
        offBladeUp1Cart[m] = np.array(mf.pol2cart(offBladeUp1Cyl[m][:,0], offBladeUp1Cyl[m][:,1], offBladeUp1Cyl[m][:,2])).T
        offBladeUp2Cart[m] = np.array(mf.pol2cart(offBladeUp2Cyl[m][:,0], offBladeUp2Cyl[m][:,1], offBladeUp2Cyl[m][:,2])).T
        offBladeDw1Cart[m] = np.array(mf.pol2cart(offBladeDw1Cyl[m][:,0], offBladeDw1Cyl[m][:,1], offBladeDw1Cyl[m][:,2])).T
        offBladeDw2Cart[m] = np.array(mf.pol2cart(offBladeDw2Cyl[m][:,0], offBladeDw2Cyl[m][:,1], offBladeDw2Cyl[m][:,2])).T
        


for m in range(int(0.5*res)):
    highLEHubCart[m] = np.array(mf.pol2cart(highLEHubCyl[m][:,0], highLEHubCyl[m][:,1], highLEHubCyl[m][:,2])).T
    highTEHubCart[m] = np.array(mf.pol2cart(highTEHubCyl[m][:,0], highTEHubCyl[m][:,1], highTEHubCyl[m][:,2])).T
    highLECasCart[m] = np.array(mf.pol2cart(highLECasCyl[m][:,0], highLECasCyl[m][:,1], highLECasCyl[m][:,2])).T
    highTECasCart[m] = np.array(mf.pol2cart(highTECasCyl[m][:,0], highTECasCyl[m][:,1], highTECasCyl[m][:,2])).T
    lowLEHubCart[m] = np.array(mf.pol2cart(lowLEHubCyl[m][:,0], lowLEHubCyl[m][:,1], lowLEHubCyl[m][:,2])).T
    lowTEHubCart[m] = np.array(mf.pol2cart(lowTEHubCyl[m][:,0], lowTEHubCyl[m][:,1], lowTEHubCyl[m][:,2])).T
    lowLECasCart[m] = np.array(mf.pol2cart(lowLECasCyl[m][:,0], lowLECasCyl[m][:,1], lowLECasCyl[m][:,2])).T
    lowTECasCart[m] = np.array(mf.pol2cart(lowTECasCyl[m][:,0], lowTECasCyl[m][:,1], lowTECasCyl[m][:,2])).T



hubCart = np.concatenate([inletHubCart, leHubCart[:, 1:-1, :], teHubCart[:, :-1, :], outletHubCart], axis=1)
casCart = np.concatenate([inletCasCart, leCasCart[:, 1:-1, :], teCasCart[:, :-1, :], outletCasCart], axis=1)

# plt.plot(highTEHubCyl[:,:,2], highTEHubCyl[:,:,0]*highTEHubCyl[:,:,1], 'm.')
#%%
Xo1L = np.zeros([newNsection, bladeRes])
Yo1L = np.zeros([newNsection, bladeRes])
Zo1L = np.zeros([newNsection, bladeRes])
Xo1H = np.zeros([newNsection, bladeRes])
Yo1H = np.zeros([newNsection, bladeRes])
Zo1H = np.zeros([newNsection, bladeRes])

Xb1L = np.zeros([newNsection, bladeRes])
Yb1L = np.zeros([newNsection, bladeRes])
Zb1L = np.zeros([newNsection, bladeRes])
Xb1H = np.zeros([newNsection, bladeRes])
Yb1H = np.zeros([newNsection, bladeRes])
Zb1H = np.zeros([newNsection, bladeRes])

Xo2L = np.zeros([newNsection, bladeRes])
Yo2L = np.zeros([newNsection, bladeRes])
Zo2L = np.zeros([newNsection, bladeRes])
Xo2H = np.zeros([newNsection, bladeRes])
Yo2H = np.zeros([newNsection, bladeRes])
Zo2H = np.zeros([newNsection, bladeRes])

Xb2L = np.zeros([newNsection, bladeRes])
Yb2L = np.zeros([newNsection, bladeRes])
Zb2L = np.zeros([newNsection, bladeRes])
Xb2H = np.zeros([newNsection, bladeRes])
Yb2H = np.zeros([newNsection, bladeRes])
Zb2H = np.zeros([newNsection, bladeRes])

Xle = np.zeros([newNsection, passageRes])
Yle = np.zeros([newNsection, passageRes])
Zle = np.zeros([newNsection, passageRes])
Xte = np.zeros([newNsection, passageRes])
Yte = np.zeros([newNsection, passageRes])
Zte = np.zeros([newNsection, passageRes])
Xmid = np.zeros([newNsection, passageRes])
Ymid = np.zeros([newNsection, passageRes])
Zmid = np.zeros([newNsection, passageRes])

XinL = np.zeros([newNsection,res*mul])
YinL = np.zeros([newNsection,res*mul])
ZinL = np.zeros([newNsection,res*mul])
XinH = np.zeros([newNsection,res*mul])
YinH = np.zeros([newNsection,res*mul])
ZinH = np.zeros([newNsection,res*mul])

XoutL = np.zeros([newNsection,res*mul])
YoutL = np.zeros([newNsection,res*mul])
ZoutL = np.zeros([newNsection,res*mul])
XoutH = np.zeros([newNsection,res*mul])
YoutH = np.zeros([newNsection,res*mul])
ZoutH = np.zeros([newNsection,res*mul])

XmidLow = np.zeros([newNsection, int(0.5*res)])
YmidLow = np.zeros([newNsection, int(0.5*res)])
ZmidLow = np.zeros([newNsection, int(0.5*res)])

XmidHigh = np.zeros([newNsection, int(0.5*res)])
YmidHigh = np.zeros([newNsection, int(0.5*res)])
ZmidHigh = np.zeros([newNsection, int(0.5*res)])

XhUpH = np.zeros([int(0.5*res),bladeRes])
YhUpH = np.zeros([int(0.5*res),bladeRes])
ZhUpH = np.zeros([int(0.5*res),bladeRes])
XhDwH = np.zeros([int(0.5*res),bladeRes])
YhDwH = np.zeros([int(0.5*res),bladeRes])
ZhDwH = np.zeros([int(0.5*res),bladeRes])

XcUpH = np.zeros([int(0.5*res),bladeRes])
YcUpH = np.zeros([int(0.5*res),bladeRes])
ZcUpH = np.zeros([int(0.5*res),bladeRes])
XcDwH = np.zeros([int(0.5*res),bladeRes])
YcDwH = np.zeros([int(0.5*res),bladeRes])
ZcDwH = np.zeros([int(0.5*res),bladeRes])

XhUpL = np.zeros([int(0.5*res),bladeRes])
YhUpL = np.zeros([int(0.5*res),bladeRes])
ZhUpL = np.zeros([int(0.5*res),bladeRes])
XhDwL = np.zeros([int(0.5*res),bladeRes])
YhDwL = np.zeros([int(0.5*res),bladeRes])
ZhDwL = np.zeros([int(0.5*res),bladeRes])

XcUpL = np.zeros([int(0.5*res),bladeRes])
YcUpL = np.zeros([int(0.5*res),bladeRes])
ZcUpL = np.zeros([int(0.5*res),bladeRes])
XcDwL = np.zeros([int(0.5*res),bladeRes])
YcDwL = np.zeros([int(0.5*res),bladeRes])
ZcDwL = np.zeros([int(0.5*res),bladeRes])


Xh = np.zeros([passageRes, mul*res*2 + bladeRes*2 -3])
Yh = np.zeros([passageRes, mul*res*2 + bladeRes*2 -3])
Zh = np.zeros([passageRes, mul*res*2 + bladeRes*2 -3])
Xc = np.zeros([passageRes, mul*res*2 + bladeRes*2 -3])
Yc = np.zeros([passageRes, mul*res*2 + bladeRes*2 -3])
Zc = np.zeros([passageRes, mul*res*2 + bladeRes*2 -3])

Xi = np.zeros([passageRes,newNsection])
Yi = np.zeros([passageRes,newNsection])
Zi = np.zeros([passageRes,newNsection])
Xo = np.zeros([passageRes,newNsection])
Yo = np.zeros([passageRes,newNsection])
Zo = np.zeros([passageRes,newNsection])

XobUpH = np.zeros([newNsection,int(res*0.5)])
YobUpH = np.zeros([newNsection,int(res*0.5)])
ZobUpH = np.zeros([newNsection,int(res*0.5)])
XobDwH = np.zeros([newNsection,int(res*0.5)])
YobDwH = np.zeros([newNsection,int(res*0.5)])
ZobDwH = np.zeros([newNsection,int(res*0.5)])
XobUpL = np.zeros([newNsection,int(res*0.5)])
YobUpL = np.zeros([newNsection,int(res*0.5)])
ZobUpL = np.zeros([newNsection,int(res*0.5)])
XobDwL = np.zeros([newNsection,int(res*0.5)])
YobDwL = np.zeros([newNsection,int(res*0.5)])
ZobDwL = np.zeros([newNsection,int(res*0.5)])
for p in range(newNsection):
    for q in range(bladeRes):
        Xo1L[p,q] = lowThetaO1Cart[p][q,0]
        Yo1L[p,q] = lowThetaO1Cart[p][q,1]
        Zo1L[p,q] = lowThetaO1Cart[p][q,2]
        Xo1H[p,q] = highThetaO1Cart[p][q,0]
        Yo1H[p,q] = highThetaO1Cart[p][q,1]
        Zo1H[p,q] = highThetaO1Cart[p][q,2]     
        Xb1L[p,q] = lowThetaB1Cart[p][q,0]
        Yb1L[p,q] = lowThetaB1Cart[p][q,1]
        Zb1L[p,q] = lowThetaB1Cart[p][q,2]
        Xb1H[p,q] = highThetaB1Cart[p][q,0]
        Yb1H[p,q] = highThetaB1Cart[p][q,1]
        Zb1H[p,q] = highThetaB1Cart[p][q,2]   
        Xo2L[p,q] = lowThetaO2Cart[p][q,0]
        Yo2L[p,q] = lowThetaO2Cart[p][q,1]
        Zo2L[p,q] = lowThetaO2Cart[p][q,2]
        Xo2H[p,q] = highThetaO2Cart[p][q,0]
        Yo2H[p,q] = highThetaO2Cart[p][q,1]
        Zo2H[p,q] = highThetaO2Cart[p][q,2]     
        Xb2L[p,q] = lowThetaB2Cart[p][q,0]
        Yb2L[p,q] = lowThetaB2Cart[p][q,1]
        Zb2L[p,q] = lowThetaB2Cart[p][q,2]
        Xb2H[p,q] = highThetaB2Cart[p][q,0]
        Yb2H[p,q] = highThetaB2Cart[p][q,1]
        Zb2H[p,q] = highThetaB2Cart[p][q,2] 
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
    for rr in range(res*mul):
        XinL[p,rr] = inletLowThetaCart[p][rr,0]
        YinL[p,rr] = inletLowThetaCart[p][rr,1]
        ZinL[p,rr] = inletLowThetaCart[p][rr,2]
        XinH[p,rr] = inletHighThetaCart[p][rr,0]
        YinH[p,rr] = inletHighThetaCart[p][rr,1]
        ZinH[p,rr] = inletHighThetaCart[p][rr,2]
        XoutL[p,rr] = outletLowThetaCart[p][rr,0]
        YoutL[p,rr] = outletLowThetaCart[p][rr,1]
        ZoutL[p,rr] = outletLowThetaCart[p][rr,2]
        XoutH[p,rr] = outletHighThetaCart[p][rr,0]
        YoutH[p,rr] = outletHighThetaCart[p][rr,1]
        ZoutH[p,rr] = outletHighThetaCart[p][rr,2]      
    for rs in range(int(0.5*res)):
        XobUpH[p,rs] = offBladeUp1Cart[p][rs,0]
        YobUpH[p,rs] = offBladeUp1Cart[p][rs,1]
        ZobUpH[p,rs] = offBladeUp1Cart[p][rs,2]
        XobDwH[p,rs] = offBladeDw1Cart[p][rs,0]
        YobDwH[p,rs] = offBladeDw1Cart[p][rs,1]
        ZobDwH[p,rs] = offBladeDw1Cart[p][rs,2]
        XobUpL[p,rs] = offBladeUp2Cart[p][rs,0]
        YobUpL[p,rs] = offBladeUp2Cart[p][rs,1]
        ZobUpL[p,rs] = offBladeUp2Cart[p][rs,2]
        XobDwL[p,rs] = offBladeDw2Cart[p][rs,0]
        YobDwL[p,rs] = offBladeDw2Cart[p][rs,1]
        ZobDwL[p,rs] = offBladeDw2Cart[p][rs,2]
        XmidLow[p,rs]  = midLowCart[p][rs,0]
        YmidLow[p,rs]  = midLowCart[p][rs,1]
        ZmidLow[p,rs]  = midLowCart[p][rs,2]
        XmidHigh[p,rs] = midHighCart[p][rs,0]
        YmidHigh[p,rs] = midHighCart[p][rs,1]
        ZmidHigh[p,rs] = midHighCart[p][rs,2]

for s in range(passageRes):
    for t in range(mul*res*2 + bladeRes*2 -3):
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
for s in range(int(0.5*res)):
    for t in range(bladeRes):
        XhUpH[s,t] = highLEHubCart[s][t,0]
        YhUpH[s,t] = highLEHubCart[s][t,1]
        ZhUpH[s,t] = highLEHubCart[s][t,2]
        XcUpH[s,t] = highLECasCart[s][t,0]
        YcUpH[s,t] = highLECasCart[s][t,1]
        ZcUpH[s,t] = highLECasCart[s][t,2]       
        XhDwH[s,t] = highTEHubCart[s][t,0]
        YhDwH[s,t] = highTEHubCart[s][t,1]
        ZhDwH[s,t] = highTEHubCart[s][t,2]
        XcDwH[s,t] = highTECasCart[s][t,0]
        YcDwH[s,t] = highTECasCart[s][t,1]
        ZcDwH[s,t] = highTECasCart[s][t,2]
        
        XhUpL[s,t] = lowLEHubCart[s][t,0]
        YhUpL[s,t] = lowLEHubCart[s][t,1]
        ZhUpL[s,t] = lowLEHubCart[s][t,2]
        XcUpL[s,t] = lowLECasCart[s][t,0]
        YcUpL[s,t] = lowLECasCart[s][t,1]
        ZcUpL[s,t] = lowLECasCart[s][t,2]       
        XhDwL[s,t] = lowTEHubCart[s][t,0]
        YhDwL[s,t] = lowTEHubCart[s][t,1]
        ZhDwL[s,t] = lowTEHubCart[s][t,2]
        XcDwL[s,t] = lowTECasCart[s][t,0]
        YcDwL[s,t] = lowTECasCart[s][t,1]
        ZcDwL[s,t] = lowTECasCart[s][t,2]

#%%
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Zo1L, Yo1L, Xo1L,  alpha=0.8,)
# ax.plot_surface(Zo1H, Yo1H, Xo1H,  alpha=0.8)
# ax.plot_surface(ZbL, YbL, XbL,  alpha=0.8,)
# ax.plot_surface(ZbH, YbH, XbH,  alpha=0.8)
# ax.plot_surface(Zle, Yle, Xle, alpha=0.8)
# ax.plot_surface(Zte, Yte, Xte, alpha=0.8)
# ax.plot_surface(Zi,Yi,Xi, alpha=0.8)
# ax.plot_surface(Zc,Yc,Xc, alpha=0.8)
#ax.plot_surface(Zc,Yc,Xc, alpha=0.8)
#ax.plot_surface(ZcDwH,YcDwH,XcDwH, alpha=0.8)
# ax.plot_surface(ZinL, YinL, XinL,  alpha=0.8)
# ax.plot_surface(ZinH, YinH, XinH,  alpha=0.8)
# ax.plot_surface(ZoutL, YoutL, XoutL,  alpha=0.8)
# ax.plot_surface(ZoutH, YoutH, XoutH,  alpha=0.8)


#%% Defining the STLs
filenames = ['blade1LowTheta','offset1LowTheta','blade1HighTheta','offset1HighTheta','blade2LowTheta','offset2LowTheta','blade2HighTheta','offset2HighTheta'
             ,'LE','TE','midChord', 'hub', 'casing', 'inlet', 'outlet', 'inletLow', 'inletHigh', 'outletLow', 'outletHigh', 'highUpHub', 'highDwHub',
             'lowUpHub', 'lowDwHub', 'highUpCas', 'highDwCas', 'lowUpCas', 'lowDwCas', 'highOBUp', 'lowOBUp', 'highOBDw', 'lowOBDw', 'midChordLow', 'midChordHigh']

scale = 0.001
Xvalues = [Xb1L, Xo1L, Xb1H, Xo1H, Xb2L, Xo2L, Xb2H, Xo2H, Xle, Xte, Xmid, Xh, Xc, Xi, Xo, XinL, XinH, XoutL, XoutH, XhUpH, XhDwH, XhUpL, XhDwL, XcUpH, XcDwH, XcUpL, XcDwL, XobUpH, XobUpL, XobDwH, XobDwL, XmidLow, XmidHigh]
Yvalues = [Yb1L, Yo1L, Yb1H, Yo1H, Yb2L, Yo2L, Yb2H, Yo2H, Yle, Yte, Ymid, Yh, Yc, Yi, Yo, YinL, YinH, YoutL, YoutH, YhUpH, YhDwH, YhUpL, YhDwL, YcUpH, YcDwH, YcUpL, YcDwL, YobUpH, YobUpL, YobDwH, YobDwL, YmidLow, YmidHigh]
Zvalues = [Zb1L, Zo1L, Zb1H, Zo1H, Zb2L, Zo2L, Zb2H, Zo2H, Zle, Zte, Zmid, Zh, Zc, Zi, Zo, ZinL, ZinH, ZoutL, ZoutH, ZhUpH, ZhDwH, ZhUpL, ZhDwL, ZcUpH, ZcDwH, ZcUpL, ZcDwL, ZobUpH, ZobUpL, ZobDwH, ZobDwL, ZmidLow, ZmidHigh]
#             0     1     2     3     4     5     6     7    8    9    10  11  12  13  14    15    16     17     18     19     20     21     22     23     24     25     26      27      28      29      30       31        32

# Note: this creates STLs in Cartesian coordinates -- good for visual checks
for qq in range(len(Xvalues)):
    filename = filePath + '/../geometry1/{}.stl'.format(filenames[qq])
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
    
    def unitVector(file, p1, p2, p3):
        # VECTORS TANGENT TO FACET
        vector1 = p3 - p2
        vector2 = p3 - p1
        
        normalVec = np.cross(vector1, vector2)
        # magnitude = (normalVec[0]**2 + normalVec[1]**2 + normalVec[2]**2)**0.5
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

bpsg.calcAndWritePassageParameters(scale, Xvalues, Yvalues, Zvalues, nrad, delHub, delCas, delBla, dy1Hub, dy1Cas, dy1Bla, gRad, gTan, dax1primeLE, rLE, dax1primeTE, rTE, rUpFar, rDnFar, dataPath, additionalTangentialRefine, additionalAxialRefine, highHub1, lowHub1, highCas1, lowCas1, highHub2, lowHub2, highCas2, lowCas2)
