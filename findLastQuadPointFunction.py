#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jeff Defoe
October 2025
"""

import numpy as np
from scipy.optimize import fsolve
import model_function as mf

""" Example input data:
A = np.array([140.346, -6.38901, 174.524])
B = np.array([140.455, -3.17045, 174.524])
C = np.array([140.49, -0.431207, 174.524])
D = np.array([140.463, 2.81166, 174.524])
E = np.array([128.239, 12.5022, 141.777])
meridCurve = np.loadtxt('hub.curve', delimiter=' ')
meridCurve = np.delete(meridCurve, 1, axis=1)
"""

def circleEqns(vars, a, b, c, delta, rc, zc, phat, qhat):
    """ Equation set to solve to get 3D coordinates of F point """
    cosTheta, sinTheta = vars
    xc = a + delta*cosTheta*phat[0] + delta*sinTheta*qhat[0]
    yc = b + delta*cosTheta*phat[1] + delta*sinTheta*qhat[1]
    eq1 = -rc + np.sqrt(xc**2 + yc**2)
    eq2 = -zc + c + delta*cosTheta*phat[2] + delta*sinTheta*qhat[2]
    return np.array([eq1, eq2])


def getFvertex(A, B, C, D, E, meridCurve):
    """
    Function to find the offset surface endpoint
    using a 3D approach.

    Points A-E are 3-element numpy arrays in Cartesian coordinates.
    Point A is the minimum theta point at midchord
    Point B is the blade surface point at midchord (near A)
    Point C is the blade surface point at midchord (near D)
    Point D is the maximum theta point at midchord
    Point E is the blade leading/trailing edge point
    and streamsurface meridional coordinates are meridCurve ((r, z) pairs)

    To find: point F, which will form two convex quads,
    ABEF and DCEF.

    F lies on the bisector plane of the angle BEC.
    F also lies on the surface of revolution of the streamsurface.
    """

    # Compute distance to go ahead of LE/TE point
    delta = np.average([np.linalg.norm(A-B), np.linalg.norm(C-D)])

    # 1. Find the bisector plane.

    u = B - E  # direction of EB
    uhat = u/np.linalg.norm(u)  # normalized unit vector
    v = C - E  # direction of EC
    vhat = v/np.linalg.norm(v)  # normaliezd unit vector
    # Find normal to plane
    n = uhat-vhat
    nhat = n / np.linalg.norm(n)  # normalized unit vector
    # The bisector plane is defined by the point E and the normal nhat.
    # from: https://www.youtube.com/watch?v=KFD3qfQV34o
    # Equation of the plane is given as follows...
    #  nhat = [p, q, r]
    #  E = [a, b, c]
    # Then
    # p*(x-a)+q*(y-b)+r*(z-c)=0
    # and we have a sphere which will have a max diameter intersection with
    # the plane if it has equation (x-a)**2+(y-b)**2+(z-c)**2=delta**2
    # p = nhat[0]
    # q = nhat[1]
    # r = nhat[2]
    a = E[0]
    b = E[1]
    c = E[2]

    # 2. Create a circle in that plane of radius delta.

    # Combining the above equations:
    # ((x-a)**2+(y-b)**2+(z-c)**2-delta**2)**2 + (p*(x-a)+q*(y-b)+r*(z-c))**2 = 0
    # ... but putting in parametric general form much better

    # Need to get 2 vectors in plane u1, v1.
    # Normalized: u1hat v1hat
    # Orthogonal directions: phat = hat(u1hat+v1hat), qhat = hat(u1hat-v1hat)
    # Then:
    # [x, y, z] = [a, b, c] + delta*np.cos(theta)*phat + delta*np.sin(theta)*qhat
    # for 0 <= theta <= 2*np.pi
    #
    # So problem becomes just getting u1, v1. These are any two non-parallel vectors
    # which lie in our plane.

    vt = np.array([0, 0, 1])  # this will never be parallel to n

    u1 = np.cross(nhat, vt)  # a vector perpendicular to n, thus in the plane
    v1 = np.cross(nhat, u1)  # a vector perpendicular to u1 and v1, thus in the plane too

    u1hat = u1/np.linalg.norm(u1)
    v1hat = v1/np.linalg.norm(v1)

    phat = (u1hat+v1hat)/np.linalg.norm(u1hat+v1hat)
    qhat = (u1hat-v1hat)/np.linalg.norm(u1hat-v1hat)

    theta = np.linspace(0, 2*np.pi, num=1000, endpoint=True)

    xc = a + delta*np.cos(theta)*phat[0] + delta*np.sin(theta)*qhat[0]
    yc = b + delta*np.cos(theta)*phat[1] + delta*np.sin(theta)*qhat[1]
    zc = c + delta*np.cos(theta)*phat[2] + delta*np.sin(theta)*qhat[2]

    # 3. Project the circle onto the meridional plane, get an ellipse.

    # To do this, we can just consider the r and z coordinates.
    rc = np.sqrt(xc**2 + yc**2)

    # 4. Find the intersections of the ellipse and the meridional curve.
    # Use function Adekola wrote for this which works (only for 2D data)
    intersectPts = mf.TwoLinesIntersect(np.column_stack((rc, zc)), meridCurve)
    intersectPts = np.array(intersectPts)

    print(f"Intersection points: z = {intersectPts[:,1]}, r = {intersectPts[:,0]}")

    # 5. Choose the correct intersection point.

    # get vector which has axial/radial part which is in the streamwise direction
    vertexVect = uhat+vhat
    vertexVectMerid = np.array([np.sqrt(vertexVect[0]**2 + vertexVect[1]**2), vertexVect[2]])
    vertexVectMeridHat = vertexVectMerid / np.linalg.norm(vertexVectMerid)

    # Create arclength along streamsurface
    localArcLengths = np.sqrt(np.diff(meridCurve[:, 0])**2 + np.diff(meridCurve[:, 1])**2)
    localCumulativeArcLength = np.cumsum(localArcLengths)
    localCumulativeArcLength = np.insert(localCumulativeArcLength, 0, 0)

    # Find arclength for each intersection; use linear interpolation of coordinates
    arcLenIntersections = np.interp(intersectPts[:, 1], meridCurve[:, 1], localCumulativeArcLength)
    # if 1st intersection point has higher arc length, swap order:
    if arcLenIntersections[0] > arcLenIntersections[1]:
        intersectPts = np.flipud(intersectPts)
        print('Swapping order of intersection points...')

    # vector from upstream intersection pt to downstream one
    intersectVect = intersectPts[1, :] - intersectPts[0, :]

    # dot product -- positive means LE, negative means TE
    projResult = np.dot(vertexVectMeridHat, intersectVect)

    if projResult > 0:
        keptIntersection = intersectPts[0, :]
        print('Upstream intersection kept -- working around LE')
    elif projResult < 0:
        keptIntersection = intersectPts[1, :]
        print('Downstream intersection kept -- working around TE')

    # 6. Extract the actual xyz coords and return as the coordinate F

    # Set up 2 equations in 2 unknowns for cos(theta) and sin(theta), using
    # the values of rc and vz for the keptIntersection.
    # End up with a system of nonlinear equations so use fsolve.

    argList = (a, b, c, delta, keptIntersection[0], keptIntersection[1], phat, qhat)
    root = fsolve(circleEqns, x0=[1/np.sqrt(2), 1/np.sqrt(2)], args=argList)
    cosTheta, sinTheta = root

    thetaF = np.arctan2(sinTheta, cosTheta)

    Fx = a + delta*np.cos(thetaF)*phat[0] + delta*np.sin(thetaF)*qhat[0]
    Fy = b + delta*np.cos(thetaF)*phat[1] + delta*np.sin(thetaF)*qhat[1]
    Fz = c + delta*np.cos(thetaF)*phat[2] + delta*np.sin(thetaF)*qhat[2]

    F = np.array([Fx, Fy, Fz])

    return F
