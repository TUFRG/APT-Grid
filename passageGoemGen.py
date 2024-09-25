#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Updated 16 September 2024

@author: Adekola Adeyemi and Jeff Defoe
"""

# Import packages
import numpy as np
from scipy.interpolate import interp1d
import model_function as mf  # custom library of functions called
import TransfiniteInterpolation as tf  # function for transfinite interpolation in here

# Set inputs
nSections = 23  # number of blade sections. Currently assumed to be < 100
# Would like to automate this to check folder to figure out how many there are
pathToSectionFiles = 'bladeSections'  # relative path, or can be absolute
pathToMeridGeomFiles = 'gasPath'  # relative path, or can be absolute
dataScale = 0.001  # scale relative to base SI units of input data, e.g. 0.001 for mm
pathForSTLs = 'constant/geometry'  # relative path, or can be absolute
# (don't change for automatic OF compatibility)
scriptFile = 'system/passageGeometry'  # filename for script to modify blockMeshDict
# Still need to clean up filenames to make it generic
Nb = 31  # number of blades in the blade row

# NOTE: Currently the assumption is that the z axis is the axis of rotation.
# f this is not the case, additional pre-processing of the input files is
# needed to conform to this.

# NOTE: A straight line is assumed for the inlet and outlet between the ends
# of the hub and casing curves (in the meridional plane)

# Resolution of extensions of camber line upstream and downsteam
# (generally don't need to change):
N = 30
# Number of points to discretize in circumferential direction
thetaPts = 360  # must yield smaller triangles than desired grid cells in CFD

# Import data from files
bladeP = []
bladeS = []
LE = np.empty([nSections, 2])
TE = np.empty([nSections, 2])
# sections; pressure and suction sides already divided:
# Note here the P stuff is assigned from the suction surfaces files due to the way
# the code is structured, naming cleanup needs to be done later
for a in range(nSections):
    if a < 10:
        fileP = np.loadtxt(pathToSectionFiles + '/' + 'EX_OGV_SuctionSide_0{}.dat'.format(a))
        bladeP.append(fileP)
        fileS = np.loadtxt(pathToSectionFiles + '/' + 'EX_OGV_PressureSide_0{}.dat'.format(a))
        bladeS.append(fileS)
    else:
        fileP = np.loadtxt(pathToSectionFiles + '/' + 'EX_OGV_SuctionSide_{}.dat'.format(a))
        bladeP.append(fileP)
        fileS = np.loadtxt(pathToSectionFiles + '/' + 'EX_OGV_PressureSide_{}.dat'.format(a))
        bladeS.append(fileS)
    LE[a] = [fileP[0, 2], fileP[0, 0]]  # meridional locus of LE points (z, r)
    TE[a] = [fileP[len(fileP)-1, 2], fileP[len(fileP)-1, 0]]  # meridional locus of TE points (z,r)
# hub and casing curves (extends: desired length of domain):
hub = np.loadtxt(pathToMeridGeomFiles + '/' + 'sHub.curve')  # these are (r, theta, z)
cas = np.loadtxt(pathToMeridGeomFiles + '/' + 'sCasing.curve')  # these are (r, theta, z)

# Get number of points per side in sections
nPtsPerSide = len(bladeS[0])  # Note all sections assumed to be comprised of the same number of points!

# Set up approximate camber surface variables
allCamber = np.zeros([len(bladeP), nPtsPerSide + 2*N, 3])  # holds full camber surface, for all profiles, from inlet to outlet (Cartesian)
allBlade = np.zeros([len(bladeP), bladeP[0].shape[0]*2, 3])  # holds blade surface (SS and PS) for all profiles (Cartesian)
justCamber = np.zeros([len(bladeP), nPtsPerSide, 3])  # holds camber surface from LE to TE for all profiles (Cartesian)
OUTxyz = np.zeros([len(bladeP), 3])  # outlet curve in Cartesian coordinates
INxyz = np.zeros([len(bladeP), 3])  # inlet curve in Cartesian coordinates
camberTheta = np.empty([len(bladeP), nPtsPerSide + 2*N, 1])  # theta coordinates of full camber surface (rad)

pBlade = np.zeros([len(bladeP), bladeP[0].shape[0], 3])  # assemble all PS sections (Cartesian)
sBlade = np.zeros([len(bladeP), bladeP[0].shape[0], 3])  # assemble all SS sections (Cartesian)

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
nIN[:, 1] = mf.scale(max(LE[:, 1]), min(LE[:, 1]),
                     casIN[1], hubIN[1], LE[:, 1])
f = interp1d(upLine[:, 1], upLine[:, 0])
nIN[:, 0] = f(nIN[:, 1])  # meridional (z,r) coordinates of inlet curve
nIN = np.flip(nIN,0)  # flip order so points are from hub to casing

nOUT = np.zeros([len(TE), 2])
nOUT[:, 1] = mf.scale(min(TE[:, 1]), max(TE[:, 1]),
                     hubOUT[1], casOUT[1], TE[:, 1])
f = interp1d(dwLine[:, 1], dwLine[:, 0])
nOUT[:, 0] = f(nOUT[:, 1])  # meridional (z,r) coordinates of outlet curve
nOUT = np.flip(nOUT,0)  # flip order so points are from hub to casing

# Before we figure out the 3D profiles of the interior points of the camber
# line extensions, we need to use transfinite interpolation to get the
# interior points on the meridional plane.
#
# For now, use linear interpolation to create a set of points on the hub and
# casing that have the save number of points. Later, improve this using cubic
# spline-based interpolation.
#
# The camber surface generation routine (within the blade row) will already
# produce the points needed within the row, so we only need concern ourselves
# with the upstream and downstream regions.
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
hubNUp[N, 1] = LE[i, 1]
# Then downstream extension
hubNDn = np.zeros([N+1, 2])  # (z,r)
hubNDn[:, 0] = np.linspace(start=TE[i, 0], stop=nOUT[i, 0], num=N+1)
hubNDn[:, 1] = np.interp(hubNDn[:, 0], hub[:, 2], hub[:, 0])
hubNDn[0, 1] = TE[i, 1]

# Casing
i = nSections-1  # last section is the casing
# Start with the upstream extension
casNUp = np.zeros([N+1, 2])  # (z,r)
casNUp[:, 0] = np.linspace(start=nIN[i, 0], stop=LE[i, 0], num=N+1)
casNUp[:, 1] = np.interp(casNUp[:, 0], cas[:, 2], cas[:, 0])
casNUp[N, 1] = LE[i, 1]
# Then downstream extension
casNDn = np.zeros([N+1, 2])  # (z,r)
casNDn[:, 0] = np.linspace(start=TE[i, 0], stop=nOUT[i, 0], num=N+1)
casNDn[:, 1] = np.interp(casNDn[:, 0], cas[:, 2], cas[:, 0])
casNDn[0, 1] = TE[i, 1]

# Do transfinite interpolation to get (z,r) coordinates of interior points
nodesUp = tf.transfinite(hubNUp, casNUp, nIN, LE)
nodesDn = tf.transfinite(hubNDn, casNDn, TE, nOUT)

# Loop over profiles
for i in range(len(bladeP)):
    # Generate approximate camber line between LE and TE

    # Convert Data to Cylindrical coordinates
    PScyl = mf.cart2pol(bladeP[i][:, 0], bladeP[i][:, 1], bladeP[i][:, 2])
    SScyl = mf.cart2pol(bladeS[i][:, 0], bladeS[i][:, 1], bladeS[i][:, 2])

    r = 0.5*(PScyl[1] + SScyl[1])
    theta = 0.5*(PScyl[0] + SScyl[0])
    z = 0.5*(PScyl[2] + SScyl[2])
    camberLinecyl = np.concatenate((theta, r, z)).reshape((-1, 3), order='F')

    # convert data to (z,r*theta)
    PSzrq = np.zeros([len(bladeP[0]), 2])
    PSzrq[:, 1] = PScyl[0]*(PScyl[1])
    PSzrq[:, 0] = PScyl[2]
    SSzrq = np.zeros([len(bladeS[0]), 2])
    SSzrq[:, 1] = SScyl[0]*SScyl[1]
    SSzrq[:, 0] = SScyl[2]
    LEzrq = PSzrq[0]
    TEzrq = PSzrq[len(PSzrq)-1]
    camberLinezrq = np.zeros([len(camberLinecyl), 2])
    camberLinezrq[:, 1] = theta*r
    camberLinezrq[:, 0] = z

    # Extend camber line to domain region bounds upstream and downstream
    LEslope = (camberLinezrq[1][1] - camberLinezrq[0][1]) / (camberLinezrq[1][0] - camberLinezrq[0][0])
    TEslope = (camberLinezrq[len(camberLinezrq)-1][1] - camberLinezrq[len(camberLinezrq)-2]
               [1]) / (camberLinezrq[len(camberLinezrq)-1][0] - camberLinezrq[len(camberLinezrq)-2][0])

    perpLEslope = -1/LEslope
    perpTEslope = -1/TEslope

    # Create upstream curve
    LEextPt = [LEzrq[0]+1, LEzrq[1]+perpLEslope]
    upLinePerp = mf.line(LEzrq, LEextPt)
    upExtPts = [nIN[::-1][i,0]+0, hubIN[1]+1]
    inletLine = mf.line(nIN[::-1][i], upExtPts)
    INcenter = mf.intersection(upLinePerp, inletLine)
    radIN = mf.dist2D(LEzrq[0], LEzrq[1], INcenter[0], INcenter[1])
    zINCurve = nodesUp[i::nSections, 0]  # axial coordinates from transfinite interpolation
    rqINCurve = INcenter[1] - np.sign(INcenter[1]) * \
        np.sqrt(np.abs(radIN**2 - (zINCurve - INcenter[0])**2))
    upCurve = np.concatenate((zINCurve[:-1], rqINCurve[:-1])).reshape((-1, 2), order='F')

    # Create downstream curve
    TEextPt = [TEzrq[0]+1, TEzrq[1]+perpTEslope]
    dwLinePerp = mf.line(TEzrq, TEextPt)
    dwExtPts = [nOUT[::-1][i,0]+0, nOUT[::-1][i,1]+1]
    outletLine = mf.line(nOUT[::-1][i], dwExtPts)
    OUTcenter = mf.intersection(dwLinePerp, outletLine)
    radOUT = mf.dist2D(TEzrq[0], TEzrq[1], OUTcenter[0], OUTcenter[1])
    zOUTCurve = nodesDn[i::nSections, 0]  # axial coordinates from transfinite interpolation
    rqOUTCurve = OUTcenter[1] - np.sign(OUTcenter[1]) * \
        np.sqrt(np.abs(radOUT**2 - (zOUTCurve - OUTcenter[0])**2))
    dwCurve = np.concatenate((zOUTCurve[1:], rqOUTCurve[1:])).reshape((-1, 2), order='F')

    # Take everything back to 3D (Cylindrical)

    # inlet
    zNew = zINCurve
    rNew = nodesUp[i::nSections, 1]  # radial coordinates from transfinite interpolation

    # outlet
    vNew = zOUTCurve
    wNew = nodesDn[i::nSections, 1]  # radial coordinates from transfinite interpolation

    # cylindrical coordinates (theta, r, z) camber surfaces for extensions
    camber3DIN = np.array([rqINCurve/rNew, rNew, zNew]).T  #[::-1]
    camber3DOUT = np.array([rqOUTCurve/wNew, wNew, vNew]).T

    # combine all Camber
    camber = np.concatenate((camber3DIN, camberLinecyl[1:len(
        camberLinecyl)-1], camber3DOUT)).reshape((-1, 3), order='F')

    # Return back to Cartesian Coordinate System
    camberTheta[i,:,:] = camber[:,0].reshape(-1,1)
    medCamber = mf.pol2cart(camber[:, 0], camber[:, 1], camber[:, 2])
    medCamber = np.array(medCamber).T
    allCamber[i, :, :] = medCamber

    # combine the blade also
    blade = np.concatenate(
        (bladeP[i], bladeS[i][::-1])).reshape((-1, 3), order='F')
    allBlade[i, :, :] = blade
    onlyCamber = mf.pol2cart(camberLinecyl[:, 0], camberLinecyl[:, 1], camberLinecyl[:, 2])
    onlyCamber = np.array(onlyCamber).T
    justCamber[i, :, :] = onlyCamber
    OUTxyz[i, :] = medCamber[::-1][0]
    INxyz[i, :] = medCamber[0]

    pBlade[i, :, :] = bladeP[i]
    sBlade[i, :, :] = bladeS[i]

# Now rotate the camber surface to create the passage boundaries
halfPitch = (360/Nb)*0.5

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

# Initialize arrays for some surfaces
Angle = np.linspace(0, 2*np.pi, thetaPts)
outlet = np.zeros([thetaPts, nSections, 3])
inlet = np.zeros([thetaPts, nSections, 3])
leadEdge = np.zeros([thetaPts, len(LE), 3])
trailEdge = np.zeros([thetaPts, len(TE), 3])
hubS = np.zeros([thetaPts, len(hub), 3])
casing = np.zeros([thetaPts, len(cas), 3])

# Fill arrays with data
for b in range(thetaPts):
    outlet[b, :, :] = np.array(mf.vectorRot3D(
        OUTxyz[:, 2], OUTxyz[:, 1], OUTxyz[:, 0], Angle[b])).T
    inlet[b, :, :] = np.array(mf.vectorRot3D(
        INxyz[:, 2], INxyz[:, 1], INxyz[:, 0], Angle[b])).T
    leadEdge[b, :, :] = np.array(mf.vectorRot3D(
        LE[:, 0], LE[:, 1], np.zeros(LE[:,1].shape), Angle[b])).T
    trailEdge[b, :, :] = np.array(mf.vectorRot3D(
        TE[:, 0], TE[:, 1], np.zeros(TE[:,1].shape), Angle[b])).T
    hubS[b, :, :] = np.array(mf.vectorRot3D(
        hub[:, 2], hub[:, 1], hub[:, 0], Angle[b])).T
    casing[b, :, :] = np.array(mf.vectorRot3D(
        cas[:, 2], cas[:, 1], cas[:, 0], Angle[b])).T

# Get 50% axial chord surface data
halfCamber = np.zeros([nSections, 3])
for c in range(len(justCamber)):
    z_min, z_max = np.min(justCamber[c][:, 2]), np.max(justCamber[c][:, 2])
    zhalf = 0.5*(z_min + z_max)
    yhalf = np.interp(zhalf, justCamber[c][:, 2], justCamber[c][:, 1])
    xhalf = np.interp(zhalf, justCamber[c][:, 2], justCamber[c][:, 0])
    halfCamber[c] = [xhalf, yhalf, zhalf]
midChord = np.zeros([thetaPts, nSections, 3])
for d in range(thetaPts):
    midChord[d, :, :] = np.array(mf.vectorRot3D(
        halfCamber[:, 2], halfCamber[:, 1], halfCamber[:, 0], Angle[d])).T

# Create surfaces used for projection of the template mesh
# Parameters are numbers of points, none of these are new,
# independent parameters
Nz = len(bladeP)
Nr = len(camber)
Nrb = len(PScyl[0])
Nle = len(LE)
Nh = len(hub)
Nc = len(cas)

Xp = np.zeros([Nz, Nr])  # pPeriodic
Yp = np.zeros([Nz, Nr])
Zp = np.zeros([Nz, Nr])

Xm = np.zeros([Nz, Nr])  # mPeriodic
Ym = np.zeros([Nz, Nr])
Zm = np.zeros([Nz, Nr])

Xn = np.zeros([Nz, Nr])  # nPeriodic
Yn = np.zeros([Nz, Nr])
Zn = np.zeros([Nz, Nr])

XBp = np.zeros([Nz, Nrb])  # Blade Pressure Side
YBp = np.zeros([Nz, Nrb])
ZBp = np.zeros([Nz, Nrb])

XBs = np.zeros([Nz, Nrb])  # Blade Sunction Side
YBs = np.zeros([Nz, Nrb])
ZBs = np.zeros([Nz, Nrb])

XLE = np.zeros([thetaPts, Nle])  # Blade LE
YLE = np.zeros([thetaPts, Nle])
ZLE = np.zeros([thetaPts, Nle])

XTE = np.zeros([thetaPts, Nle])  # Blade TE
YTE = np.zeros([thetaPts, Nle])
ZTE = np.zeros([thetaPts, Nle])

Xin = np.zeros([thetaPts, Nz])  # Block inlet
Yin = np.zeros([thetaPts, Nz])
Zin = np.zeros([thetaPts, Nz])

Xout = np.zeros([thetaPts, Nz])  # Block outlet
Yout = np.zeros([thetaPts, Nz])
Zout = np.zeros([thetaPts, Nz])

Xmid = np.zeros([thetaPts, Nz])  # Mid chord
Ymid = np.zeros([thetaPts, Nz])
Zmid = np.zeros([thetaPts, Nz])

Xh = np.zeros([thetaPts, Nh])  # Hub
Yh = np.zeros([thetaPts, Nh])
Zh = np.zeros([thetaPts, Nh])

Xc = np.zeros([thetaPts, Nc])  # Casing
Yc = np.zeros([thetaPts, Nc])
Zc = np.zeros([thetaPts, Nc])

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
    for m in range(Nrb):
        XBp[k, m] = pBlade[k][m, 0]
        YBp[k, m] = pBlade[k][m, 1]
        ZBp[k, m] = pBlade[k][m, 2]
        XBs[k, m] = sBlade[k][m, 0]
        YBs[k, m] = sBlade[k][m, 1]
        ZBs[k, m] = sBlade[k][m, 2]
for c in range(thetaPts):
    for d in range(Nz):
        Xin[c, d] = inlet[c][d, 2]
        Yin[c, d] = inlet[c][d, 1]
        Zin[c, d] = inlet[c][d, 0]
        Xout[c, d] = outlet[c][d, 2]
        Yout[c, d] = outlet[c][d, 1]
        Zout[c, d] = outlet[c][d, 0]
        Xmid[c, d] = midChord[c][d, 2]
        Ymid[c, d] = midChord[c][d, 1]
        Zmid[c, d] = midChord[c][d, 0]
    for e in range(Nle):
        XLE[c, e] = leadEdge[c][e, 2]
        YLE[c, e] = leadEdge[c][e, 1]
        ZLE[c, e] = leadEdge[c][e, 0]
        XTE[c, e] = trailEdge[c][e, 2]
        YTE[c, e] = trailEdge[c][e, 1]
        ZTE[c, e] = trailEdge[c][e, 0]
    for p in range(Nh):
        Xh[c, p] = hubS[c][p, 2]
        Yh[c, p] = hubS[c][p, 1]
        Zh[c, p] = hubS[c][p, 0]
    for q in range(Nc):
        Xc[c, q] = casing[c][q, 2]
        Yc[c, q] = casing[c][q, 1]
        Zc[c, q] = casing[c][q, 0]

# Set up STL filenames
filenames = ['mPeriodic','nPeriodic','pPeriodic','pBlade','sBlade','LE','TE','inlet','outlet','Casing','Hub','midChord']
Xvalues = [Xm, Xn, Xp, XBp, XBs, XLE, XTE, Xin, Xout, Xc, Xh, Xmid]
Yvalues = [Ym, Yn, Yp, YBp, YBs, YLE, YTE, Yin, Yout, Yc, Yh, Ymid]
Zvalues = [Zm, Zn, Zp, ZBp, ZBs, ZLE, ZTE, Zin, Zout, Zc, Zh, Zmid]

# Create STLs and write them to disk
for qq in range(len(Xvalues)):
    filename = pathForSTLs + '/{}.stl'.format(filenames[qq])
    rows = Zvalues[qq].shape[0]  # Use Z because it is the axis of rotation
    columns = Zvalues[qq].shape[1]
    X = Xvalues[qq]
    Y = Yvalues[qq]
    Z = Zvalues[qq]

    numFacets = 0

    file = open(filename, 'w')
    file.write('solid \n')

    def unitVector(file, p1, p2, p3):
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
    
    for i in range(rows - 1):
        for j in range(columns - 1):
            # FACET A VERTICES
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

# Generate input file for Bash script which modifies blockMeshDict to be compatible with the geometry
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
