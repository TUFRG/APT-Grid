#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:57:20 2022

@author: adekola
"""

def analytical_model(A, B, alpha, k=0):
    import numpy as np
    Temp = np.zeros(len(range(B)))
    for k in range(B):
        R = A/B
        Xk = (alpha*2*np.pi/B) + (2*np.pi*k/B)
        Jl = np.floor((R*(alpha + k))) 
        Jh = np.ceil((R*(alpha + k))) 
        Xjl = 2*np.pi*Jl/A
        Yjl = np.sin(Xjl)
        Xjh = 2*np.pi*Jh/A
        Yjh = np.sin(Xjh)
        deltaXjk = Xk - Xjl
        deltaXj = 2 * np.pi / A
        
        Yk = Yjl + deltaXjk * (Yjh - Yjl) / deltaXj
        
        Error = (Yk - np.sin(Xk))
        Temp[k] = Error
    RMSe = np.sqrt(np.mean(np.square(Temp)))
    return RMSe

def without_loop(A,B,alpha,k):
    import numpy as np
    #R = A/B
    Xk = (alpha*2*np.pi/B) + (2*np.pi*k/B)
    Jl = np.floor((A*(alpha + k)/B)) 
    Jh = np.ceil((A*(alpha + k)/B)) 
    Xjl = 2*np.pi*Jl/A
    Yjl = np.sin(Xjl)
    Xjh = 2*np.pi*Jh/A
    Yjh = np.sin(Xjh)
    deltaXjk = Xk - Xjl
    deltaXj = 2 * np.pi / A
    
    Yk = Yjl + deltaXjk * (Yjh - Yjl) / deltaXj
    
    Error = (Yk - np.sin(Xk))
    return Error

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
def draw_curve(p1, p2):
    import numpy as np
    a = (p2[1] - p1[1]) / (np.cosh(p2[0]) - np.cosh(p1[0]))
    b = p1[1] - a * np.cosh(p1[0])
    x = np.linspace(p1[0], p2[0], 100)
    y = a * np.cosh(x) + b
    return x, y

def vectorRot(x,y,theta):
    import numpy as np
    xRot = x*np.cos(theta) - y*np.sin(theta)
    yRot = x*np.sin(theta) + y*np.cos(theta)
    return xRot, yRot

def vectorRotP(x,y,P,theta):
    import numpy as np
    xRot = ((x-P[0])*np.cos(theta) - (P[1]-y)*np.sin(theta)) + P[0]
    yRot = P[1] -((x-P[0])*np.sin(theta) + (P[1]-y)*np.cos(theta))
    return xRot, yRot, 0

def cart2pol(x,y,z):
    import numpy as np
    theta = np.arctan(y/x)
    rho = np.sqrt(np.square(x)+np.square(y))
    Z = z
    return(theta, rho, Z)

def pol2cart(theta, r, Z):
    import numpy as np
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = Z
    return x, y, z   

def dist3D(x1,x2,y1,y2,z1,z2):
    import numpy as np
    distance = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1) + np.square(z2- z1))
    return distance

def dist2D(x1,y1,x2,y2):
    import numpy as np
    distance = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
    return distance

def RMS(P):
    import numpy as np
    AddSqure = np.sum(P**2)/len(P)
    rms = np.sqrt(AddSqure)
    return rms
    
def AngRadCenPts(x1,y1,x2,y2,slope,m1,m2):
    import numpy as np
    slope_perp = -1/slope
    Cinterp1 = y2 - slope_perp*x2
    P1 = [x1, x1+m1]
    P2 = [x2+m2, (slope_perp*(x2+m2)+Cinterp1)]
    Line1 = line(P1,[x1,y1])
    Line2 = line(P2,[x2,y2])
    InterPt = intersection(Line1, Line2)
    radius1 = np.sqrt((InterPt[0] - x1)**2 + (InterPt[1] - y1)**2)
    radius2 = np.sqrt((InterPt[0] - x2)**2 + (InterPt[1] - y2)**2)
    return (radius1, radius2, InterPt)
    
def ExtentionIntersection(x1,y1,Line2,theta,m1):
    import numpy as np
    from shapely.geometry import LineString
    slope = np.tan(theta*2*np.pi/180)
    slope_perp = -1/slope
    Cinterp = y1 - slope_perp*x1
    NewPt = [[x1-m1,slope_perp*(x1-m1)+Cinterp]]
    originalPt = [[x1,y1]]
    Line1 = np.concatenate((originalPt, NewPt))
    L1 = LineString(np.column_stack((Line1[:,0],Line1[:,1])))
    L2 = LineString(np.column_stack((Line2[:,0],Line2[:,1])))
    InterPt = L1.intersection(L2)
    [x,y] =InterPt.xy
    Center = [x[0],y[0],0]
    return (Center, Line1)

def simulEqn(x1,y1,x2,y2):
    import numpy as np
    A = np.array([[((2*x2)-(2*x1)), ((2*y2)-(2*y1))],[(2*x1), (2*y1)]])
    G = np.linalg.det(A)
    B = np.array([(x2**2+y2**2-x1**2-y1**2),(x1**2+y1**2)])
    result = np.linalg.inv(A).dot(B)
    return result

def RadAndAng(x1,y1,x2,y2):
    import numpy as np
    center = simulEqn(x1, y1, x2, y2)
    radius = np.sqrt((center[0] - x1)**2 + (center[1] - y1)**2)
    base = dist2D(x1, y1, x2, y2)
    cosAngle = ((radius**2 + radius**2) - base**2)/(2*radius*radius)
    Angle = np.arccos(cosAngle)
    AngleDeg = np.rad2deg(Angle)
    arcLen = Angle * radius
    return arcLen
    
def scale(rmin,rmax,tmin,tmax,m):
    NewM = (((rmin - m)/(rmax - rmin)) * (tmax - tmin)) + tmax
    return NewM
# def scale(rmin, rmax, tmin, tmax, m):
#     # Ensure rmin and rmax are not equal to avoid division by zero
#     if rmin == rmax:
#         raise ValueError("rmin and rmax cannot be equal")
        
#     NewM = ((m - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
#     return NewM

def TwoLinesIntersect (Line1, Line2): #Get the intersection point if the lines already intersects
    import numpy as np
    from shapely.geometry import LineString, Point, MultiPoint
    L1 = LineString(np.column_stack((Line1[:,0],Line1[:,1])))
    L2 = LineString(np.column_stack((Line2[:,0],Line2[:,1])))
    InterPt = L1.intersection(L2)
    if InterPt.is_empty:
        return 0
    else:
        if InterPt.geom_type == 'MultiPoint':
            return [[point.x, point.y] for point in InterPt.geoms]
        else:
            return [InterPt.x, InterPt.y]

def MidPts (Line):
    x1 = Line[0][0]
    x2 = Line[len(Line)-1][0]
    y1 = Line[0][1]
    y2 = Line[len(Line)-1][1]
    x = (x1+x2)/2
    y = (y1+y2)/2
    MidPoint = [x,y,0]
    return MidPoint
    
def Slope (x1,y1,x2,y2):
    grad = (y2 - y1)/ (x2 - x1)
    return grad
def vectorRot3D (x,y,z,theta):
    import numpy as np
    yRot = y*np.cos(theta) - z*np.sin(theta)
    zRot = y*np.sin(theta) + z*np.cos(theta)
    return x, yRot, zRot
def vectorRotZ3D (x,y,z,theta):
    import numpy as np
    yRot = x*np.cos(theta) - y*np.sin(theta)
    xRot = x*np.sin(theta) + y*np.cos(theta)
    return xRot, yRot, z
def TwoLinesIntersectExtsn (Line1, Line2, m): #Get the intersection point if the lines DO NOT intersects
    import numpy as np
    from shapely.geometry import LineString
    slope1 = (Line1[1][1] - Line1[0][1])/(Line1[1][0] - Line1[0][0])
    slope2 = (Line2[1][1] - Line2[0][1])/(Line2[1][0] - Line2[0][0])
    Intercept1 = Line1[1][1] - (slope1 * Line1[1][0])
    Intercept2 = Line2[1][1] - (slope2 * Line2[1][0])
    NLine1 = np.vstack(((Line1[1][0],Line1[1][1]),(Line1[1][0]+m, (slope1*(Line1[1][0]+m))+Intercept1)))
    NLine2 = np.vstack(((Line2[1][0],Line2[1][1]),(Line2[1][0]+m, (slope2*(Line2[1][0]+m))+Intercept2)))
    L1 = LineString(np.column_stack((NLine1[:,0],NLine1[:,1])))
    L2 = LineString(np.column_stack((NLine2[:,0],NLine2[:,1])))
    InterPt = L1.intersection(L2)
    [x,y] =InterPt.xy
    Center = [x[0],y[0],0]
    return Center    

def redistribute_point(x, y, num_points):
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.integrate import cumtrapz
    assert len(x) == len(y), "x and y arrays must have the same length"
    # Calculate cumulative distance along the curve
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_distances = np.concatenate(([0], cumtrapz(distances)))
    cumulative_distances = np.append(cumulative_distances, cumulative_distances[-1])
    # Linearly interpolate the cumulative distances to get a function f(s) = t,
    # where s is the arc length parameter and t is the parameter along the curve
    interp_func = interp1d(cumulative_distances, np.arange(len(x)), kind='linear')
    
    # Equally distribute points along the curve
    new_distances = np.linspace(0, cumulative_distances[-1], num_points)
    new_indices = interp_func(new_distances).astype(int)
    
    # Extract redistributed points
    redistributed_x = x[new_indices]
    redistributed_y = y[new_indices]
    
    return redistributed_x, redistributed_y

def pairwise_distances(points):
    import numpy as np
    """
    Calculate pairwise distances between points.
    """
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            distances[i, j] = distances[j, i] = np.linalg.norm(points[i] - points[j])
    return distances

def longest_distance_on_curve(x, y):
    import numpy as np
    """
    Find the longest distance between two points on a closed curve defined by x and y coordinates.
    Return the maximum distance and the indices of the two points.
    """
    # Combine x and y coordinates into a single array of points
    points = np.column_stack([x, y])
    
    # Compute pairwise distances between points
    distances = pairwise_distances(points)
    
    # Find the indices of the maximum distance
    idx_i, idx_j = np.unravel_index(np.argmax(distances), distances.shape)
    
    # Find the maximum distance
    max_distance = distances[idx_i, idx_j]
    
    return max_distance, idx_i, idx_j

def find_arc_center(P1, P2, r):
    import numpy as np
    x1, y1 = P1
    x2, y2 = P2

    # Midpoint
    Mx, My = (x1 + x2) / 2, (y1 + y2) / 2

    # Distance between points
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Ensure the points are within the valid range
    if d > 2 * r:
        raise ValueError("The points are too far apart for the given radius")

    # Distance from midpoint to the center
    h = np.sqrt(r**2 - (d / 2)**2)

    # Calculate the direction vector for the perpendicular line
    dx, dy = x2 - x1, y2 - y1

    # Normalize the direction vector
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / length, dy / length

    # Calculate the two possible centers
    C1 = (Mx + h * dy, My - h * dx)
    C2 = (Mx - h * dy, My + h * dx)

    return C1, C2

def plot_arc(P1, P2, C, r):
    import numpy as np
    # Find the two possible centers


    # Calculate angles for the arc
    theta1 = np.arctan2(P1[1] - C[1], P1[0] - C[0])
    theta2 = np.arctan2(P2[1] - C[1], P2[0] - C[0])

    # Ensure the angles are in the correct order
    if theta1 > theta2:
        theta1, theta2 = theta2, theta1

    # Generate points for the arc
    theta = np.linspace(theta1, theta2, 100)
    x = C[0] + r * np.cos(theta)
    y = C[1] + r * np.sin(theta)
    
    return x,y


def find_tangent_arc_center(P1, P2, r, line):
    import numpy as np
    x1, y1 = P1
    x2, y2 = P2
    a, b, c = line

    # Midpoint
    Mx, My = (x1 + x2) / 2, (y1 + y2) / 2

    # Distance between points
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Ensure the points are within the valid range
    if d > 2 * r:
        raise ValueError("The points are too far apart for the given radius")

    # Distance from midpoint to the center
    h = np.sqrt(r**2 - (d / 2)**2)

    # Calculate the direction vector for the perpendicular line
    dx, dy = x2 - x1, y2 - y1

    # Normalize the direction vector
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / length, dy / length

    # Find the perpendicular distance from the midpoint to the line
    dist_to_line = np.abs(a * Mx + b * My + c) / np.sqrt(a**2 + b**2)

    # Ensure the line is within the valid range
    if dist_to_line > h:
        raise ValueError("The tangent line is too far for the given radius and points")

    # Calculate the two possible centers considering the tangent constraint
    C1 = (Mx + h * dy, My - h * dx)
    C2 = (Mx - h * dy, My + h * dx)

    # Choose the center such that the circle is tangent to the line
    # We calculate the distance from the center to the line and ensure it equals the radius
    def is_tangent(center):
        cx, cy = center
        return np.isclose(np.abs(a * cx + b * cy + c) / np.sqrt(a**2 + b**2), r)

    if is_tangent(C1):
        return C1
    elif is_tangent(C2):
        return C2
    else:
        raise ValueError("No valid center found that satisfies both conditions")

def plot_arc_tangent(P1, P2, r, line):
    import numpy as np
    # Find the center of the circle
    C = find_tangent_arc_center(P1, P2, r, line)

    # Calculate angles for the arc
    theta1 = np.arctan2(P1[1] - C[1], P1[0] - C[0])
    theta2 = np.arctan2(P2[1] - C[1], P2[0] - C[0])

    # Ensure the angles are in the correct order
    if theta1 > theta2:
        theta1, theta2 = theta2, theta1

    # Generate points for the arc
    theta = np.linspace(theta1, theta2, 100)
    x = C[0] + r * np.cos(theta)
    y = C[1] + r * np.sin(theta)
    
    return x,y

def cart2Mises(x,y,z):
    import numpy as np
    theta = np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2)
    m = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0:
            m[i] = 1
        else:
            m[i] = m[i-1] + ((2/(r[i]+r[i-1]))*np.sqrt((r[i]-r[i-1])**2 + (z[i]-z[i-1])**2))
    return theta, r, m

def mises2Cart(theta, r, m):
    import numpy as np
    x = r*np.sin(theta)
    y = r*np.cos(theta)
    z = m*r
    return x,y,z

def plot_slices(A,B):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_slices = A.shape[0]
    num_sliced = B.shape[0]

    for i in range(num_slices):
        slices = A[i]
        sliced = B[i]
        ax.plot(slices[:, 2], slices[:, 1], slices[:, 0], label=f'Slice {i + 1}')
        ax.plot(sliced[:, 2], sliced[:, 1], sliced[:, 0], label=f'Slice {i + 1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('All {} Slices in One Plot'.format(len(A)))
    # plt.legend()
    plt.show()
    
    
def compute_surface_normals(X, Y, Z):
    import numpy as np
    # Calculate the gradients in x and y directions
    dX, dY = np.gradient(X), np.gradient(Y)
    dZx, dZy = np.gradient(Z)
    
    # Cross product of the gradients
    nx = -dZx
    ny = -dZy
    nz = np.ones_like(Z)
    
    # Normalize the normal vectors
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length
    ny /= length
    nz /= length
    
    return nx, ny, nz

def rotate_points(points, angle):
    import numpy as np
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(points, rotation_matrix.T)

def radius_of_curvature(cs):
    import numpy as np
    x = cs.x
    y = cs(x)
    dydx = cs.derivative(nu=1)(x)
    d2ydx2 = cs.derivative(nu=2)(x)    
    curvature = np.abs(d2ydx2) / (1 + dydx**2)**(3/2)
    return 1 / curvature

def circle_from_points(p1, p2, p3):
    import numpy as np
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        raise ValueError("Points are collinear or too close to each other")
    
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    
    return np.array([cx, cy]), radius

def generate_circle_points(center, radius, num_points=100000):
    import numpy as np
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.zeros((num_points, 2))
    circle_points[:, 0] = center[0] + radius * np.cos(angles)
    circle_points[:, 1] = center[1] + radius * np.sin(angles)
    return circle_points

def distance(p1, p2):
    import numpy as np
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def closest_point(point, points):
    import numpy as np
    return points[np.argmin([distance(point, p) for p in points])]

def justin(bladeP, bladeS):
    import numpy as np
    from scipy.interpolate import CubicSpline
    # from scipy.optimize import fsolve
    # from scipy.optimize import minimize
    
    preS = cart2pol(bladeP[:, 0], bladeP[:, 1], bladeP[:, 2])
    suncS = cart2pol(bladeS[:, 0], bladeS[:, 1], bladeS[:, 2])
    PS = np.empty([len(bladeP), 2])
    PS[:, 1] = preS[0]*(preS[1])
    PS[:, 0] = preS[2]
    
    SS = np.empty([len(bladeS), 2])
    SS[:, 1] = suncS[0]*(suncS[1])
    SS[:, 0] = suncS[2]

    blade = np.vstack((PS[0:len(PS)+1],SS[::-1][1:len(SS)+1]))
    blade = blade[blade[:, 0].argsort()]
    
    leftmost_points = blade[:int(0.15 * len(blade))]
    rightmost_points = blade[int(0.85 * len(blade)):]
    
    centroid = np.mean(blade, axis=0)
    blade_centered = blade - centroid
    dx = blade_centered[-1, 0] - blade_centered[0, 0] # x2 - x1
    dy = blade_centered[-1, 1] - blade_centered[0, 1] # y2 - y1
    angle_to_vertical = np.arctan2(dy, dx) - np.pi/2
    
    rotated_blade = rotate_points(blade_centered, angle_to_vertical)
    rotated_blade += centroid  # Translate back to the original position
    
    rotated_PS = rotate_points(PS - centroid, angle_to_vertical) + centroid
    rotated_SS = rotate_points(SS - centroid, angle_to_vertical) + centroid
    
    rotated_leftmost_points = rotated_blade[:int(0.1 * len(rotated_blade))]
    rotated_rightmost_points = rotated_blade[int(0.9 * len(rotated_blade)):]
    
    rotated_leftmost_points = rotated_leftmost_points[rotated_leftmost_points[:, 0].argsort()]
    rotated_rightmost_points = rotated_rightmost_points[rotated_rightmost_points[:, 0].argsort()]
    
    rotated_leftmost_points = np.unique(rotated_leftmost_points, axis=0)
    rotated_rightmost_points = np.unique(rotated_rightmost_points, axis=0)

    cs_leftmost = CubicSpline(rotated_leftmost_points[:, 0], rotated_leftmost_points[:, 1])
    cs_rightmost = CubicSpline(rotated_rightmost_points[:, 0], rotated_rightmost_points[:, 1])
    
    x_fine_leftmost = np.linspace(rotated_leftmost_points[:, 0].min(), rotated_leftmost_points[:, 0].max(), 1000)
    y_fine_leftmost = cs_leftmost(x_fine_leftmost)
    x_fine_rightmost = np.linspace(rotated_rightmost_points[:, 0].min(), rotated_rightmost_points[:, 0].max(), 1000)
    y_fine_rightmost = cs_rightmost(x_fine_rightmost)
    
    radii_leftmost = radius_of_curvature(cs_leftmost)
    radii_rightmost = radius_of_curvature(cs_rightmost)
    
    min_radius_leftmost_index = np.argmin(radii_leftmost)
    min_radius_rightmost_index = np.argmin(radii_rightmost)
    min_radius_leftmost_x = x_fine_leftmost[min_radius_leftmost_index]
    min_radius_rightmost_x = x_fine_rightmost[min_radius_rightmost_index]
    
    min_radius_leftmost_y = cs_leftmost(min_radius_leftmost_x)
    min_radius_rightmost_y = cs_rightmost(min_radius_rightmost_x)
    
    left = rotated_blade[rotated_blade[:, 1] > min_radius_leftmost_y]
    right = rotated_blade[rotated_blade[:, 1] < min_radius_rightmost_y]
    middle_PS = rotated_PS[rotated_PS[:, 1] < min_radius_leftmost_y]
    middle_PS = middle_PS[middle_PS[:, 1] > min_radius_rightmost_y]
    middle_SS = rotated_SS[rotated_SS[:, 1] < min_radius_leftmost_y]
    middle_SS = middle_SS[middle_SS[:, 1] > min_radius_rightmost_y]
    
    left = rotate_points(left, -angle_to_vertical) + centroid
    right = rotate_points(right, -angle_to_vertical) + centroid
    middle_PS = rotate_points(middle_PS, -angle_to_vertical) + centroid
    middle_SS = rotate_points(middle_SS, -angle_to_vertical) + centroid
    
    left = left[left[:, 1].argsort()]
    right = right[right[:, 1].argsort()]
    
    min_x_PS = middle_PS[:, 0].min()
    max_x_PS = middle_PS[:, 0].max()
    min_x_SS = middle_SS[:, 0].min()
    max_x_SS = middle_SS[:, 0].max()
    
    common_min_x = max(min_x_PS, min_x_SS)
    common_max_x = min(max_x_PS, max_x_SS)  
    cs_middle_PS = CubicSpline(middle_PS[:, 0], middle_PS[:, 1])
    cs_middle_SS = CubicSpline(middle_SS[:, 0], middle_SS[:, 1])
    x_common = np.linspace(common_min_x, common_max_x, 1000)
    
    y_camber_line = 0.5 * (cs_middle_PS(x_common) + cs_middle_SS(x_common))
    camber_line = np.column_stack((x_common, y_camber_line))
    camber_line = camber_line[camber_line[:, 0].argsort()]
    cs_camber_line = CubicSpline(camber_line[:, 0], camber_line[:, 1])   
    d2ydx2_camber = cs_camber_line.derivative(nu=2)(camber_line[:, 0])
    boundary_points = np.where(np.diff(np.sign(d2ydx2_camber)))[0]
    left_boundary_points = boundary_points[boundary_points[:] < 500]
    right_boundary_points = boundary_points[boundary_points[:] > 500]
    
    left_boundary_points_x = camber_line[left_boundary_points, 0]
    left_boundary_points_y = camber_line[left_boundary_points, 1]
    right_boundary_points_x = camber_line[right_boundary_points, 0]
    right_boundary_points_y = camber_line[right_boundary_points, 1]
    left_boundary_points = np.column_stack((left_boundary_points_x, left_boundary_points_y))
    right_boundary_points = np.column_stack((right_boundary_points_x, right_boundary_points_y))  
    camber_line = camber_line[(camber_line[:, 0] > left_boundary_points[:, 0].max()) & (camber_line[:, 0] < right_boundary_points[:, 0].min())]
    
    leftmost_camber_points = camber_line[:3]
    rightmost_camber_points = camber_line[-3:]
    
    center, radius = circle_from_points(leftmost_camber_points[0], leftmost_camber_points[1], leftmost_camber_points[2])
    circle_points = generate_circle_points(center, radius)
    arc_leftmost = circle_points[(circle_points[:, 0] < leftmost_camber_points[:, 0].max()) & (circle_points[:, 0] > left[:, 0].min()) & (circle_points[:, 1] > left[:, 1].min())]
    intersection_point_left = TwoLinesIntersect(circle_points,left)
    intersection_point_left = np.array(intersection_point_left)
    
    center1, radius1 = circle_from_points(rightmost_camber_points[0], rightmost_camber_points[1], rightmost_camber_points[2])
    circle_points1 = generate_circle_points(center1, radius1)
    arc_rightmost = circle_points1[(circle_points1[:, 0] > rightmost_camber_points[:, 0].min()) & 
                                  (circle_points1[:, 0] < right[:, 0].max()) & 
                                  (circle_points1[:, 1] > right[:, 1].min())]

    intersection_point_right = TwoLinesIntersect(circle_points1,right)
    intersection_point_right = np.array(intersection_point_right)
    
    camber_line = np.vstack((camber_line, intersection_point_left, arc_leftmost, arc_rightmost, intersection_point_right))
    camber_line = camber_line[camber_line[:, 0].argsort()]
    camber_line = np.unique(camber_line, axis=0)
    
    return intersection_point_left, intersection_point_right, camber_line
        
    
    
    
    