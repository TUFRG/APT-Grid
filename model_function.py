#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:57:20 2022

@author: adekola
"""
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
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
    A = (p2[1] - p1[1])
    B = (p1[0] - p2[0])
    C = (p2[0]*p1[1] - p1[0]*p2[1])
    return A, B, C

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
    theta = np.arctan2(y,x)
    # if np.all(x < 0) and np.any(y < 0) and np.any(y > 0): # This is important for situations of discontinuities. This happens when you 'just' cross the +ve y to -ve y via the -ve x-axis
    #     theta = np.where(theta < 0, theta + 2*np.pi, theta) #Basically you adjust theta range from (-np.pi,np.pi) to (0, 2*np.pi)
    rho = np.sqrt(np.square(x)+np.square(y))
    Z = z
    return(theta, rho, Z)
def densify_curve_simple(points, n_points_new):
    if n_points_new <= len(points):
        return points[:n_points_new]
    # If 1D, reshape to 2D
    points = np.atleast_2d(points)
    if points.shape[0] == 1:
        points = points.T   
    current_points = points.copy()
    n_to_add = n_points_new - len(points)
    for _ in range(n_to_add):
        # Calculate distances between consecutive points
        diffs = np.diff(current_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # Find the largest gap
        max_gap_idx = np.argmax(distances)
        # Insert a point at the midpoint of the largest gap
        midpoint = (current_points[max_gap_idx] + current_points[max_gap_idx + 1]) / 2
        # Insert the midpoint
        current_points = np.insert(current_points, max_gap_idx + 1, midpoint, axis=0)
    
    return current_points

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
    MidPoint = [x,y]
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

def arclength(points):
   import numpy as np
   diffs = np.diff(points, axis=0)
   distances = np.sqrt(np.sum(diffs**2, axis=1))
   arclengths = np.cumsum(distances)
   arclengths = np.insert(arclengths, 0, 0)  # Include starting point (0 arclength)
   return arclengths
   
def unit_vector(v):
    import numpy as np
    return v / np.linalg.norm(v)

def angle_bisector(p1, p2, p3, reflex=False):
    v1 = unit_vector(p1 - p2)  # Vector from p2 to p1
    v2 = unit_vector(p3 - p2)  # Vector from p2 to p3
    
    bisector = unit_vector(v1 + v2)  # Sum and normalize to get bisector direction
    if reflex:
        bisector = -bisector  # Negate to get bisector of reflex angle
    return bisector
def find_angle(p1, p2, p3):
    import numpy as np
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    c = np.asarray(p3, dtype=float)
   
    v1 = a - b
    v2 = c - b
    # v1 = p1 - p2  # Vector from p2 to p1
    # v2 = p3 - p2  # Vector from p2 to p3
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Compute the acute/obtuse angle (0°–180°)
    theta_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    theta_deg = np.degrees(theta_rad)
    return theta_deg
def scale(rmin, rmax, tmin, tmax, m):
    import numpy as np
    m = np.asarray(m)  # Ensure m is an array
    return ((m - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
def find_optimal_ellipse_LE(points, slopes,initial_guess):
    from scipy.optimize import minimize
    if len(points) < 2:
        raise ValueError("At least 2 points are required")  
    def rotate_point(x, y, theta, inverse=False):
        import numpy as np
        if inverse:
            theta = -theta  # Reverse the rotation if inverse is True
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])  # Rotation matrix
        return R @ np.array([x, y])  #@ is a matrix multiplication. Just learning this
    def get_ellipse_coeffs_local(x3_y3):
        import numpy as np
        x3, y3 = x3_y3
        local_points = points.copy()
        local_points.append((x3, y3))     
        A_matrix = []    
        # Points constraints
        for x, y in local_points:
            A_matrix.append([x**2, x*y, y**2, x, y])     
        # Slope constraints
        for (x_s, y_s), m_s in slopes:
            # Implicit differentiation: 2Ax + By + D + (Bx + 2Cy + E)m  = 0
            A_matrix.append([2*x_s, y_s + x_s*m_s, 2*y_s*m_s, 1, m_s,])
        # print("A_matrix contents:", A_matrix)
        A_matrix = np.array(A_matrix)       
        B_matrix = -np.ones(len(A_matrix))
        B_matrix[3] = 0
        B_matrix[4] = 0
        # print(A_matrix)
        solution = np.linalg.solve(A_matrix, B_matrix)
        solution = np.append(solution, 1)    
        return solution     
    def objective(x3_y3):
        import numpy as np
        
        try:
            coeffs = get_ellipse_coeffs_local(x3_y3)
            A, B, C, D, E, F = coeffs          
            # Check if this is an ellipse
            disc = B**2 - 4*A*C
            if disc >= 0:
                # print('Discriminant is not negative!')
                # print(x3_y3)
                return 1e6 + 100*disc
            # Calculate semi-axes directly from the coefficients
            center = [(2*C*D - B*E) / (B**2 - 4*A*C), (2*A*E - B*D) / (B**2 - 4*A*C)] #Center of the ellipse 
            angle = 0.5 * np.arctan2(-B, C - A) 
            expr1 = 2 * (A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)
            expr2 = np.sqrt((A - C)**2 + B**2)           
            if expr1 <= 0:
                return 1e6                
            a = np.sqrt(abs(expr1 * ((A + C) + expr2))) / abs(B**2 - 4*A*C)
            b = np.sqrt(abs(expr1 * ((A + C) - expr2))) / abs(B**2 - 4*A*C)            
            if a < b:  # Ensure a is major axis
                a, b = b, a    
            if b <  1e-10:
                return float('inf') #1e6 
            # Minimize maximum curvature (ab/(b**2cos**2(t)+a**2sin**2(t))**(3/2))
            shifted1 = [(points[0][0] - center[0]), (points[0][1] - center[1])] #The first point is shifted by subtracting from the center to move to the origin
            shifted2 = [(points[1][0] - center[0]), (points[1][1] - center[1])] # same thing is done for the second point 
            rot1 = rotate_point(shifted1[0], shifted1[1], angle, inverse=True) # The first point is then rotated counterclockwise sort of like rotating back to the origin 
            rot2 = rotate_point(shifted2[0], shifted2[1], angle, inverse=True) # same thing is done for the second point 
            PointT1 = np.arctan2(rot1[1]/b, rot1[0]/a) # A parameter t is defined to determine the start and end angle of the segment of the ellipse needed here
            PointT2 = np.arctan2(rot2[1]/b, rot2[0]/a) # The vertical and horizontal axis are normalized by the major and minor axis because unlike a circle an ellipse is streched
            if abs(PointT1 - PointT2) <= 2*np.pi -  abs(PointT1 - PointT2): # in a weird way. So there is a need to 'un-scale' the ellipse for accurate start and end angle
                tPoints = np.linspace(PointT1, PointT2, 1000) # This is the short path. kind of like the right way
            else:
                if PointT1 < PointT2: 
                    PointT2 = PointT2 - 2*np.pi # Go in the clockwise direction 
                else:
                    PointT2 =PointT2 + 2*np.pi # Go in the anti-clockwise direction 
                tPoints = np.linspace(PointT1, PointT2, 1000)  
            R = (b**2*(np.cos(tPoints))**2 + a**2*(np.sin(tPoints)**2))**(3/2)/(a*b)
            return max(1/R)/min(1/R)  
        except Exception as e:
            print(f"Error: {e}")
            return float('inf') #1e6   
        
    result = minimize(objective, initial_guess, method='L-BFGS-B',  # Better for bounded problems
        options={'ftol': 1e-8, 'maxiter': 1000} )             
    third_point = result.x
    # third_point = initial_guess
    A, B, C, D, E, F = get_ellipse_coeffs_local(third_point)   
    # Calculate ellipse properties
    disc = B**2 - 4*A*C
    if disc >= 0:
        raise ValueError("The computed conic section is not an ellipse.")     
    return A, B, C, D, E, F#, third_point

def find_optimal_ellipse_TE(points, slopes,initial_guess):
    import numpy as np
    from scipy.optimize import minimize
    if len(points) < 2:
        raise ValueError("At least 2 points are required")  
    def rotate_point(x, y, theta, inverse=False):
        if inverse:
            theta = -theta  # Reverse the rotation if inverse is True
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])  # Rotation matrix
        return R @ np.array([x, y])  #@ is a matrix multiplication. Just learning this
    def get_ellipse_coeffs_local(x3_y3):
        x3, y3 = x3_y3
        local_points = points.copy()
        local_points.append((x3, y3))     
        A_matrix = []    
        # Points constraints
        for x, y in local_points:
            A_matrix.append([x**2, x*y, y**2, x, y])     
        # Slope constraints
        for (x_s, y_s), m_s in slopes:
            # Implicit differentiation: 2Ax + By + D + (Bx + 2Cy + E)m  = 0
            A_matrix.append([2*x_s, y_s + x_s*m_s, 2*y_s*m_s, 1, m_s,])
        A_matrix = np.array(A_matrix)       
        B_matrix = -np.ones(len(A_matrix))
        B_matrix[3] = 0
        B_matrix[4] = 0
        # print(A_matrix)
        solution = np.linalg.solve(A_matrix, B_matrix)
        solution = np.append(solution, 1)    
        return solution     
    def objective(x3_y3):
        try:
            coeffs = get_ellipse_coeffs_local(x3_y3)
            A, B, C, D, E, F = coeffs          
            # Check if this is an ellipse
            disc = B**2 - 4*A*C        
            if disc >= 0:
                # print('Discriminant is not negative!')
                # print(x3_y3)
                return 1e6 + 100*disc
            # Calculate semi-axes directly from the coefficients
            center = [(2*C*D - B*E) / (B**2 - 4*A*C), (2*A*E - B*D) / (B**2 - 4*A*C)] #Center of the ellipse 
            angle = 0.5 * np.arctan2(-B, C - A) 
            expr1 = 2 * (A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)
            expr2 = np.sqrt((A - C)**2 + B**2) 
            # print(expr1)
            # if expr1 <= 0:
            #     return 1e6                
            a = np.sqrt(abs(expr1 * ((A + C) + expr2))) / abs(B**2 - 4*A*C)
            b = np.sqrt(abs(expr1 * ((A + C) - expr2))) / abs(B**2 - 4*A*C)            
            if a < b:  # Ensure a is major axis
                a, b = b, a    
            if b <  1e-10:
                return float('inf') #1e6 
            # Minimize maximum curvature (ab/(b**2cos**2(t)+a**2sin**2(t))**(3/2))
            shifted1 = [(points[0][0] - center[0]), (points[0][1] - center[1])] #The first point is shifted by subtracting from the center to move to the origin
            shifted2 = [(points[1][0] - center[0]), (points[1][1] - center[1])] # same thing is done for the second point 
            rot1 = rotate_point(shifted1[0], shifted1[1], angle, inverse=True) # The first point is then rotated counterclockwise sort of like rotating back to the origin 
            rot2 = rotate_point(shifted2[0], shifted2[1], angle, inverse=True) # same thing is done for the second point 
            PointT1 = np.arctan2(rot1[1]/b, rot1[0]/a) # A parameter t is defined to determine the start and end angle of the segment of the ellipse needed here
            PointT2 = np.arctan2(rot2[1]/b, rot2[0]/a) # The vertical and horizontal axis are normalized by the major and minor axis because unlike a circle an ellipse is streched
    
            if abs(PointT1 - PointT2) <= 2*np.pi -  abs(PointT1 - PointT2): # in a weird way. So there is a need to 'un-scale' the ellipse for accurate start and end angle
                tPoints = np.linspace(PointT1, PointT2, 1000) # This is the short path. kind of like the right way
            else:
                if PointT1 < PointT2: 
                    PointT2 = PointT2 - 2*np.pi # Go in the clockwise direction 
                else:
                    PointT2 =PointT2 + 2*np.pi # Go in the anti-clockwise direction 
                tPoints = np.linspace(PointT1, PointT2, 1000) 
            
            R = (b**2*(np.cos(tPoints))**2 + a**2*(np.sin(tPoints)**2))**(3/2)/(a*b)
            # print(R)
            return min(1/R)/max(1/R)
        except Exception as e:
            print(f"Error: {e}")
            return float('inf') #1e6   
   
    result = minimize(objective, initial_guess, method='L-BFGS-B',  # Better for bounded problems
        options={'ftol': 1e-8, 'maxiter': 1000} )             
    third_point = result.x
    # third_point = initial_guess
    A, B, C, D, E, F = get_ellipse_coeffs_local(third_point)   
    # Calculate ellipse properties
    disc = B**2 - 4*A*C
    if disc >= 0:
        raise ValueError("The computed conic section is not an ellipse.")     
    return A, B, C, D, E, F
def rotate_point(x, y, theta, inverse=False):
    import numpy as np
    if inverse:
        theta = -theta  # Reverse the rotation if inverse is True
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])  # Rotation matrix
    return R @ np.array([x, y])  #@ is a matrix multiplication. Just learning this 
def cubicPolynomial(x, xPoints, yPoints, slopes):
    import numpy as np
    x1,x2 = xPoints
    y1,y2 = yPoints
    m1,m2 = slopes
    A = np.array([
    [x1**3, x1**2, x1, 1],
    [x2**3, x2**2, x2, 1],
    [3*x1**2, 2*x1, 1, 0],
    [3*x2**2, 2*x2, 1, 0]])
    # Create the solution vector b
    b = np.array([y1, y2, m1, m2])
    # Solve Ax = b for x (coefficients A, B, C, D)
    coeffs = np.linalg.solve(A, b)
    A, B, C, D = coeffs
    return A*x**3 + B*x**2 + C*x + D
def compute_span_fractions(points, A, B):
    import numpy as np
    points = np.array(points)
    A = np.array(A)
    B = np.array(B)
    AB = B - A
    AB_len_sq = np.dot(AB, AB)
    # Vector from A to each point
    AP = points - A

    # Projection of AP onto AB gives the span fraction
    s = np.dot(AP, AB) / AB_len_sq
    return s
def map_span_fractions_to_line(span_fractions, A_new, B_new):
    import numpy as np
    A_new = np.array(A_new)
    B_new = np.array(B_new)
    AB_new = B_new - A_new
    return np.array([A_new + s * AB_new for s in span_fractions])
def cartesian_to_cylindrical(coords):
    import numpy as np
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    r = np.sqrt(z**2 + y**2)
    theta = np.arctan2(z, y)

    cylindrical_coords = np.stack((theta, r, x), axis=-1)
    return cylindrical_coords
def rearrange_curve_by_arc_length(curve_points, reference_arc_params):
    import numpy as np
    # Compute arc length parameterization of the new curve
    diffs = np.diff(curve_points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_lengths[-1]
    normalized_arc_length = cumulative_lengths / total_length
    
    # Create interpolation functions for x and y
    interp_x = interp1d(normalized_arc_length, curve_points[:, 0], 
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
    interp_y = interp1d(normalized_arc_length, curve_points[:, 1], 
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Sample at the reference arc length parameters
    new_x = interp_x(reference_arc_params)
    new_y = interp_y(reference_arc_params)
    
    return np.column_stack([new_x, new_y])

def resample_curve_preserve_density(A: np.ndarray, N_out: int = 1000) -> np.ndarray:
    import numpy as np
    # Step 1: Compute cumulative arc length
    diffs = np.diff(A, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    # Step 2: Create interpolators for x, y, z vs arc length
    fx = interp1d(arc_lengths, A[:, 0], kind='linear')
    fy = interp1d(arc_lengths, A[:, 1], kind='linear')
    fz = interp1d(arc_lengths, A[:, 2], kind='linear')

    # Step 3: Sample N_out arc length values spaced like the original (not uniform)
    # We'll do this by using the original arc length density
    arc_sampled = np.interp(
        np.linspace(0, 1, N_out),
        np.linspace(0, 1, len(arc_lengths)),
        arc_lengths
    )

    # Step 4: Interpolate positions at sampled arc lengths
    x_new = fx(arc_sampled)
    y_new = fy(arc_sampled)
    z_new = fz(arc_sampled)

    return np.vstack([x_new, y_new, z_new]).T 
    
def findOptimalBeta(alpha,N, initial_guess):
    def getSSE(beta):
        SSE = np.zeros(N)
        for i in range(N):
            SSE[i] = (beta[i] - alpha[i])**2
        sumSSE = np.sum(SSE)
        return sumSSE
    def objective(beta):
        try:
            Error = getSSE(beta)
            return Error
        except Exception as e:
            print(f"Error: {e}")
            return float('inf')
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds= None, options={'ftol': 1e-8, 'maxiter': 1000})
    betaValues = result.x
    return betaValues

def optimalBeta(Mprime1, Mprime2, theta2, alpha):
    def getTheta(beta):
        theta1 = theta2 + np.tan(beta)*(Mprime1 - Mprime2)
        return theta1
    
def densify_curve_robust(points, n_points_new):
    # Handle edge cases
    if n_points_new <= 0:
        raise ValueError("n_points_new must be positive")
    points = np.atleast_2d(points)
    if points.shape[0] == 1:
        points = points.T
    # If we already have the desired number of points, return copy
    if n_points_new == len(points):
        return points.copy()
    # Case 1: Need to add points (densify)
    elif n_points_new > len(points):
        return _add_points(points, n_points_new)
    # Case 2: Need to remove points (sparsify)
    else:
        return _remove_points(points, n_points_new)
    
def _add_points(points, n_points_new):
    """Add points by inserting at midpoints of largest gaps."""
    current_points = points.copy()
    n_to_add = n_points_new - len(points)
    for _ in range(n_to_add):
        # Calculate distances between consecutive points
        diffs = np.diff(current_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # Find the largest gap
        max_gap_idx = np.argmax(distances)
        # Insert a point at the midpoint of the largest gap
        midpoint = (current_points[max_gap_idx] + current_points[max_gap_idx + 1]) / 2
        # Insert the midpoint
        current_points = np.insert(current_points, max_gap_idx + 1, midpoint, axis=0)
    return current_points

def _remove_points(points, n_points_new):
    """Remove points by eliminating those that create smallest distances."""
    current_points = points.copy()
    n_to_remove = len(points) - n_points_new
    # Always keep the first and last points
    if n_points_new < 2:
        raise ValueError("Cannot reduce to fewer than 2 points")
    for _ in range(n_to_remove):
        if len(current_points) <= 2:
            break
        # Calculate distances between consecutive points
        diffs = np.diff(current_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # Don't remove first or last point, so exclude first and last distances
        # from consideration (they involve the endpoints)
        if len(distances) <= 2:
            break
        # Find the smallest distance among removable points (not endpoints)
        removable_distances = distances[1:-1]  # Exclude first and last distances
        min_distance_idx = np.argmin(removable_distances) + 1  # Adjust index
        # The point to remove is the one that creates this small distance
        # We remove the second point of the pair (min_distance_idx + 1)
        point_to_remove = min_distance_idx + 1
        # Remove the point
        current_points = np.delete(current_points, point_to_remove, axis=0)
    return current_points
    
    
    
    
    
    
    
    
    
    