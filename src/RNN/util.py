import numpy as np
try:
    from itertools import izip
except ImportError:
    izip = zip

PLY_HEADER = """ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {1}
property list uchar int vertex_indices
end_header"""

PLY_HEADER_NO_COLOR = """ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
element face {1}
property list uchar int vertex_indices
end_header"""

WRL_HEADER = """
#VRML V2.0 utf8
Background { skyColor [1.0 1.0 1.0] } 
Shape{ appearance Appearance {
 material Material {emissiveColor 1 1 1} }
 geometry PointSet {
 coord Coordinate {
 point [
"""

def calculate_normal(camera, S):
  h, w = S.shape[:2]

  Sx, Sy = np.empty_like(S), np.empty_like(S)
  Sx[:,1:-1,:] = (S[:,2:,:] - S[:,:-2,:]) * camera.fx
  Sx[:,0,:] = Sx[:,1,:]
  Sx[:,-1,:] = Sx[:,-2,:]
  Sy[1:-1,:,:] = (S[2:,:,:] - S[:-2,:,:]) * camera.fy
  Sy[0,:,:] = Sy[1,:,:]
  Sy[-1,:,:] = Sy[-2,:,:]
  N = np.cross(Sx, Sy, axis=-1)

  N /= np.linalg.norm(N, axis=-1)[:,:,np.newaxis]

  return N

def calculate_ndotl(camera, S):
  h, w = S.shape[:2]

  x, y = camera.get_image_grid()

  L = np.dstack((x, y, np.ones_like(x)))
  L /= np.linalg.norm(L, axis=-1)[:,:,np.newaxis]

  return np.sum(calculate_normal(camera, S) * L, axis=-1)

# generate a surface from depth values
def generate_surface(camera, z):
  return np.dstack(
      (camera.get_image_grid() + [np.ones_like(z)])) * z[:,:,np.newaxis]

#
#
#

# www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix
def quaternion_to_rotation_matrix(q):
  qxsq, qysq, qzsq = q[1] * q[1], q[2] * q[2], q[3] * q[3]
  qxqy, qxqz, qyqz = q[1] * q[2], q[1] * q[3], q[2] * q[3]
  qxqw, qyqw, qzqw = q[1] * q[0], q[2] * q[0], q[3] * q[0]
  return np.eye(3) + 2 * np.array((
    (-qysq - qzsq,  qxqy - qzqw,  qxqz + qyqw),
    ( qxqy + qzqw, -qxsq - qzsq,  qyqz - qxqw),
    ( qxqz - qyqw,  qyqz + qxqw, -qxsq - qysq)))

def cross_prod_matrix(v):
  M = np.zeros((3, 3))
  M[0, 1], M[0, 2], M[1, 2] = -v[2], v[1], -v[0]
  return M - M.T

# www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
def axis_angle_to_rotation_matrix(axis, angle):
  cp_axis = cross_prod_matrix(axis)
  return np.eye(3) + (
    np.sin(angle) * cp_axis + (1 - np.cos(angle)) * cp_axis.dot(cp_axis))

# www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle
def rotation_matrix_to_axis_angle(R):
  angle = np.arccos((np.trace(R) - 1.) / 2.)
  axis = np.array((R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]))
  axis /= np.linalg.norm(axis)

  return axis, angle

# read an undistorted pinhole camera file
def get_camera_params(camera_file):
  with open(camera_file, 'r') as f:
    f.readline()
    f.readline()
    line = f.readline()

  return map(float, line.split()[1:])

# returns: focal axis, up vector, and translation of camera relative to the
#          original camera world coordinate system
def get_camera_pose(colmap_file, image_name):
  with open(colmap_file, 'r') as f:
    camera_description = False # whether the current line describes a camera

    while True:
      line = f.readline().strip()
      if not line or line.startswith('#'):
        continue

      camera_description = not camera_description

      if not camera_description:
        continue

      data = line.split()

      name = data[-1][data[-1].rfind('/')+1:]

      if name == image_name: # found the camera we're looking for
        camera_id = data[0]

        R = quaternion_to_rotation_matrix(np.array(map(float, data[1:5]))).T
        t = -R.dot(np.array(map(float, data[5:8])))

        break
    else:
      raise Exception('Camera not found!')

    # load 2D points (with an associated 3D point) for the camera
    point_data = f.readline().split()
    points2D = dict(
      (int(point_data[i+2]), (float(point_data[i]), float(point_data[i+1])))
      for i in xrange(0, len(point_data), 3) if point_data[i+2] != '-1')

  return R, t, camera_id, points2D

def load_colmap_camera_positions(colmap_images_file):
  cameras = { } # camera id => camera world coordinates

  with open(colmap_images_file, 'r') as f:
    camera_description = False

    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue

      camera_description = not camera_description
      if not camera_description:
        continue

      # store camera world coordinates
      data = line.split()
      axis = np.array(map(float, data[1:4]))
      angle = np.linalg.norm(axis)
      R = axis_angle_to_rotation_matrix(axis / angle, angle)
      cameras[data[0]] = -R.T.dot(np.array(map(float, data[4:7])))

  return cameras

def undistort_points(x, k, p):
  # iterative undistortion
  xx = x.copy()

  for _ in xrange(20):
    xx2 = xx * xx
    xy = (xx[:,0] * xx[:,1])[:,np.newaxis]
    r2 = (xx2[:,0] + xx2[:,1])[:,np.newaxis]
    radial = k[0] * r2 + k[1] * r2 * r2

    xx = x - (xx * radial + 2 * xy * p.T + (r2 + 2 * xx2) * p[::-1].T)

  return xx

def load_point_ply(ply_file, max_residual=np.inf):
  # get order of properties from header
  xidx, nidx, cidx, ididx = [-1] * 6
  idx = 0

  with open(ply_file, 'r') as f:
    while True:
      line = f.readline().rstrip()
      if line.startswith('end_header'):
        break
      elif line.startswith('element vertex'):
        num_vertices = int(line.split()[-1])
      elif line.startswith('property'):
        if line.endswith(' x'): # xyz
          xidx = idx
        elif line.endswith(' nx'): # normals
          nidx = idx
        elif line.endswith(' red'): # colors; assume order RGB
          cidx = idx
        elif line.endswith(' point_id'):
          ididx = idx

        idx += 1

    assert(xidx != -1 and nidx != -1 and cidx != -1 and ididx != -1)

    points3D = []
    point3D_ids = []
    point3D_lighting = []
    for i in xrange(num_vertices):
      data = f.readline().split()
      residual = float(data[ridx])
      if residual < max_residual:
        points3D.append(map(float, data[xidx:xidx+3]))
        point3D_ids.append(int(data[ididx]))
        point3D_lighting.append(float(data[lidx]))

  return np.array(points3D), np.array(point3D_ids), np.array(point3D_lighting)

# save an SFS surface, automatically generating triangle (face) data
# assumes im is already in range [0,1]
def save_sfs_ply(filename, S, im=None):
  h, w = S.shape[:2]

  S = S.reshape(-1, 3)
  if im is not None:
    im = (im * 255).astype(np.uint8).reshape(-1, 3)

  with open(filename, 'w') as f:
    if im is not None:
      print(PLY_HEADER.format(w * h, (w - 1) * (h - 1) * 2),file=f)
      # write vertex data
      for p, c in izip(S, im):
        print(p[0], p[1], p[2], c[0], c[1], c[2],file=f)
    else:
      print(PLY_HEADER_NO_COLOR.format(w * h, (w - 1) * (h - 1) * 2),file=f)
      # write vertex data
      for p in S:
        print(p[0], p[1], p[2],file=f)

    # write triangle data
    idx = 0
    for i in range(h):
      for j in range(w - 1):
        if i < h - 1: # upper triangle, starting from top left
          print('3', idx, idx + 1, idx + w,file=f)
        if i > 0: # lower triangle, starting from bottom left
          print('3', idx, idx - w + 1, idx + 1,file=f)

        idx += 1
      idx += 1

# assumes colors is already in range [0,1]
def save_ply(filename, points3D, tri_data, colors=None):
  if colors is not None:
    colors = (colors * 255).astype(np.uint8)

  with open(filename, 'w') as f:
    if colors is not None:
      print>>f, PLY_HEADER.format(points3D.shape[0], tri_data.shape[0])
      np.savetxt(f, points3D, '%.4f') # write vertex data
      np.savetxt(f, colors, '%d') # write color data
    else:
      print>>f, PLY_HEADER_NO_COLOR.format(points3D.shape[0], tri_data.shape[0])
      np.savetxt(f, points3D, '%.4f') # write vertex data

    # write triangle data
    np.savetxt(f, tri_data, '3 %u %u %u') # write color data

def save_wrl(wrl_file, points, colors):
  with open(wrl_file, 'w') as f:
    print>>f, WRL_HEADER

    for point in points:
      print>>f, ', '.join(map(str, point))

    print>>f, ' ] }'
    print>>f, ' color Color { color ['

    for color in colors:
      print>>f, ', '.join(map(str, color / 255.))

    print>>f, ' ] } } }'


def save_xyz(filename,points3D):
  with open(filename, 'w') as f:
    np.savetxt(f, points3D, '%.4f') # write vertex data



def bilinear_interpolate(im, x, y):

    mask = np.zeros_like(im)

    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1


    
    # indexx = x.where(x<0)
    # indexy = y.where(y<0)

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    wmask = wa+wb+wc+wd

    #import pdb;pdb.set_trace()
    return np.repeat(wa[:,:,np.newaxis],3,2)*Ia + np.repeat(wb[:,:,np.newaxis],3,2)*Ib + np.repeat(wc[:,:,np.newaxis],3,2)*Ic + np.repeat(wd[:,:,np.newaxis],3,2)*Id
    #wa.reshape([len(wa),1])*Ia + wb.reshape([len(wb),1])*Ib + wc.reshape([len(wc),1])*Ic + wd.reshape([len(wd),1])*Id#,wmask


def get_camera_grid(width,height,cx,cy,fx,fy):
    return np.meshgrid(
        (np.arange(width)-cx)/fx,
        (np.arange(height)-cy)/fy)



def world2cam(points,cx,cy,fx,fy):
    camcorr = points[0:2,:]/points[2,:]
    camcorr[0,:]=camcorr[0,:]*fx+cx
    camcorr[1,:]=camcorr[1,:]*fy+cy
    return camcorr


def readFlow(name):

    
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)