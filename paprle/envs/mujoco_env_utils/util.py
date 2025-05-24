import math,time,os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import shapely as sp # handle polygon
from shapely import Polygon,LineString,Point # handle polygons
from scipy.spatial.distance import cdist

try:
    import cvxpy as cp
except:
    print ("[WARNING] cvxpy is not installed.")

def rot_mtx(deg):
    """
        2 x 2 rotation matrix
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def pr2t(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def t2pr(T):
    """
        T to p and R
    """   
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        T to p 
    """   
    p = T[:3,3]
    return p

def t2r(T):
    """
        T to R
    """   
    R = T[:3,:3]
    return R    

def rpy2r(rpy_rad):
    """
        roll,pitch,yaw in radian to R
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = math.cos(roll)
    Sphi  = math.sin(roll)
    Cthe  = math.cos(pitch)
    Sthe  = math.sin(pitch)
    Cpsi  = math.cos(yaw)
    Spsi  = math.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def rpy2r_order(r0, order=[0,1,2]):
    """ 
        roll,pitch,yaw in radian to R with ordering
    """
    c1 = math.cos(r0[0]); c2 = math.cos(r0[1]); c3 = math.cos(r0[2])
    s1 = math.sin(r0[0]); s2 = math.sin(r0[1]); s3 = math.sin(r0[2])
    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])
    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])
    assert a.shape == (3,3)
    return a

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out    

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def r2quat(R):
    """ 
        Convert Rotation Matrix to Quaternion.  See rotation.py for notes 
        (https://gist.github.com/machinaut/dab261b78ac19641e91c6490fb9faa96)
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def skew(x):
    """ 
        Get a skew-symmetric matrix
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
        Compute the rotation matrix from an angular velocity vector
    """
    a_norm = np.linalg.norm(a)
    if abs(a_norm-1) > 1e-6:
        print ("[rodrigues] norm of a should be 1.0 not [%.2e]."%(a_norm))
        return np.eye(3)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew(a)
    
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R
    
def np_uv(vec):
    """
        Get unit vector
    """
    x = np.array(vec)
    len = np.linalg.norm(x+1e-8)
    if len <= 1e-6:
        return np.array([0,0,1])
    else:
        return x/len

def get_rotation_matrix_from_two_points(p_fr,p_to):
    p_a  = np.copy(np.array([0,0,1]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # if two points are too close
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R
    

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
    x      = np.random.randn(100,5),
    x_min  = -np.ones(5),
    x_max  = np.ones(5),
    margin = 0.1):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def is_point_in_polygon(point,polygon):
    """
        Is the point inside the polygon
    """
    if isinstance(point,np.ndarray):
        point_check = Point(point)
    else:
        point_check = point
    return sp.contains(polygon,point_check)

def is_point_feasible(point,obs_list):
    """
        Is the point feasible w.r.t. obstacle list
    """
    result = is_point_in_polygon(point,obs_list) # is the point inside each obstacle?
    if sum(result) == 0:
        return True
    else:
        return False

def is_point_to_point_connectable(point1,point2,obs_list):
    """
        Is the line connecting two points connectable
    """
    result = sp.intersects(LineString([point1,point2]),obs_list)
    if sum(result) == 0:
        return True
    else:
        return False
    
class TicTocClass(object):
    """
        Tic toc
        tictoc = TicTocClass()
        tictoc.tic()
        ~~
        tictoc.toc()
    """
    def __init__(self,name='tictoc',print_every=1):
        """
            Initialize
        """
        self.name         = name
        self.time_start   = time.time()
        self.time_end     = time.time()
        self.print_every  = print_every
        self.time_elapsed = 0.0

    def tic(self):
        """
            Tic
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=0,VERBOSE=True,RETURN=False):
        """
            Toc
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if VERBOSE:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if (cnt % self.print_every) == 0:
                if str is None:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (str,time_show,time_unit))
        if RETURN:
            return self.time_elapsed

def get_interp_const_vel_traj(traj_anchor,vel=1.0,HZ=100,ord=np.inf):
    """
        Get linearly interpolated constant velocity trajectory
    """
    L = traj_anchor.shape[0]
    D = traj_anchor.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = traj_anchor[tick-1,:],traj_anchor[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    traj_interp = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D):
        traj_interp[:,d_idx] = np.interp(times_interp,times_anchor,traj_anchor[:,d_idx])
    return times_interp,traj_interp

def meters2xyz(depth_img,cam_matrix):
    """
        Scaled depth image to pointcloud
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img # [H x W x 3]

def compute_view_params(camera_pos,target_pos,up_vector=np.array([0,0,1])):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def sample_xyzs(n_sample,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        Sample a point in three dimensional space with the minimum distance between points
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    return xyzs

def create_folder_if_not_exists(file_path):
    """ 
        Create folder if not exist
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print ("[%s] created."%(folder_path))
        
class MultiSliderClass(object):
    """
        GUI with multiple sliders
    """
    def __init__(self,
                 n_slider      = 10,
                 title         = 'Multiple Sliders',
                 window_width  = 500,
                 window_height = None,
                 x_offset      = 500,
                 y_offset      = 100,
                 slider_width  = 400,
                 label_texts   = None,
                 slider_mins   = None,
                 slider_maxs   = None,
                 slider_vals   = None,
                 resolution    = 0.1,
                 VERBOSE       = True
        ):
        """
            Initialze multiple sliders
        """
        self.n_slider      = n_slider
        self.title         = title
        
        self.window_width  = window_width
        if window_height is None:
            self.window_height = self.n_slider*40
        else:
            self.window_height = window_height
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.slider_width  = slider_width
        
        self.resolution    = resolution
        self.VERBOSE       = VERBOSE
        
        # Slider values
        self.slider_values = np.zeros(self.n_slider)
        
        # Initial/default slider settings
        self.label_texts   = label_texts
        self.slider_mins   = slider_mins
        self.slider_maxs   = slider_maxs
        self.slider_vals   = slider_vals
        
        # Create main window
        self.gui = tk.Tk()
        self.gui.title("%s"%(self.title))
        self.gui.geometry("%dx%d+%d+%d"%
                          (self.window_width,self.window_height,self.x_offset,self.y_offset))

        # Create vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.gui,orient=tk.VERTICAL)
        
        # Create a Canvas widget with the scrollbar attached
        self.canvas = tk.Canvas(self.gui,yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure the scrollbar to control the canvas
        self.scrollbar.config(command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a frame inside the canvas to hold the sliders
        self.sliders_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0),window=self.sliders_frame,anchor=tk.NW)
        
        # Create sliders
        self.sliders = self.create_sliders()
        
        # Update the canvas scroll region when the sliders_frame changes size
        self.sliders_frame.bind("<Configure>",self.cb_scroll)

        # You may want to do this in the main script
        for _ in range(100): self.update() # to avoid GIL-related error 
        
    def cb_scroll(self,event):    
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def cb_slider(self,slider_idx,slider_value):
        """
            Slider callback function
        """
        self.slider_values[slider_idx] = slider_value # append
        if self.VERBOSE:
            print ("slider_idx:[%d] slider_value:[%.1f]"%(slider_idx,slider_value))
        
    def create_sliders(self):
        """
            Create sliders
        """
        sliders = []
        for s_idx in range(self.n_slider):
            # Create label
            if self.label_texts is None:
                label_text = "Slider %02d "%(s_idx)
            else:
                label_text = "[%d/%d]%s"%(s_idx,self.n_slider,self.label_texts[s_idx])
            slider_label = tk.Label(self.sliders_frame, text=label_text)
            slider_label.grid(row=s_idx,column=0,padx=0,pady=0)
            
            # Create slider
            if self.slider_mins is None: slider_min = 0
            else: slider_min = self.slider_mins[s_idx]
            if self.slider_maxs is None: slider_max = 100
            else: slider_max = self.slider_maxs[s_idx]
            if self.slider_vals is None: slider_val = 50
            else: slider_val = self.slider_vals[s_idx]
            slider = tk.Scale(
                self.sliders_frame,
                from_      = slider_min,
                to         = slider_max,
                orient     = tk.HORIZONTAL,
                command    = lambda value,idx=s_idx:self.cb_slider(idx,float(value)),
                resolution = self.resolution,
                length     = self.slider_width
            )
            slider.grid(row=s_idx,column=1,padx=0,pady=0,sticky=tk.W)
            slider.set(slider_val)
            sliders.append(slider)
            
        return sliders
    
    def update(self):
        if self.is_window_exists():
            self.gui.update()
        
    def run(self):
        self.gui.mainloop()
        
    def is_window_exists(self):
        try:
            return self.gui.winfo_exists()
        except tk.TclError:
            return False
        
    def get_slider_values(self):
        return self.slider_values
    
    def set_slider_values(self,slider_values):
        self.slider_values = slider_values
        for slider,slider_value in zip(self.sliders,self.slider_values):
            slider.set(slider_value)
    
    def close(self):
        if self.is_window_exists():
            # some loop
            for _ in range(100): self.update() # to avoid GIL-related error 
            # Close 
            self.gui.destroy()
            self.gui.quit()
            self.gui.update()
        
def get_colors(cmap_name='gist_rainbow',n_color=10,alpha=1.0):
    colors = [plt.get_cmap(cmap_name)(idx) for idx in np.linspace(0,1,n_color)]
    for idx in range(n_color):
        color = colors[idx]
        colors[idx] = color
    return colors

def uv_T_joi(T_joi,joi_fr,joi_to):
    """ 
        Get unit vector between to JOI poses
    """
    return np_uv(t2p(T_joi[joi_to])-t2p(T_joi[joi_fr]))

def len_T_joi(T_joi,joi_fr,joi_to):
    """ 
        Get length between two JOI poses
    """
    return np.linalg.norm(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def get_idxs(list_query,list_domain):
    return [list_query.index(item) for item in list_domain if item in list_query]

def finite_difference_matrix(n, dt, order):
    """
    n: number of points
    dt: time interval
    order: (1=velocity, 2=acceleration, 3=jerk)
    """ 
    # Order
    if order == 1:  # velocity
        coeffs = np.array([-1, 1])
    elif order == 2:  # acceleration
        coeffs = np.array([1, -2, 1])
    elif order == 3:  # jerk
        coeffs = np.array([-1, 3, -3, 1])
    else:
        raise ValueError("Order must be 1, 2, or 3.")

    # Fill-in matrix
    mat = np.zeros((n, n))
    for i in range(n - order):
        for j, c in enumerate(coeffs):
            mat[i, i + j] = c
    return mat / (dt ** order)

def get_A_vel_acc_jerk(n=100,dt=1e-2):
    """
        Get matrices to compute velocities, accelerations, and jerks
    """
    A_vel  = finite_difference_matrix(n,dt,order=1)
    A_acc  = finite_difference_matrix(n,dt,order=2)
    A_jerk = finite_difference_matrix(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def optimization_based_smoothing_1d(
        traj,
        dt=0.1,
        x_init=None,
        x_final=None,
        vel_limit=None,
        acc_limit=None,
        jerk_limit=None,
        p_norm=2,
    ):
    """
        1-D smoothing based on optimization
    """
    n = len(traj)
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    # Convex optimization
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x-traj,p_norm))
    # Boundary condition
    A_list,b_list = [],[]
    if x_init is not None:
        A_list.append(np.eye(n,n)[0,:])
        b_list.append(x_init)
    if x_final is not None:
        A_list.append(np.eye(n,n)[-1,:])
        b_list.append(x_final)
    # Velocity, acceleration, and jerk limits
    C_list,d_list = [],[]
    if vel_limit is not None:
        C_list.append(A_vel)
        C_list.append(-A_vel)
        d_list.append(vel_limit*np.ones(n))
        d_list.append(vel_limit*np.ones(n))
    if acc_limit is not None:
        C_list.append(A_acc)
        C_list.append(-A_acc)
        d_list.append(acc_limit*np.ones(n))
        d_list.append(acc_limit*np.ones(n))
    if jerk_limit is not None:
        C_list.append(A_jerk)
        C_list.append(-A_jerk)
        d_list.append(jerk_limit*np.ones(n))
        d_list.append(jerk_limit*np.ones(n))
    constraints = []
    if A_list:
        A = np.vstack(A_list)
        b = np.hstack(b_list).squeeze()
        constraints.append(A @ x == b) 
    if C_list:
        C = np.vstack(C_list)
        d = np.hstack(d_list).squeeze()
        constraints.append(C @ x <= d)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    traj_smt = x.value
    return traj_smt

def plot_traj_vel_acc_jerk(
        t,traj,
        traj_smt=None,figsize=(6,6),title='Trajectory',
        ):
    """ 
        Plot trajectory, velocity, acceleration, and jerk
    """
    n  = len(t)
    dt = t[1]-t[0]
    # Compute velocity, acceleration, and jerk
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    vel  = A_vel @ traj
    acc  = A_acc @ traj
    jerk = A_jerk @ traj
    if traj_smt is not None:
        vel_smt  = A_vel @ traj_smt
        acc_smt  = A_acc @ traj_smt
        jerk_smt = A_jerk @ traj_smt
    # Plot
    plt.figure(figsize=figsize)
    plt.subplot(4, 1, 1)
    plt.plot(t,traj,'o-',ms=1,color='k',lw=1/5,label='Trajectory')
    if traj_smt is not None:
        plt.plot(t,traj_smt,'o-',ms=1,color='r',lw=1/5,label='Smoothed Trajectory')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 2)
    plt.plot(t,vel,'o-',ms=1,color='k',lw=1/5,label='Velocity')
    if traj_smt is not None:
        plt.plot(t,vel_smt,'o-',ms=1,color='r',lw=1/5,label='Smoothed Velocity')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 3)
    plt.plot(t,acc,'o-',ms=1,color='k',lw=1/5,label='Acceleration')
    if traj_smt is not None:
        plt.plot(t,acc_smt,'o-',ms=1,color='r',lw=1/5,label='Smoothed Acceleration')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 4)
    plt.plot(t,jerk,'o-',ms=1,color='k',lw=1/5,label='Jerk')
    if traj_smt is not None:
        plt.plot(t,jerk_smt,'o-',ms=1,color='r',lw=1/5,label='Smoothed Jerk')
    plt.legend(fontsize=8,loc='upper right')
    plt.suptitle(title,fontsize=10)
    plt.subplots_adjust(hspace=0.2,top=0.95)
    plt.show()

def get_vel_from_traj(traj,dt):
    """ 
        Get velocities from trajectory
    """
    L = traj.shape[0]
    vel = np.zeros(L)
    for tick in range(L-1):
        vel[tick] = np.linalg.norm(traj[tick+1,:]-traj[tick,:])/dt
    vel[-1] = vel[-2] # last two velocities to be the same
    return vel

def get_consecutive_subarrays(array,min_element=1):
    """ 
        Get consecutive sub arrays from an array
    """
    split_points = np.where(np.diff(array) != 1)[0] + 1
    subarrays = np.split(array,split_points)    
    return [subarray for subarray in subarrays if len(subarray) >= min_element]

class PID_ControllerClass(object):
    def __init__(self,
                 name      = 'PID',
                 k_p       = 0.01,
                 k_i       = 0.0,
                 k_d       = 0.001,
                 dt        = 0.01,
                 dim       = 1,
                 dt_min    = 1e-6,
                 out_min   = -np.inf,
                 out_max   = np.inf,
                 ANTIWU    = True,   # anti-windup
                 out_alpha = 0.0    # output EMA (0: no EMA)
                ):
        """
            Initialize PID Controller
        """
        self.name      = name
        self.k_p       = k_p
        self.k_i       = k_i
        self.k_d       = k_d
        self.dt        = dt
        self.dim       = dim
        self.dt_min    = dt_min
        self.out_min   = out_min
        self.out_max   = out_max
        self.ANTIWU    = ANTIWU
        self.out_alpha = out_alpha
        # Buffers
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = 0.0
        self.t_prev   = 0.0
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def reset(self,t_curr=0.0):
        """
            Reset PID Controller
        """
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = t_curr
        self.t_prev   = t_curr
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def update(
        self,
        t_curr  = None,
        x_trgt  = None,
        x_curr  = None,
        VERBOSE = False
        ):
        """
            Update PID controller
            u(t) = K_p e(t) + K_i int e(t) {dt} + K_d {de}/{dt}
        """
        if x_trgt is not None:
            self.x_trgt  = x_trgt
        if t_curr is not None:
            self.t_curr  = t_curr
        if x_curr is not None:
            self.x_curr  = x_curr
            # PID controller updates here
            self.dt       = max(self.t_curr - self.t_prev,self.dt_min)
            self.err_curr = self.x_trgt - self.x_curr     
            self.err_intg = self.err_intg + (self.err_curr*self.dt)
            self.err_diff = self.err_curr - self.err_prev
            
            if self.ANTIWU: # anti-windup
                self.err_out = self.err_curr * self.out_val
                self.err_intg[self.err_out<0.0] = 0.0
            
            if self.dt > self.dt_min:
                self.p_term   = self.k_p * self.err_curr
                self.i_term   = self.k_i * self.err_intg
                self.d_term   = self.k_d * self.err_diff / self.dt
                self.out_val  = np.clip(
                    a     = self.p_term + self.i_term + self.d_term,
                    a_min = self.out_min,
                    a_max = self.out_max)
                # Smooth the output control value using EMA
                self.out_val = self.out_alpha*self.out_val_prev + \
                    (1.0-self.out_alpha)*self.out_val
                self.out_val_prev = self.out_val

                if VERBOSE:
                    print ("cnt:[%d] t_curr:[%.5f] dt:[%.5f]"%
                           (self.cnt,self.t_curr,self.dt))
                    print (" x_trgt:   %s"%(self.x_trgt))
                    print (" x_curr:   %s"%(self.x_curr))
                    print (" err_curr: %s"%(self.err_curr))
                    print (" err_intg: %s"%(self.err_intg))
                    print (" p_term:   %s"%(self.p_term))
                    print (" i_term:   %s"%(self.i_term))
                    print (" d_term:   %s"%(self.d_term))
                    print (" out_val:  %s"%(self.out_val))
                    print (" err_out:  %s"%(self.err_out))
            # Backup
            self.t_prev   = self.t_curr
            self.err_prev = self.err_curr
        # Counter
        if (t_curr is not None) and (x_curr is not None):
            self.cnt = self.cnt + 1
            
    def out(self):
        """
            Get control output
        """
        return self.out_val
    

def rpy2R(r0, order=[0,1,2]):
    c1 = np.math.cos(r0[0]); c2 = np.math.cos(r0[1]); c3 = np.math.cos(r0[2])
    s1 = np.math.sin(r0[0]); s2 = np.math.sin(r0[1]); s3 = np.math.sin(r0[2])

    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])

    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])

    assert a.shape == (3,3)
    return a