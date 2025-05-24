import cv2,os,mujoco,mujoco_viewer,time
import numpy as np
from paprle.envs.mujoco_env_utils.util import (compute_view_params, get_rotation_matrix_from_two_points,
                                  meters2xyz, pr2t, r2w, rpy2r, trim_scale, r2quat,
                                  t2p,t2r,get_colors,get_idxs,
                                  )

class MuJoCoParserClass(object):
    """
        MuJoCo Parser class
    """
    def __init__(self, name='Robot',rel_xml_path=None,USE_MUJOCO_VIEWER=False,VERBOSE=True):
        """
            Initialize MuJoCo parser
        """
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.VERBOSE      = VERBOSE
        # Constants
        self.tick         = 0
        self.render_tick  = 0
        # Parse an xml file
        if self.rel_xml_path is not None:
            self._parse_xml()
        # Viewer
        self.USE_MUJOCO_VIEWER = USE_MUJOCO_VIEWER
        if self.USE_MUJOCO_VIEWER:
            self.init_viewer()
        # Initial joint position
        self.qpos0 = self.data.qpos
        # Reset
        self.reset()
        # Time
        self.init_sim_time  = self.data.time
        self.init_wall_time = time.time()
        # Print
        if self.VERBOSE:
            self.print_info()


    def _parse_xml(self):
        """
            Parse an xml file
        """
        self.full_xml_path    = os.path.abspath(os.path.join(os.getcwd(),self.rel_xml_path))
        self.model            = mujoco.MjModel.from_xml_path(self.full_xml_path)
        self.data             = mujoco.MjData(self.model)
        self.dt               = self.model.opt.timestep
        self.HZ               = int(1/self.dt)
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,x)
                                for x in range(self.model.ngeom)]
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,x)
                                for x in range(self.n_body)]
        self.n_dof            = self.model.nv # degree of freedom
        self.n_joint          = self.model.njnt     # number of joints 
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtJoint.mjJNT_HINGE,x)
                                 for x in range(self.n_joint)]
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[x] for x in self.pri_joint_idxs]
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins
        self.n_pri_joint      = len(self.pri_joint_idxs)
        # Actuator
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = []
        for addr in self.model.name_actuatoradr:
            ctrl_name = self.model.names[addr:].decode().split('\x00')[0]
            self.ctrl_names.append(ctrl_name) # get ctrl name
        self.ctrl_joint_idxs = []
        self.ctrl_joint_names = []
        self.ctrl_joint_mins = []
        self.ctrl_joint_maxs = []
        for ctrl_idx in range(self.n_ctrl):
            transmission_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid # transmission index
            joint_idx = self.model.jnt_qposadr[transmission_idx][0] # index of the joint when the actuator acts on a joint
            self.ctrl_joint_idxs.append(joint_idx)
            self.ctrl_joint_names.append(self.joint_names[transmission_idx[0]])
            self.ctrl_joint_mins.append(self.joint_ranges[transmission_idx[0],0])
            self.ctrl_joint_maxs.append(self.joint_ranges[transmission_idx[0],1])
            
        self.ctrl_qpos_idxs = self.ctrl_joint_idxs
        self.ctrl_qvel_idxs = []
        for ctrl_idx in range(self.n_ctrl):
            transmission_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid # transmission index
            joint_idx = self.model.jnt_dofadr[transmission_idx][0] # index of the joint when the actuator acts on a joint
            self.ctrl_qvel_idxs.append(joint_idx)
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range
        # Sensors
        self.n_sensor         = self.model.nsensor
        self.sensor_names     = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,x)
                                for x in range(self.n_sensor)]
        # Site (sites are where sensors usually located)
        self.n_site           = self.model.nsite
        self.site_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,x)
                                for x in range(self.n_site)]

    def print_info(self):
        """
            Printout model information
        """
        print ("dt:[%.4f] HZ:[%d]"%(self.dt,self.HZ))
        print ("n_dof (=nv):[%d]"%(self.n_dof))
        print ("n_geom:[%d]"%(self.n_geom))
        print ("geom_names:%s"%(self.geom_names))
        print ("n_body:[%d]"%(self.n_body))
        print ("body_names:%s"%(self.body_names))
        print ("n_joint:[%d]"%(self.n_joint))
        print ("joint_names:%s"%(self.joint_names))
        print ("joint_types:%s"%(self.joint_types))
        
        # Print joint range
        # print ("joint_ranges:\n%s"%(self.joint_ranges))
        for (joint_name,joint_range) in zip(self.joint_names,self.joint_ranges):
            print ("[%s] range: [%.3f]deg~[%.3f]deg"%
                   (joint_name,np.degrees(joint_range[0]),np.degrees(joint_range[1])))

        print ("n_rev_joint:[%d]"%(self.n_rev_joint))
        print ("rev_joint_idxs:%s"%(self.rev_joint_idxs))
        print ("rev_joint_names:%s"%(self.rev_joint_names))
        print ("rev_joint_mins:%s"%(self.rev_joint_mins))
        print ("rev_joint_maxs:%s"%(self.rev_joint_maxs))
        print ("rev_joint_ranges:%s"%(self.rev_joint_ranges))
        print ("n_pri_joint:[%d]"%(self.n_pri_joint))
        print ("pri_joint_idxs:%s"%(self.pri_joint_idxs))
        print ("pri_joint_names:%s"%(self.pri_joint_names))
        print ("pri_joint_mins:%s"%(self.pri_joint_mins))
        print ("pri_joint_maxs:%s"%(self.pri_joint_maxs))
        print ("pri_joint_ranges:%s"%(self.pri_joint_ranges))
        print ("n_ctrl:[%d]"%(self.n_ctrl))
        print ("ctrl_names:%s"%(self.ctrl_names))
        print ("ctrl_joint_idxs:%s"%(self.ctrl_joint_idxs))
        print ("ctrl_joint_names:%s"%(self.ctrl_joint_names))
        print ("ctrl_qvel_idxs:%s"%(self.ctrl_qvel_idxs))
        print ("ctrl_ranges:\n%s"%(self.ctrl_ranges))
        print ("n_sensor:[%d]"%(self.n_sensor))
        print ("sensor_names:%s"%(self.sensor_names))
        print ("n_site:[%d]"%(self.n_site))
        print ("site_names:%s"%(self.site_names))

    def init_viewer(self,viewer_title='MuJoCo',viewer_width=1200,viewer_height=800,
                    viewer_hide_menus=True,
                    FONTSCALE_VALUE=mujoco.mjtFontScale.mjFONTSCALE_100.value):
        """
            Initialize viewer
            - FONTSCALE_VALUE:[50,100,150,200,250,300]
        """
        self.USE_MUJOCO_VIEWER = True
        self.viewer = mujoco_viewer.MujocoViewer(
                self.model,self.data,mode='window',title=viewer_title,
                width=viewer_width,height=viewer_height,hide_menus=viewer_hide_menus)
        # Modify the fontsize
        self.viewer.ctx = mujoco.MjrContext(self.model,FONTSCALE_VALUE)

    def update_viewer(self,azimuth=None,distance=None,elevation=None,lookat=None,
                      VIS_TRANSPARENT=None,VIS_CONTACTPOINT=None,
                      contactwidth=None,contactheight=None,contactrgba=None,
                      VIS_JOINT=None,jointlength=None,jointwidth=None,jointrgba=None,
                      CALL_MUJOCO_FUNC=True):
        """
            Initialize viewer
        """
        if azimuth is not None:
            self.viewer.cam.azimuth = azimuth
        if distance is not None:
            self.viewer.cam.distance = distance
        if elevation is not None:
            self.viewer.cam.elevation = elevation
        if lookat is not None:
            self.viewer.cam.lookat = lookat
        if VIS_TRANSPARENT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = VIS_TRANSPARENT
        if VIS_CONTACTPOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = VIS_CONTACTPOINT
        if contactwidth is not None:
            self.model.vis.scale.contactwidth = contactwidth
        if contactheight is not None:
            self.model.vis.scale.contactheight = contactheight
        if contactrgba is not None:
            self.model.vis.rgba.contactpoint = contactrgba
        if VIS_JOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = VIS_JOINT
        if jointlength is not None:
            self.model.vis.scale.jointlength = jointlength
        if jointwidth is not None:
            self.model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None:
            self.model.vis.rgba.joint = jointrgba
        # Call MuJoCo functions for immediate modification
        if CALL_MUJOCO_FUNC:
            # Forward
            mujoco.mj_forward(self.model,self.data)
            # Update scene and render
            mujoco.mjv_updateScene(
                self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
            
    def update_font_scale_from_cam_dist(
        self,cam_dists=[2.0,2.5,3.0,4.0],font_scales=[300,250,200,150,100],VERBOSE=False):
        """ 
            Update font scale from cam distance
        """
        def map_x_to_output(numbers,outputs,x):
            if x < numbers[0]:
                return outputs[0]
            if x >= numbers[-1]:
                return outputs[-1]
            for i in range(len(numbers) - 1):
                if numbers[i] <= x < numbers[i + 1]:
                    return outputs[i + 1]
        
        cam_dist = self.viewer.cam.distance
        font_scale_new = map_x_to_output(numbers=cam_dists,outputs=font_scales,x=cam_dist)
        font_scale_curr = self.viewer.ctx.fontScale
        
        if np.abs(font_scale_curr-font_scale_new) > 1.0: # if font scale changes
            self.viewer.ctx = mujoco.MjrContext(self.model,font_scale_new)
            if VERBOSE:
                print ("font_scale modified. [%d]=>[%d]"%(font_scale_curr,font_scale_new))

    def get_viewer_cam_info(self,VERBOSE=False):
        """
            Get viewer cam information
        """
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance
        cam_elevation = self.viewer.cam.elevation
        cam_lookat    = self.viewer.cam.lookat.copy()
        if VERBOSE:
            print ("cam_azimuth:[%.2f] cam_distance:[%.2f] cam_elevation:[%.2f] cam_lookat:%s]"%
                (cam_azimuth,cam_distance,cam_elevation,cam_lookat))
        return cam_azimuth,cam_distance,cam_elevation,cam_lookat

    def is_viewer_alive(self):
        """
            Check whether a viewer is alive
        """
        return self.viewer.is_alive

    def reset(self):
        """
            Reset
        """
        mujoco.mj_resetData(self.model,self.data)
        # To initial position
        self.data.qpos = self.qpos0
        mujoco.mj_forward(self.model,self.data)
        # Reset ticks
        self.tick        = 0
        self.render_tick = 0

    def step(self,ctrl=None,ctrl_idxs=None,nstep=1,INCREASE_TICK=True):
        """
            Forward dynamics
        """
        if ctrl is not None:
            if ctrl_idxs is None:
                self.data.ctrl[:] = ctrl
            else:
                self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model,self.data,nstep=nstep)
        if INCREASE_TICK:
            self.tick = self.tick + 1

    def forward(self,q=None,joint_idxs=None,INCREASE_TICK=True):
        """
            Forward kinematics
        """
        if q is not None:
            if joint_idxs is not None:
                self.data.qpos[joint_idxs] = q
            else:
                self.data.qpos = q
        mujoco.mj_forward(self.model,self.data)
        if INCREASE_TICK:
            self.tick = self.tick + 1

    def get_state(
            self,
        ):
        """ 
            Get MuJoCo state
        """
        state = {
            'qpos':self.data.qpos,
            'qvel':self.data.qvel,
            'qacc':self.data.qacc,
            'act':self.data.act,
            'ctrl':self.data.ctrl,
        }
        return state
    
    def set_state(
            self,
            qpos = None,
            qvel = None,
            qacc = None,
            act  = None, # used for simulating tendons and muscles
            ctrl = None,
        ):
        """ 
            Set MuJoCo state
        """
        if qpos is not None: self.data.qpos = qpos
        if qvel is not None: self.data.qvel = qvel
        if qacc is not None: self.data.qacc = qacc
        if act is not None: self.data.act = act
        if ctrl is not None: self.data.ctrl = ctrl


    def set_init_sim_time(self,init_sim_time=None):
        """
            Initialize simulation time (sec)
        """
        if init_sim_time:
            self.init_sim_time = init_sim_time
        else:
            self.init_sim_time = self.data.time
    
    def set_init_wall_time(self,init_wall_time=None):
        """ 
            Initialize wall clock time
        """
        if init_wall_time:
            self.init_wall_time = init_wall_time
        else:
            self.init_wall_time = time.time()
    
    def get_sim_time(self,init_flag=False):
        """
            Get simulation time (sec)
        """
        if init_flag:
            self.init_sim_time = self.data.time
        elapsed_time = self.data.time - self.init_sim_time
        return elapsed_time
    
    def get_wall_time(self,init_flag=False):
        """ 
            Get wall clock time
        """
        if init_flag:
            self.init_wall_time = time.time()
        elapsed_time = time.time() - self.init_wall_time # second
        return elapsed_time

    def render(self,render_every=1):
        """
            Render
        """
        if self.USE_MUJOCO_VIEWER:
            if ((self.render_tick % render_every) == 0) or (self.render_tick == 0):
                self.viewer.render()
            self.render_tick = self.render_tick + 1
        else:
            print ("[%s] Viewer NOT initialized."%(self.name))

    def grab_image(self,resize_rate=None,interpolation=cv2.INTER_NEAREST):
        """
            Grab the rendered iamge
        """
        img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
        mujoco.mjr_readPixels(img, None,self.viewer.viewport,self.viewer.ctx)
        img = np.flipud(img) # flip image
        # Resize
        if resize_rate is not None:
            h = int(img.shape[0]*resize_rate)
            w = int(img.shape[1]*resize_rate)
            img = cv2.resize(img,(w,h),interpolation=interpolation)
        return img.copy()

    def close_viewer(self):
        """
            Close viewer
        """
        self.USE_MUJOCO_VIEWER = False
        self.viewer.close()

    def get_p_body(self,body_name):
        """
            Get body position
        """
        return self.data.body(body_name).xpos.copy()

    def get_R_body(self,body_name):
        """
            Get body rotation matrix
        """
        return self.data.body(body_name).xmat.reshape([3,3]).copy()

    def get_pR_body(self,body_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p,R
    
    def get_p_joint(self,joint_name):
        """
            Get joint position
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_p_body(self.body_names[body_id])

    def get_R_joint(self,joint_name):
        """
            Get joint rotation matrix
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_R_body(self.body_names[body_id])
    
    def get_pR_joint(self,joint_name):
        """
            Get joint position and rotation matrix
        """
        p = self.get_p_joint(joint_name)
        R = self.get_R_joint(joint_name)
        return p,R

    def get_p_geom(self,geom_name):
        """ 
            Get geom position
        """
        return self.data.geom(geom_name).xpos
    
    def get_R_geom(self,geom_name):
        """ 
            Get geom rotation
        """
        return self.data.geom(geom_name).xmat.reshape((3,3))
    
    def get_pR_geom(self,geom_name):
        """
            Get geom position and rotation matrix
        """
        p = self.get_p_geom(geom_name)
        R = self.get_R_geom(geom_name)
        return p,R
    
    def get_p_sensor(self,sensor_name):
        """
             Get sensor position
        """
        sensor_id = self.model.sensor(sensor_name).id # get sensor ID
        sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
        sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
        p = self.data.site(site_name).xpos.copy() # get the position of the site
        return p
    
    def get_R_sensor(self,sensor_name):
        """
             Get sensor position
        """
        sensor_id = self.model.sensor(sensor_name).id
        sensor_objtype = self.model.sensor_objtype[sensor_id]
        sensor_objid = self.model.sensor_objid[sensor_id]
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid)
        R = self.data.site(site_name).xmat.reshape([3,3]).copy()
        return R
    
    def get_pR_sensor(self,sensor_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_sensor(sensor_name)
        R = self.get_R_sensor(sensor_name)
        return p,R

    def get_q(self,joint_idxs=None):
        """
            Get joint position in (radian)
        """
        if joint_idxs is None:
            q = self.data.qpos
        else:
            q = self.data.qpos[joint_idxs]
        return q.copy()

    def get_J_body(self,body_name):
        """
            Get Jocobian matrices of a body
        """
        J_p = np.zeros((3,self.model.nv)) # nv: nDoF
        J_R = np.zeros((3,self.model.nv))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full
    
    def get_J_geom(self,geom_name):
        """
            Get Jocobian matrices of a geom
        """
        J_p = np.zeros((3,self.model.nv)) # nv: nDoF
        J_R = np.zeros((3,self.model.nv))
        mujoco.mj_jacGeom(self.model,self.data,J_p,J_R,self.data.geom(geom_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_ik_ingredients(
            self,
            body_name=None,
            geom_name=None,
            p_trgt=None,
            R_trgt=None,
            IK_P=True,
            IK_R=True
        ):
        """
            Get IK ingredients
        """
        if body_name is not None:
            J_p,J_R,J_full = self.get_J_body(body_name=body_name)
            p_curr,R_curr = self.get_pR_body(body_name=body_name)
        if geom_name is not None:
            J_p,J_R,J_full = self.get_J_geom(geom_name=geom_name)
            p_curr,R_curr = self.get_pR_geom(geom_name=geom_name)
        if (body_name is not None) and (geom_name is not None):
            print ("[get_ik_ingredients] body_name:[%s] geom_name:[%s] are both not None!"%(body_name,geom_name))
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_full
            err   = np.concatenate((p_err,w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err
    
    def get_ik_ingredients_geom(self,geom_name,p_trgt=None,R_trgt=None,IK_P=True,IK_R=True):
        """
            Get IK ingredients
        """
        J_p,J_R,J_full = self.get_J_geom(geom_name=geom_name)
        p_curr,R_curr = self.get_pR_geom(geom_name=geom_name)
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_full
            err   = np.concatenate((p_err,w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err

    def damped_ls(self,J,err,eps=1e-6,stepsize=1.0,th=5*np.pi/180.0):
        """
            Dampled least square for IK
        """
        dq = stepsize*np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq

    def onestep_ik(self,body_name,p_trgt=None,R_trgt=None,IK_P=True,IK_R=True,
                   joint_idxs=None,stepsize=1,eps=1e-1,th=5*np.pi/180.0):
        """
            Solve IK for a single step
        """
        J,err = self.get_ik_ingredients(
            body_name=body_name,p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R)
        dq = self.damped_ls(J,err,stepsize=stepsize,eps=eps,th=th)
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        q = self.get_q(joint_idxs=joint_idxs)
        q = q + dq[joint_idxs]
        # FK
        self.forward(q=q,joint_idxs=joint_idxs)
        return q, err
    
    def solve_ik(self,body_name,p_trgt,R_trgt,IK_P,IK_R,q_init,rev_joint_idxs,
                 RESET=False,DO_RENDER=False,render_every=1,th=1*np.pi/180.0,err_th=1e-6):
        """
            Solve IK
        """
        if RESET:
            self.reset()
        q_backup = self.get_q(joint_idxs=rev_joint_idxs)
        q = q_init.copy()
        self.forward(q=q,joint_idxs=rev_joint_idxs)
        tick = 0
        while True:
            tick = tick + 1
            J,err = self.get_ik_ingredients(
                body_name=body_name,p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R)
            dq = self.damped_ls(J,err,stepsize=1,eps=1e-1,th=th)
            q = q + dq[rev_joint_idxs]
            self.forward(q=q,joint_idxs=rev_joint_idxs)
            # Terminate condition
            err_norm = np.linalg.norm(err)
            if err_norm < err_th:
                break
            # Render
            if DO_RENDER:
                if ((tick-1)%render_every) == 0:
                    p_tcp,R_tcp = self.get_pR_body(body_name=body_name)
                    self.plot_T(p=p_tcp,R=R_tcp,PLOT_AXIS=True,axis_len=0.1,axis_width=0.005)
                    self.plot_T(p=p_trgt,R=R_trgt,PLOT_AXIS=True,axis_len=0.2,axis_width=0.005)
                    self.render()
        # Back to back-uped position
        q_ik = self.get_q(joint_idxs=rev_joint_idxs)
        self.forward(q=q_backup,joint_idxs=rev_joint_idxs)
        return q_ik

    def plot_sphere(self,p,r,rgba=[1,1,1,1],label=''):
        """
            Add sphere
        """
        self.viewer.add_marker(
            pos   = p,
            size  = [r,r,r],
            rgba  = rgba,
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = label)

    def plot_T(self,p,R,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],axis_rgba=None,
               label=None):
        """
            Plot coordinate axes
        """
        if PLOT_AXIS:
            if axis_rgba is None:
                rgba_x = [1.0,0.0,0.0,0.9]
                rgba_y = [0.0,1.0,0.0,0.9]
                rgba_z = [0.0,0.0,1.0,0.9]
            else:
                rgba_x = axis_rgba[0]
                rgba_y = axis_rgba[1]
                rgba_z = axis_rgba[2]
            # X axis
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p + R_x[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = ''
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = ''
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = ''
            )
        if PLOT_SPHERE:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')
        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label)
            
    def plot_box(self,p=np.array([0,0,0]),R=np.eye(3),
                 xlen=1.0,ylen=1.0,zlen=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_BOX,
            size  = [xlen,ylen,zlen],
            rgba  = rgba,
            label = ''
        )
        
    def plot_capsule(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size  = [r,r,h],
            rgba  = rgba,
            label = ''
        )
        
    def plot_cylinder(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,h],
            rgba  = rgba,
            label = ''
        )
    
    def plot_ellipsoid(self,p=np.array([0,0,0]),R=np.eye(3),rx=1.0,ry=1.0,rz=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size  = [rx,ry,rz],
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,h*2],
            rgba  = rgba,
            label = ''
        )
        
    def plot_line(self,p=np.array([0,0,0]),R=np.eye(3),h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = h,
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow_fr2to(self,p_fr,p_to,r=1.0,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,np.linalg.norm(p_to-p_fr)*2],
            rgba  = rgba,
            label = ''
        )

    def plot_line_fr2to(self,p_fr,p_to,rgba=[0.5,0.5,0.5,0.5], label=''):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = np.linalg.norm(p_to-p_fr),
            rgba  = rgba,
            label = label
        )
    
    def plot_cylinder_fr2to(self,p_fr,p_to,r=0.01,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = (p_fr+p_to)/2,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,np.linalg.norm(p_to-p_fr)/2],
            rgba  = rgba,
            label = ''
        )
            
    def plot_body_T(self,body_name,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],axis_rgba=None,
               label=None):
        """
            Plot coordinate axes on a body
        """
        p,R = self.get_pR_body(body_name=body_name)
        self.plot_T(p,R,PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                    PLOT_SPHERE=PLOT_SPHERE,sphere_r=sphere_r,sphere_rgba=sphere_rgba,axis_rgba=axis_rgba,
                    label=label)
        
    def plot_joint_T(self,joint_name,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],axis_rgba=None,
               label=None):
        """
            Plot coordinate axes on a joint
        """
        p,R = self.get_pR_joint(joint_name=joint_name)
        self.plot_T(p,R,PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                    PLOT_SPHERE=PLOT_SPHERE,sphere_r=sphere_r,sphere_rgba=sphere_rgba,axis_rgba=axis_rgba,
                    label=label)
        
    def plot_geom_T(self,geom_name,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],axis_rgba=None,
               label=None):
        """
            Plot coordinate axes on a goem
        """
        p,R = self.get_pR_geom(geom_name=geom_name)
        self.plot_T(p,R,PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                    PLOT_SPHERE=PLOT_SPHERE,sphere_r=sphere_r,sphere_rgba=sphere_rgba,axis_rgba=axis_rgba,
                    label=label)

    def plot_arrow_contact(self,p,uv,r_arrow=0.03,h_arrow=0.3,rgba=[1,0,0,1],label=''):
        """
            Plot arrow
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r_arrow,r_arrow,h_arrow],
            rgba  = rgba,
            label = label
        )
        
    def plot_joint_axis(self,axis_len=0.1,axis_r=0.01,joint_names=None):
        """ 
            Plot revolute joint axis 
        """
        rev_joint_idxs  = self.rev_joint_idxs
        rev_joint_names = self.rev_joint_names

        if joint_names is not None:
            idxs = get_idxs(self.rev_joint_names,joint_names)
            rev_joint_idxs_to_use  = rev_joint_idxs[idxs]
            rev_joint_names_to_use = [rev_joint_names[i] for i in idxs]
        else:
            rev_joint_idxs_to_use  = rev_joint_idxs
            rev_joint_names_to_use = rev_joint_names

        for rev_joint_idx,rev_joint_name in zip(rev_joint_idxs_to_use,rev_joint_names_to_use):
            axis_joint      = self.model.jnt_axis[rev_joint_idx]
            p_joint,R_joint = self.get_pR_joint(joint_name=rev_joint_name)
            axis_world      = R_joint@axis_joint
            axis_rgba       = np.append(np.eye(3)[:,np.argmax(axis_joint)],0.2)
            self.plot_arrow_fr2to(
                p_fr = p_joint,
                p_to = p_joint+axis_len*axis_world,
                r    = axis_r,
                rgba = axis_rgba
            )
            
    def plot_joint_info(
        self,PLOT_AXIS=True,axis_len=0.05,axis_width=0.05):
        """ 
            Plot joint information
        """
        for joint_name in self.joint_names:
            p_joint,R_joint = self.get_pR_joint(joint_name=joint_name)
            self.plot_T(
                p_joint,R_joint,
                PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                label='%s'%(joint_name))
            
    def plot_body_info(
        self,PLOT_AXIS=True,axis_len=0.05,axis_width=0.05,PLOT_BODY_NAME=True):
        """
            Plot body information
        """
        for body_name in self.body_names:
            p_body,R_body = self.get_pR_body(body_name=body_name)
            if PLOT_BODY_NAME:
                label = '%s'%(body_name)
            else:
                label = ''
            self.plot_T(
                p_body,R_body,
                PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                label=label)
            
    def plot_joi(
        self,T_joi,
        PLOT_AXIS=True,axis_len=0.05,axis_width=0.01,
        PLOT_SPHERE=False,sphere_r=0.0075,sphere_rgba=[1,0,0,0.25],
        PLOT_NAME=True,
        ):
        """ 
            Plot joints of interest (JOI) 
        """
        for key in T_joi.keys():
            if PLOT_NAME:
                label = key
            else:
                label = ''
            self.plot_T(
                p=t2p(T_joi[key]),R=t2r(T_joi[key]),
                PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                PLOT_SPHERE=PLOT_SPHERE,sphere_r=sphere_r,sphere_rgba=sphere_rgba,
                label=label, # '', key
            )

    def plot_traj(
        self,
        traj,
        rgba        = [1,0,0,1],
        plot_line   = True,
        plot_sphere = True,
        sphere_r    = 0.01,
        ):
        """ 
            Plot trajectory
        """
        L = traj.shape[0]
        if plot_line:
            for idx in range(L-1):
                p_fr = traj[idx,:]
                p_to = traj[idx+1,:]
                self.plot_line_fr2to(p_fr=p_fr,p_to=p_to,rgba=rgba)
        if plot_sphere:
            for idx in range(L):
                p = traj[idx,:]
                self.plot_sphere(p=p,r=sphere_r,rgba=rgba)
            
    def get_body_names(self,prefix='obj_'):
        """
            Get body names with prefix
        """
        body_names = [x for x in self.body_names if x[:len(prefix)]==prefix]
        return body_names

    def get_contact_info(self,must_include_prefix=None,must_exclude_prefix=None):
        """
            Get contact information
        """
        p_contacts = []
        f_contacts = []
        geom1s = []
        geom2s = []
        body1s = []
        body2s = []
        for c_idx in range(self.data.ncon):
            contact   = self.data.contact[c_idx]
            # Contact position and frame orientation
            p_contact = contact.pos # contact position
            R_frame   = contact.frame.reshape(( 3,3))
            # Contact force
            f_contact_local = np.zeros(6,dtype=np.float64)
            mujoco.mj_contactForce(self.model,self.data,0,f_contact_local)
            f_contact = R_frame @ f_contact_local[:3] # in the global coordinate
            # Contacting geoms
            contact_geom1 = self.geom_names[contact.geom1]
            contact_geom2 = self.geom_names[contact.geom2]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
            # Append
            if must_include_prefix is not None:
                if (contact_geom1[:len(must_include_prefix)] == must_include_prefix) or (contact_geom2[:len(must_include_prefix)] == must_include_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            elif must_exclude_prefix is not None:
                if (contact_geom1[:len(must_exclude_prefix)] != must_exclude_prefix) and (contact_geom2[:len(must_exclude_prefix)] != must_exclude_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            else:
                p_contacts.append(p_contact)
                f_contacts.append(f_contact)
                geom1s.append(contact_geom1)
                geom2s.append(contact_geom2)
                body1s.append(contact_body1)
                body2s.append(contact_body2)
        return p_contacts,f_contacts,geom1s,geom2s,body1s,body2s

    def plot_contact_info(self,must_include_prefix=None,h_arrow=0.3,rgba_arrow=[1,0,0,1],
                          PRINT_CONTACT_BODY=False,PRINT_CONTACT_GEOM=False,VERBOSE=False):
        """
            Plot contact information
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        # Render contact informations
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            f_norm = np.linalg.norm(f_contact)
            f_uv = f_contact / (f_norm+1e-8)
            # h_arrow = 0.3 # f_norm*0.05
            self.plot_arrow_contact(p=p_contact,uv=f_uv,r_arrow=0.01,h_arrow=h_arrow,rgba=rgba_arrow,
                        label='')
            self.plot_arrow_contact(p=p_contact,uv=-f_uv,r_arrow=0.01,h_arrow=h_arrow,rgba=rgba_arrow,
                        label='')
            if PRINT_CONTACT_BODY:
                label = '[%s]-[%s]'%(body1,body2)
            elif PRINT_CONTACT_GEOM:
                label = '[%s]-[%s]'%(geom1,geom2)
            else:
                label = '' 
            self.plot_sphere(p=p_contact,r=0.02,rgba=[1,0.2,0.2,1],label=label)
        # Print
        if VERBOSE:
            self.print_contact_info(must_include_prefix=must_include_prefix)
            
    def print_contact_info(self,must_include_prefix=None):
        """ 
            Print contact information
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            print ("Tick:[%d] Body contact:[%s]-[%s]"%(self.tick,body1,body2))

    def open_interactive_viewer(self):
        """
            Open interactive viewer
        """
        from mujoco import viewer
        viewer.launch(self.model)

    def get_T_viewer(self,fovy=45):
        """
            Get viewer pose
        """
        cam_lookat    = self.viewer.cam.lookat
        cam_elevation = self.viewer.cam.elevation
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance

        p_lookat = cam_lookat
        R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth]))
        T_lookat = pr2t(p_lookat,R_lookat)
        T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3))
        return T_viewer

    def grab_rgb_depth_img(self):
        """
            Grab RGB and Depth images
        """
        rgb_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img)

        # Rescale depth image
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - depth_img * (1 - near / far))
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img
    
    def get_pcd_from_depth_img(self,depth_img,fovy=45):
        """
            Get point cloud data from depth image
        """
        # Get camera pose
        T_viewer = self.get_T_viewer(fovy=fovy)

        # Camera intrinsic
        img_height = depth_img.shape[0]
        img_width = depth_img.shape[1]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
        cam_matrix = np.array(((focal_scaling,0,img_width/2),
                            (0,focal_scaling,img_height/2),
                            (0,0,1)))

        # Estimate 3D point from depth image
        xyz_img = meters2xyz(depth_img,cam_matrix) # [H x W x 3]
        xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]

        # To world coordinate
        xyzone_world_transpose = T_viewer @ xyzone_transpose
        xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
        xyz_world = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]
        return xyz_world,xyz_img
    
    def get_egocentric_rgb_depth_pcd(self,p_ego=None,p_trgt=None,rsz_rate=50,fovy=45,
                                     BACKUP_AND_RESTORE_VIEW=False):
        """
            Get egocentric 1) RGB image, 2) Depth image, 3) Point Cloud Data
        """
        if BACKUP_AND_RESTORE_VIEW:
            # Backup camera information
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = compute_view_params(
                camera_pos=p_ego,target_pos=p_trgt,up_vector=np.array([0,0,1]))
            self.update_viewer(azimuth=cam_azimuth,distance=cam_distance,
                               elevation=cam_elevation,lookat=cam_lookat)
        
        # Grab RGB and depth image
        rgb_img,depth_img = self.grab_rgb_depth_img() # get rgb and depth images

        # Resize
        h_rsz,w_rsz = depth_img.shape[0]//rsz_rate,depth_img.shape[1]//rsz_rate
        depth_img_rsz = cv2.resize(depth_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)

        # Get PCD
        pcd,xyz_img = self.get_pcd_from_depth_img(depth_img_rsz,fovy=fovy) # [N x 3]

        if BACKUP_AND_RESTORE_VIEW:
            # Restore camera information
            self.update_viewer(azimuth=viewer_azimuth,distance=viewer_distance,
                               elevation=viewer_elevation,lookat=viewer_lookat)
        return rgb_img,depth_img,pcd,xyz_img

    def get_tick(self):
        """
            Get tick
        """
        tick = int(self.get_sim_time()/self.dt)
        return tick

    def loop_every(self,HZ=None,tick_every=None):
        """
            Loop every
        """
        # tick = int(self.get_sim_time()/self.dt)
        FLAG = False
        if HZ is not None:
            FLAG = (self.tick-1)%(int(1/self.dt/HZ))==0
        if tick_every is not None:
            FLAG = (self.tick-1)%(tick_every)==0
        return FLAG
    
    def get_sensor_value(self,sensor_name):
        """
            Read sensor value
        """
        data = self.data.sensor(sensor_name).data
        return data.copy()

    def get_sensor_values(self,sensor_names=None):
        """
            Read multiple sensor values
        """
        if sensor_names is None:
            sensor_names = self.sensor_names
        data = np.array([self.get_sensor_value(sensor_name) for sensor_name in sensor_names]).squeeze()
        return data.copy()
    
    def get_qpos_joint(self,joint_name):
        """
            Get joint position
        """
        addr = self.model.joint(joint_name).qposadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        qpos = self.data.qpos[addr:addr+L]
        return qpos
    
    def get_qvel_joint(self,joint_name):
        """
            Get joint velocity
        """
        addr = self.model.joint(joint_name).dofadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        if L > 1: L = 6
        qvel = self.data.qvel[addr:addr+L]
        return qvel
    
    def get_qpos_joints(self,joint_names):
        """
            Get multiple joint positions from 'joint_names'
        """
        return np.array([self.get_qpos_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_qvel_joints(self,joint_names):
        """
            Get multiple joint velocities from 'joint_names'
        """
        return np.array([self.get_qvel_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def viewer_pause(self):
        """
            Viewer pause
        """
        self.viewer._paused = True
        
    def viewer_resume(self):
        """
            Viewer resume
        """
        self.viewer._paused = False

    def is_viewer_paused(self):
        return self.viewer._paused
        
    def get_idxs_fwd(self,joint_names):
        """ 
            Get indices for using env.forward()
            Example)
            env.forward(q=q,joint_idxs=idxs_fwd) # <= HERE
        """
        return [self.model.joint(jname).qposadr[0] for jname in joint_names]
    
    def get_idxs_jac(self,joint_names):
        """ 
            Get indices for solving inverse kinematics
            Example)
            J,ik_err = env.get_ik_ingredients(...)
            dq = env.damped_ls(J,ik_err,stepsize=1,eps=1e-2,th=np.radians(1.0))
            q = q + dq[idxs_jac] # <= HERE
        """
        return [self.model.joint(jname).dofadr[0] for jname in joint_names]
    
    def get_idxs_step(self,joint_names):
        """ 
            Get indices for using env.step()
            Example)
            env.step(ctrl=q,ctrl_idxs=idxs_step) # <= HERE
        """
        return [self.ctrl_joint_names.index(jname) for jname in joint_names]

    def get_geom_idxs_from_body_name(self,body_name):
        """ 
            Get geometry indices for a body name to modify the properties of geom attached to a body
        """
        body_idx = self.body_names.index(body_name)
        geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx] 
        return geom_idxs
    
    def set_p_root(self,root_name='torso',p=np.array([0,0,0])):
        """ 
             Set the position of a specific body
             FK must be called after
        """
        jntadr  = self.model.body(root_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr:qposadr+3] = p
        
    def set_R_root(self,root_name='torso',R=np.eye(3,3)):
        """ 
            Set the rotation of a root joint
            FK must be called after
        """
        jntadr  = self.model.body(root_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr+3:qposadr+7] = r2quat(R)
        
    def set_quat_root(self,root_name='torso',quat=np.array([0,0,0,0])):
        """ 
            Set the rotation of a root joint
            FK must be called after
        """
        jntadr  = self.model.body(root_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr+3:qposadr+7] = quat
        
    def get_q_couple(
        self,
        q_raw,
        coupled_joint_idxs_list    = None,
        coupled_joint_names_list   = None,
        coupled_joint_weights_list = None,
        ):
        """ 
            Coupled joint positions
        """
        q_couple = q_raw.copy()
        if coupled_joint_idxs_list is not None:
            for i in range(len(coupled_joint_idxs_list)): # for each couple
                coupled_joint_idxs    = coupled_joint_idxs_list[i]
                coupled_joint_weights = coupled_joint_weights_list[i]
                joint_sum = 0
                for j in range(len(coupled_joint_idxs)):
                    joint_sum += q_raw[coupled_joint_idxs[j]]
                joint_sum /= np.sum(coupled_joint_weights)
                for k in range(len(coupled_joint_idxs)):
                    q_couple[coupled_joint_idxs[k]] = joint_sum*coupled_joint_weights[k] # distribute coupled joint positions
        if coupled_joint_names_list is not None:
            for i in range(len(coupled_joint_names_list)): # for each couple
                coupled_joint_names   = coupled_joint_names_list[i]
                coupled_joint_idxs    = get_idxs(self.joint_names,coupled_joint_names)
                coupled_joint_weights = coupled_joint_weights_list[i]
                joint_sum = 0
                for j in range(len(coupled_joint_idxs)):
                    joint_sum += q_raw[coupled_joint_idxs[j]]
                joint_sum /= np.sum(coupled_joint_weights)
                for k in range(len(coupled_joint_idxs)):
                    q_couple[coupled_joint_idxs[k]] = joint_sum*coupled_joint_weights[k] # distribute coupled joint positions
        return q_couple
    
    def set_geom_color(self,rgba=[0.75,0.95,0.15,1.0],body_names_to_color=None,body_names_to_exclude=['world']):
        """
            Set body color
        """
        if body_names_to_color is None:
            body_names_to_color = self.body_names
        for body_name in body_names_to_color: # for all bodies
            if body_name in body_names_to_exclude: 
                continue 
            body_idx = self.body_names.index(body_name)
            geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx]
            for geom_idx in geom_idxs: # for geoms attached to the body
                self.model.geom(geom_idx).rgba = rgba
    
def get_joi_body_name_of_common_rig_hand():
    """
        Get the body name of JOI
    """
    joi_body_name = {
        'torso':'torso',
        'spine':'spine',
        'rs':'right_shoulder',
        're':'right_elbow',
        'rw':'right_wrist',
        'ls':'left_shoulder',
        'le':'left_elbow',
        'lw':'left_wrist',
        'rp':'right_pelvis',
        'rk':'right_knee',
        'ra':'right_ankle',
        'lp':'left_pelvis',
        'lk':'left_knee',
        'la':'left_ankle',
    }
    return joi_body_name

def init_ik_info():
    """
        Initialize IK information
    """
    ik_info = {
        'body_names':[],
        'geom_names':[],
        'p_trgts':[],
        'R_trgts':[],
        'n_trgt':0,
    }
    return ik_info

def add_ik_info(
        ik_info,
        body_name=None,
        geom_name=None,
        p_trgt=None,
        R_trgt=None
    ):
    """ 
        Add IK information
    """
    ik_info['body_names'].append(body_name)
    ik_info['geom_names'].append(geom_name)
    ik_info['p_trgts'].append(p_trgt)
    ik_info['R_trgts'].append(R_trgt)
    ik_info['n_trgt'] = ik_info['n_trgt'] + 1

def get_dq_from_augmented_jacobian_method(
        env,ik_info,
        stepsize=1,eps=1e-2,th=np.radians(1.0),
        joint_idxs_jac=None,
    ):
    """
        Get delta q from augmented Jacobian method
    """
    J_list,ik_err_list = [],[]
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None
        J,ik_err = env.get_ik_ingredients(
            body_name=ik_body_name,
            geom_name=ik_geom_name,
            p_trgt=ik_p_trgt,
            R_trgt=ik_R_trgt,
            IK_P=IK_P,
            IK_R=IK_R
        )
        J_list.append(J)
        ik_err_list.append(ik_err)

    J_stack      = np.vstack(J_list)
    ik_err_stack = np.hstack(ik_err_list)

    # Select Jacobian columns that are within the joints to use
    if joint_idxs_jac is not None:
        J_stack_backup = J_stack.copy()
        J_stack = np.zeros_like(J_stack)
        J_stack[:,joint_idxs_jac] = J_stack_backup[:,joint_idxs_jac]

    # Compute dq from damped least square
    dq = env.damped_ls(J_stack,ik_err_stack,stepsize=stepsize,eps=eps,th=th)
    return dq,ik_err_stack

def get_dq_from_ik_info(
        env,
        ik_info,
        stepsize       = 1,
        eps            = 1e-2,
        th             = np.radians(1.0),
        joint_idxs_jac = None,
    ):
    """
        Get delta q from augmented Jacobian method
    """
    J_list,ik_err_list = [],[]
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None
        J,ik_err = env.get_ik_ingredients(
            body_name = ik_body_name,
            geom_name = ik_geom_name,
            p_trgt    = ik_p_trgt,
            R_trgt    = ik_R_trgt,
            IK_P      = IK_P,
            IK_R      = IK_R,
        )
        J_list.append(J)
        ik_err_list.append(ik_err)

    J_stack      = np.vstack(J_list)
    ik_err_stack = np.hstack(ik_err_list)

    # Select Jacobian columns that are within the joints to use
    if joint_idxs_jac is not None:
        J_stack_backup = J_stack.copy()
        J_stack = np.zeros_like(J_stack)
        J_stack[:,joint_idxs_jac] = J_stack_backup[:,joint_idxs_jac]

    # Compute dq from damped least square
    dq = env.damped_ls(J_stack,ik_err_stack,stepsize=stepsize,eps=eps,th=th)
    return dq,ik_err_stack

def plot_ik_info(
        env,ik_info,
        axis_len=0.05,axis_width=0.005,
        sphere_r=0.01
        ):
    """
        Plot IK information
    """
    colors = get_colors(cmap_name='gist_rainbow',n_color=ik_info['n_trgt'])
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        color = colors[ik_idx]
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None

        if ik_body_name is not None:
            # Plot current 
            env.plot_body_T(
                body_name=ik_body_name,
                PLOT_AXIS=IK_R,axis_len=axis_len,axis_width=axis_width,
                PLOT_SPHERE=IK_P,sphere_r=sphere_r,sphere_rgba=color,
                label='' # ''/ik_body_name
            )
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='') 
                env.plot_line_fr2to(p_fr=env.get_p_body(body_name=ik_body_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,PLOT_AXIS=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R:
                p_curr = env.get_p_body(body_name=ik_body_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,PLOT_AXIS=True,axis_len=axis_len,axis_width=axis_width)
            
        if ik_geom_name is not None:
            # Plot current 
            env.plot_geom_T(
                geom_name=ik_geom_name,
                PLOT_AXIS=IK_R,axis_len=axis_len,axis_width=axis_width,
                PLOT_SPHERE=IK_P,sphere_r=sphere_r,sphere_rgba=color,
                label='' # ''/ik_geom_name
            )
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='') 
                env.plot_line_fr2to(p_fr=env.get_p_geom(geom_name=ik_geom_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,PLOT_AXIS=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R:
                p_curr = env.get_p_geom(geom_name=ik_geom_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,PLOT_AXIS=True,axis_len=axis_len,axis_width=axis_width)
    
def get_T_joi_from_common_rig_hand(env):
    """ 
        Get joints of interest of common-rig-hand model
    """
    
    p_rs,R_rs = env.get_pR_body(body_name='right_shoulder')
    p_re,R_re = env.get_pR_body(body_name='right_elbow')
    p_rw,R_rw = env.get_pR_body(body_name='right_wrist')
    
    p_ls,R_ls = env.get_pR_body(body_name='left_shoulder')
    p_le,R_le = env.get_pR_body(body_name='left_elbow')
    p_lw,R_lw = env.get_pR_body(body_name='left_wrist')
    
    p_rc,_ = env.get_pR_body(body_name='right_clavicle')
    p_lc,_ = env.get_pR_body(body_name='left_clavicle')

    # p_neck,R_neck = env.get_pR_body(body_name='neck') # location of the neck body is problematic
    # p_neck = 0.5 * (p_rs+p_ls)
    p_neck = 0.5 * (p_rc+p_lc)
    
    p_rp,R_rp = env.get_pR_body(body_name='right_pelvis')
    p_rk,R_rk = env.get_pR_body(body_name='right_knee')
    p_ra,R_ra = env.get_pR_body(body_name='right_ankle')
    
    p_lp,R_lp = env.get_pR_body(body_name='left_pelvis')
    p_lk,R_lk = env.get_pR_body(body_name='left_knee')
    p_la,R_la = env.get_pR_body(body_name='left_ankle')
    
    # Right hand
    p_r_thumb_cmc = env.get_p_body(body_name='rthumb_l1')
    p_r_thumb_mcp = env.get_p_body(body_name='rthumb_l2')
    p_r_thumb_dip = env.get_p_body(body_name='rthumb_l3')
    p_r_thumb_end = env.get_p_body(body_name='rthumb_end')

    p_r_index_cmc = env.get_p_body(body_name='rindex_l0')
    p_r_index_mcp = env.get_p_body(body_name='rindex_l1')
    p_r_index_pip = env.get_p_body(body_name='rindex_l2')
    p_r_index_dip = env.get_p_body(body_name='rindex_l3')
    p_r_index_end = env.get_p_body(body_name='rindex_end')

    p_r_middle_cmc = env.get_p_body(body_name='rmiddle_l0')
    p_r_middle_mcp = env.get_p_body(body_name='rmiddle_l1')
    p_r_middle_pip = env.get_p_body(body_name='rmiddle_l2')
    p_r_middle_dip = env.get_p_body(body_name='rmiddle_l3')
    p_r_middle_end = env.get_p_body(body_name='rmiddle_end')

    p_r_ring_cmc = env.get_p_body(body_name='rring_l0')
    p_r_ring_mcp = env.get_p_body(body_name='rring_l1')
    p_r_ring_pip = env.get_p_body(body_name='rring_l2')
    p_r_ring_dip = env.get_p_body(body_name='rring_l3')
    p_r_ring_end = env.get_p_body(body_name='rring_end')

    p_r_pinky_cmc = env.get_p_body(body_name='rpinky_l0')
    p_r_pinky_mcp = env.get_p_body(body_name='rpinky_l1')
    p_r_pinky_pip = env.get_p_body(body_name='rpinky_l2')
    p_r_pinky_dip = env.get_p_body(body_name='rpinky_l3')
    p_r_pinky_end = env.get_p_body(body_name='rpinky_end')

    # Left hand
    p_l_thumb_cmc = env.get_p_body(body_name='lthumb_l1')
    p_l_thumb_mcp = env.get_p_body(body_name='lthumb_l2')
    p_l_thumb_dip = env.get_p_body(body_name='lthumb_l3')
    p_l_thumb_end = env.get_p_body(body_name='lthumb_end')

    p_l_index_cmc = env.get_p_body(body_name='lindex_l0')
    p_l_index_mcp = env.get_p_body(body_name='lindex_l1')
    p_l_index_pip = env.get_p_body(body_name='lindex_l2')
    p_l_index_dip = env.get_p_body(body_name='lindex_l3')
    p_l_index_end = env.get_p_body(body_name='lindex_end')

    p_l_middle_cmc = env.get_p_body(body_name='lmiddle_l0')
    p_l_middle_mcp = env.get_p_body(body_name='lmiddle_l1')
    p_l_middle_pip = env.get_p_body(body_name='lmiddle_l2')
    p_l_middle_dip = env.get_p_body(body_name='lmiddle_l3')
    p_l_middle_end = env.get_p_body(body_name='lmiddle_end')

    p_l_ring_cmc = env.get_p_body(body_name='lring_l0')
    p_l_ring_mcp = env.get_p_body(body_name='lring_l1')
    p_l_ring_pip = env.get_p_body(body_name='lring_l2')
    p_l_ring_dip = env.get_p_body(body_name='lring_l3')
    p_l_ring_end = env.get_p_body(body_name='lring_end')

    p_l_pinky_cmc = env.get_p_body(body_name='lpinky_l0')
    p_l_pinky_mcp = env.get_p_body(body_name='lpinky_l1')
    p_l_pinky_pip = env.get_p_body(body_name='lpinky_l2')
    p_l_pinky_dip = env.get_p_body(body_name='lpinky_l3')
    p_l_pinky_end = env.get_p_body(body_name='lpinky_end')

    T_joi = {
        'hip': pr2t(env.get_p_body(body_name='base'),env.get_R_body(body_name='base')),
        'spine': pr2t(env.get_p_body(body_name='spine'),env.get_R_body(body_name='spine')),
        'neck': pr2t(p_neck,np.eye(3,3)),
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
        'r_thumb_cmc':pr2t(p_r_thumb_cmc,np.eye(3,3)),
        'r_thumb_mcp':pr2t(p_r_thumb_mcp,np.eye(3,3)),
        'r_thumb_dip':pr2t(p_r_thumb_dip,np.eye(3,3)),
        'r_thumb_end':pr2t(p_r_thumb_end,np.eye(3,3)),
        'r_index_cmc':pr2t(p_r_index_cmc,np.eye(3,3)),
        'r_index_mcp':pr2t(p_r_index_mcp,np.eye(3,3)),
        'r_index_pip':pr2t(p_r_index_pip,np.eye(3,3)),
        'r_index_dip':pr2t(p_r_index_dip,np.eye(3,3)),
        'r_index_end':pr2t(p_r_index_end,np.eye(3,3)),
        'r_middle_cmc':pr2t(p_r_middle_cmc,np.eye(3,3)),
        'r_middle_mcp':pr2t(p_r_middle_mcp,np.eye(3,3)),
        'r_middle_pip':pr2t(p_r_middle_pip,np.eye(3,3)),
        'r_middle_dip':pr2t(p_r_middle_dip,np.eye(3,3)),
        'r_middle_end':pr2t(p_r_middle_end,np.eye(3,3)),
        'r_ring_cmc':pr2t(p_r_ring_cmc,np.eye(3,3)),
        'r_ring_mcp':pr2t(p_r_ring_mcp,np.eye(3,3)),
        'r_ring_pip':pr2t(p_r_ring_pip,np.eye(3,3)),
        'r_ring_dip':pr2t(p_r_ring_dip,np.eye(3,3)),
        'r_ring_end':pr2t(p_r_ring_end,np.eye(3,3)),
        'r_pinky_cmc':pr2t(p_r_pinky_cmc,np.eye(3,3)),
        'r_pinky_mcp':pr2t(p_r_pinky_mcp,np.eye(3,3)),
        'r_pinky_pip':pr2t(p_r_pinky_pip,np.eye(3,3)),
        'r_pinky_dip':pr2t(p_r_pinky_dip,np.eye(3,3)),
        'r_pinky_end':pr2t(p_r_pinky_end,np.eye(3,3)),
        'l_thumb_cmc':pr2t(p_l_thumb_cmc,np.eye(3,3)),
        'l_thumb_mcp':pr2t(p_l_thumb_mcp,np.eye(3,3)),
        'l_thumb_dip':pr2t(p_l_thumb_dip,np.eye(3,3)),
        'l_thumb_end':pr2t(p_l_thumb_end,np.eye(3,3)),
        'l_index_cmc':pr2t(p_l_index_cmc,np.eye(3,3)),
        'l_index_mcp':pr2t(p_l_index_mcp,np.eye(3,3)),
        'l_index_pip':pr2t(p_l_index_pip,np.eye(3,3)),
        'l_index_dip':pr2t(p_l_index_dip,np.eye(3,3)),
        'l_index_end':pr2t(p_l_index_end,np.eye(3,3)),
        'l_middle_cmc':pr2t(p_l_middle_cmc,np.eye(3,3)),
        'l_middle_mcp':pr2t(p_l_middle_mcp,np.eye(3,3)),
        'l_middle_pip':pr2t(p_l_middle_pip,np.eye(3,3)),
        'l_middle_dip':pr2t(p_l_middle_dip,np.eye(3,3)),
        'l_middle_end':pr2t(p_l_middle_end,np.eye(3,3)),
        'l_ring_cmc':pr2t(p_l_ring_cmc,np.eye(3,3)),
        'l_ring_mcp':pr2t(p_l_ring_mcp,np.eye(3,3)),
        'l_ring_pip':pr2t(p_l_ring_pip,np.eye(3,3)),
        'l_ring_dip':pr2t(p_l_ring_dip,np.eye(3,3)),
        'l_ring_end':pr2t(p_l_ring_end,np.eye(3,3)),
        'l_pinky_cmc':pr2t(p_l_pinky_cmc,np.eye(3,3)),
        'l_pinky_mcp':pr2t(p_l_pinky_mcp,np.eye(3,3)),
        'l_pinky_pip':pr2t(p_l_pinky_pip,np.eye(3,3)),
        'l_pinky_dip':pr2t(p_l_pinky_dip,np.eye(3,3)),
        'l_pinky_end':pr2t(p_l_pinky_end,np.eye(3,3)),
    }
    return T_joi