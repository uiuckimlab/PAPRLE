import os
import pyrealsense2 as rs
import numpy as np
from threading import Thread
import copy
from multiprocessing import RawArray, Value, Process
import cv2
import time

def stream_camera(cam_info, color_img, depth_img, time_stamp, depth_units, shutdown=False):

    cam = rs.pipeline()
    config = rs.config()
    config.enable_device(str(cam_info.serial_number))
    width, height = cam_info.depth_resolution
    fps = getattr(cam_info, 'fps', 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    width, height = cam_info.rgb_resolution
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    profile = cam.start(config)

    while not shutdown.value:
        try:
            frames = cam.wait_for_frames()
        except:
            return
        if cam_info.get_aligned:
            align = rs.align(rs.stream.color)
            frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_img[:] = np.array(color_frame.get_data()).flatten()
        depth_img[:] = np.array(depth_frame.get_data()).flatten()
        time_stamp.value = time.time()
        depth_units.value = depth_frame.get_units()
    return


class RealSenseReader:
    def __init__(self, camera_config, render_depth=False):
        self.cameras, self.processes = {}, []
        self.images = {}
        self.shutdown = Value('b', False)
        self.cam_infos = {}
        fps = 30
        for idx, (name, cam_info) in enumerate(camera_config.items()):
            try:
                self.cam_infos[name] = cam_info
                dr = cam_info.depth_resolution if not cam_info.get_aligned else cam_info.rgb_resolution
                self.images[name] = {
                    'color': RawArray('d', np.zeros([cam_info.rgb_resolution[1], cam_info.rgb_resolution[0], 3], dtype=np.uint8).flatten()),
                    'depth': RawArray('d', np.zeros([dr[1], dr[0]], dtype=np.uint16).flatten()),
                    'timestamp': Value('d', 0.0),
                    'depth_units': Value('d', 0.001)
                }
                p = Process(target=stream_camera, args=(cam_info, self.images[name]['color'], self.images[name]['depth'],
                                                        self.images[name]['timestamp'], self.images[name]['depth_units'], self.shutdown))
                p.start()
                self.processes.append(p)
                iterations = 0
                while self.images[name] == {}:
                    iterations += 1
                    if iterations % 100 == 0:
                        print("Waiting for camera to initialize: ", name, cam_info.serial_number)
                    time.sleep(0.01)
                    pass
                print("Initialized camera: ", name, cam_info.serial_number)
            except:
                print("Failed to initialize camera: ", name, cam_info.serial_number)
        self.render_depth = render_depth

    def get_status(self):
        out_dict = {}
        for name in self.images.keys():
            out_dict[name] = {}
            w, h = self.cam_infos[name].rgb_resolution
            out_dict[name]['color'] = np.array(self.images[name]['color']).reshape(h,w, 3).astype(np.uint8)[...,[2,1,0]]
            dw, dh = self.cam_infos[name].depth_resolution if not self.cam_infos[name].get_aligned else self.cam_infos[
                name].rgb_resolution
            out_dict[name]['depth'] = np.array(self.images[name]['depth']).reshape(dh, dw).astype(np.uint16)
            out_dict[name]['timestamp'] = self.images[name]['timestamp'].value
            out_dict[name]['depth_units'] = self.images[name]['depth_units'].value
        return out_dict

    def render(self):
        view_ims = []
        H = None
        for name in self.images.keys():
            w, h = self.cam_infos[name].rgb_resolution
            im = np.ascontiguousarray(np.array(self.images[name]['color']).reshape(h,w, 3).astype(np.uint8))
            im = cv2.putText(im, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if self.render_depth:
                dw, dh = self.cam_infos[name].depth_resolution if not self.cam_infos[name].get_aligned else self.cam_infos[name].rgb_resolution
                depth = np.ascontiguousarray(np.array(self.images[name]['depth']).reshape(dh, dw).astype(np.uint16))
                depth = np.clip(depth * self.images[name]['depth_units'].value, 0, 3.0)/3.0 * 255
                depth = cv2.applyColorMap(depth.astype(np.uint8),  cv2.COLORMAP_JET)
                depth = cv2.resize(depth, (im.shape[1], im.shape[0]))
                im = np.concatenate([im, depth], 0)
            if H is None:
                H = im.shape[0]
            else:
                im = cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
            view_ims.append(im)
        view_im = np.concatenate(view_ims, 1)
        view_im = cv2.resize(view_im, dsize=None, fx=0.6, fy=0.6)
        return view_im

    def close(self):
        self.shutdown.value = True
        for p in self.processes:
            p.join()
        self.processes = []
        self.images = {}
        self.cam_infos = {}
        print("Closed RealSenseReader and released resources.")

if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.follower import Robot
    import time

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)

    reader = RealSenseReader(robot.camera_config, render_depth=True)

    while True:
        im = reader.render()
        cv2.imshow('Camera Feed', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.close()