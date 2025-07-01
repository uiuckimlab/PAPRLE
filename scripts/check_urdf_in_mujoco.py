from paprle.utils.config_utils import change_working_directory
change_working_directory()

import argparse
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass
from paprle.envs.mujoco_env_utils.util import MultiSliderClass,get_idxs
parser = argparse.ArgumentParser()
parser.add_argument("xml", type=str, help="path to xml file.")
args = parser.parse_args()

env = MuJoCoParserClass("Viewer", rel_xml_path=args.xml, VERBOSE=False)
env.init_viewer(
            viewer_title='Viewer', viewer_width=1200, viewer_height=800,
            viewer_hide_menus=True,
        )

joint_names_to_slide = env.rev_joint_names
idxs_fwd         = env.get_idxs_fwd(joint_names=joint_names_to_slide) # idxs for qpos
idxs_range       = get_idxs(env.joint_names,joint_names_to_slide) # idxs for joint_ranges
q_init           = env.get_qpos_joints(joint_names_to_slide)
n_joint_to_slide = len(joint_names_to_slide)
sliders = MultiSliderClass(
    n_slider      = n_joint_to_slide,
    title         = 'Sliders for Joint Control',
    window_width  = 600,
    window_height = 800,
    x_offset      = 50,
    y_offset      = 100,
    slider_width  = 350,
    label_texts   = joint_names_to_slide,
    slider_mins   = env.joint_ranges[idxs_range,0],
    slider_maxs   = env.joint_ranges[idxs_range,1],
    slider_vals   = q_init,
    resolution    = 0.01,
    VERBOSE       = False,
)
env.reset()

visualize_links = ['.*end_effector.*']

import re
poi_links = []
for link_name in env.body_names:
    if any([re.match(visualize_link, link_name) for visualize_link in visualize_links]):
        poi_links.append(link_name)
while env.is_viewer_alive() and sliders.is_window_exists():
    sliders.update() # update slider
    qpos_curr = env.data.qpos.copy()
    qpos_curr[idxs_fwd] = sliders.get_slider_values()
    env.forward(qpos_curr)
    sliders.set_slider_values(slider_values=env.data.qpos[idxs_fwd])
    #env.plot_joint_axis(axis_len=0.02, axis_r=0.004, joint_names=joint_names_to_slide)  # joint axis
    env.plot_contact_info(h_arrow=0.3, rgba_arrow=[1, 0, 0, 1], PRINT_CONTACT_BODY=True)  # contact

    for link_name in poi_links:
       p, R = env.get_pR_body(link_name)
       env.plot_T(p, R, PLOT_AXIS=True, PLOT_SPHERE=True, sphere_r=0.02, axis_len=0.12, axis_width=0.005)
    env.render()