import sys
import os
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QTextEdit, QGridLayout, QStatusBar, QLineEdit,
    QStackedWidget, QAbstractItemView, QMessageBox, QTreeWidget, QTreeWidgetItem, QFormLayout,
    QComboBox, QSlider, QSizePolicy, QDoubleSpinBox
)
from PySide6.QtCore import Qt
import pyqtgraph.opengl as gl
from yourdfpy import URDF
import numpy as np
import time
from omegaconf import OmegaConf
from trimesh.creation import axis as create_axis
from paprle.utils.config_utils import change_working_directory
change_working_directory()

from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader

ShaderProgram('custom_shaded', [
    VertexShader("""
        varying vec3 normal;
        void main() {
            // compute here for use in fragment shader
            normal = normalize(gl_NormalMatrix * gl_Normal);
            gl_FrontColor = gl_Color;
            gl_BackColor = gl_Color;
            gl_Position = ftransform();
        }
    """),
    FragmentShader("""
        varying vec3 normal;
        void main() {
            float p = dot(normal, normalize(vec3(1.0, 1.0, 1.0)));
            p = p < 0. ? 0. : p * 0.8;
            vec4 color = gl_Color;
            color.x = color.x * (0.2 + p);
            color.y = color.y * (0.2 + p);
            color.z = color.z * (0.2 + p);
            gl_FragColor = color;
        }
    """)
])


class PageWidget(QWidget):
    def set_show_event(self, func):
        parent_show_event = super().showEvent
        def wrapper(event):
            parent_show_event(event)
            func()
        self.showEvent = wrapper

    def set_hide_event(self, func):
        parent_hide_event = super().hideEvent
        def wrapper(event):
            parent_hide_event(event)
            func()
        self.hideEvent = wrapper


class URDFConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("[PAPRLE] New Robot Config Setup")
        self.setGeometry(100, 100, 1500, 600)

        main_layout = QGridLayout(self)

        sidebar = QListWidget()
        sidebar.addItems(["1. Load URDF",
                          "2. Define Limbs",
                          "3. Define Hands",
                          "4. Select End Effector Link",
                          "5. Configure Init Pose",
                          "6. Configure IK",
                          # "7. (Optional) ROS1 Config",
                          # "8. (Optional) ROS2 Config",
                          "7. Export Config"])
        sidebar.setFixedWidth(150)

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)


        self.setup_page1() # Load URDF
        self.URDF_LOADED = False
        self.setup_page2() # Define Limbs
        self.setup_page3() # Define Hands
        self.setup_page4() # Select End Effector Link
        self.setup_page5() # Configure Init Pose
        self.setup_page6() # Configure IK
        self.setup_page7() # Save Config

        for i in range(self.stack.count()):
            if i < 1: # Page 1 is always visible
                self.stack.widget(i).setEnabled(True)
            else:
                self.stack.widget(i).setEnabled(False)

        # --- Page 7: (Optional) ROS1 Config ---

        # --- Page 8 : (Optional) ROS2 Config ---

        # --- Page 9 : Export Config ---


        sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        main_layout.addWidget(sidebar, 0, 0)
        main_layout.addWidget(self.stack, 0, 1)

        # Rendering View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=2)
        self.gl_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gl_view.setBackgroundColor(1.0)
        self.gl_grid = gl.GLGridItem()
        self.gl_grid.setSize(4, 4)
        self.gl_grid.setSpacing(0.1, 0.1)
        self.gl_grid.setColor(0.3)
        self.gl_view.addItem(self.gl_grid)
        self.gl_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gl_view.setBaseSize(300, 600)
        main_layout.addWidget(self.gl_view, 0, 2, 2, 1)

        # # Config editor
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QTextEdit.NoWrap)
        main_layout.addWidget(QLabel("Edit Config"))
        main_layout.addWidget(self.log_box, 1, 0, 1, 2)

    def change_page(self, index):
        self.stack.setCurrentIndex(index)
    def enable_pages(self):
        for i in range(self.stack.count()):
            self.stack.widget(i).setEnabled(True)
        return

    def print_log(self, message):
        message = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
        self.log_box.append(message)
        self.log_box.ensureCursorVisible()
        return
    ############## Page 1: Load URDF ##############
    def setup_page1(self):
        page1 = PageWidget()
        layout1 = QVBoxLayout()

        label = QLabel("Step 1: Load Robot")
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout1.addWidget(label)

        label = QLabel("Load the URDF file of the robot. If you already have a config file and want to edit it, you can load the config file directly.")
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout1.addWidget(label)

        button_layout = QHBoxLayout()

        self.urdf_load_btn = QPushButton("Load URDF")
        self.urdf_load_btn.clicked.connect(self.load_urdf)
        button_layout.addWidget(self.urdf_load_btn)

        self.config_load_btn = QPushButton("Load Config")
        self.config_load_btn.clicked.connect(self.load_config)
        button_layout.addWidget(self.config_load_btn)
        self.config = None # default config is None

        layout1.addLayout(button_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Enter Robot Name:"))
        self.robot_name_input = QLineEdit()
        name_layout.addWidget(self.robot_name_input)
        layout1.addLayout(name_layout)

        page1.setLayout(layout1)
        self.stack.addWidget(page1)
        return

    def load_urdf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open URDF File", "", "URDF Files (*.urdf)")
        self.urdf_file_name = file_name
        self.parse_urdf(file_name)
        self.URDF_LOADED = True
        self.enable_pages()


    def load_config(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Config File", "", "YAML Files (*.yaml)")
        try:
            self.config = OmegaConf.load(file_name)
            self.robot_name_input.setText(self.config.robot_cfg.name)
            self.urdf_file_name = self.config.robot_cfg.asset_cfg.urdf_path
            self.parse_urdf(self.urdf_file_name)

            self.group_dict = {}
            for limb_name, limb_joint_list in self.config.robot_cfg.limb_joint_names.items():
                self.group_dict[limb_name] = limb_joint_list

            self.hand_group_dict = {}
            for hand_name, hand_joint_list in self.config.robot_cfg.hand_joint_names.items():
                self.hand_group_dict[hand_name] = hand_joint_list

            self.end_effector_dict = dict(self.config.robot_cfg.end_effector_link)

            init_qpos = self.config.robot_cfg.init_qpos
            for idx in range(self.joint_list.count()):
                joint_name = self.joint_list.item(idx).text()
                self.joint_values[joint_name] = init_qpos[idx]

            ## ik_info
            self.ik_dict, self.ik_info = {}, {}
            for group_name in self.group_dict:
                self.ik_info[group_name] = dict(self.config.ik_cfg.get(group_name, {}))
            self.URDF_LOADED = True
            self.enable_pages()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
            self.config = None
            self.robot_name_input.clear()
            self.joint_list.clear()
            self.link_list = []
            self.group_dict = {}
            self.hand_group_dict = {}
            self.end_effector_dict = {}
            self.ik_dict, self.ik_info = {}, {}
            self.log_box.clear()
        return

    def parse_urdf(self, file_name):
        if not file_name:
            return
        try:
            tree = ET.parse(file_name)
            root = tree.getroot()
            if root.attrib.get('name'):
                self.robot_name_input.setText(root.attrib['name'])
            joints = root.findall("joint")
            links = root.findall("link")
            self.joint_list.clear()
            self.joint_dict = {}
            for joint in joints:
                name = joint.attrib.get("name", "")
                if joint.attrib.get('type', "") == 'fixed': continue
                if joint.find("mimic") is not None: continue
                self.joint_list.addItem(name)
                # get limits
                self.joint_dict[name] = {}
                limit = joint.find("limit")
                if limit is not None:
                    self.joint_dict[name]['lower'] = float(limit.attrib.get("lower", -2*np.pi))
                    self.joint_dict[name]['upper'] = float(limit.attrib.get("upper", 2*np.pi))
                else:
                    self.joint_dict[name]['lower'] = -2 * np.pi
                    self.joint_dict[name]['upper'] = 2 * np.pi

            self.link_list = [link.attrib.get("name", "") for link in links]
            self.print_log(f" Loaded {len(joints)} joints from {file_name}")

        except Exception as e:
            self.print_log(f"Failed to load URDF: {e}")


        # Load URDF + mesh
        self.robot = URDF.load(file_name)
        self.joint_values = {j.name: 0.0 for j in self.robot.actuated_joints}
        scene = self.robot.scene
        # Create mesh item
        self.gl_view.clear()
        self.gl_view.addItem(self.gl_grid)
        geom_idx = 0
        self.link_gl_items, self.link_gl_default_colors = {}, {}
        for link_name, link_info in self.robot.link_map.items():
            for v in link_info.visuals:
                if v.geometry is None:
                    continue
                mesh = scene.geometry[scene.graph.nodes_geometry[geom_idx]]
                mesh_data = gl.MeshData(vertexes=mesh.vertices, faces=mesh.faces,
                                        vertexColors=mesh.visual.vertex_colors, faceColors=mesh.visual.face_colors)
                item = gl.GLMeshItem(meshdata=mesh_data, smooth=False, shader='custom_shaded')
                self.link_gl_items[link_name] = item
                Rt = self.robot.scene.graph.get(link_name)[0]
                origin = v.origin if v.origin is not None else np.eye(4)
                item.setTransform(Rt@origin)
                self.gl_view.addItem(item)
                geom_idx += 1
        return

    def reset_colors(self):
        for i in range(self.joint_list.count()):
            joint_item = self.joint_list.item(i)
            joint_name = joint_item.text()
            if self.robot.joint_map[joint_name].parent in self.link_gl_default_colors:
                orig_color = self.link_gl_default_colors[self.robot.joint_map[joint_name].parent]
                self.link_gl_items[self.robot.joint_map[joint_name].parent].colors = orig_color
            if self.robot.joint_map[joint_name].child in self.link_gl_default_colors:
                orig_color = self.link_gl_default_colors[self.robot.joint_map[joint_name].child]
                self.link_gl_items[self.robot.joint_map[joint_name].child].colors = orig_color
            self.gl_view.update()
        return

    ################# Page 2: Define Limbs ##############
    def setup_page2(self):
        page2 = PageWidget()
        layout2 = QGridLayout()
        layout2.addWidget(QLabel("Step 2: Setup Limbs"), 0, 0, 1, 2)
        layout2.addWidget(QLabel("Set up the limbs in the URDF and assign joints to each limb (excluding gripper joints)."), 1, 0, 1, 2)

        self.limb_group_tree = QTreeWidget()
        self.limb_group_tree.setHeaderHidden(True)
        self.joint_list = QListWidget()
        self.joint_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.group_dict = {}
        add_btn = QPushButton("Add a New Group")
        add_btn.clicked.connect(self.add_group)
        remove_btn = QPushButton("Remove a Group")
        remove_btn.clicked.connect(self.remove_group)

        save_btn = QPushButton("Save Joint Assignment")
        save_btn.clicked.connect(self.save_assignment)
        self.limb_group_tree.itemClicked.connect(lambda item, column: self.highlight_group_joints(item, self.group_dict, self.joint_list))
        self.limb_group_tree.itemChanged.connect(self.rename_group)

        layout2.addWidget(QLabel("Limb Groups"), 2, 0)
        layout2.addWidget(self.limb_group_tree, 3, 0)
        layout2.addWidget(add_btn, 4, 0)
        layout2.addWidget(remove_btn, 5, 0)
        layout2.addWidget(QLabel("Assign Joints to Selected Group"), 2, 1)
        layout2.addWidget(self.joint_list, 3, 1)
        layout2.addWidget(save_btn, 4, 1)

        page2.set_show_event(self.load_page2)
        page2.set_hide_event(self.reset_colors)
        page2.setLayout(layout2)
        self.stack.addWidget(page2)
        return

    def load_page2(self):
        if not self.URDF_LOADED: return
        self.limb_group_tree.clear()
        for name, joints in self.group_dict.items():
            top_item = QTreeWidgetItem([name])
            self.limb_group_tree.addTopLevelItem(top_item)
            for joint in joints:
                child_item = QTreeWidgetItem([joint])
                top_item.addChild(child_item)
        self.print_log(f"Current limb groups: {list(self.group_dict.keys())}")
        return

    def add_group(self):
        default_name = 'New Group'
        i = 1
        name = default_name
        while name in self.group_dict:
            name = f"{default_name}_{i}"
            i += 1

        self.group_dict[name] = []
        self.hand_group_dict[name] = []

        top_item = QTreeWidgetItem([name])
        top_item.setFlags(top_item.flags() | Qt.ItemIsEditable)
        self.limb_group_tree.addTopLevelItem(top_item)
        self.limb_group_tree.setCurrentItem(top_item)
        self.limb_group_tree.editItem(top_item, 0)
        self.print_log("Added new group: " + name)

    def remove_group(self):
        current_item = self.limb_group_tree.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "Error", "No group selected to remove.")
            return
        group_name = current_item.text(0)
        del self.group_dict[group_name]
        del self.hand_group_dict[group_name]
        index = self.limb_group_tree.indexOfTopLevelItem(current_item)
        self.limb_group_tree.takeTopLevelItem(index)
        self.print_log(f"Removed group: {group_name}")

    def save_assignment(self):
        current_group = self.limb_group_tree.selectedItems()
        if not current_group:
            QMessageBox.warning(self, "Error", "Select a group to assign joints.")
            return
        current_group = current_group[0]
        group_name = current_group.text(0)
        selected_joints = [item.text() for item in self.joint_list.selectedItems()]
        self.group_dict[group_name] = selected_joints

        # Update tree view
        current_group.takeChildren()
        for joint in selected_joints:
            child = QTreeWidgetItem([joint])
            current_group.addChild(child)
        current_group.setExpanded(True)
        self.highlight_group_joints(current_group, self.group_dict, self.joint_list)
        self.print_log(f"Assigned joints {selected_joints} to group '{group_name}'.")


    def rename_group(self, item, column):
        old_names = list(self.group_dict.keys())
        new_name = item.text(0)

        # If it's a new unique name
        if new_name not in self.group_dict:
            # Find which group this item used to be
            for old in old_names:
                if old != new_name and self.group_dict[old] == self.group_dict.get(item.text(0), []):
                    self.group_dict[new_name] = self.group_dict.pop(old)
                    self.hand_group_dict[new_name] = self.hand_group_dict.pop(old)
                    break
        else:
            # Prevent name collision
            QMessageBox.warning(self, "Duplicate Name", f"Group '{new_name}' already exists.")
            return

    def highlight_group_joints(self, item, current_dict, joint_list):
        # If a joint is clicked, ignore
        if item.parent() is not None:
            return
        group_name = item.text(0)
        assigned = set(current_dict.get(group_name, []))

        for i in range(joint_list.count()):
            joint_item = joint_list.item(i)
            joint_name = joint_item.text()

            parent_joint = self.robot.joint_map[joint_name].parent
            if parent_joint not in self.link_gl_default_colors:
                self.link_gl_default_colors[parent_joint] = self.link_gl_items[parent_joint].colors.copy()
            child_joint = self.robot.joint_map[joint_name].child
            if child_joint not in self.link_gl_default_colors:
                self.link_gl_default_colors[child_joint] = self.link_gl_items[child_joint].colors.copy()
            if joint_name in assigned:
                joint_item.setSelected(True)
                orig_color = self.link_gl_default_colors[parent_joint].copy()
                orig_color[...,0] = 255
                orig_color[...,1] = 0
                orig_color[...,2] = 0
                self.link_gl_items[self.robot.joint_map[joint_name].parent].colors = orig_color
                orig_color = self.link_gl_default_colors[child_joint].copy()
                orig_color[:,:,0] = 255
                orig_color[:,:,1] = 0
                orig_color[:,:,2] = 0
                self.link_gl_items[self.robot.joint_map[joint_name].child].colors = orig_color
            else:
                joint_item.setSelected(False)
                orig_color = self.link_gl_default_colors[parent_joint].copy()
                self.link_gl_items[self.robot.joint_map[joint_name].parent].colors = orig_color
                orig_color = self.link_gl_default_colors[child_joint].copy()
                self.link_gl_items[self.robot.joint_map[joint_name].child].colors = orig_color
        self.gl_view.update()
    ################ Page 3: Define Hands ##############
    def setup_page3(self):
        # --- Page 3: Define hand Joints ---
        self.hand_group_dict = {}

        page3 = PageWidget()
        layout3 = QGridLayout()
        self.hand_group_tree = QTreeWidget()
        self.hand_group_tree.setHeaderHidden(True)
        self.hand_group_tree.itemClicked.connect(lambda item, column: self.highlight_group_joints(item, self.hand_group_dict, self.hand_joint_list))

        layout3.addWidget(QLabel("Step 3: Setup Hands"), 0, 0, 1, 2)
        layout3.addWidget(QLabel("Assign gripper joints to each limb, if it does not have hands, please skip"), 1, 0, 1, 2)
        layout3.addWidget(QLabel("Hand Groups"), 2, 0)
        layout3.addWidget(self.hand_group_tree, 3, 0)
        self.hand_joint_list = QListWidget()
        self.hand_joint_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout3.addWidget(QLabel("Hand Joints"), 2, 1)
        layout3.addWidget(self.hand_joint_list, 3, 1)

        save_btn = QPushButton("Save Joint Assignment")
        save_btn.clicked.connect(self.save_hand_assignment)
        layout3.addWidget(save_btn, 4, 1)

        page3.setLayout(layout3)
        page3.set_show_event(self.load_page3)
        page3.set_hide_event(self.reset_colors)
        self.stack.addWidget(page3)
        return

    def save_hand_assignment(self):
        current_group = self.hand_group_tree.selectedItems()
        if not current_group:
            QMessageBox.warning(self, "Error", "Select a group to assign joints.")
            return
        current_group = current_group[0]
        group_name = current_group.text(0)
        selected_joints = [item.text() for item in self.hand_joint_list.selectedItems()]
        self.hand_group_dict[group_name] = selected_joints

        if len(selected_joints) == 0:
            QMessageBox.warning(self, "Error", "No joints selected for this group.")
            return
        # Update tree view
        current_group.takeChildren()
        for joint in selected_joints:
            child = QTreeWidgetItem([joint])
            current_group.addChild(child)
        current_group.setExpanded(True)
        self.highlight_group_joints(current_group, self.hand_group_dict, self.hand_joint_list)

    def load_page3(self):
        # load_hand_group_from_limb_group
        if not self.URDF_LOADED: return
        self.hand_group_tree.clear()
        for name in self.group_dict:
            top_item = QTreeWidgetItem([name])
            self.hand_group_tree.addTopLevelItem(top_item)

            if self.hand_group_dict[name]:
                for joint in self.hand_group_dict[name]:
                    child_item = QTreeWidgetItem([joint])
                    top_item.addChild(child_item)

        self.hand_joint_list.clear()
        # copy self.joint_list items to hand_joint_list
        for i in range(self.joint_list.count()):
            joint_name = self.joint_list.item(i).text()
            self.hand_joint_list.addItem(joint_name)

        self.print_log("Current hand joints ")
        for name, joints in self.hand_group_dict.items():
            self.print_log(f"    {name}: {joints}")

    ################ Page 4: Select End Effector Link ##############
    def setup_page4(self):
        page4 = PageWidget()
        layout4 = QVBoxLayout()
        layout4.addWidget(QLabel("Step 4: Select End Effector Link"))
        self.eef_form_layout = QFormLayout()
        self.link_list = []
        self.end_effector_dict = {}
        layout4.addLayout(self.eef_form_layout)
        page4.setLayout(layout4)
        page4.set_show_event(self.load_page4)
        self.stack.addWidget(page4)
        return

    def load_page4(self):
        if not self.URDF_LOADED: return
        default_dict = {}
        for group_name in self.group_dict:
            default_value = ''
            if group_name in self.end_effector_dict:
                if isinstance(self.end_effector_dict[group_name], str) and self.end_effector_dict[group_name] in self.link_list:
                    default_value = self.end_effector_dict[group_name]
                elif isinstance(self.end_effector_dict[group_name], QComboBox):
                    default_value = self.end_effector_dict[group_name].currentText()
            default_dict[group_name] = default_value


        while self.eef_form_layout.rowCount() > 0:
            self.eef_form_layout.removeRow(0)
        for group_name in self.group_dict:
            combo = QComboBox()
            combo.addItems(self.link_list)
            combo.currentTextChanged.connect(
                lambda value, group=group_name: self.on_eef_change(value, group_name=group)
            )
            self.eef_form_layout.addRow(group_name, combo)
            default_value = default_dict[group_name]
            if default_value:
                combo.setCurrentText(default_value)
            self.end_effector_dict[group_name] = combo

        self.print_log("Current end effector links:")
        for group_name, link in self.end_effector_dict.items():
            link_name = link.currentText() if isinstance(link, QComboBox) else link
            self.print_log(f"    {group_name}: {link_name}")


    def on_eef_change(self, value, group_name=''):
        if f'eef_{group_name}' not in self.link_gl_items:
            axis = create_axis(origin_size=0.01)
            self.robot._scene.add_geometry(axis, f'eef_{group_name}', geom_name=f'eef_{group_name}', parent_node_name=value)
            mesh = self.robot._scene.geometry[f'eef_{group_name}']
            mesh_data = gl.MeshData(vertexes=mesh.vertices, faces=mesh.faces,
                                    vertexColors=mesh.visual.vertex_colors, faceColors=mesh.visual.face_colors)
            item = gl.GLMeshItem(meshdata=mesh_data, smooth=False, shader='custom_shaded')
            self.link_gl_items[f'eef_{group_name}'] = item
            self.gl_view.addItem(item)
        item = self.link_gl_items[f'eef_{group_name}']
        Rt = self.robot.scene.graph.get(value)[0]
        item.setTransform(Rt)
        self.print_log("Set end effector link for group '{}': {}".format(group_name, value))
        return

    ################ Page 5: Configure Init Pose ################
    def setup_page5(self):
        # --- Page 5: Configure Init Pose ---
        page5 = PageWidget()
        self.slider_layout = QVBoxLayout()
        self.slider_layout.addWidget(QLabel("Step 5: Configure Init Pose"))
        self.slider_layout.addWidget(QLabel("Set the initial pose of the robot."))

        page5.setLayout(self.slider_layout)
        page5.set_show_event(self.load_sliders)
        self.stack.addWidget(page5)
        return

    def clear_layout(self, layout, offset=0):
        while layout.count() > offset:
            item = layout.takeAt(offset)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout(), offset=0)
                item.layout().deleteLater()

    def load_sliders(self):
        if not self.URDF_LOADED: return
        self.clear_layout(self.slider_layout, offset=2)
        for group_name, joint_list in self.group_dict.items():
            for joint_name in joint_list:
                self.add_joint_slider(joint_name)
            if self.hand_group_dict[group_name]:
                for joint_name in self.hand_group_dict[group_name]:
                    self.add_joint_slider(joint_name)
        self.update_fk()

    def add_joint_slider(self, joint_name):
        hbox = QHBoxLayout()
        label = QLabel(joint_name)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(self.joint_dict[joint_name]['lower'] * 100)
        slider.setMaximum(self.joint_dict[joint_name]['upper'] * 100)
        slider.setValue(0)
        slider.setSingleStep(1)
        slider.valueChanged.connect(
            lambda val, name=joint_name: self.on_slider_change(name, val / 100.0)
        )

        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(self.joint_dict[joint_name]['lower'])
        spinbox.setMaximum(self.joint_dict[joint_name]['upper'])
        spinbox.setSingleStep(0.01)
        spinbox.setDecimals(2)
        spinbox.setValue(self.joint_values[joint_name])
        spinbox.setFixedWidth(70)
        slider.valueChanged.connect(lambda val, sb=spinbox: sb.setValue(val / 100.0))
        spinbox.valueChanged.connect(lambda val, sl=slider: sl.setValue(int(val * 100)))
        spinbox.valueChanged.connect(
            lambda val, name=joint_name: self.on_slider_change(name, val)
        )
        hbox.addWidget(label)
        hbox.addWidget(slider)
        hbox.addWidget(spinbox)
        self.slider_layout.addLayout(hbox)

    def on_slider_change(self, joint_name, value):
        self.joint_values[joint_name] = value
        self.update_fk()

    def update_fk(self):
        self.robot.update_cfg(self.joint_values)
        geom_idx = 0
        for link_name, link_info in self.robot.link_map.items():
            for v in link_info.visuals:
                if v.geometry is None:
                    continue
                item = self.link_gl_items[link_name]
                Rt = self.robot.scene.graph.get(link_name)[0]
                origin = v.origin if v.origin is not None else np.eye(4)
                item.resetTransform()
                item.setTransform(Rt@origin)
                geom_idx += 1
        if self.end_effector_dict:
            for group_name, link in self.end_effector_dict.items():
                if isinstance(link, QComboBox):
                    link_name = link.currentText()
                else:
                    link_name = link
                if f'eef_{group_name}' in self.link_gl_items:
                    item = self.link_gl_items[f'eef_{group_name}']
                    Rt = self.robot.scene.graph.get(link_name)[0]
                    item.setTransform(Rt)
    def setup_page6(self):
        # --- Page 6: Configure IK ---
        page6 = PageWidget()
        layout6 = QVBoxLayout()
        label = QLabel("Step 6: Configure Inverse Kinematics (IK)")
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout6.addWidget(label)

        label = QLabel("Set up the IK configuration for each limb group. Load separate URDF files for IK if needed, or use the same URDF as the robot.")
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout6.addWidget(label)

        load_layout = QHBoxLayout()
        self.ik_group_tree = QTreeWidget()
        self.ik_group_tree.setHeaderHidden(True)
        self.ik_group_tree.itemClicked.connect(self.load_ik_config)
        load_layout.addWidget(self.ik_group_tree)

        self.ik_form = QFormLayout()
        self.ik_urdf_path_line = QLineEdit()
        self.ik_dict, self.ik_info = {}, {}
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.load_urdf_for_ik)
        urdf_hbox = QHBoxLayout()
        urdf_hbox.addWidget(self.ik_urdf_path_line)
        urdf_hbox.addWidget(browse_btn)
        urdf_widget = QWidget()
        urdf_widget.setLayout(urdf_hbox)
        self.ik_form.addRow("URDF Path", urdf_widget)

        self.ik_joint_list_widget = QListWidget()
        self.ik_joint_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ik_form.addRow("Joint Names", self.ik_joint_list_widget)

        self.ik_ee_link_combo = QComboBox()
        self.ik_form.addRow("End-effector Link", self.ik_ee_link_combo)

        self.ik_base_link_combo = QComboBox()
        self.ik_form.addRow("Base Link (from orig. URDF)", self.ik_base_link_combo)

        self.ik_save_btn = QPushButton("Save")
        self.ik_save_btn.clicked.connect(self.save_ik_config)
        self.ik_form.addRow(self.ik_save_btn)

        right_widget = QWidget()
        right_widget.setLayout(self.ik_form)
        load_layout.addWidget(right_widget)
        layout6.addLayout(load_layout)
        page6.setLayout(layout6)
        page6.set_show_event(self.load_page6)
        page6.set_hide_event(self.reset_colors)
        self.stack.addWidget(page6)
        return

    def load_page6(self):
        if not self.URDF_LOADED: return
        self.ik_group_tree.clear()
        for group_name in self.group_dict:
            top_item = QTreeWidgetItem([group_name])
            self.ik_group_tree.addTopLevelItem(top_item)
            for joint in self.group_dict[group_name]:
                child_item = QTreeWidgetItem([joint])
                top_item.addChild(child_item)

            self.ik_dict[group_name], self.ik_info[group_name] = {}, {}
            if self.config is not None:
                self.ik_info[group_name] = dict(self.config.ik_cfg.get(group_name, {}))
                file_name = self.ik_info[group_name]['urdf_path']
                self.parse_urdf_for_ik(file_name, group_name)
                self.ik_ee_link_combo.setCurrentText(self.ik_info[group_name].get('ee_link', ''))
                self.ik_base_link_combo.setCurrentText(self.ik_info[group_name].get('base_link', ''))
        return

    def load_urdf_for_ik(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open URDF File", "", "URDF Files (*.urdf)")
        if not file_name:
            return
        selected_items = self.ik_group_tree.selectedItems()
        if len(selected_items) == 0:
            QMessageBox.warning(self, "Error", "Please select a group to load URDF for IK.")
            return
        group_name = selected_items[0].text(0)
        self.ik_info[group_name]['urdf_path'] = file_name
        self.parse_urdf_for_ik(file_name, group_name)

    def parse_urdf_for_ik(self, file_name, group_name):
        try:
            self.ik_dict[group_name]['robot'] = ik_robot = URDF.load(file_name)
            self.ik_urdf_path_line.setText(file_name)

            self.ik_joint_list_widget.clear()
            for joint in ik_robot.actuated_joints:
                self.ik_joint_list_widget.addItem(joint.name)

            # Populate end-effector links
            self.ik_ee_link_combo.clear()
            self.ik_ee_link_combo.addItems(ik_robot.link_map.keys())

            # Populate base links
            self.ik_base_link_combo.clear()
            self.ik_base_link_combo.addItems(self.robot.link_map.keys())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load URDF: {e}")

    def load_ik_config(self, item):
        group_name = item.text(0)
        if group_name not in self.ik_info: return
        file_name = self.ik_info[group_name].get('urdf_path', '')
        if file_name:
            self.parse_urdf_for_ik(file_name, group_name)
        elif self.ik_urdf_path_line.text():
            self.parse_urdf_for_ik(self.ik_urdf_path_line.text(), group_name)

        for i in range(self.ik_joint_list_widget.count()):
            name =  self.ik_joint_list_widget.item(i).text()
            if 'joint_names' in self.ik_info[group_name] and name in self.ik_info[group_name]['joint_names']:
                self.ik_joint_list_widget.item(i).setSelected(True)
        self.ik_ee_link_combo.setCurrentText(self.ik_info[group_name].get('ee_link', ''))
        self.ik_base_link_combo.setCurrentText(self.ik_info[group_name].get('base_link', ''))
        self.highlight_group_joints(item, self.group_dict, self.joint_list)
        return

    def save_ik_config(self):
        selected_items = self.ik_group_tree.selectedItems()
        if len(selected_items) == 0:
            QMessageBox.warning(self, "Error", "Please select a group to load URDF for IK.")
            return
        group_name = selected_items[0].text(0)
        # get selected joint names in self.ik_joint_list_widget
        selected_joints = [item.text() for item in self.ik_joint_list_widget.selectedItems()]
        if not selected_joints:
            QMessageBox.warning(self, "Error", "No joints selected for IK configuration.")
            return
        self.ik_info[group_name]['urdf_path'] = self.ik_urdf_path_line.text()
        self.ik_info[group_name]['joint_names'] = selected_joints
        self.ik_info[group_name]['ee_link'] = self.ik_ee_link_combo.currentText()
        self.ik_info[group_name]['base_link'] = self.ik_base_link_combo.currentText()
        return

    def setup_page7(self):
        # --- Page 7: Save Config ---
        page7 = PageWidget()
        layout7 = QVBoxLayout()

        label = QLabel("Step 7: Save Config")
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout7.addWidget(label)

        label = QLabel("Save the robot configuration to a YAML file.")
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout7.addWidget(label)

        path_layout = QHBoxLayout()
        label = QLabel("Save Folder:")
        path_layout.addWidget(label)
        self.save_folder_edit = QLineEdit()

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_save_folder)
        path_layout.addWidget(self.save_folder_edit)
        path_layout.addWidget(browse_button)
        layout7.addLayout(path_layout)

        file_layout = QHBoxLayout()
        label = QLabel("Save File:")
        file_layout.addWidget(label)
        self.save_file_name_edit = QLineEdit()
        file_layout.addWidget(self.save_file_name_edit)
        save_button = QPushButton("Save Config")
        save_button.clicked.connect(self.save_config)
        file_layout.addWidget(save_button)
        layout7.addLayout(file_layout)

        page7.setLayout(layout7)
        page7.set_show_event(self.load_page7)
        self.stack.addWidget(page7)

    def load_page7(self):
        if not self.URDF_LOADED: return
        self.save_folder_edit.setText("configs/follower/")
        self.save_file_name_edit.setText(f"configs/follower/{self.robot_name_input.text()}.yaml")
        return

    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Configuration Folder")
        if folder:
            self.save_folder_edit.setText(folder)
            if not self.save_file_name_edit.text():
                self.save_file_name_edit.setText(f"{folder}/{self.robot_name_input.text()}.yaml")
            else:
                file_name = os.path.basename(self.save_file_name_edit.text())
                self.save_file_name_edit.setText(os.path.join(folder, file_name))
        return

    def save_config(self):
        config = OmegaConf.load("configs/follower/default.yaml")
        config.robot_cfg.name = self.robot_name_input.text()
        robot_urdf_path = os.path.relpath(self.urdf_file_name)
        config.robot_cfg.asset_cfg.urdf_path = robot_urdf_path
        config.robot_cfg.asset_cfg.xml_path = robot_urdf_path
        config.robot_cfg.limb_joint_names = {}
        for limb_group_name, limb_joint_list in self.group_dict.items():
            config.robot_cfg.limb_joint_names[limb_group_name] = limb_joint_list
        config.robot_cfg.hand_joint_names = {}
        config.robot_cfg.hand_limits = {}
        for hand_group_name, hand_joint_list in self.hand_group_dict.items():
            config.robot_cfg.hand_joint_names[hand_group_name] = hand_joint_list
            config.robot_cfg.hand_limits[hand_group_name] = []
            for hand_joint_name in hand_joint_list:
                limit_info = self.robot.joint_map[hand_joint_name].limit
                if limit_info is None:
                    config.robot_cfg.hand_limits[hand_group_name].append([0.0, 1.0])
                else:
                    config.robot_cfg.hand_limits[hand_group_name].append([limit_info.lower, limit_info.upper])
        config.robot_cfg.end_effector_link = {}
        for group_name, combo in self.end_effector_dict.items():
            config.robot_cfg.end_effector_link[group_name] = combo.currentText() if not isinstance(combo, str) else combo
        config.robot_cfg.init_qpos = []
        for joint_name, value in self.joint_values.items():
            config.robot_cfg.init_qpos.append(value)

        config.ik_cfg = {}
        for group_name, info in self.ik_info.items():
            config.ik_cfg[group_name] = {}
            config.ik_cfg[group_name]['urdf_path'] = os.path.relpath(info.get('urdf_path', ''))
            config.ik_cfg[group_name]['joint_names'] = info.get('joint_names', [])
            config.ik_cfg[group_name]['ee_link'] = info.get('ee_link', '')
            config.ik_cfg[group_name]['base_link'] = info.get('base_link', '')
            config.ik_cfg[group_name]['dt'] = 0.05
            config.ik_cfg[group_name]['asset_dir'] = '${mesh_dir}'
            config.ik_cfg[group_name]['ik_damping'] = 0.075
            config.ik_cfg[group_name]['eps'] = 1e-3

        file_name = self.save_file_name_edit.text()
        OmegaConf.save(config, file_name)
        self.print_log(f"Configuration saved to {file_name}")


if __name__ == "__main__":
    app = QApplication([])
    gui = URDFConfigGUI()
    gui.show()
    sys.exit(app.exec())

