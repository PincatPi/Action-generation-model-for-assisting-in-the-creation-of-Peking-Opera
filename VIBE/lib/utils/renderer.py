import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from lib.models.smpl import get_smpl_faces


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

        self.camera_nodes = []
        self.mesh_nodes = []

    def render(self, img, verts, cam):
        if len(self.camera_nodes) > 0:
            for node in self.camera_nodes:
                self.scene.remove_node(node)
            self.camera_nodes = []

        if len(self.mesh_nodes) > 0:
            for node in self.mesh_nodes:
                self.scene.remove_node(node)
            self.mesh_nodes = []

        for idx in range(len(verts)):
            mesh = trimesh.Trimesh(verts[idx], self.faces)
            mesh = pyrender.Mesh.from_trimesh(mesh)
            node = self.scene.add(mesh)
            self.mesh_nodes.append(node)

            camera = WeakPerspectiveCamera(
                scale=cam[idx, 0] * np.ones(2),
                translation=cam[idx, 1:]
            )
            camera_node = self.scene.add(camera)
            self.camera_nodes.append(camera_node)

        color, _ = self.renderer.render(self.scene, flags=RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * img)

        return output_img
