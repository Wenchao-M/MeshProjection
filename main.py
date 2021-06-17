import os, sys
import numpy as np
import cv2
import trimesh
import sklearn.preprocessing
import matplotlib.pyplot as plt
from trimesh.grouping import *
from trimesh.graph import *
from argparse import ArgumentParser
import os
import os.path as osp
import json
import ctypes as ct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import external.libsimplify as libsimplify



class MeshProjector():

    def __init__(self,data_root,num_face=None,mode="perspective_boundary",visible=False):
        self.data_root = data_root
        self.num_face = num_face
        self.mode = mode
        if self.mode not in ('perspective_boundary','zbuffer'):
            raise Exception('Only zbuffer or perspective_boundary are supported')
        self.obj_path = osp.join(self.data_root,'model.obj')
        self.mesh = self.as_mesh(trimesh.load_mesh(self.obj_path))
        #import pdb;pdb.set_trace()
        if self.num_face != None and len(self.mesh.faces)>self.num_face:
            self.mesh = self.mesh.simplify_quadratic_decimation(self.num_face)
        else:
            self.num_face = len(self.mesh.faces)
        self.normal = np.array(sklearn.preprocessing.normalize(self.mesh.face_normals, axis=1))
        #import pdb;pdb.set_trace()
        self.view_path = osp.join(self.data_root,'rendering/rendering_metadata.txt')
        self.cam_params = np.loadtxt(self.view_path)
        self.visible = visible
        self.out_path = osp.join(self.data_root,'out')
        if not osp.exists(self.out_path):
            os.mkdir(self.out_path)
        so_p = 'rastertriangle_so.so'
        self.dll = np.ctypeslib.load_library(so_p, '.')

    def as_mesh(self,scene_or_mesh):
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                                             for m in scene_or_mesh.geometry.values()])
        else:
            mesh = scene_or_mesh
        return mesh

    def camera_info(self,param):
        theta = np.deg2rad(param[0])
        phi = np.deg2rad(param[1])

        camY = param[3] * np.sin(phi)
        temp = param[3] * np.cos(phi)
        camX = temp * np.cos(theta)
        camZ = temp * np.sin(theta)
        cam_pos = np.array([camX, camY, camZ])

        axisZ = cam_pos.copy()
        axisY = np.array([0, 1, 0])
        axisX = np.cross(axisY, axisZ)
        axisY = np.cross(axisZ, axisX)

        cam_mat = np.array([axisX, axisY, axisZ])
        cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
        return cam_mat, cam_pos



    def project(self):
        if self.mode == 'perspective_boundary':
            self.perspective_boundary()
        else:
            self.zbuffer()




    def perspective_boundary(self):
        for index,param in enumerate(self.cam_params):
            img_path = os.path.join(os.path.split(self.view_path)[0], '%02d.png' % index)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            cam_mat, cam_pos = self.camera_info(param)
            nom_trans = np.dot(self.normal, cam_mat.transpose())
            side = nom_trans[:, 2] < -0.2
            adjacency_check = side[self.mesh.face_adjacency].all(axis=1)
            adjacency = self.mesh.face_adjacency[adjacency_check]
            face_groups = connected_components(
                adjacency, nodes=np.nonzero(side)[0])

            edges = self.mesh.edges_sorted.reshape((-1, 6))
            boundarys = []
            for faces in face_groups:
                # index edges by face then shape back to individual edges
                edge = edges[faces].reshape((-1, 2))
                # edges that occur only once are on the boundary
                group = grouping.group_rows(edge, require_count=1)
                boundarys.extend(edge[group])

            points = self.mesh.vertices * 0.57
            pt_trans = np.dot(points - cam_pos, cam_mat.transpose())
            X, Y, Z = pt_trans.T
            F = 250
            h = (-Y) / (-Z) * F + 224 / 2.0
            w = X / (-Z) * F + 224 / 2.0
            h = np.minimum(np.maximum(h, 0), 223)
            w = np.minimum(np.maximum(w, 0), 223)
            pj_points = np.stack((w, h), axis=1)
            edges = pj_points[boundarys].reshape((-1, 4))
            results = {'projected_vertices=pj_points':pj_points.tolist(),
                        #'boundarys':boundarys,
                        'edges':edges.tolist()}
            with open(osp.join(self.out_path,'{}.json'.format(index)),'w') as _:
                rs = json.dumps(results)
                _.write(rs)
                _.close()
            if self.visible:
                self.display(index,img,edges)



    def zbuffer(self):

        #import pdb;pdb.set_trace()
        for index,param in enumerate(self.cam_params):
            img_path = os.path.join(os.path.split(self.view_path)[0], '%02d.png' % index)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            cam_mat, cam_pos = self.camera_info(param)
            points = self.mesh.vertices * 0.57
            pt_trans = np.dot(points - cam_pos, cam_mat.transpose())
            X, Y, Z = pt_trans.T
            F = 250
            h = (-Y) / (-Z) * F + 224 / 2.0
            w = X / (-Z) * F + 224 / 2.0
            h = np.minimum(np.maximum(h, 0), 223)
            w = np.minimum(np.maximum(w, 0), 223)
            pj_points = np.stack((w, h), axis=1)
            pj_points = np.require(pj_points.flatten(),'float32','C')
            zs = np.require(-Z,'float32','C')
            zbuf = np.require(np.zeros(224 * 224), 'float32', 'C')
            triangle = np.require(np.zeros(224 * 224), 'int32', 'C')
            triangle_area = np.require(np.zeros(self.num_face), 'int32', 'C')
            faces = np.require(self.mesh.faces.flatten(), 'int32', 'C')

            self.dll.rgbzbuffer(
                ct.c_int(224),
                ct.c_int(224),
                pj_points.ctypes.data_as(ct.c_void_p),
                pt_trans.ctypes.data_as(ct.c_void_p),
                zs.ctypes.data_as(ct.c_void_p),
                ct.c_int(self.num_face),
                faces.ctypes.data_as(ct.c_void_p),
                zbuf.ctypes.data_as(ct.c_void_p),
                triangle.ctypes.data_as(ct.c_void_p),
                triangle_area.ctypes.data_as(ct.c_void_p)
            )

            pj_points = pj_points.reshape((-1,2))
            #import pdb;pdb.set_trace()
            visible_faces = []
            edges = []
            for i,face in enumerate(self.mesh.faces):
                visible_points = np.sum(triangle==i)
                if visible_points != 0 and triangle_area[i]!= 0:
                    rate = float(visible_points)/float(triangle_area[i])
                    if rate > 0.8:
                        #import pdb;pdb.set_trace()
                        visible_faces.append(face)
                        edges.append([pj_points[face[0]],pj_points[face[1]]])
                        edges.append([pj_points[face[1]], pj_points[face[2]]])
                        edges.append([pj_points[face[0]], pj_points[face[2]]])

            edges = np.array(edges).reshape((-1,4)).tolist()
            results = {'projected_vertices=pj_points': pj_points.tolist(),
                       'edges': edges}
            with open(osp.join(self.out_path, '{}.json'.format(index)),'w') as _:
                rs = json.dumps(results)
                _.write(rs)
                _.close()

            if self.visible:
                self.display(index,img,np.array(edges))


    def display(self,index,img,edges):

        if self.mode=='perspective_boundary':
            plt.imshow(img)
            plt.plot((edges[:, 0], edges[:, 2]), (edges[:, 1], edges[:, 3]), 'r', markersize=0.001)
            plt.savefig(osp.join(self.data_root,'out/pj_{}.png'.format(index)))
        else:
            img = cv2.resize(img,(448,448))
            edges = edges *2
            plt.imshow(img)
            plt.plot((edges[:, 0], edges[:, 2]), (edges[:, 1], edges[:, 3]), 'r', markersize=0.001)
            plt.savefig(osp.join(self.data_root,'out/zb_{}.png'.format(index)))
        #plt.show()
        plt.close()








if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="perspective_boundary",
        help='chose the projection mode, there are two modes supported, perspective_boundary or z-buffer'
    )
    parser.add_argument(
        '--num_face', type=int, default=None,
        help="Number of faces if you want to downsample the meshes to alleviate the compution time"
    )

    parser.add_argument(
        '--data_path', type=str,
            default="1a0bc9ab92c915167ae33d942430658c")
    parser.add_argument('--display',type=bool,
                        default=False,help='Display the projected 2D mesh on the image')
    #parser.add_argument('--output',type=str, default='output',help='output dictionary')

    args = parser.parse_args()
    projector = MeshProjector(args.data_path,args.num_face,args.mode,args.display)
    projector.project()







