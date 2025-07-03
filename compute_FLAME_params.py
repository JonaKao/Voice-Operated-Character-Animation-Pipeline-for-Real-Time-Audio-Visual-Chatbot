import os
import glob
import argparse
import numpy as np
import chumpy as ch
from scipy.sparse.linalg import cg
import trimesh
from utils.inference import output_sequence_meshes
from smpl_webuser.serialization import load_model

parser = argparse.ArgumentParser(description='Edit VOCA motion sequences')

parser.add_argument('--source_path', default='', help='input sequence path')
parser.add_argument('--params_fname', default='', help='path of the computed parameter file')
parser.add_argument('--out_path', default='', help='FLAME meshes output path')
parser.add_argument('--flame_model_path', default='./flame/generic_model.pkl', help='path to the FLAME model')
parser.add_argument('--template_fname', default='./template/FLAME_sample.ply', help='Path of "zero pose" template mesh in FLAME topology used for the animation')

args = parser.parse_args()
source_path = args.source_path
params_fname = args.params_fname
out_path = args.out_path
flame_model_fname = args.flame_model_path
template_fname = args.template_fname

# === 🔧 Utility to load OBJ using trimesh
def load_vertices_from_obj(filename):
    mesh = trimesh.load(filename, process=False)
    return np.asarray(mesh.vertices)

def compute_FLAME_params(source_path, params_out_fname, flame_model_fname, template_fname):
    if not os.path.exists(os.path.dirname(params_out_fname)):
        os.makedirs(os.path.dirname(params_out_fname))

    sequence_fnames = sorted(glob.glob(os.path.join(source_path, '*.obj')))
    num_frames = len(sequence_fnames)
    if num_frames == 0:
        print('No sequence meshes found')
        return

    model = load_model(flame_model_fname)

    print('Optimize for template identity parameters')
    template_vertices = load_vertices_from_obj(template_fname)
    ch.minimize(template_vertices - model, x0=[model.betas[:300]], options={'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]})

    betas = model.betas.r[:300].copy()
    model.betas[:] = 0.
    model.v_template[:] = template_vertices

    model_pose = np.zeros((num_frames, model.pose.shape[0]))
    model_exp = np.zeros((num_frames, 100))

    for frame_idx in range(num_frames):
        print('Process frame %d/%d' % (frame_idx+1, num_frames))
        model.betas[:] = 0.
        model.pose[:] = 0.

        frame_vertices = load_vertices_from_obj(sequence_fnames[frame_idx])

        ch.minimize(frame_vertices - model, x0=[model.pose[6:9], model.betas[300:]], options={'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]})
        model_pose[frame_idx] = model.pose.r.copy()
        model_exp[frame_idx] = model.betas.r[300:].copy()

    np.save(params_out_fname, {'shape': betas, 'pose': model_pose, 'expression': model_exp})


def output_FLAME_meshes(flame_model_fname, params_fname, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model = load_model(flame_model_fname)
    params = np.load(params_fname, allow_pickle=True).item()

    shape = params['shape']
    pose = params['pose']
    exp = params['expression']

    model.betas[:300] = shape

    num_frames = pose.shape[0]
    for frame_idx in range(num_frames):
        model.pose[:] = pose[frame_idx, :]
        model.betas[300:] = exp[frame_idx, :]
        out_fname = os.path.join(out_path, '%05d_FLAME.obj' % frame_idx)

        mesh = trimesh.Trimesh(vertices=model.r, faces=model.f)
        mesh.export(out_fname)

if os.path.exists(params_fname) and out_path != '':
    output_FLAME_meshes(flame_model_fname, params_fname, out_path)
else:
    compute_FLAME_params(source_path, params_fname, flame_model_fname, template_fname)
