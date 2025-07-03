import os
import cv2
import scipy
import tempfile
import numpy as np
import tensorflow as tf
from subprocess import call
from scipy.io import wavfile
import trimesh
from utils.audio_handler import AudioHandler

def process_audio(ds_path, audio, sample_rate):
    config = {
        'deepspeech_graph_fname': ds_path,
        'audio_feature_type': 'deepspeech',
        'num_audio_features': 29,
        'audio_window_size': 16,
        'audio_window_stride': 1
    }
    tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
    audio_handler = AudioHandler(config)
    return audio_handler.process(tmp_audio)['subj']['seq']['audio']

def output_sequence_meshes(sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    mesh_out_path = os.path.join(out_path, 'meshes')
    os.makedirs(mesh_out_path, exist_ok=True)
    for i, verts in enumerate(sequence_vertices):
        mesh = trimesh.Trimesh(vertices=verts, faces=template.faces)
        mesh.export(os.path.join(mesh_out_path, f'{i:05d}.obj'))

def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    # (you can leave this here even if you never visualize)
    os.makedirs(out_path, exist_ok=True)
    tmp_video = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_video.name, fourcc, 60, (800, 800), True)
    tex_img = cv2.imread(texture_img_fname)[:, :, ::-1] if texture_img_fname else None
    center = np.mean(sequence_vertices[0], axis=0)
    for verts in sequence_vertices:
        mesh = trimesh.Trimesh(vertices=verts, faces=template.faces)
        img = render_mesh_helper(mesh, center, tex_img=tex_img)
        writer.write(img)
    writer.release()
    out_vid = os.path.join(out_path, 'video.mp4')
    cmd = f'ffmpeg -i {audio_fname} -i {tmp_video.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {out_vid}'.split()
    call(cmd)

def inference(tf_model_fname, ds_fname, audio_fname, template_fname, condition_idx, out_path,
              render_sequence=True, uv_template_fname='', texture_img_fname=''):
    # load template
    tmesh = trimesh.load(template_fname, process=False)
    Template = type('T', (), {'v': np.asarray(tmesh.vertices),
                              'faces': np.asarray(tmesh.faces)})()

    # read & preprocess audio
    sr, audio = wavfile.read(audio_fname)
    audio = audio if audio.ndim==1 else audio[:,0]
    proc_audio = process_audio(ds_fname, audio, sr)

    # set up TF graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    g = tf.get_default_graph()
    sf = g.get_tensor_by_name('VOCA/Inputs_encoder/speech_features:0')
    cs = g.get_tensor_by_name('VOCA/Inputs_encoder/condition_subject_id:0')
    it = g.get_tensor_by_name('VOCA/Inputs_encoder/is_training:0')
    ti = g.get_tensor_by_name('VOCA/Inputs_decoder/template_placeholder:0')
    od = g.get_tensor_by_name('VOCA/output_decoder:0')

    n = len(proc_audio)
    feed = {
      sf: np.expand_dims(np.stack(proc_audio), -1),
      cs: np.repeat(condition_idx-1, n),
      it: False,
      ti: np.repeat(Template.v[np.newaxis,...,np.newaxis], n, axis=0)
    }

    with tf.Session() as sess:
        saver.restore(sess, tf_model_fname)
        verts = np.squeeze(sess.run(od, feed))
        output_sequence_meshes(verts, Template, out_path, uv_template_fname, texture_img_fname)
        if render_sequence:
            render_sequence_meshes(audio_fname, verts, Template, out_path, uv_template_fname, texture_img_fname)
    tf.reset_default_graph()
