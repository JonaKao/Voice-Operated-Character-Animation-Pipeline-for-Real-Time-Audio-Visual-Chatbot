import open3d as o3d
import numpy as np
import os

# Paths
MESH_DIR = 'animation_output/meshes'
FRAME_DIR = 'frames'
RESOLUTION = (800, 800)

os.makedirs(FRAME_DIR, exist_ok=True)

# Offscreen renderer
renderer = o3d.visualization.rendering.OffscreenRenderer(RESOLUTION[0], RESOLUTION[1])
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = 'defaultLit'

mesh_files = sorted(f for f in os.listdir(MESH_DIR) if f.endswith('.obj'))
print(f'Rendering {len(mesh_files)} frames with Open3D.')

for i, fname in enumerate(mesh_files):
    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_DIR, fname))
    mesh.compute_vertex_normals()
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat)
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.max(bbox.get_extent())
    # Position camera in front of the mesh
    renderer.setup_camera(60.0, center, center + [0, 0, extent*2], [0, 1, 0])
    img = renderer.render_to_image()
    out_path = os.path.join(FRAME_DIR, f'{i:05d}.png')
    o3d.io.write_image(out_path, img)
    if i % 50 == 0:
        print(f'  rendered frame {i}/{len(mesh_files)}')

print('Done rendering frames!')
