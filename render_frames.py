import trimesh
import os

# Paths (edit if yours differ)
MESH_DIR   = 'animation_output/meshes'
FRAME_DIR  = 'frames'
RESOLUTION = (800, 800)

os.makedirs(FRAME_DIR, exist_ok=True)

mesh_files = sorted(f for f in os.listdir(MESH_DIR) if f.endswith('.obj'))
print(f'Found {len(mesh_files)} mesh files; rendering to {FRAME_DIR}/…')

for i, fname in enumerate(mesh_files):
    path = os.path.join(MESH_DIR, fname)
    mesh = trimesh.load(path, process=False)
    scene = mesh.scene()
    png = scene.save_image(resolution=RESOLUTION)
    out_path = os.path.join(FRAME_DIR, f'{i:05d}.png')
    with open(out_path, 'wb') as f:
        f.write(png)
    if i % 50 == 0:
        print(f'  rendered frame {i}/{len(mesh_files)}')

print('Done rendering frames.')
