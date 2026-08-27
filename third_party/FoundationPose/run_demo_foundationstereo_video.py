# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import glob
import os
import re


def load_intrinsics(K_file):
  with open(K_file, 'r') as ff:
    lines = [line.strip() for line in ff if line.strip()]

  if len(lines)>=3 and len(lines[0].split())==3:
    values = []
    for line in lines[:3]:
      values.extend(float(x) for x in line.split())
  else:
    values = [float(x) for x in lines[0].split()]

  if len(values)!=9:
    raise ValueError(f'Expected 9 intrinsic values in {K_file}, got {len(values)}')
  return np.asarray(values, dtype=np.float32).reshape(3, 3)


def mask_score(mask_file):
  match = re.search(r'_score([-+]?\d*\.?\d+)\.png$', os.path.basename(mask_file))
  return float(match.group(1)) if match else float('-inf')


def load_mesh(mesh_file):
  mesh = trimesh.load(mesh_file, process=False)
  if isinstance(mesh, trimesh.Scene):
    if len(mesh.geometry)==0:
      raise RuntimeError(f'No geometry found in mesh scene: {mesh_file}')
    mesh = mesh.dump(concatenate=True)
  if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError(f'Unsupported mesh type {type(mesh)} from {mesh_file}')

  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    logging.info('Loaded mesh with UV texture')
  else:
    n_vertex_colors = 0 if mesh.visual.vertex_colors is None else len(mesh.visual.vertex_colors)
    logging.info(f'Loaded mesh without UV texture, using vertex colors ({n_vertex_colors} vertices)')

  return mesh


def maybe_simplify_mesh(mesh, max_faces=30000):
  if max_faces is None or max_faces<=0 or len(mesh.faces)<=max_faces:
    return mesh

  logging.info(f'Simplifying mesh from {len(mesh.faces)} faces to {max_faces} faces')
  mesh = mesh.simplify_quadric_decimation(max_faces)

  if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError(f'Unexpected simplified mesh type: {type(mesh)}')

  logging.info(f'Simplified mesh now has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces')
  return mesh


def load_scale_median(sequence_dir, object_name, frame_ids, max_frames):
  meta_dir = os.path.join(sequence_dir, 'scale', object_name)
  scale_files = []
  scales = []

  for frame_id in frame_ids[:max_frames]:
    meta_file = os.path.join(meta_dir, f'{frame_id}_meta.npz')
    if not os.path.exists(meta_file):
      logging.info(f'Scale meta not found: {meta_file}')
      continue

    meta = np.load(meta_file)
    if 'scale' not in meta:
      logging.info(f'Scale meta does not contain scale field: {meta_file}')
      continue

    scale_files.append(meta_file)
    scales.append(float(meta['scale']))

  if len(scales)==0:
    return None, scale_files

  scale = float(np.median(scales))
  return scale, scale_files


def apply_uniform_scale(mesh, scale):
  mesh = mesh.copy()
  mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * float(scale)
  return mesh


def compute_mask_iou(mask_a, mask_b):
  mask_a = np.asarray(mask_a).astype(bool)
  mask_b = np.asarray(mask_b).astype(bool)
  union = np.logical_or(mask_a, mask_b).sum()
  if union==0:
    return 0.0
  inter = np.logical_and(mask_a, mask_b).sum()
  return float(inter) / float(union)


@torch.inference_mode()
def render_current_pose_mask(est, K, H, W):
  if est.pose_last is None:
    return None

  ob_in_cam = est.pose_last
  if not torch.is_tensor(ob_in_cam):
    ob_in_cam = torch.as_tensor(ob_in_cam, device='cuda', dtype=torch.float)
  else:
    ob_in_cam = ob_in_cam.to(device='cuda', dtype=torch.float)
  if ob_in_cam.ndim==2:
    ob_in_cam = ob_in_cam[None]

  _, depth, _ = nvdiffrast_render(
    K=K,
    H=H,
    W=W,
    ob_in_cams=ob_in_cam,
    context='cuda',
    get_normal=False,
    glctx=est.glctx,
    mesh_tensors=est.mesh_tensors,
    output_size=(H, W),
  )
  return (depth[0].data.cpu().numpy()>0)


class FoundationStereoReader:
  def __init__(self, sequence_dir, mask_root, object_name, shorter_side=None, zfar=np.inf):
    self.sequence_dir = os.path.abspath(sequence_dir)
    self.mask_root = os.path.abspath(mask_root)
    self.object_name = object_name
    self.zfar = zfar

    self.color_files = sorted(glob.glob(f'{self.sequence_dir}/frames/left/*.png'))
    if len(self.color_files)==0:
      raise FileNotFoundError(f'No input frames found in {self.sequence_dir}/frames/left')

    self.K = load_intrinsics(f'{self.sequence_dir}/K.txt')
    self.id_strs = [os.path.splitext(os.path.basename(color_file))[0] for color_file in self.color_files]

    H0, W0 = imageio.imread(self.color_files[0]).shape[:2]
    self.downscale = 1.0
    if shorter_side is not None:
      self.downscale = float(shorter_side) / float(min(H0, W0))

    self.H = int(round(H0*self.downscale))
    self.W = int(round(W0*self.downscale))
    self.K[:2] *= self.downscale

  def __len__(self):
    return len(self.color_files)

  def get_color(self, i):
    color = imageio.imread(self.color_files[i])[...,:3]
    if self.downscale!=1:
      color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
    return color

  def get_depth(self, i):
    depth_file = f'{self.sequence_dir}/depth/{self.id_strs[i]}_depth.npy'
    depth = np.load(depth_file).astype(np.float32)
    if self.downscale!=1:
      depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
    depth[~np.isfinite(depth)] = 0
    depth[(depth<0.001) | (depth>=self.zfar)] = 0
    return depth

  def get_mask_frame_stems(self, i):
    frame_id = self.id_strs[i]
    stems = [frame_id]

    if frame_id.startswith('frame_'):
      stems.append(frame_id[len('frame_'):])

    match = re.search(r'(\d+)$', frame_id)
    if match:
      stems.append(match.group(1))

    deduped = []
    for stem in stems:
      if stem not in deduped:
        deduped.append(stem)
    return deduped

  def get_best_mask_file(self, i):
    new_mask_dir = os.path.join(self.mask_root, self.object_name, 'masks')
    for frame_stem in self.get_mask_frame_stems(i):
      mask_file = os.path.join(new_mask_dir, f'{frame_stem}.png')
      if os.path.exists(mask_file):
        return mask_file

    mask_dir = os.path.join(self.mask_root, f'{self.id_strs[i]}_masks')
    mask_files = glob.glob(os.path.join(mask_dir, f'{self.object_name}_mask*_score*.png'))
    if len(mask_files)==0:
      return None
    return sorted(mask_files, key=mask_score, reverse=True)[0]

  def get_mask(self, i):
    mask_file = self.get_best_mask_file(i)
    if mask_file is None:
      return None
    mask = cv2.imread(mask_file, -1)
    if mask is None:
      return None
    if len(mask.shape)==3:
      channel_sums = [mask[...,c].sum() for c in range(mask.shape[2])]
      mask = mask[...,int(np.argmax(channel_sums))]
    if self.downscale!=1:
      mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
    return (mask>0).astype(np.uint8)


def find_init_frame(reader, requested_init_frame):
  if requested_init_frame>=0:
    if requested_init_frame>=len(reader):
      raise IndexError(f'init_frame {requested_init_frame} is out of range [0, {len(reader)-1}]')
    mask = reader.get_mask(requested_init_frame)
    if mask is None or mask.sum()==0:
      raise RuntimeError(f'No valid {reader.object_name} mask at init_frame={requested_init_frame}')
    return requested_init_frame

  for i in range(len(reader)):
    mask = reader.get_mask(i)
    if mask is not None and mask.sum()>0:
      return i
  raise RuntimeError(f'Could not find any valid {reader.object_name} mask under {reader.mask_root}')


def discover_object_names(mask_root):
  mask_root = os.path.abspath(mask_root)
  object_names = []
  if not os.path.isdir(mask_root):
    return object_names

  for name in sorted(os.listdir(mask_root)):
    object_dir = os.path.join(mask_root, name)
    if not os.path.isdir(object_dir):
      continue
    if os.path.isdir(os.path.join(object_dir, 'masks')):
      object_names.append(name)
  return object_names


def parse_object_names(object_name_arg, mask_root):
  object_name_arg = (object_name_arg or '').strip()
  if object_name_arg=='' or object_name_arg.lower()=='all':
    object_names = discover_object_names(mask_root)
  else:
    object_names = [name.strip() for name in object_name_arg.split(',') if name.strip()]

  if len(object_names)==0:
    raise RuntimeError(f'Could not find any objects under {os.path.abspath(mask_root)}')
  return object_names


def resolve_mesh_file(mesh_file_arg, mesh_dir, object_name):
  if mesh_file_arg is None:
    return f'{mesh_dir}/{object_name}.glb'
  if '{object_name}' in mesh_file_arg:
    return mesh_file_arg.format(object_name=object_name)
  return mesh_file_arg


def run_tracking_for_object(args, object_name, has_display):
  mesh_file = resolve_mesh_file(args.mesh_file, args.mesh_dir, object_name)
  run_debug_dir = os.path.join(os.path.abspath(args.debug_dir), object_name)
  os.makedirs(run_debug_dir, exist_ok=True)
  os.makedirs(f'{run_debug_dir}/track_vis', exist_ok=True)
  os.makedirs(f'{run_debug_dir}/ob_in_cam', exist_ok=True)

  reader = FoundationStereoReader(
    sequence_dir=args.sequence_dir,
    mask_root=args.mask_root,
    object_name=object_name,
    shorter_side=args.shorter_side,
    zfar=np.inf,
  )

  init_frame = find_init_frame(reader, args.init_frame)
  end_frame = len(reader)-1 if args.end_frame<0 else min(args.end_frame, len(reader)-1)
  if init_frame>end_frame:
    raise RuntimeError(f'init_frame {init_frame} is larger than end_frame {end_frame}')

  mesh = load_mesh(mesh_file)
  mesh_scale = 1.0
  mesh_scale_meta_files = []
  if args.use_icp_init_scale:
    mesh_scale, mesh_scale_meta_files = load_scale_median(
      args.sequence_dir,
      object_name,
      reader.id_strs,
      max_frames=args.icp_scale_num_frames,
    )
    if mesh_scale is None:
      mesh_scale = 1.0
    else:
      logging.info(f'Applying median ICP scale {mesh_scale:.6f} from {len(mesh_scale_meta_files)} frame(s)')
      mesh = apply_uniform_scale(mesh, mesh_scale)

  mesh = maybe_simplify_mesh(mesh, max_faces=args.max_mesh_faces)
  with open(f'{run_debug_dir}/init_scale.txt', 'w') as ff:
    ff.write(f'init_frame_id {reader.id_strs[init_frame]}\n')
    ff.write(f'scale {mesh_scale:.12f}\n')
    ff.write(f'icp_scale_num_frames {args.icp_scale_num_frames}\n')
    ff.write(f'icp_scale_files_used {len(mesh_scale_meta_files)}\n')
    if len(mesh_scale_meta_files)==0:
      ff.write('source none\n')
    else:
      for meta_file in mesh_scale_meta_files:
        ff.write(f'source {meta_file}\n')
  mesh.export(f'{run_debug_dir}/scaled_mesh.ply')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=run_debug_dir,
    debug=args.debug,
    glctx=glctx,
  )
  logging.info(f'estimator initialization done for object={object_name}')
  logging.info(f'Using init_frame={init_frame}, end_frame={end_frame}, object={object_name}')

  reinit_log_file = f'{run_debug_dir}/reinit_log.csv'
  with open(reinit_log_file, 'w') as ff:
    ff.write('frame_id,used_mask,mask_iou,reinitialized,reason\n')

  for i in range(init_frame, end_frame+1):
    logging.info(f'object={object_name}, i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    mask = reader.get_mask(i) if (i==init_frame or args.reinit_with_mask) else None
    reinitialized = False
    reinit_reason = ''
    mask_iou = None

    if i==init_frame:
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask.astype(bool), iteration=args.est_refine_iter)
      reinitialized = True
      reinit_reason = 'init_register'

      if args.debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{run_debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{run_debug_dir}/scene_complete.ply', pcd)
    else:
      try:
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
      except Exception as e:
        if args.reinit_with_mask and mask is not None and mask.sum()>0:
          logging.info(f'track_one failed with {type(e).__name__}: {e}, falling back to register with mask')
          pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask.astype(bool), iteration=args.est_refine_iter)
          reinitialized = True
          reinit_reason = f'track_exception:{type(e).__name__}'
        else:
          raise

      if args.reinit_with_mask and mask is not None and mask.sum()>0 and not reinitialized:
        pred_mask = render_current_pose_mask(est, K=reader.K, H=color.shape[0], W=color.shape[1])
        if pred_mask is not None:
          mask_iou = compute_mask_iou(pred_mask, mask)
          logging.info(f'mask_iou:{mask_iou:.4f}')
          if mask_iou < args.reinit_iou_thresh:
            logging.info(f'mask IoU {mask_iou:.4f} < {args.reinit_iou_thresh}, re-registering')
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask.astype(bool), iteration=args.est_refine_iter)
            reinitialized = True
            reinit_reason = 'mask_iou'

    np.savetxt(f'{run_debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

    if args.debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      status_lines = [f'object:{object_name}']
      if args.reinit_with_mask:
        status_lines.append(f'mask:{int(mask is not None and mask.sum()>0)}')
        if mask_iou is not None:
          status_lines.append(f'iou:{mask_iou:.3f}')
        status_lines.append(f'reinit:{int(reinitialized)}')
        if reinit_reason:
          status_lines.append(reinit_reason)
      if len(status_lines)>0:
        vis = cv_draw_text(vis, text='\n'.join(status_lines), uv_top_left=(10, 10), color=(0,255,0), fontScale=0.6, outline_color=(0,0,0))
      if has_display:
        cv2.imshow(f'FoundationPose-{object_name}', vis[...,::-1])
        cv2.waitKey(1)

    if args.debug>=1:
      imageio.imwrite(f'{run_debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    with open(reinit_log_file, 'a') as ff:
      used_mask = int(mask is not None and mask.sum()>0)
      mask_iou_str = '' if mask_iou is None else f'{mask_iou:.6f}'
      ff.write(f'{reader.id_strs[i]},{used_mask},{mask_iou_str},{int(reinitialized)},{reinit_reason}\n')


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  repo_dir = os.path.dirname(code_dir)
  parser.add_argument('--object_name', type=str, default='all', help='Object name, comma-separated object names, or "all" to process every object folder under --mask_root.')
  parser.add_argument('--sequence_dir', type=str, default=f'{repo_dir}/FoundationStereo/output_droid_25947356')
  parser.add_argument('--mask_root', type=str, default=f'{repo_dir}/FoundationStereo/segment_results_video_25947356')
  parser.add_argument('--mesh_dir', type=str, default=f'{repo_dir}/FoundationStereo/meshes')
  parser.add_argument('--mesh_file', type=str, default=None, help='Optional mesh override. If omitted, uses <mesh_dir>/<object_name>.glb. If provided, it can include {object_name}.')
  parser.add_argument('--max_mesh_faces', type=int, default=15000, help='Automatically simplify very dense meshes before pose estimation. Set to 0 to disable.')
  parser.add_argument('--use_icp_init_scale', type=int, default=1, help='If 1, estimate mesh scale from ICP metadata under <sequence_dir>/icp_results/<object>/')
  parser.add_argument('--icp_scale_num_frames', type=int, default=20, help='Use the median ICP scale from the first N frames of the sequence.')
  parser.add_argument('--shorter_side', type=int, default=None)
  parser.add_argument('--init_frame', type=int, default=-1, help='Frame index used for registration. -1 means auto-select the first frame with a valid mask.')
  parser.add_argument('--end_frame', type=int, default=-1, help='Last frame index to process. -1 means run to the end of the sequence.')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--reinit_with_mask', type=int, default=1, help='If 1, compare tracked pose against the current segmentation mask and re-register when IoU falls below --reinit_iou_thresh.')
  parser.add_argument('--reinit_iou_thresh', type=float, default=0.3, help='Mask IoU threshold used by --reinit_with_mask.')
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_foundationstereo_video')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
  object_names = parse_object_names(args.object_name, args.mask_root)
  logging.info(f'Processing objects: {object_names}')
  for object_name in object_names:
    run_tracking_for_object(args, object_name=object_name, has_display=has_display)
