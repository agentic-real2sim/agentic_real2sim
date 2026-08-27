import ipdb
import torch
import kornia


def quat2mat(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.shape[0], 3, 3)).to(q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def mat2quat(rot):
    t = torch.clamp(rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2], min=-1)
    q = torch.zeros((rot.shape[0], 4)).to(rot.device)

    mask_0 = t > -1
    t_0 = torch.sqrt(t[mask_0] + 1)
    q[mask_0, 0] = 0.5 * t_0
    t_0 = 0.5 / t_0
    q[mask_0, 1] = (rot[mask_0, 2, 1] - rot[mask_0, 1, 2]) * t_0
    q[mask_0, 2] = (rot[mask_0, 0, 2] - rot[mask_0, 2, 0]) * t_0
    q[mask_0, 3] = (rot[mask_0, 1, 0] - rot[mask_0, 0, 1]) * t_0

    # i = 0, j = 1, k = 2
    mask_1 = ~mask_0 & (rot[:, 0, 0] >= rot[:, 1, 1]) & (rot[:, 0, 0] >= rot[:, 2, 2])
    t_1 = torch.sqrt(1 + rot[mask_1, 0, 0] - rot[mask_1, 1, 1] - rot[mask_1, 2, 2])
    t_1 = 0.5 / t_1
    q[mask_1, 0] = (rot[mask_1, 2, 1] - rot[mask_1, 1, 2]) * t_1
    q[mask_1, 1] = 0.5 * t_1
    q[mask_1, 2] = (rot[mask_1, 1, 0] + rot[mask_1, 0, 1]) * t_1
    q[mask_1, 3] = (rot[mask_1, 2, 0] + rot[mask_1, 0, 2]) * t_1

    # i = 1, j = 2, k = 0
    mask_2 = ~mask_0 & (rot[:, 1, 1] >= rot[:, 2, 2]) & (rot[:, 1, 1] > rot[:, 0, 0])
    t_2 = torch.sqrt(1 + rot[mask_2, 1, 1] - rot[mask_2, 0, 0] - rot[mask_2, 2, 2])
    t_2 = 0.5 / t_2
    q[mask_2, 0] = (rot[mask_2, 0, 2] - rot[mask_2, 2, 0]) * t_2
    q[mask_2, 1] = (rot[mask_2, 2, 1] + rot[mask_2, 1, 2]) * t_2
    q[mask_2, 2] = 0.5 * t_2
    q[mask_2, 3] = (rot[mask_2, 0, 1] + rot[mask_2, 1, 0]) * t_2

    # i = 2, j = 0, k = 1
    mask_3 = ~mask_0 & (rot[:, 2, 2] > rot[:, 0, 0]) & (rot[:, 2, 2] > rot[:, 1, 1])
    t_3 = torch.sqrt(1 + rot[mask_3, 2, 2] - rot[mask_3, 0, 0] - rot[mask_3, 1, 1])
    t_3 = 0.5 / t_3
    q[mask_3, 0] = (rot[mask_3, 1, 0] - rot[mask_3, 0, 1]) * t_3
    q[mask_3, 1] = (rot[mask_3, 0, 2] + rot[mask_3, 2, 0]) * t_3
    q[mask_3, 2] = (rot[mask_3, 1, 2] + rot[mask_3, 2, 1]) * t_3
    q[mask_3, 3] = 0.5 * t_3

    assert torch.allclose(mask_1 + mask_2 + mask_3 + mask_0, torch.ones_like(mask_0))
    return q

def interpolate_motions(bones, motions, relations, xyz, rot=None, quat=None, weights=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3)
    # motions: (n_bones, 3)
    # relations: (n_bones, k)
    # indices: (n_bones,)
    # xyz: (n_particles, 3)
    # rot: (n_particles, 3, 3)
    # quat: (n_particles, 4)
    # weights: (n_particles, n_bones)

    n_bones, _ = bones.shape
    n_particles, _ = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]

    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)

    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)

    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    if weights is None:
        weights = torch.ones((n_particles, n_bones), device=device)

        dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
        dist = torch.clamp(dist, min=1e-4)
        weights = 1 / dist
        # weights_topk = torch.topk(weights, 5, dim=1, largest=True, sorted=True)
        # weights[weights < weights_topk.values[:, -1:]] = 0.
        weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
        # weights[weights < 0.01] = 0.
        # weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)

    # Compute the transformed particles
    xyz_transformed = torch.zeros((n_particles, n_bones, 3), device=device)

    xyz_transformed = xyz[:, None] - bones[None]  # (n_particles, n_bones, 3)
    # xyz_transformed = (bone_transforms[:, :3, :3][None].repeat(n_particles, 1, 1, 1)\
    #         .reshape(n_particles * n_bones, 3, 3) @ xyz_transformed.reshape(n_particles * n_bones, 3, 1)).reshape(n_particles, n_bones, 3)
    xyz_transformed = torch.einsum('ijk,jkl->ijl', xyz_transformed, bone_transforms[:, :3, :3].permute(0, 2, 1))  # (n_particles, n_bones, 3)
    xyz_transformed = xyz_transformed + bone_transforms[:, :3, 3][None] + bones[None]  # (n_particles, n_bones, 3)
    xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)

    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        # base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = mat2quat(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)
        rot = quaternion_multiply(quats, quat)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights


def create_relation_matrix(points, K=5):
    """
    Create an NxN relation matrix where each row has 1s for the top K closest neighbors and 0s elsewhere.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        K (int): Number of closest neighbors to mark as 1.

    Returns:
        torch.Tensor: NxN relation matrix with dtype int.
    """
    N = points.shape[0]

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(points, points, p=2)  # (N, N)

    # Get the indices of the top K closest neighbors (excluding self)
    topk_indices = torch.topk(dist_matrix, K + 1, largest=False).indices[:, 1:]  # Skip self (0 distance)

    # Create the NxN relation matrix
    relation_matrix = torch.zeros((N, N), dtype=torch.int)

    # Scatter 1s for the top K neighbors
    batch_indices = torch.arange(N).unsqueeze(1).expand(-1, K)
    relation_matrix[batch_indices, topk_indices] = 1

    return relation_matrix


def get_topk_indices(points, K=5):
    """
    Compute the indices of the top K closest neighbors for each point.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        K (int): Number of closest neighbors to retrieve.

    Returns:
        torch.Tensor: Tensor of shape (N, K) containing the indices of the top K closest neighbors.
    """
    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(points, points, p=2)  # (N, N)

    # Get the indices of the top K closest neighbors (excluding self)
    topk_indices = torch.topk(dist_matrix, K + 1, largest=False).indices[:, 1:]  # Skip self (0 distance)

    return topk_indices


def knn_weights(bones, pts, K=5):
    dist = torch.norm(pts[:, None] - bones, dim=-1)  # (n_pts, n_bones)
    _, indices = torch.topk(dist, K, dim=-1, largest=False)
    bones_selected = bones[indices]  # (N, k, 3)
    dist = torch.norm(bones_selected - pts[:, None], dim=-1)  # (N, k)
    weights = 1 / (dist + 1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # (N, k)
    weights_all = torch.zeros((pts.shape[0], bones.shape[0]), device=pts.device)  # TODO: prevent init new one
    # weights_all[torch.arange(pts.shape[0])[:, None], indices] = weights
    weights_all[torch.arange(pts.shape[0], device=pts.device)[:, None], indices] = weights
    return weights_all



def calc_weights_vals_from_indices(bones, pts, indices):
    # bones: (n_bones, 3)
    # pts: (n_particles, 3)
    # indices: (n_particles, k) indices of k nearest bones per particle

    nearest_bones = bones[indices]  # (n_particles, k, 3)
    pts_expanded = pts.unsqueeze(1)  # (n_particles, 1, 3)
    distances = torch.norm(pts_expanded - nearest_bones, dim=2)
    weights_vals = 1.0 / (distances + 1e-6)
    weights_vals = weights_vals / weights_vals.sum(dim=1, keepdim=True)  # (n_particles, k)
    return weights_vals


def knn_weights_sparse(bones, pts, K=5):
    dist = torch.norm(pts[:, None].cpu() - bones.cpu(), dim=-1)  # (n_pts, n_bones)
    weights_vals, indices = torch.topk(dist, K, dim=-1, largest=False)
    weights_vals = weights_vals.to(pts.device)
    indices = indices.to(pts.device)
    weights_vals = 1 / (weights_vals + 1e-6)
    weights_vals = weights_vals / weights_vals.sum(dim=-1, keepdim=True)  # (N, k)
    torch.cuda.empty_cache()
    return weights_vals, indices

def interpolate_motions_speedup(bones, motions, relations, xyz, rot=None, quat=None, weights=None, weights_indices=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3) bone positions
    # motions: (n_bones, 3) bone motions/displacements
    # relations: (n_bones, k_adj) bone adjacency relationships - which bones are connected to each other
    # xyz: (n_particles, 3) particle positions
    # weights: (n_particles, k) weights for k nearest bones per particle
    # weights_indices: (n_particles, k) indices of k nearest bones per particle
    # rot: (n_particles, 3, 3) optional rotation matrices
    # quat: (n_particles, 4) optional quaternions

    n_bones, _ = bones.shape
    n_particles, k_nearest = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]

    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)

    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)

    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    # if weights is None:
    #     weights = torch.ones((n_particles, n_bones), device=device)

    #     dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
    #     dist = torch.clamp(dist, min=1e-4)
    #     weights = 1 / dist
    #     # weights_topk = torch.topk(weights, 5, dim=1, largest=True, sorted=True)
    #     # weights[weights < weights_topk.values[:, -1:]] = 0.
    #     weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    #     # weights[weights < 0.01] = 0.
    #     # weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)

    # Compute the transformed particles
    # xyz_transformed = torch.zeros((n_particles, n_bones, 3), device=device)

    # xyz_transformed = xyz[:, None] - bones[None]  # (n_particles, n_bones, 3)
    # # xyz_transformed = (bone_transforms[:, :3, :3][None].repeat(n_particles, 1, 1, 1)\
    # #         .reshape(n_particles * n_bones, 3, 3) @ xyz_transformed.reshape(n_particles * n_bones, 3, 1)).reshape(n_particles, n_bones, 3)
    # xyz_transformed = torch.einsum('ijk,jkl->ijl', xyz_transformed, bone_transforms[:, :3, :3].permute(0, 2, 1))  # (n_particles, n_bones, 3)
    # xyz_transformed = xyz_transformed + bone_transforms[:, :3, 3][None] + bones[None]  # (n_particles, n_bones, 3)
    # xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)


    selected_bones = bones[weights_indices]  # (n_particles, k, 3)
    selected_transforms = bone_transforms[weights_indices]  # (n_particles, k, 4, 4)

    # Transform each point with only its k nearest bones
    # xyz_expanded = xyz[:, None].unsqueeze(1).expand(-1, k_nearest, -1)  # (n_particles, k, 3)
    # xyz_local = xyz_expanded - selected_bones  # (n_particles, k, 3)
    xyz_local = xyz.unsqueeze(1) - selected_bones  # (n_particles, k, 3)

    # Apply rotation to local coordinates
    rotated_local = torch.einsum('nkij,nkj->nki', selected_transforms[:, :, :3, :3], xyz_local)  # (n_particles, k, 3)

    # Apply translation and add back bone positions
    transformed_pts = rotated_local + selected_transforms[:, :, :3, 3] + selected_bones  # (n_particles, k, 3)

    # Apply weights to get final positions
    xyz_transformed = torch.sum(transformed_pts * weights[:, :, None], dim=1)  # (n_particles, 3)


    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        # base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        # base_quats = mat2quat(bone_transforms[:, :3, :3])  # (n_bones, 4)
        # base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        # quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        # quats = torch.nn.functional.normalize(quats, dim=-1)

        from kornia.geometry.conversions import rotation_matrix_to_quaternion

        selected_rot_matrices = selected_transforms[:, :, :3, :3]  # (n_particles, k, 3, 3)
        n_particles, k_weights = weights_indices.shape
        batch_rot_matrices = selected_rot_matrices.reshape(-1, 3, 3)  # (n_particles*k, 3, 3)

        try:
            base_quats = rotation_matrix_to_quaternion(batch_rot_matrices)  # (n_particles*k, 4)
        except:
            print('use mat2quat')
            base_quats = mat2quat(batch_rot_matrices)  # (n_particles*k, 4)

        base_quats = base_quats.reshape(n_particles, k_weights, 4)  # (n_particles, k, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)
        quats = torch.sum(base_quats * weights[:, :, None], dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)

        rot = quaternion_multiply(quats, quat)

    # Return sparse weights representation for reuse
    weights_sparse = (weights, weights_indices)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights_sparse
