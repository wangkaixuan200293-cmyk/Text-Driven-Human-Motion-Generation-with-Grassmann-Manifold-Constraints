"""
HumanML3D 数据集的格拉斯曼流形表示与损失函数

格拉斯曼流形 Gr(k, n) 是 n 维空间中所有 k 维子空间的集合。
对于旋转表示，我们使用 Gr(2, 3)，即 3D 空间中所有 2D 平面的集合。

6D 旋转表示（两个正交单位向量）天然地对应于 Gr(2, 3) 上的点，
因为这两个向量张成一个 2D 子空间。

数据维度分布:
- 0: 根旋转角速度 (旋转)
- 1-2: 根XZ线速度 (位移速度)
- 3: 根Y高度 (位置)
- 4-66: 21个关节的局部位置 (相对于根) - 21*3=63维
- 67-192: 21个关节的6D旋转 - 21*6=126维
- 193-258: 22个关节的速度 - 22*3=66维
- 259-262: 脚部接触 (状态) - 4维
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ==================== 格拉斯曼流形基础操作 ====================

class GrassmannManifold:
    """
    格拉斯曼流形 Gr(k, n) 的操作类
    对于旋转表示，我们使用 Gr(2, 3)
    """

    @staticmethod
    def project_to_grassmann(vectors: torch.Tensor) -> torch.Tensor:
        """
        将向量投影到格拉斯曼流形上
        使用 QR 分解确保正交性

        Args:
            vectors: shape (..., n, k) - k个n维向量
        Returns:
            正交化后的向量 shape (..., n, k)
        """
        # QR分解保证正交性
        Q, R = torch.linalg.qr(vectors)
        # 确保方向一致性（R的对角线为正）
        signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        signs = signs.unsqueeze(-2)
        Q = Q * signs
        return Q

    @staticmethod
    def gram_schmidt(v1: torch.Tensor, v2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gram-Schmidt 正交化

        Args:
            v1: 第一个向量 (..., 3)
            v2: 第二个向量 (..., 3)
        Returns:
            正交化后的两个单位向量
        """
        # 归一化 v1
        u1 = F.normalize(v1, dim=-1, eps=1e-8)

        # v2 减去在 u1 方向的投影
        proj = (v2 * u1).sum(dim=-1, keepdim=True) * u1
        u2 = v2 - proj
        u2 = F.normalize(u2, dim=-1, eps=1e-8)

        return u1, u2

    @staticmethod
    def compute_projector(basis: torch.Tensor) -> torch.Tensor:
        """
        计算子空间的投影矩阵 P = V @ V^T
        这是格拉斯曼流形上点的标准表示

        Args:
            basis: 正交基 shape (..., n, k)
        Returns:
            投影矩阵 shape (..., n, n)
        """
        return torch.matmul(basis, basis.transpose(-2, -1))

    @staticmethod
    def grassmann_distance(P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
        """
        计算格拉斯曼流形上两点之间的测地距离 ================================       测地线这个别用了，我那个用测地线了，咱两个用一样的会冲突，你看一下其他的损失函数 ，172行有其他备注，注意查看 ========================================================
        使用主角（principal angles）

        Args:
            P1, P2: 投影矩阵 shape (..., n, n)
        Returns:
            测地距离 shape (...)
        """
        # 计算 P1 @ P2 的奇异值
        product = torch.matmul(P1, P2)
        # 奇异值的 arccos 给出主角
        singular_values = torch.linalg.svdvals(product)
        # 裁剪到 [-1, 1] 避免数值问题
        singular_values = torch.clamp(singular_values, -1.0, 1.0)
        principal_angles = torch.acos(singular_values)
        # 测地距离是主角的 L2 范数
        return torch.norm(principal_angles, dim=-1)

    @staticmethod
    def grassmann_log(P_base: torch.Tensor, P_target: torch.Tensor) -> torch.Tensor:
        """
        格拉斯曼流形上的对数映射（Log map）
        将流形上的点映射到切空间

        Args:
            P_base: 基点的投影矩阵 (..., n, n)
            P_target: 目标点的投影矩阵 (..., n, n)
        Returns:
            切向量 (..., n, n)
        """
        # 计算方向
        diff = P_target - P_base
        # 投影到切空间: (I - P_base) @ diff @ P_base + P_base @ diff @ (I - P_base)
        I = torch.eye(P_base.shape[-1], device=P_base.device, dtype=P_base.dtype)
        I_minus_P = I - P_base

        tangent = torch.matmul(torch.matmul(I_minus_P, diff), P_base) + \
                  torch.matmul(torch.matmul(P_base, diff), I_minus_P)

        return tangent

    @staticmethod
    def grassmann_exp(P_base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """
        格拉斯曼流形上的指数映射（Exp map）
        将切空间的向量映射回流形

        Args:
            P_base: 基点的投影矩阵 (..., n, n)
            tangent: 切向量 (..., n, n)
        Returns:
            流形上的新点 (..., n, n)
        """
        # 使用矩阵指数的近似
        # P_new = expm(tangent) @ P_base @ expm(-tangent)
        # 对于小的切向量，可以使用一阶近似
        P_new = P_base + tangent

        # 重新投影到流形上（确保是有效的投影矩阵）
        # 通过特征分解
        eigenvalues, eigenvectors = torch.linalg.eigh(P_new)
        # 选取最大的k个特征值对应的特征向量
        k = 2  # 对于 Gr(2,3)
        idx = torch.argsort(eigenvalues, dim=-1, descending=True)[..., :k]

        # 重构投影矩阵
        batch_shape = eigenvectors.shape[:-2]
        n = eigenvectors.shape[-1]

        # 获取对应的特征向量
        selected_vecs = torch.gather(
            eigenvectors,
            dim=-1,
            index=idx.unsqueeze(-2).expand(*batch_shape, n, k)
        )

        P_new = torch.matmul(selected_vecs, selected_vecs.transpose(-2, -1))
        return P_new


# ==================== HumanML3D 数据转换 ====================
# ========================================================================================= 这里我只转了humanml3d数据集的非根节点，67-192维，kit数据集是64-124维表示非根节点旋转 ==============================          =========================

class HumanML3DGrassmann:
    """
    将 HumanML3D 数据集转换为格拉斯曼流形表示
    """

    # 数据维度定义
    ROOT_ROT_VEL_IDX = 0
    ROOT_XZ_VEL_IDX = slice(1, 3)
    ROOT_Y_IDX = 3
    JOINT_POS_IDX = slice(4, 67)  # 21 joints * 3
    JOINT_ROT_6D_IDX = slice(67, 193)  # 21 joints * 6
    JOINT_VEL_IDX = slice(193, 259)  # 22 joints * 3
    FOOT_CONTACT_IDX = slice(259, 263)  # 4

    NUM_JOINTS = 21

    @staticmethod
    def extract_6d_rotations(data: torch.Tensor) -> torch.Tensor:
        """
        从 HumanML3D 数据中提取 6D 旋转表示

        Args:
            data: shape (batch, seq_len, 263) 或 (seq_len, 263)
        Returns:
            6D 旋转 shape (..., seq_len, 21, 6)
        """
        rot_6d = data[..., HumanML3DGrassmann.JOINT_ROT_6D_IDX]
        # 重塑为 (batch, seq_len, 21, 6)
        original_shape = rot_6d.shape[:-1]
        rot_6d = rot_6d.reshape(*original_shape, HumanML3DGrassmann.NUM_JOINTS, 6)
        return rot_6d

    @staticmethod
    def rot6d_to_grassmann_basis(rot_6d: torch.Tensor) -> torch.Tensor:
        """
        将 6D 旋转转换为格拉斯曼流形的正交基表示

        Args:
            rot_6d: shape (..., 6)，包含两个3D向量
        Returns:
            正交基 shape (..., 3, 2)
        """
        v1 = rot_6d[..., :3]
        v2 = rot_6d[..., 3:6]

        # Gram-Schmidt 正交化
        u1, u2 = GrassmannManifold.gram_schmidt(v1, v2)

        # 堆叠为基矩阵
        basis = torch.stack([u1, u2], dim=-1)  # (..., 3, 2)
        return basis

    @staticmethod
    def rot6d_to_projector(rot_6d: torch.Tensor) -> torch.Tensor:
        """
        将 6D 旋转转换为格拉斯曼流形的投影矩阵表示

        Args:
            rot_6d: shape (..., 6)
        Returns:
            投影矩阵 shape (..., 3, 3)
        """
        basis = HumanML3DGrassmann.rot6d_to_grassmann_basis(rot_6d)
        projector = GrassmannManifold.compute_projector(basis)
        return projector

    @staticmethod
    def projector_to_rot6d(projector: torch.Tensor) -> torch.Tensor:
        """
        将投影矩阵转换回 6D 旋转表示

        Args:
            projector: shape (..., 3, 3)
        Returns:
            6D 旋转 shape (..., 6)
        """
        # 特征分解获取主成分
        eigenvalues, eigenvectors = torch.linalg.eigh(projector)

        # 取最大的两个特征值对应的特征向量
        # eigh 返回升序排列的特征值
        v1 = eigenvectors[..., -1]  # 最大特征值
        v2 = eigenvectors[..., -2]  # 第二大特征值

        # 拼接为 6D 表示
        rot_6d = torch.cat([v1, v2], dim=-1)
        return rot_6d

    @staticmethod
    def data_to_grassmann(data: torch.Tensor) -> dict:
        """
        将完整的 HumanML3D 数据转换为包含格拉斯曼表示的字典

        Args:
            data: shape (batch, seq_len, 263) 或 (seq_len, 263)
        Returns:
            包含各种表示的字典
        """
        result = {
            'root_rot_vel': data[..., HumanML3DGrassmann.ROOT_ROT_VEL_IDX:HumanML3DGrassmann.ROOT_ROT_VEL_IDX + 1],
            'root_xz_vel': data[..., HumanML3DGrassmann.ROOT_XZ_VEL_IDX],
            'root_y': data[..., HumanML3DGrassmann.ROOT_Y_IDX:HumanML3DGrassmann.ROOT_Y_IDX + 1],
            'joint_positions': data[..., HumanML3DGrassmann.JOINT_POS_IDX].reshape(*data.shape[:-1], 21, 3),
            'joint_velocities': data[..., HumanML3DGrassmann.JOINT_VEL_IDX].reshape(*data.shape[:-1], 22, 3),
            'foot_contact': data[..., HumanML3DGrassmann.FOOT_CONTACT_IDX],
        }

        # 提取并转换旋转
        rot_6d = HumanML3DGrassmann.extract_6d_rotations(data)  # (..., 21, 6)

        # 转换为格拉斯曼表示
        original_shape = rot_6d.shape[:-1]
        rot_6d_flat = rot_6d.reshape(-1, 6)

        # 投影矩阵表示
        projectors = HumanML3DGrassmann.rot6d_to_projector(rot_6d_flat)
        projectors = projectors.reshape(*original_shape, 3, 3)

        # 正交基表示
        basis = HumanML3DGrassmann.rot6d_to_grassmann_basis(rot_6d_flat)
        basis = basis.reshape(*original_shape, 3, 2)

        result['rot_6d'] = rot_6d
        result['grassmann_projector'] = projectors  # (..., 21, 3, 3)
        result['grassmann_basis'] = basis  # (..., 21, 3, 2)

        return result


# ==================== 格拉斯曼流形损失函数 ====================

class GrassmannLoss(nn.Module):
    """
    格拉斯曼流形上的损失函数
    """

    def __init__(self,
                 geodesic_weight: float = 1.0,
                 projection_weight: float = 1.0,
                 orthogonality_weight: float = 1.0,
                 smoothness_weight: float = 0.1):
        """
        Args:
            geodesic_weight: 测地距离损失权重
            projection_weight: 投影矩阵 Frobenius 范数损失权重
            orthogonality_weight: 正交性约束损失权重
            smoothness_weight: 时序平滑性损失权重
        """
        super().__init__()
        self.geodesic_weight = geodesic_weight
        self.projection_weight = projection_weight
        self.orthogonality_weight = orthogonality_weight
        self.smoothness_weight = smoothness_weight

    def geodesic_distance_loss(self,
                               pred_rot6d: torch.Tensor,
                               target_rot6d: torch.Tensor) -> torch.Tensor:
        """
        格拉斯曼流形上的测地距离损失

        Args:
            pred_rot6d: 预测的 6D 旋转 (..., 6)
            target_rot6d: 目标 6D 旋转 (..., 6)
        Returns:
            测地距离损失
        """
        # 转换为投影矩阵
        pred_P = HumanML3DGrassmann.rot6d_to_projector(pred_rot6d)
        target_P = HumanML3DGrassmann.rot6d_to_projector(target_rot6d)

        # 计算测地距离
        dist = GrassmannManifold.grassmann_distance(pred_P, target_P)

        return dist.mean()

    def projection_frobenius_loss(self,
                                  pred_rot6d: torch.Tensor,
                                  target_rot6d: torch.Tensor) -> torch.Tensor:
        """
        投影矩阵的 Frobenius 范数损失
        比测地距离计算更稳定

        Args:
            pred_rot6d: 预测的 6D 旋转 (..., 6)
            target_rot6d: 目标 6D 旋转 (..., 6)
        Returns:
            Frobenius 范数损失
        """
        pred_P = HumanML3DGrassmann.rot6d_to_projector(pred_rot6d)
        target_P = HumanML3DGrassmann.rot6d_to_projector(target_rot6d)

        # Frobenius 范数: ||P1 - P2||_F
        diff = pred_P - target_P
        frob_norm = torch.norm(diff, p='fro', dim=(-2, -1))

        return frob_norm.mean()

    def orthogonality_loss(self, rot6d: torch.Tensor) -> torch.Tensor:
        """
        正交性约束损失
        确保 6D 表示的两个向量正交且单位长度

        Args:
            rot6d: 6D 旋转 (..., 6)
        Returns:
            正交性损失
        """
        v1 = rot6d[..., :3]
        v2 = rot6d[..., 3:6]

        # 单位长度约束
        norm1 = torch.norm(v1, dim=-1)
        norm2 = torch.norm(v2, dim=-1)
        unit_loss = (norm1 - 1).pow(2) + (norm2 - 1).pow(2)

        # 正交性约束
        dot_product = (v1 * v2).sum(dim=-1)
        ortho_loss = dot_product.pow(2)

        return (unit_loss + ortho_loss).mean()

    def temporal_smoothness_loss(self, rot6d_seq: torch.Tensor) -> torch.Tensor:
        """
        时序平滑性损失（在切空间中计算）

        Args:
            rot6d_seq: 时序 6D 旋转 (batch, seq_len, num_joints, 6)
        Returns:
            平滑性损失
        """
        if rot6d_seq.shape[1] < 2:
            return torch.tensor(0.0, device=rot6d_seq.device)

        # 转换为投影矩阵
        original_shape = rot6d_seq.shape[:-1]
        rot6d_flat = rot6d_seq.reshape(-1, 6)
        P = HumanML3DGrassmann.rot6d_to_projector(rot6d_flat)
        P = P.reshape(*original_shape, 3, 3)

        # 计算相邻帧之间的切向量
        P_current = P[:, :-1]  # (batch, seq_len-1, joints, 3, 3)
        P_next = P[:, 1:]

        # 计算切向量的范数（表示旋转变化量）
        log_map = GrassmannManifold.grassmann_log(P_current, P_next)
        tangent_norm = torch.norm(log_map, p='fro', dim=(-2, -1))

        # 二阶平滑性：加速度应该小
        if rot6d_seq.shape[1] >= 3:
            accel = tangent_norm[:, 1:] - tangent_norm[:, :-1]
            smoothness = accel.pow(2).mean()
        else:
            smoothness = tangent_norm.pow(2).mean()

        return smoothness

    def forward(self,
                pred_rot6d: torch.Tensor,
                target_rot6d: torch.Tensor,
                is_sequence: bool = True) -> dict:
        """
        计算完整的格拉斯曼损失

        Args:
            pred_rot6d: 预测的 6D 旋转
                - 如果 is_sequence=True: (batch, seq_len, num_joints, 6)
                - 如果 is_sequence=False: (..., 6)
            target_rot6d: 目标 6D 旋转，形状同上
            is_sequence: 是否为时序数据
        Returns:
            包含各项损失的字典
        """
        losses = {}

        # 展平用于计算点对点损失
        pred_flat = pred_rot6d.reshape(-1, 6)
        target_flat = target_rot6d.reshape(-1, 6)

        # 测地距离损失
        if self.geodesic_weight > 0:
            losses['geodesic'] = self.geodesic_weight * self.geodesic_distance_loss(
                pred_flat, target_flat
            )

        # 投影矩阵 Frobenius 损失
        if self.projection_weight > 0:
            losses['projection'] = self.projection_weight * self.projection_frobenius_loss(
                pred_flat, target_flat
            )

        # 正交性损失
        if self.orthogonality_weight > 0:
            losses['orthogonality'] = self.orthogonality_weight * self.orthogonality_loss(
                pred_flat
            )

        # 时序平滑性损失
        if is_sequence and self.smoothness_weight > 0:
            losses['smoothness'] = self.smoothness_weight * self.temporal_smoothness_loss(
                pred_rot6d
            )

        # 总损失
        losses['total'] = sum(losses.values())

        return losses


class ChordDistanceLoss(nn.Module):
    """
    弦距离损失 - 比测地距离更稳定的替代方案
    d_chord(P1, P2) = ||P1 - P2||_F
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor) -> torch.Tensor:
        pred_P = HumanML3DGrassmann.rot6d_to_projector(pred_rot6d)
        target_P = HumanML3DGrassmann.rot6d_to_projector(target_rot6d)

        diff = pred_P - target_P
        chord_dist = torch.norm(diff, p='fro', dim=(-2, -1))

        return chord_dist.mean()


class GrassmannFlowLoss(nn.Module):
    """
    格拉斯曼流形上的流损失
    用于生成模型（如扩散模型）的速度场学习
    """

    def __init__(self,
                 velocity_weight: float = 1.0,
                 manifold_weight: float = 0.5):
        """
        Args:
            velocity_weight: 速度场损失权重
            manifold_weight: 流形约束损失权重
        """
        super().__init__()
        self.velocity_weight = velocity_weight
        self.manifold_weight = manifold_weight

    def tangent_space_projection(self,
                                 velocity: torch.Tensor,
                                 base_rot6d: torch.Tensor) -> torch.Tensor:
        """
        将速度向量投影到切空间

        Args:
            velocity: 速度场 (..., 6)
            base_rot6d: 基点 6D 旋转 (..., 6)
        Returns:
            投影后的速度 (..., 6)
        """
        # 将 6D 转换为 (3, 2) 基
        v1 = base_rot6d[..., :3]
        v2 = base_rot6d[..., 3:6]

        # 正交化
        u1, u2 = GrassmannManifold.gram_schmidt(v1, v2)

        # 计算法向量（正交补空间）
        n = torch.cross(u1, u2, dim=-1)
        n = F.normalize(n, dim=-1, eps=1e-8)

        # 将速度分解为两部分
        vel1 = velocity[..., :3]
        vel2 = velocity[..., 3:6]

        # 投影：移除法向分量
        vel1_proj = vel1 - (vel1 * n).sum(dim=-1, keepdim=True) * n
        vel2_proj = vel2 - (vel2 * n).sum(dim=-1, keepdim=True) * n

        return torch.cat([vel1_proj, vel2_proj], dim=-1)

    def velocity_field_loss(self,
                            pred_velocity: torch.Tensor,
                            target_velocity: torch.Tensor,
                            base_rot6d: torch.Tensor) -> torch.Tensor:
        """
        速度场损失（在切空间中计算）

        Args:
            pred_velocity: 预测的速度场 (..., 6)
            target_velocity: 目标速度场 (..., 6)
            base_rot6d: 基点 (..., 6)
        Returns:
            速度损失
        """
        # 投影到切空间
        pred_proj = self.tangent_space_projection(pred_velocity, base_rot6d)
        target_proj = self.tangent_space_projection(target_velocity, base_rot6d)

        # L2 损失
        return F.mse_loss(pred_proj, target_proj)

    def manifold_consistency_loss(self, rot6d: torch.Tensor) -> torch.Tensor:
        """
        流形一致性损失
        确保输出在流形上

        Args:
            rot6d: 6D 旋转 (..., 6)
        Returns:
            一致性损失
        """
        v1 = rot6d[..., :3]
        v2 = rot6d[..., 3:6]

        # 检查是否在流形上（正交且单位）
        norm1 = torch.norm(v1, dim=-1)
        norm2 = torch.norm(v2, dim=-1)
        dot = (v1 * v2).sum(dim=-1)

        loss = (norm1 - 1).pow(2).mean() + (norm2 - 1).pow(2).mean() + dot.pow(2).mean()
        return loss

    def forward(self,
                pred_velocity: torch.Tensor,
                target_velocity: torch.Tensor,
                base_rot6d: torch.Tensor,
                output_rot6d: Optional[torch.Tensor] = None) -> dict:
        """
        计算完整的流损失

        Args:
            pred_velocity: 预测的速度场
            target_velocity: 目标速度场
            base_rot6d: 基点
            output_rot6d: （可选）输出点，用于流形约束
        Returns:
            损失字典
        """
        losses = {}

        # 速度场损失
        losses['velocity'] = self.velocity_weight * self.velocity_field_loss(
            pred_velocity, target_velocity, base_rot6d
        )

        # 流形一致性损失
        if output_rot6d is not None and self.manifold_weight > 0:
            losses['manifold'] = self.manifold_weight * self.manifold_consistency_loss(
                output_rot6d
            )

        losses['total'] = sum(losses.values())

        return losses


# ==================== 示例使用 ====================

def example_usage():
    """示例：如何使用这些工具"""

    # 加载数据
    data = np.load('/mnt/user-data/uploads/000004.npy')
    data_tensor = torch.from_numpy(data).float()

    # 添加 batch 维度
    data_tensor = data_tensor.unsqueeze(0)  # (1, 199, 263)

    print("=" * 60)
    print("HumanML3D 格拉斯曼流形表示示例")
    print("=" * 60)
    print(f"输入数据形状: {data_tensor.shape}")

    # 转换为格拉斯曼表示
    grassmann_data = HumanML3DGrassmann.data_to_grassmann(data_tensor)

    print(f"\n转换后的数据:")
    for key, value in grassmann_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # 验证格拉斯曼表示
    projectors = grassmann_data['grassmann_projector']  # (1, 199, 21, 3, 3)
    print(f"\n验证投影矩阵属性（第一帧第一个关节）:")
    P = projectors[0, 0, 0]
    print(f"  P @ P = P (幂等性): {torch.allclose(P @ P, P, atol=1e-5)}")
    print(f"  P^T = P (对称性): {torch.allclose(P.T, P, atol=1e-5)}")
    print(f"  trace(P) = 2 (秩为2): {P.trace().item():.4f}")

    # 测试损失函数
    print(f"\n测试损失函数:")

    # 模拟预测和目标
    rot6d = grassmann_data['rot_6d']  # (1, 199, 21, 6)

    # 添加一些噪声作为"预测"
    pred_rot6d = rot6d + 0.1 * torch.randn_like(rot6d)
    target_rot6d = rot6d

    # 计算损失
    loss_fn = GrassmannLoss(
        geodesic_weight=1.0,
        projection_weight=1.0,
        orthogonality_weight=0.5,
        smoothness_weight=0.1
    )

    losses = loss_fn(pred_rot6d, target_rot6d, is_sequence=True)

    print(f"  测地距离损失: {losses['geodesic'].item():.6f}")
    print(f"  投影矩阵损失: {losses['projection'].item():.6f}")
    print(f"  正交性损失: {losses['orthogonality'].item():.6f}")
    print(f"  平滑性损失: {losses['smoothness'].item():.6f}")
    print(f"  总损失: {losses['total'].item():.6f}")

    # 测试流损失
    print(f"\n测试格拉斯曼流损失:")
    flow_loss_fn = GrassmannFlowLoss(velocity_weight=1.0, manifold_weight=0.5)

    # 模拟速度场
    base_rot6d = rot6d[:, :-1]  # 基点
    target_velocity = rot6d[:, 1:] - rot6d[:, :-1]  # 简单差分作为目标速度
    pred_velocity = target_velocity + 0.05 * torch.randn_like(target_velocity)

    flow_losses = flow_loss_fn(pred_velocity, target_velocity, base_rot6d)
    print(f"  速度场损失: {flow_losses['velocity'].item():.6f}")
    print(f"  总流损失: {flow_losses['total'].item():.6f}")

    return grassmann_data, losses


if __name__ == "__main__":
    example_usage()