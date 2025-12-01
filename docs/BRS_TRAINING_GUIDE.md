# BRS Policy Training Guide

BigYM SaucepanToHob 태스크를 위한 BRS (Bi-Manual Robot System) Policy 학습 가이드

## 목차
1. [개요](#1-개요)
2. [데이터셋 구조](#2-데이터셋-구조)
3. [데이터 로딩 파이프라인](#3-데이터-로딩-파이프라인)
4. [모델 아키텍처](#4-모델-아키텍처)
5. [학습 과정](#5-학습-과정)
6. [실행 방법](#6-실행-방법)
7. [설정 파라미터](#7-설정-파라미터)
8. [Rollout 평가](#8-rollout-평가)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. 개요

### 1.1 BRS Policy란?
BRS Policy는 Point Cloud와 Proprioception을 입력으로 받아 로봇의 행동을 예측하는 **Diffusion 기반 모방 학습** 모델입니다.

```
입력: Point Cloud (4096, 3) + Proprioception (16D)
     ↓
모델: Transformer + Conditional Diffusion
     ↓
출력: Action Chunks (H, 16) - H개의 미래 행동 예측
```

### 1.2 주요 특징
- **Temporal Windowing**: 최근 T개 프레임의 관측을 사용 (기본 T=2)
- **Action Chunking**: H개의 미래 행동을 한번에 예측 (기본 H=8)
- **Autoregressive Action**: 행동을 mobile_base → torso → arms 순서로 분해
- **Conditional Diffusion**: DDPM 기반 노이즈 제거로 행동 생성

---

## 2. 데이터셋 구조

### 2.1 파일 구조
```
data/demonstrations/saucepan_to_hob/
├── demos.hdf5              # 주 데이터 파일 (13GB)
├── pcd/                    # Point Cloud 파일들
│   ├── demo_000_pcd.npy    # (T_demo, 4096, 3) per demo
│   ├── demo_001_pcd.npy
│   └── ...
├── action_stats.json       # Action 정규화 통계
├── prop_stats.json         # Proprioception 정규화 통계
└── pcd_stats.json          # PCD XYZ 정규화 통계
```

### 2.2 HDF5 구조
```
demos.hdf5
├── demo_0/
│   ├── actions                    # (T, 16) - 16D 행동
│   ├── proprioception             # (T, 60) - 전체 관절 상태
│   ├── proprioception_floating_base  # (T, 4) - [x, y, z, rz]
│   ├── proprioception_grippers    # (T, 2) - [left, right]
│   ├── rgb_head                   # (T, H, W, 3) - 헤드 카메라 RGB
│   └── depth_head                 # (T, H, W) - 헤드 카메라 깊이
├── demo_1/
└── ... (총 31개 데모, ~27k timesteps)
```

### 2.3 Action 구조 (16D)
BRS Policy는 BigYM의 16D 행동을 다음과 같이 분해합니다:

| 구성요소 | 차원 | 인덱스 | 설명 |
|---------|------|--------|------|
| mobile_base | 3D | [0:3] | dx, dy, drz (이동) |
| torso | 1D | [3:4] | dz (높이) |
| arms | 12D | [4:16] | left_arm(5) + left_grip(1) + right_arm(5) + right_grip(1) |

#### 🔄 BRS → BigYM 액션 리매핑
BRS와 BigYM은 액션 순서가 다릅니다:

```
BRS 16D:    [X, Y, RZ, Z, left_arm(5), left_grip, right_arm(5), right_grip]
             0  1   2  3     4-8           9          10-14         15

BigYM 16D:  [X, Y, Z, RZ, left_arm(5), right_arm(5), left_grip, right_grip]
             0  1  2   3     4-8          9-13           14         15
```

**중요**: BigYM 환경에서 rollout 평가 시, `_brs_to_bigym_action()` 함수에서 이 변환이 수행됩니다:
```python
def _brs_to_bigym_action(brs_action):
    # BRS: [X, Y, RZ, Z, left_arm(5), left_grip, right_arm(5), right_grip]
    # BigYM: [X, Y, Z, RZ, left_arm(5), right_arm(5), left_grip, right_grip]
    bigym_action = np.zeros(16)
    bigym_action[0] = brs_action[0]   # X
    bigym_action[1] = brs_action[1]   # Y
    bigym_action[2] = brs_action[3]   # Z (BRS의 index 3)
    bigym_action[3] = brs_action[2]   # RZ (BRS의 index 2)
    bigym_action[4:9] = brs_action[4:9]    # left_arm(5)
    bigym_action[9:14] = brs_action[10:15] # right_arm(5)
    bigym_action[14] = brs_action[9]       # left_gripper
    bigym_action[15] = brs_action[15]      # right_gripper
    return bigym_action
```

### 2.4 Proprioception 구조 (16D)
| 구성요소 | 차원 | 소스 | 설명 |
|---------|------|------|------|
| mobile_base_vel | 3D | diff(floating_base)/dt | 속도 [vx, vy, vrz] |
| torso | 1D | floating_base[2] | 높이 z |
| left_arm | 5D | qpos[0,1,2,3,12] | 왼팔 관절 |
| left_gripper | 1D | grippers[0] | 왼손 그리퍼 |
| right_arm | 5D | qpos[13,14,15,16,25] | 오른팔 관절 |
| right_gripper | 1D | grippers[1] | 오른손 그리퍼 |

### 2.5 정규화 통계 파일

#### action_stats.json
```json
{
  "mobile_base": {"min": [...], "max": [...], "mean": [...], "std": [...]},
  "torso": {"min": ..., "max": ..., "mean": ..., "std": ...},
  "arms": {"min": [...], "max": [...], "mean": [...], "std": [...]},
  "full": {"min": [16D], "max": [16D], "mean": [16D], "std": [16D]}
}
```

#### prop_stats.json
```json
{
  "mobile_base_vel": {"min": [3D], "max": [3D], ...},
  "torso": {"min": scalar, "max": scalar, ...},
  "left_arm": {"min": [5D], "max": [5D], ...},
  "left_gripper": {"min": scalar, "max": scalar, ...},
  "right_arm": {"min": [5D], "max": [5D], ...},
  "right_gripper": {"min": scalar, "max": scalar, ...},
  "full": {"min": [16D], "max": [16D], ...}
}
```

#### pcd_stats.json
```json
{
  "xyz": {
    "min": [-0.828, -0.828, 0.051],
    "max": [0.821, 0.601, 2.0],
    "mean": [-0.063, -0.018, 0.305],
    "std": [0.278, 0.233, 0.781]
  }
}
```

---

## 3. 데이터 로딩 파이프라인

### 3.1 전체 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                     PCDDataModule                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Train/Val    │───▶│ PCDBRSDataset│───▶│  DataLoader  │       │
│  │ Demo Split   │    │              │    │ (8 workers)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PCDBRSDataset.__getitem__                    │
│                                                                  │
│  1. Sample Index → (demo_id, frame_idx)                         │
│  2. Load T frames of:                                            │
│     - PCD: pcd/demo_XXX_pcd.npy[frame_idx:frame_idx+T]          │
│     - Proprioception: HDF5에서 추출 및 변환                       │
│     - Actions: HDF5에서 H개 미래 행동 로드                        │
│  3. Normalize (if enabled):                                      │
│     - PCD: (xyz - min) / (max - min) * 2 - 1 → [-1, 1]          │
│     - Prop/Action: (x - min) / (max - min) * 2 - 1 → [-1, 1]    │
│  4. Return batch dict                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 샘플 인덱스 구조

각 샘플은 특정 데모의 특정 시점을 나타냅니다:
```python
# _build_sample_index()에서 생성
samples = [
    (demo_id="demo_0", frame_idx=0),
    (demo_id="demo_0", frame_idx=1),
    ...
    (demo_id="demo_1", frame_idx=0),
    ...
]
# 유효 범위: T-1 ≤ frame_idx ≤ len(demo) - H
```

### 3.3 __getitem__ 상세

```python
def __getitem__(self, idx):
    demo_id, frame_idx = self.samples[idx]
    
    # 1. Point Cloud 로드 (T frames)
    pcd_file = f"pcd/{demo_id}_pcd.npy"
    pcd = np.load(pcd_file)[frame_idx-T+1:frame_idx+1]  # (T, N, 3)
    
    # 2. Proprioception 추출 (T frames)
    with h5py.File(hdf5_path, 'r') as f:
        demo = f[demo_id]
        # Mobile base velocity (차분 계산)
        fb = demo['proprioception_floating_base'][frame_idx-T:frame_idx+1]
        mobile_base_vel = np.diff(fb[:, [0,1,3]], axis=0) / dt
        
        # Torso (z position)
        torso = fb[1:, 2:3]
        
        # Arms (QPOS에서 특정 인덱스 추출)
        qpos = demo['proprioception'][frame_idx-T+1:frame_idx+1]
        left_arm = qpos[:, [0,1,2,3,12]]
        right_arm = qpos[:, [13,14,15,16,25]]
        
        # Grippers
        grippers = demo['proprioception_grippers'][frame_idx-T+1:frame_idx+1]
    
    # 3. Action Chunks (H frames)
    actions = demo['actions'][frame_idx:frame_idx+H]  # (H, 16)
    
    # 4. 정규화
    if self.normalize:
        pcd = normalize_to_minus1_plus1(pcd, self.pcd_xyz_min, self.pcd_xyz_max)
        prop = normalize_to_minus1_plus1(prop, self.prop_min, self.prop_max)
        actions = normalize_to_minus1_plus1(actions, self.action_min, self.action_max)
    
    return {
        'pointcloud': {'xyz': pcd, 'rgb': rgb},
        'qpos': {'torso': torso, 'left_arm': left_arm, ...},
        'odom': {'mobile_base': mobile_base_vel},
        'action_chunks': {'mobile_base': actions[:,:3], 'torso': actions[:,3:4], 'arms': actions[:,4:]},
        'pad_mask': pad_mask
    }
```

### 3.4 Batch Collation

```python
def pcd_brs_collate_fn(batch):
    """
    여러 샘플을 배치로 묶음
    
    Input: List of dicts
    Output: Nested dict with batched tensors
    
    Shape conventions:
    - pcd: (B, num_cams, T, N, 3)
    - qpos: (B, num_cams, T, dim)
    - actions: (B, num_cams, T, H, dim)
    """
```

### 3.5 데이터 로딩 최적화

| 최적화 | 설명 |
|--------|------|
| Per-worker HDF5 handles | 각 worker가 독립적인 HDF5 파일 핸들 유지 |
| Prefetch factor=4 | 4배치를 미리 로드 |
| Persistent workers | Worker 재생성 오버헤드 제거 |
| Pin memory | GPU 전송 가속 |
| Chunk cache | HDF5 청크 캐싱 |

**성능**: ~690 samples/sec (8 workers)

---

## 4. 모델 아키텍처

### 4.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        BRS Policy                                │
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────────────────┐   │
│  │ PointNet   │   │ Prop MLP   │   │    Transformer         │   │
│  │ Encoder    │   │ Encoder    │   │    (2 layers)          │   │
│  │            │   │            │   │                        │   │
│  │ PCD(N,3)   │   │ Prop(16)   │   │  ┌──────────────────┐  │   │
│  │    ↓       │   │    ↓       │   │  │ Self-Attention   │  │   │
│  │ (256,)     │   │ (256,)     │   │  │ + Cross-Attn     │  │   │
│  └─────┬──────┘   └─────┬──────┘   │  │ to observations  │  │   │
│        │                │          │  └──────────────────┘  │   │
│        └────────┬───────┘          │           │            │   │
│                 │                  │           ▼            │   │
│                 ▼                  │  ┌──────────────────┐  │   │
│         ┌──────────────┐           │  │ Action Readout   │  │   │
│         │ Observation  │           │  │ Token            │  │   │
│         │ Tokens       │──────────▶│  └──────────────────┘  │   │
│         │ (T, 256)     │           │           │            │   │
│         └──────────────┘           │           ▼            │   │
│                                    │  ┌──────────────────┐  │   │
│                                    │  │ Diffusion Head   │  │   │
│                                    │  │ (Conditional     │  │   │
│                                    │  │  U-Net 1D)       │  │   │
│                                    │  └──────────────────┘  │   │
│                                    └────────────────────────┘   │
│                                                 │                │
│                                                 ▼                │
│                                    ┌────────────────────────┐   │
│                                    │  Action Chunks (H, 16) │   │
│                                    └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 인코더

#### PointNet Encoder
```python
# 입력: (B, T, N, 3) - T개 프레임, N개 포인트, XYZ
# 출력: (B, T, 256) - 프레임별 특징

class PointNetEncoder:
    def __init__(self):
        self.mlp = MLP([3, 64, 128, 256])  # Per-point features
        self.max_pool = GlobalMaxPool()     # Permutation invariant
    
    def forward(self, x):
        x = self.mlp(x)           # (B, T, N, 256)
        x = self.max_pool(x)      # (B, T, 256)
        return x
```

#### Proprioception MLP Encoder
```python
# 입력: (B, T, 16) - 16D proprioception
# 출력: (B, T, 256) - 프레임별 특징

class PropMLP:
    def __init__(self):
        self.layers = MLP([16, 256, 256])  # 2-layer MLP
```

### 4.3 Transformer

```python
class ObservationTransformer:
    def __init__(self):
        self.n_embd = 256
        self.n_layer = 2
        self.n_head = 8
        self.dropout = 0.1
    
    def forward(self, obs_tokens, action_readout_token):
        # obs_tokens: (B, T, 256) - PCD + Prop 결합
        # action_readout_token: (B, 1, 256) - learnable or fixed
        
        # Concat and apply transformer
        tokens = concat([obs_tokens, action_readout_token])
        output = self.transformer(tokens)
        
        # Extract action condition
        action_cond = output[:, -1, :]  # (B, 256)
        return action_cond
```

### 4.4 Diffusion Head

```python
class ConditionalUNet1D:
    """
    DDPM 기반 조건부 행동 생성
    
    Training:
        1. 깨끗한 action에 노이즈 추가
        2. 노이즈 레벨 + observation condition으로 노이즈 예측
        3. 예측 노이즈와 실제 노이즈의 MSE Loss
    
    Inference:
        1. 순수 노이즈에서 시작
        2. 반복적으로 노이즈 제거 (16 steps)
        3. 최종 action chunks 출력
    """
    
    def __init__(self):
        self.down_dims = [64, 128]
        self.kernel_size = 5
        self.n_groups = 8
        self.num_train_timesteps = 100
        self.num_inference_steps = 16
```

---

## 5. 학습 과정

### 5.1 전체 학습 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Loop                                │
│                                                                  │
│  for epoch in range(max_epochs):                                │
│      for batch in train_dataloader:                             │
│          ┌──────────────────────────────────────────────────┐   │
│          │ 1. Forward Pass                                   │   │
│          │    - Encode PCD → pcd_features                   │   │
│          │    - Encode Prop → prop_features                 │   │
│          │    - Transformer → action_condition              │   │
│          │    - Sample noise timestep t                     │   │
│          │    - Add noise to GT actions                     │   │
│          │    - Predict noise with UNet                     │   │
│          └──────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│          ┌──────────────────────────────────────────────────┐   │
│          │ 2. Compute Loss                                   │   │
│          │    loss = MSE(predicted_noise, actual_noise)     │   │
│          └──────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│          ┌──────────────────────────────────────────────────┐   │
│          │ 3. Backward Pass                                  │   │
│          │    - loss.backward()                             │   │
│          │    - gradient_clip(1.0)                          │   │
│          │    - optimizer.step()                            │   │
│          │    - lr_scheduler.step()                         │   │
│          └──────────────────────────────────────────────────┘   │
│                                                                  │
│      # Validation                                               │
│      if epoch % eval_interval == 0:                             │
│          val_loss = validate(val_dataloader)                    │
│          log_to_wandb(train_loss, val_loss, lr)                │
│                                                                  │
│      # Rollout Evaluation (optional)                            │
│      if epoch % rollout_interval == 0:                          │
│          success_rate = evaluate_in_env()                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Forward Pass 상세

```python
def training_step(self, batch):
    # 1. 입력 추출
    pcd = batch['pointcloud']['xyz']           # (B, num_cams, T, N, 3)
    prop = self._flatten_prop(batch['qpos'])   # (B, T, 16)
    gt_actions = self._flatten_actions(batch['action_chunks'])  # (B, H, 16)
    
    # 2. 인코딩
    pcd_features = self.pointnet(pcd)          # (B, T, 256)
    prop_features = self.prop_mlp(prop)        # (B, T, 256)
    obs_features = pcd_features + prop_features  # (B, T, 256)
    
    # 3. Transformer
    action_cond = self.transformer(obs_features)  # (B, 256)
    
    # 4. Diffusion Loss
    # Sample random timestep
    t = torch.randint(0, self.num_train_timesteps, (B,))
    
    # Add noise to actions
    noise = torch.randn_like(gt_actions)
    noisy_actions = self.scheduler.add_noise(gt_actions, noise, t)
    
    # Predict noise
    pred_noise = self.unet(noisy_actions, t, action_cond)
    
    # MSE Loss
    loss = F.mse_loss(pred_noise, noise)
    
    return loss
```

### 5.3 Learning Rate Schedule

```
LR
 │
 │    ┌────────────────────────────────────────────────
 │    │ Warmup (1000 steps)
0.0007├────┘
 │    
 │                    Cosine Decay
 │                         ╲
 │                          ╲
 │                           ╲
0.000005├──────────────────────────────────────────────
 │
 └────────────────────────────────────────────────────▶ Steps
      0    1000                              300000
```

### 5.4 Inference (Action Generation)

```python
def predict_action(self, observation):
    # 1. 인코딩
    pcd_features = self.pointnet(observation['pcd'])
    prop_features = self.prop_mlp(observation['prop'])
    action_cond = self.transformer(pcd_features + prop_features)
    
    # 2. Diffusion Sampling (16 steps)
    actions = torch.randn(B, H, 16)  # Start from noise
    
    for t in reversed(range(16)):
        pred_noise = self.unet(actions, t, action_cond)
        actions = self.scheduler.step(pred_noise, t, actions)
    
    # 3. Denormalize
    actions = denormalize(actions, self.action_min, self.action_max)
    
    return actions  # (B, H, 16)
```

---

## 6. 실행 방법

### 6.1 학습 실행

```bash
cd /home/hyunjin/bigym_ws/robobase

# 기본 실행
bash train_brs.sh

# 또는 직접 실행
python -m robobase.method.brs_lightning \
    --config robobase/cfgs/brs_config.yaml \
    --use-pcd \
    --bs 64 \
    --vbs 64 \
    --dataloader-num-workers 8 \
    --wandb-name brs_experiment_1
```

### 6.2 주요 CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | Config YAML 경로 | brs_config.yaml |
| `--use-pcd` | PCD 데이터셋 사용 | False |
| `--hdf5-path` | HDF5 파일 경로 | config 값 |
| `--pcd-root` | PCD 디렉토리 경로 | config 값 |
| `--bs` | 배치 크기 | 256 |
| `--vbs` | 검증 배치 크기 | 256 |
| `--lr` | 학습률 | 0.0007 |
| `--seed` | 랜덤 시드 | 42 |
| `--no-wandb` | WandB 비활성화 | False |
| `--wandb-name` | WandB 실행 이름 | config 값 |
| `--dataloader-num-workers` | 데이터 로더 워커 수 | 16 |

### 6.3 WandB 로깅

학습 중 다음 메트릭이 WandB에 로깅됩니다:
- `train/loss`: 학습 손실
- `val/loss`: 검증 손실
- `lr`: 현재 학습률
- `epoch`: 현재 에폭
- GPU/CPU 사용률

#### Run Directory 구조
학습 결과물은 `runs/{wandb_name}/` 디렉토리에 저장됩니다:
```
runs/
└── brs_experiment_1/           # --wandb-name 값과 동일
    ├── checkpoints/
    │   ├── last.ckpt
    │   └── best-epoch=XX-val_loss=X.XX.ckpt
    ├── tb/                     # TensorBoard 로그
    ├── logs/                   # CSV 로그
    └── wandb/                  # WandB 로컬 캐시
```

**참고**: `--wandb-name` 옵션을 지정하면 로컬 run directory와 WandB run 이름이 동일하게 설정됩니다.

---

## 7. 설정 파라미터

### 7.1 brs_config.yaml 주요 설정

```yaml
# ====== Training ======
seed: 42
gpus: 1
lr: 0.0007
bs: 256                    # batch_size
vbs: 256                   # val_batch_size
val_split_ratio: 0.1
max_epochs: 10000
gradient_clip_val: 1.0

# ====== LR Schedule ======
use_cosine_lr: true
lr_warmup_steps: 1000
lr_cosine_steps: 300000
lr_cosine_min: 0.000005

# ====== Model ======
action_dim: 16
prop_dim: 16
num_latest_obs: 2          # Temporal window T
action_prediction_horizon: 8  # Action horizon H

# ====== PointNet ======
pointnet_hidden_dim: 256
pcd_downsample_points: 2048

# ====== Transformer ======
xf_n_embd: 256
xf_n_layer: 2
xf_n_head: 8
xf_dropout_rate: 0.1

# ====== Diffusion ======
noise_scheduler:
  num_train_timesteps: 100
  beta_schedule: "squaredcos_cap_v2"
num_denoise_steps_per_inference: 16

# ====== Data ======
hdf5_path: ".../demos.hdf5"
pcd_root: ".../pcd"
action_stats_path: ".../action_stats.json"
prop_stats_path: ".../prop_stats.json"
pcd_stats_path: ".../pcd_stats.json"
normalize: true
normalize_pcd: true
```

### 7.2 하이퍼파라미터 권장값

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| bs | 64-256 | GPU 메모리에 따라 조절 |
| lr | 5e-4 ~ 1e-3 | 큰 배치에선 높게 |
| num_latest_obs | 2 | 관측 윈도우 |
| action_prediction_horizon | 8 | 예측 호라이즌 |
| pcd_downsample_points | 2048-4096 | 계산량과 정확도 트레이드오프 |
| xf_n_layer | 2-4 | 모델 용량 |
| num_denoise_steps | 16-50 | 속도와 품질 트레이드오프 |

---

## 8. Rollout 평가

### 8.1 개요
학습된 정책을 BigYM 환경에서 실제로 실행하여 성능을 평가합니다.

### 8.2 BigYM 환경 설정

#### PelvisDof 설정 (중요!)
BigYM에서 torso Z 축 제어를 위해 `PelvisDof.Z`를 반드시 포함해야 합니다:

```python
from bigym.action_modes import PelvisDof

env = BiGymEnv(
    action_mode=JointPositionActionMode(
        floating_base=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],  # 4D
        absolute=False,
    ),
    # ...
)
```

**주의**: 기본 BigYM 설정은 `floating_dofs=[X, Y, RZ]` (3D)입니다. BRS 정책의 16D 액션과 호환되려면 Z축을 포함한 4D가 필요합니다.

| floating_dofs | 총 액션 차원 | 호환성 |
|---------------|-------------|--------|
| [X, Y, RZ] (기본) | 15D | ❌ 불일치 |
| [X, Y, Z, RZ] | 16D | ✅ BRS 호환 |

### 8.3 Rollout Callback 설정

`robobase/rollout_callback.py`에서 환경 생성 시 올바른 설정을 사용합니다:

```python
def _create_env(self):
    env = BiGymEnv(
        task=SaucepanToHob,
        action_mode=JointPositionActionMode(
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
            absolute=False,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(name="head", resolution=(84, 84)),
            ],
            proprioception=True,
        ),
        render_mode="rgb_array",
    )
    return env
```

### 8.4 평가 메트릭
- `rollout/success_rate`: 성공률 (0~1)
- `rollout/avg_return`: 평균 리턴
- `rollout/avg_episode_length`: 평균 에피소드 길이

---

## 9. 트러블슈팅

### 9.1 액션 차원 불일치 오류
```
Error: Action dimension mismatch: expected 15, got 16
```

**원인**: BigYM 환경이 기본 `floating_dofs=[X, Y, RZ]`로 설정되어 15D 액션만 받음

**해결**: `floating_dofs`에 `PelvisDof.Z` 추가:
```python
floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
```

### 9.2 메모리 부족
```bash
# 배치 크기 줄이기
--bs 32 --vbs 32

# 포인트 수 줄이기 (config에서)
pcd_downsample_points: 1024
```

### 9.3 데이터 로딩 느림
```bash
# 워커 수 늘리기
--dataloader-num-workers 8

# Prefetch 늘리기 (config에서)
prefetch_factor: 4
persistent_workers: true
```

### 9.4 학습 불안정
```yaml
# Config에서 gradient clipping 조절
gradient_clip_val: 0.5

# 학습률 낮추기
lr: 0.0003
```

---

## 참고 자료

- [BRS-Algo Repository](https://github.com/brs-algo)
- [BigYM Documentation](https://github.com/chernyadev/bigym)
- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
