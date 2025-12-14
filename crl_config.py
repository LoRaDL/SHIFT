"""
CRL统一配置文件
所有CRL相关脚本共享此配置
"""

# ============ 视频配置 ============
VIDEO_PATH = "dataset/vsi-super-recall/10mins/00000000.mp4"
SAMPLE_RATE = 1  # 每N帧取1帧（1=每帧都取）

# ============ CRL配置 ============
CRITERION = "pikachu detection"

# 基底定义：每个语义维度由多个描述性文本共同构成
BASIS_TEXTS = {
    "chair detection": {
        "chair": [
            "chair",
            "office chair"
        ]
    },
    "computer detection": {
        "computer": [
            "computer",
            "computer screen"
        ]
    },
    "pikachu detection": {
        "pikachu": [
            "pikachu",
            "pokemon pikachu",
            "yellow toy",
            "plush",
            "Teddy",
            "electric pokemon",
            "cute yellow creature",
            "pikachu character"
        ]
    },
    "furniture detection": {
        "trash_can": [
            "trash can",
            "garbage bin",
            "waste basket",
            "rubbish bin"
        ],
        "bed": [
            "bed",
            "bedroom bed",
            "sleeping bed",
            "mattress"
        ],
        "chair": [
            "chair",
            "office chair",
            "dining chair",
            "seat"
        ],
        "basket": [
            "basket",
            "wicker basket",
            "storage basket",
            "laundry basket"
        ]
    },
    "hello kitty detection": {
        "hello_kitty": [
            "hello kitty",
            "hello kitty character",
            "white cat with bow",
            "sanrio hello kitty",
            "kitty white",
            "cute white cat",
            "hello kitty plush",
            "hello kitty toy"
        ]
    }
}

# ============ 处理参数 ============
PROCESS_EVERY_N_FRAMES = 1  # 视频导出时每N帧计算一次（提高性能）
BATCH_SIZE = 128  # 批处理大小（并行推理，根据显存调整）

# ============ CLIP模型配置 ============
# 可选模型（按性能从低到高）：
# "RN50"       - ResNet50, 快速, 显存小 (~2GB)
# "RN101"      - ResNet101, 中等, 显存中 (~3GB)
# "ViT-B/32"   - Vision Transformer Base/32, 默认 (~4GB)
# "ViT-B/16"   - Vision Transformer Base/16, 更好 (~6GB)
# "ViT-L/14"   - Vision Transformer Large/14, 最强 (~12GB) ⭐推荐
CLIP_MODEL = "ViT-B/32"  # 使用最强模型

# ============ 滤波参数 ============
SMOOTH_SIGMA = 0.1  # 高斯滤波强度（越大越平滑）

# ============ 关键帧检测参数 ============
CHANGE_THRESHOLD_PERCENTILE = 93  # 变化率阈值（百分位数）
TOP_K_KEYFRAMES = 6  # 每个维度显示的关键帧数量

# 加权检测参数
SCORE_WEIGHT = 0.6  # 相似度权重
CHANGE_WEIGHT = 0.0  # 变化率权重
MIN_SCORE_THRESHOLD = 0.15  # 最低相似度阈值（过滤噪声）

# ============ 输出配置 ============
OUTPUT_VIDEO = "output_crl.mp4"  # 视频导出路径

# ============ 辅助函数 ============
def get_basis_names():
    """获取当前准则的所有维度名称"""
    return list(BASIS_TEXTS[CRITERION].keys())

def print_config():
    """打印当前配置"""
    print("="*60)
    print("CRL CONFIGURATION")
    print("="*60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Sample Rate: {SAMPLE_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Criterion: {CRITERION}")
    print(f"\nSemantic Dimensions: {len(BASIS_TEXTS[CRITERION])}")
    for name, texts in BASIS_TEXTS[CRITERION].items():
        print(f"  - {name}: {len(texts)} descriptions")
    print(f"\nSmoothing: σ={SMOOTH_SIGMA}")
    print(f"Keyframe Detection: {CHANGE_THRESHOLD_PERCENTILE}th percentile, top-{TOP_K_KEYFRAMES}")
    print("="*60)

if __name__ == "__main__":
    print_config()
