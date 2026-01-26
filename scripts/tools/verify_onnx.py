import onnx
import onnxruntime as ort
import numpy as np
import os

# 修改为你的 onnx 路径
ONNX_PATH = "/home/ouge/Software/robot_lab/logs/moe_training/sirius_split_moe_parallel/2026-01-25_07-20-45/exported/policy.onnx"

def verify():
    print(f"Checking model: {ONNX_PATH}")
    
    # 1. 检查模型结构完整性
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("[Pass] ONNX structure check.")

    # 2. 创建推理会话
    ort_session = ort.InferenceSession(ONNX_PATH)
    
    # 获取输入信息
    input_obs_name = ort_session.get_inputs()[0].name
    input_hidden_name = ort_session.get_inputs()[1].name
    obs_shape = ort_session.get_inputs()[0].shape
    hidden_shape = ort_session.get_inputs()[1].shape
    
    print(f"Input Obs Shape: {obs_shape}")     # 应该是 ['batch_size', obs_dim]
    print(f"Input Hidden Shape: {hidden_shape}") # 应该是 [num_layers, 'batch_size', hidden_dim]

    # 3. 测试动态 Batch Size (例如 Batch = 5)
    test_batch_size = 5
    # 注意：obs_dim 和 hidden_dim 需要根据你的打印结果填入，或者自动获取
    obs_dim = obs_shape[1] if isinstance(obs_shape[1], int) else 48 # 假设
    num_layers = hidden_shape[0]
    hidden_dim = hidden_shape[2]
    
    # 构造随机输入
    dummy_obs = np.random.randn(test_batch_size, obs_dim).astype(np.float32)
    # GRU Hidden State 形状通常是 [num_layers, batch, hidden_dim]
    dummy_hidden = np.zeros((num_layers, test_batch_size, hidden_dim), dtype=np.float32)

    print(f"Running inference with Batch Size = {test_batch_size}...")
    
    try:
        outputs = ort_session.run(
            None, 
            {
                input_obs_name: dummy_obs,
                input_hidden_name: dummy_hidden
            }
        )
        action = outputs[0]
        next_hidden = outputs[1]
        
        print(f"[Success] Inference output shape: {action.shape}")
        if action.shape[0] == test_batch_size:
            print("✅ 动态 Batch Size 测试通过！ONNX 模型完好。")
        else:
            print("❌ Batch Size 未随输入改变。")
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")

if __name__ == "__main__":
    if os.path.exists(ONNX_PATH):
        verify()
    else:
        print("找不到模型文件，请修改路径。")