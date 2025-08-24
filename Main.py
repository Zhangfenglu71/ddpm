from Diffusion.Train import train, eval
import torch

def main(model_config = None):
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
    }

    if model_config is not None:
        modelConfig = model_config
    
    # 动态选择设备：如果有 CUDA 使用 CUDA，否则如果有 MPS 使用 MPS，否则使用 CPU
    if torch.cuda.is_available():
        modelConfig["device"] = "cuda:0"
    elif torch.backends.mps.is_available():
        modelConfig["device"] = "mps"
    else:
        modelConfig["device"] = "cpu"

    print(f"Using device: {modelConfig['device']}")

    # 强制使用 float32
    modelConfig["dtype"] = torch.float32

    # 调用训练或评估函数
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    main()
