from DiffusionFreeGuidence.TrainCondition import train, eval
import torch

def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_63_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config

    # 动态选择设备：优先使用 CUDA，其次是 Apple MPS，最后回退到 CPU
    if torch.cuda.is_available():
        modelConfig["device"] = "cuda:0"
    elif torch.backends.mps.is_available():
        modelConfig["device"] = "mps"
    else:
        modelConfig["device"] = "cpu"

    print(f"Using device: {modelConfig['device']}")

    # 强制使用 float32，确保在 MPS 上运行
    modelConfig["dtype"] = torch.float32

    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
