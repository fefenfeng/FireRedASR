# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/11/13 17:32

@Description: simple python usage example for using FireRedASR-AED-L model to transcribe audio files
"""
from fireredasr.models.fireredasr import FireRedAsr

# 准备输入数据
batch_uttid = ["test_audio", "test_audio1", "test_audio2", "test_audio3"]
batch_wav_path = ["test/test_audio.wav", "test/test_audio1.wav", "test/test_audio2.wav", "test/test_audio3.wav"]

# 加载AED模型
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")

# 执行转录
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,  # 使用GPU推理的bool
        "beam_size": 1,  # 束搜索宽度，越大越准确但越慢
        "nbest": 1,  # 返回前n个结果
        "decode_max_len": 0,  # 最大解码长度，0表示不限制
        "softmax_smoothing": 1.0,  # softmax平滑系数，增大可提升罕见词识别
        "aed_length_penalty": 0.0,  # 长度惩罚系数，增大鼓励生成更长文本
        "eos_penalty": 1.0  # 结束符惩罚系数，增大鼓励更早结束解码
    }
)

# 打印结果
for result in results:
    print(f"音频ID: {result['uttid']}")
    print(f"识别文本: {result['text']}")
