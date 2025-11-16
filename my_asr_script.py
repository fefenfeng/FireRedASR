# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/11/13 17:32

@Description: simple python usage example for using FireRedASR-AED-L and LLM-L model to transcribe audio files
"""
import time
import wave

from fireredasr.models.fireredasr import FireRedAsr


def get_audio_duration(wav_path: str) -> float:
    """获取音频时长(秒)"""
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration


# 准备输入数据
batch_uttid = ["test_audio", "test_audio1", "test_audio2", "test_audio3"]
batch_wav_path = ["test/test_audio.wav", "test/test_audio1.wav", "test/test_audio2.wav", "test/test_audio3.wav"]

# 计算总音频时长
total_audio_duration = sum(get_audio_duration(path) for path in batch_wav_path)
print(f"总音频时长: {total_audio_duration:.2f} 秒")

# 加载AED模型
print("=" * 50)
print("使用 FireRedASR-AED-L 模型进行推理")
print("=" * 50)
model_aed = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")

# 执行转录
start_time = time.time()
results_aed = model_aed.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,  # 使用GPU推理的bool
        "beam_size": 1,  # 束搜索宽度，越大越准确但越慢
        "nbest": 1,  # 返回前n个结果
        "decode_max_len": 0,  # 最大解码长度，0表示不限制
        "softmax_smoothing": 1.25,  # softmax平滑系数，增大可提升罕见词识别
        "aed_length_penalty": 0.6,  # 长度惩罚系数，增大鼓励生成更长文本
        "eos_penalty": 1.0  # 结束符惩罚系数，增大鼓励更早结束解码
    }
)
aed_inference_time = time.time() - start_time

# 计算AED的RTF
aed_rtf = aed_inference_time / total_audio_duration

# 打印结果
for result in results_aed:
    print(f"音频ID: {result['uttid']}")
    print(f"识别文本: {result['text']}")
print(f"\nAED推理时间: {aed_inference_time:.2f} 秒")
print(f"AED RTF: {aed_rtf:.4f}")

# 加载LLM模型
print("\n" + "=" * 50)
print("使用 FireRedASR-LLM-L 模型进行推理")
print("=" * 50)
model_llm = FireRedAsr.from_pretrained("llm", "pretrained_models/FireRedASR-LLM-L")

# 执行LLM转录
start_time = time.time()
results_llm = model_llm.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,  # 使用GPU推理
        "beam_size": 1,  # 束搜索宽度
        "decode_max_len": 0,  # 最大解码长度，0表示不限制
        "decode_min_len": 0,  # 最小解码长度
        "repetition_penalty": 3.0,  # 重复惩罚系数，增大可减少重复
        "llm_length_penalty": 1.0,  # LLM长度惩罚系数
        "temperature": 1.0  # 采样温度，越低越确定性
    }
)
llm_inference_time = time.time() - start_time

# 计算LLM的RTF
llm_rtf = llm_inference_time / total_audio_duration

# 打印LLM结果
for result in results_llm:
    print(f"音频ID: {result['uttid']}")
    print(f"识别文本: {result['text']}")
print(f"\nLLM推理时间: {llm_inference_time:.2f} 秒")
print(f"LLM RTF: {llm_rtf:.4f}")

# 打印对比总结
print("\n" + "=" * 50)
print("性能对比")
print("=" * 50)
print(f"总音频时长: {total_audio_duration:.2f} 秒")
print(f"AED推理时间: {aed_inference_time:.2f} 秒, RTF: {aed_rtf:.4f}")
print(f"LLM推理时间: {llm_inference_time:.2f} 秒, RTF: {llm_rtf:.4f}")