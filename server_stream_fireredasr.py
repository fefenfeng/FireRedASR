# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/11/16 11:12

@Description: FireRedASR实时ASR推理服务器端， websocket音频流接受及ASR处理结果返回
"""
import asyncio
import wave
import os
import time
import struct
import uuid
import threading
import argparse
import configparser
from queue import Queue

import websockets
import numpy as np
import webrtcvad

from fireredasr.models.fireredasr import FireRedAsr

# 全局配置
SAMPLE_RATE = 16000
TMP_DIR = "stream_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# 全局变量
model = None
inference_queue = Queue()
active_connections = {}  # {websocket: connection_id}
inference_lock = threading.Lock()  # 模型推理互斥锁
main_loop = None  # 主事件循环


def is_speech(int16_samples: np.ndarray, sample_rate: int = 16000, aggressiveness: int = 3,
              speech_ratio_threshold: float = 0.3) -> bool:
    """
    使用WebRTC VAD检测音频是否包含有效语音
    :param int16_samples: int16 PCM样本数组
    :param sample_rate: 采样率
    :param aggressiveness: VAD激进度 (0-3),越高越严格,建议3
    :param speech_ratio_threshold: 语音帧占比阈值,超过此值认为包含有效语音
    :return: True表示包含语音,False表示静音
    """
    vad = webrtcvad.Vad(aggressiveness)

    # WebRTC VAD 要求帧长为 10/20/30ms
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)

    # 统计包含语音的帧数
    num_speech_frames = 0
    num_frames = 0

    # 按帧检测
    for i in range(0, len(int16_samples) - frame_size, frame_size):
        frame = int16_samples[i:i + frame_size].tobytes()
        num_frames += 1
        if vad.is_speech(frame, sample_rate):
            num_speech_frames += 1

    # 使用配置的阈值判断
    if num_frames == 0:
        return False

    speech_ratio = num_speech_frames / num_frames
    return speech_ratio > speech_ratio_threshold


def load_config(config_path: str) -> argparse.Namespace:
    """
    从.cfg文件加载配置并返回argparse.Namespace对象
    :param config_path: 配置文件路径
    :return: 包含所有配置参数的Namespace对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    # 服务器配置
    host = config.get('server', 'host')
    port = config.getint('server', 'port')
    chunk_seconds = config.getfloat('server', 'chunk_seconds')
    num_workers = config.getint('server', 'num_workers')

    # 模型配置
    asr_type = config.get('model', 'asr_type')
    model_dir = config.get('model', 'model_dir')

    # 解码配置
    decode_config = {
        'use_gpu': config.getint('decode', 'use_gpu'),
        'beam_size': config.getint('decode', 'beam_size'),
        'decode_max_len': config.getint('decode', 'decode_max_len')
    }

    # 根据模型类型添加特定参数
    if asr_type == 'aed':
        decode_config.update({
            'nbest': config.getint('decode', 'nbest'),
            'softmax_smoothing': config.getfloat('decode', 'softmax_smoothing'),
            'aed_length_penalty': config.getfloat('decode', 'aed_length_penalty'),
            'eos_penalty': config.getfloat('decode', 'eos_penalty')
        })
    else:  # llm
        decode_config.update({
            'decode_min_len': config.getint('decode', 'decode_min_len'),
            'repetition_penalty': config.getfloat('decode', 'repetition_penalty'),
            'llm_length_penalty': config.getfloat('decode', 'llm_length_penalty'),
            'temperature': config.getfloat('decode', 'temperature')
        })

    # vad配置
    vad_aggressiveness = config.getint('vad', 'aggressiveness')
    vad_speech_ratio_threshold = config.getfloat('vad', 'speech_ratio_threshold')

    # 构建Namespace对象
    args = argparse.Namespace(
        host=host,
        port=port,
        chunk_seconds=chunk_seconds,
        num_workers=num_workers,
        asr_type=asr_type,
        model_dir=model_dir,
        decode_config=decode_config,
        vad_aggressiveness=vad_aggressiveness,
        vad_speech_ratio_threshold=vad_speech_ratio_threshold
    )

    return args


def write_wav(int16_samples: np.ndarray, wav_path: str) -> None:
    """
    将int16 PCM数组写入WAV文件
    :param int16_samples: int16 PCM样本数组
    :param wav_path: wav文件保存路径
    :return:
    """
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16_samples.tobytes())


def inference_worker(decode_config: dict):
    """
    推理工作线程函数: 从队列中获取任务，执行模型推理，并将结果发送回客户端
    :param decode_config: 解码配置参数字典
    :return:
    """
    global main_loop
    print("[推理线程] 启动")

    while True:
        task = inference_queue.get()
        if task is None:
            break

        websocket, uttid, wav_path = task

        try:
            start_time = time.time()

            # 使用锁保护模型推理
            with inference_lock:
                results = model.transcribe(
                    [uttid],
                    [wav_path],
                    decode_config
                )

            elapsed = time.time() - start_time
            text = results[0]['text']
            print(f"[识别结果] {uttid} -> {text} (耗时={elapsed:.2f}s)")

            # 发送结果到客户端
            if main_loop is not None:
                asyncio.run_coroutine_threadsafe(
                    websocket.send(f"RESULT|{uttid}|{text}"),
                    main_loop
                )

        except Exception as e:
            print(f"[推理错误] {uttid}: {e}")

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        inference_queue.task_done()


async def handle_audio(websocket, args):
    """
    处理来自单个客户端的音频流
    :param websocket: websocket连接对象
    :param args: 配置参数
    :return:
    """
    conn_id = uuid.uuid4().hex[:8]
    active_connections[websocket] = conn_id
    print(f"[连接] 客户端 {conn_id} 已连接")

    buffer_samples = []
    samples_per_chunk = int(SAMPLE_RATE * args.chunk_seconds)
    chunk_count = 0

    try:
        async for message in websocket:
            # 解析二进制音频数据
            num_samples = len(message) // 2
            int16_samples = struct.unpack("<" + "h" * num_samples, message)
            buffer_samples.extend(int16_samples)

            # 当累积到足够的样本时,生成切片
            while len(buffer_samples) >= samples_per_chunk:
                chunk_count += 1
                chunk_samples = buffer_samples[:samples_per_chunk]
                buffer_samples = buffer_samples[samples_per_chunk:]

                # 生成唯一ID和临时文件
                uttid = f"{conn_id}_chunk{chunk_count:04d}"

                # VAD 检测
                chunk_array = np.array(chunk_samples, dtype=np.int16)
                if not is_speech(chunk_array, SAMPLE_RATE, args.vad_aggressiveness, args.vad_speech_ratio_threshold):
                    print(f"[静音跳过] {uttid} 未检测到有效语音")
                    continue  # 跳过静音切片

                # 只处理包含语音的切片
                wav_path = os.path.join(TMP_DIR, f"{uttid}.wav")
                write_wav(np.array(chunk_samples, dtype=np.int16), wav_path)

                # 提交推理任务
                inference_queue.put((websocket, uttid, wav_path))
                await websocket.send(f"SUBMIT|{uttid}")
                print(f"[切片] {uttid} 已生成 ({args.chunk_seconds}s, {len(chunk_samples)} samples)")

    except websockets.ConnectionClosed:
        print(f"[断开] 客户端 {conn_id} 断开连接")

    except Exception as e:
        print(f"[错误] 客户端 {conn_id}: {e}")

    finally:
        # 清理资源
        if websocket in active_connections:
            del active_connections[websocket]
        print(f"[清理] 客户端 {conn_id} 资源已释放")


async def main(args):
    """启动WebSocket服务器"""
    global model, main_loop

    # 保存主事件循环引用
    main_loop = asyncio.get_running_loop()

    # 加载ASR模型
    print("=" * 60)
    print(f"正在加载 FireRedASR-{args.asr_type.upper()} 模型...")
    print(f"模型路径: {args.model_dir}")
    model = FireRedAsr.from_pretrained(args.asr_type, args.model_dir)
    print("模型加载完成!")
    print("=" * 60)

    # 启动推理工作线程
    for i in range(args.num_workers):
        worker = threading.Thread(
            target=inference_worker,
            args=(args.decode_config,),
            daemon=True
        )
        worker.start()

    # 启动WebSocket服务器
    async with websockets.serve(
            lambda ws: handle_audio(ws, args),
            args.host,
            args.port,
            max_size=None,
            ping_interval=20,
            ping_timeout=20
    ):
        print(f"[服务器] 已启动在 ws://{args.host}:{args.port}/audio")
        print(f"[配置] 切片长度={args.chunk_seconds}s, 工作线程={args.num_workers}")
        print("等待客户端连接...\n")
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FireRedASR WebSocket 流式推理服务器")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (如: server_llm.cfg)"
    )

    cmd_args = parser.parse_args()

    # 加载配置
    try:
        args = load_config(cmd_args.config)
        print(f"\n[配置] 已加载: {cmd_args.config}")
        print(f"[配置] 模型类型: {args.asr_type}")
        print(f"[配置] 服务器: {args.host}:{args.port}\n")
    except Exception as e:
        print(f"[错误] 配置加载失败: {e}")
        exit(1)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n[服务器] 正在关闭...")
