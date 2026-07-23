from dataclasses import asdict
from collections import deque
from typing import Dict, Callable, Any
import torch
import numpy as np
import cv2
import rospy
from tqdm import tqdm
import time
import sys
from kuavo_deploy.config import KuavoConfig
from sensor_msgs.msg import CompressedImage, Image, JointState
from torchvision.transforms.functional import to_tensor
# Prefer live-robot kuavo_msgs (PYTHONPATH workspace). SDK-bundled sensorsData
# often has a different MD5 and will never receive /sensors_data_raw.
try:
    from kuavo_msgs.msg import sensorsData, lejuClawState
except ImportError:  # pragma: no cover
    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData, lejuClawState
from kuavo_deploy.utils.signal_controller import ControlSignalManager
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.utils.ros_manager import ROSManager
from kuavo_data.common.config_platform import get_arm_joint_slice

log_robot = setup_logger("robot")

# Idle leju claw often has a publisher but no stream; seed so wait_buffer_ready can finish.
_IDLE_SEED_KEYS = frozenset({"gripper", "leju_claw", "rq2f85", "qiangnao"})


class ObsBuffer:
    def __init__(
        self, 
        config: KuavoConfig,  
        obs_key_map: Dict[str, Dict[str, Any]] = None,
        compute_func_map: Dict[str, Callable] = None,
    ) -> None:
        self.control_signal_manager = ControlSignalManager()
        self.ros_manager = ROSManager()
        # === 从 KuavoConfig 中提取环境配置 ===
        # env_cfg = config.env
        env_cfg = config
        self.which_arm = env_cfg.which_arm
        self.platform_type = env_cfg.platform_type

        # === 观测定义 ===
        self.obs_key_map = obs_key_map or env_cfg.obs_key_map or {}
        self.compute_func_map = compute_func_map or {}

        # === 区分订阅型与计算型观测 ===
        self.subscribe_keys = {k: v for k, v in self.obs_key_map.items() if v.get("type") != "computed"}
        self.computed_keys  = {k: v for k, v in self.obs_key_map.items() if v.get("type") == "computed"}

        # === 反向依赖索引 ===
        self.source_to_computed = {}
        for comp_key, comp_info in self.computed_keys.items():
            src = comp_info.get("source")
            if src:
                self.source_to_computed.setdefault(src, []).append(comp_key)
                log_robot.info(f"Registered computed obs '{comp_key}' depends on '{src}'")

        # === 初始化观测缓存 ===
        self.obs_buffer_size = {k: v["frequency"] for k, v in self.obs_key_map.items()}
        self.obs_buffer_data = {
            k: {"data": deque(maxlen=v["frequency"]), "timestamp": deque(maxlen=v["frequency"])}
            for k, v in self.obs_key_map.items()
        }

        # === ROS topic 对应表 ===
        # v3 production: /cam_*/color/h265_stream (CompressedImage format=h265)
        # Legacy JPEG: /cam_*/color/image_raw/compressed
        # Sim mujoco: /camera|/wrist_cam_*/color/image_raw
        self._h265_decoders: Dict[str, Any] = {}
        self.callback_key_map = {
            '/cam_h/color/h265_stream': self.rgb_callback,
            '/cam_l/color/h265_stream': self.rgb_callback,
            '/cam_r/color/h265_stream': self.rgb_callback,
            '/cam_h/color/image_raw/compressed': self.rgb_callback,
            '/cam_l/color/image_raw/compressed': self.rgb_callback,
            '/cam_r/color/image_raw/compressed': self.rgb_callback,
            '/camera/color/image_raw': self.rgb_raw_callback,
            '/left_wrist_camera/color/image_raw': self.rgb_raw_callback,
            '/right_wrist_camera/color/image_raw': self.rgb_raw_callback,
            '/wrist_cam_l/color/image_raw': self.rgb_raw_callback,
            '/wrist_cam_r/color/image_raw': self.rgb_raw_callback,
            '/cam_h/depth/image_raw/compressedDepth': self.depth_callback,
            '/cam_l/depth/image_rect_raw/compressedDepth': self.depth_callback,
            '/cam_r/depth/image_rect_raw/compressedDepth': self.depth_callback,
            '/sensors_data_raw': self.sensorsData_callback,
            '/dexhand/state': self.qiangnaoState_callback,
            '/leju_claw_state': self.lejuClawState_callback,
            '/gripper/state': self.rq2f85State_callback,
        }
        self.setup_subscribers()

    # ===== ROS订阅 =====
    def create_callback(self, callback, topic_key, handle):
        return lambda msg: callback(msg, topic_key, handle)

    def setup_subscribers(self):
        """仅订阅来自 ROS 的观测"""
        msg_type_dict = {
            "CompressedImage": CompressedImage,
            "Image": Image,
            "sensorsData": sensorsData,
            "JointState": JointState,
            "lejuClawState": lejuClawState,
        }
        for topic_key, info in self.subscribe_keys.items():
            topic_name = info["topic"]
            assert info["msg_type"] in msg_type_dict, f"msg_type '{info['msg_type']}' is not supported; valid keys: {list(msg_type_dict.keys())}"
            msg_type = msg_type_dict[info["msg_type"]]
            callback = self.callback_key_map.get(topic_name)
            # Fallback by msg type so new topic names in yaml still work.
            if callback is None and info["msg_type"] == "Image":
                callback = self.rgb_raw_callback
            elif callback is None and info["msg_type"] == "CompressedImage":
                callback = self.rgb_callback

            if not msg_type or not callback:
                log_robot.warning(f"Missing msg_type or callback for {topic_name}")
                continue

            handle = info.get("handle", {})
            self.ros_manager.register_subscriber(
                topic_name, msg_type, self.create_callback(callback, topic_key, handle)
            )
            log_robot.info(f"Subscribed to {topic_name} for key '{topic_key}'")

    # ===== 数据预处理 =====
    def img_preprocess(self, image):
        """图像预处理"""
        return to_tensor(image).unsqueeze(0)

    def depth_preprocess(self, depth, depth_range=[0, 1500]):
        """深度图像预处理"""
        depth_float32 = depth.astype(np.float32)
        depth_float32 = torch.tensor(depth_float32, dtype=torch.float32).clamp(*depth_range).unsqueeze(0)
        max_depth = depth_float32.max()
        min_depth = depth_float32.min()
        depth_normalized = (depth_float32 - min_depth) / (max_depth - min_depth + 1e-9)
        return depth_normalized

    # ===== Callback 函数群 =====
    def _h265_decoder_for(self, key: str):
        dec = self._h265_decoders.get(key)
        if dec is None:
            from kuavo_rl.h265_decode import H265StreamDecoder

            dec = H265StreamDecoder()
            self._h265_decoders[key] = dec
        return dec

    def rgb_callback(self, msg: CompressedImage, key: str, handle: dict):
        """JPEG compressed or KuavoBrain v3 H.265 CompressedImage."""
        from kuavo_rl.h265_decode import is_h265_compressed

        fmt = getattr(msg, "format", None)
        if is_h265_compressed(fmt):
            rgb = self._h265_decoder_for(key).decode(bytes(msg.data))
            if rgb is None:
                return
            cv_img = rgb
        else:
            img_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                # Topic may be h265_stream without format set — try HEVC.
                rgb = self._h265_decoder_for(key).decode(bytes(msg.data))
                if rgb is None:
                    return
                cv_img = rgb
            else:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        if resize_wh:
            cv_img = cv2.resize(cv_img, resize_wh)
        data = self.img_preprocess(cv_img)
        self._append_data(key, data, msg.header.stamp.to_sec())

    def rgb_raw_callback(self, msg: Image, key: str, handle: dict):
        """Decode sensor_msgs/Image from mujoco publish_camera:=true."""
        encoding = (msg.encoding or "").lower()
        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }.get(encoding)
        if channels is None:
            raise ValueError(f"unsupported Image encoding: {msg.encoding!r}")
        cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
        if encoding == "bgr8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif encoding == "bgra8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
        elif encoding == "rgba8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
        elif encoding == "mono8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        if resize_wh:
            cv_img = cv2.resize(cv_img, resize_wh)
        data = self.img_preprocess(cv_img)
        self._append_data(key, data, msg.header.stamp.to_sec())

    def depth_callback(self, msg: CompressedImage, key: str, handle: dict):
        png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        idx = msg.data.find(png_magic)
        if idx == -1:
            raise ValueError("Invalid depth message, PNG header not found")
        np_arr = np.frombuffer(msg.data[idx:], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        if resize_wh:
            image = cv2.resize(image, resize_wh)
        image = image[np.newaxis, ...]
        data = self.depth_preprocess(image, depth_range=handle.get("params", {}).get("depth_range", [0, 1500]))
        self._append_data(key, data, msg.header.stamp.to_sec())

    def sensorsData_callback(self, msg: sensorsData, key: str, handle = dict):
        # Float64Array ()
        joint = msg.joint_data.joint_q
        timestamp = msg.header.stamp.to_sec()

        # FK 计算需要双臂的14个关节（索引12-26）
        # 计算依赖于此数据源的观测（例如 eef_pose）
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        arm_joints = joint[arm_start:arm_end]  # 提取双臂关节

        self.compute_dependent_obs(key, arm_joints, timestamp)

        slice_value = handle.get("params", {}).get("slice", None)  
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, timestamp)

    def lejuClawState_callback(self, msg: lejuClawState, key: str, handle = dict):
        # Float64Array ()
        joint = msg.data.position
        slice_value = handle.get("params", {}).get("slice", None)  
        joint = [x / 100 for slc in slice_value for x in joint[slc[0]:slc[1]]] # 注意缩放
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    def qiangnaoState_callback(self, msg: JointState, key: str, handle = dict):
        joint = msg.position
        joint = [figure / 100 for figure in joint]
        slice_value = handle.get("params", {}).get("slice", None)
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    def rq2f85State_callback(self, msg: JointState, key: str, handle = dict):
        joint = msg.position
        joint = [figure / 0.8 for figure in joint]
        slice_value = handle.get("params", {}).get("slice", None)
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    # ===== 公共方法 =====
    def _append_data(self, key, data, timestamp):
        self.obs_buffer_data[key]["data"].append(data)
        self.obs_buffer_data[key]["timestamp"].append(timestamp)

    def compute_dependent_obs(self, source_key, source_data, timestamp):
        for comp_key in self.source_to_computed.get(source_key, []):
            func = self.compute_func_map.get(comp_key)
            if not func:
                log_robot.warning(f"No compute function for {comp_key}")
                continue
            try:
                data = func(source_data, which_arm=self.which_arm)
                if data is not None:
                    self._append_data(comp_key, data, timestamp)
            except Exception as e:
                log_robot.error(f"Error computing {comp_key} from {source_key}: {e}")

    def obs_buffer_is_ready(self):
        return all(len(self.obs_buffer_data[k]["data"]) == self.obs_key_map[k]["frequency"] for k in self.obs_key_map)

    def stop_subscribers(self):
        self.ros_manager.close()

    def wait_buffer_ready(self, *, idle_seed_after_s: float = 3.0):
        progress = {k: 0 for k in self.obs_key_map}
        total = {k: v["frequency"] for k, v in self.obs_key_map.items()}
        last_log_time = 0.0
        started = time.time()
        seeded = set()

        while not self.obs_buffer_is_ready():
            if not self.control_signal_manager.check_control_signals():
                log_robot.info("🛑 停止信号已接收，退出")
                sys.exit(1)

            now = time.time()
            if now - started >= idle_seed_after_s:
                for k in self.obs_key_map:
                    if k in seeded:
                        continue
                    if k not in _IDLE_SEED_KEYS:
                        continue
                    if len(self.obs_buffer_data[k]["data"]) > 0:
                        continue
                    # both-arm leju_claw / rq2f85 → 2 values in [0, 1]
                    fill = [0.0, 0.0]
                    need = int(total[k])
                    ts = now
                    for _ in range(need):
                        self._append_data(k, list(fill), ts)
                    seeded.add(k)
                    log_robot.warning(
                        f"No messages for '{k}' after {idle_seed_after_s:.0f}s; "
                        f"seeded {need} zero frames so collect can start"
                    )

            if now - last_log_time > 2.0:
                logs = []
                for k in progress:
                    new_len = len(self.obs_buffer_data[k]["data"])
                    progress[k] = new_len
                    if new_len < total[k]:
                        logs.append(f"{k}: {new_len}/{total[k]}")
                if logs:
                    log_robot.info("waiting buffers: " + " | ".join(logs))
                last_log_time = now

            time.sleep(0.1)

        log_robot.info("✅ All buffers ready!")
        return True

    def get_latest_obs(self):
        obs = {}
        for k, buf in self.obs_buffer_data.items():
            obs[k] = list(buf["data"])[-1]  # 取最新一帧
        return obs

    def get_aligned_obs(self, reference_keys=["/cam_h/color/image_raw/compressed"], max_dt=0.01, ratio=1.0):
        """
        返回各观测时间上对齐的最新帧
        reference_keys: 以哪些key作为时间参考，默认 None -> 所有 key 最小的最新时间戳
        max_dt: 最大允许时间偏差（秒），超出则返回 None
        """
        # ===== 获取参考时间戳 =====
        if reference_keys:
            # 用指定 key 的最新时间戳
            ref_times = []
            for k in reference_keys:
                buf = self.obs_buffer_data[k]
                if len(buf) == 0:
                    continue
                _, ts = buf["data"], buf["timestamp"]
                ref_times.append(ts[-1])
            if not ref_times:
                return None
            ref_time = min(ref_times)  # 也可以取 min 或 max，根据需要
        else:
            # 没有指定 key -> 所有观测的最新 timestamp 的最小值
            last_timestamps = []
            for buf in self.obs_buffer_data.values():
                if len(buf) == 0:
                    continue
                _, ts = buf["data"], buf["timestamp"]
                last_timestamps.append(ts[-1])
            if not last_timestamps:
                return None
            ref_time = np.min(last_timestamps)

        # ===== 对齐各观测 =====
        aligned_obs = {}
        for k, buf in self.obs_buffer_data.items():
            n = int(len(buf["data"]) * ratio)
            data, ts = list(buf["data"])[-n:], list(buf["timestamp"])[-n:]
            ts = np.array(ts)
            if len(ts) == 0:
                aligned_obs[k] = None
                continue
            idx = np.argmin(np.abs(ts - ref_time))
            if abs(ts[idx] - ref_time) > max_dt:
                aligned_obs[k] = None
            else:
                aligned_obs[k] = data[idx]

        return aligned_obs

