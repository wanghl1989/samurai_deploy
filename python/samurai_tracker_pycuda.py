import os

import cv2
import numpy as np
import onnx
import pycuda.driver as cuda
import scipy.linalg
import tensorrt as trt

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric="maha"):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")

    def compute_iou(self, pred_bbox, bboxes):
        """
        Compute the IoU between the bbox and the bboxes
        """
        ious = []
        pred_bbox = self.xyah_to_xyxy(pred_bbox)
        for bbox in bboxes:
            iou = self._compute_iou(pred_bbox, bbox)
            ious.append(iou)
        return ious

    def _compute_iou(self, bbox1, bbox2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bbox1 : list
            The first bounding box in the format [x1, y1, x2, y2].
        bbox2 : list
            The second bounding box in the format [x1, y1, x2, y2].
        Returns
        -------
        float
            The IoU of the two bounding boxes.
        """
        if bbox2 == [0, 0, 0, 0]:
            return 0
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        # Calculate intersection area
        intersection_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(
            0, min(y2, y2_) - max(y1, y1_)
        )
        # Calculate union area
        union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection_area
        # Calculate IoU
        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def xyxy_to_xyah(self, bbox):
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            h = 1
        return [xc, yc, w / h, h]

    def xyah_to_xyxy(self, bbox):
        xc, yc, a, h = bbox
        x1 = xc - a * h / 2
        y1 = yc - h / 2
        x2 = xc + a * h / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]


class SAM2TrackerTRT:
    def __init__(
        self,
        onnx_model_path: str | None,
        trt_engine_path: str | None = None,
        use_fp16: bool = False,
    ):
        self.onnx_file_prefix = onnx_model_path
        self.trt_engine_prefix = trt_engine_path
        self.fp16_mode = use_fp16

        # Create a TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # load and deserialize TRT engine
        self.engines = self.init_engine()

        self.buffers = []
        self.contexts = []
        for engine in self.engines:
            profile_idx = range(engine.num_optimization_profiles)[0]
            inputs, outputs, bindings, stream = self.allocate_buffers(
                engine, profile_idx=profile_idx
            )
            context = engine.create_execution_context()
            self.buffers.append((inputs, outputs, bindings, stream))
            self.contexts.append(context)

        self.image_size = 512
        self.video_W, self.video_H = 0, 0

        self.stable_frames_threshold = 15
        self.stable_ious_threshold = 0.3
        self.kf_score_weight = 0.25
        self.memory_bank_iou_threshold = 0.5
        self.memory_bank_obj_score_threshold = 0.0
        self.memory_bank_kf_score_threshold = 0.0
        self.max_obj_ptrs_in_encoder = 16
        self.num_maskmem = 7

        self.init_state()

    def reset(self):
        self.init_state()

    def init_state(self):
        self.maskmem_tpos_enc = None
        self.memory_bank = {}
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0
        self.frame_idx = 0

    def init_video(self, video_width, video_height):
        self.video_H = video_height
        self.video_W = video_width

    def init_engine(self):
        """
        Initialize TensorRT engines from ONNX files.
        """
        # Check if the engine file exists
        onnx_models = [
            "image_encoder.onnx",
            "memory_attention.onnx",
            "memory_encoder.onnx",
            "mask_decoder.onnx",
        ]
        trt_engines = [
            model.replace(".onnx", "_fp16.engine")
            if self.fp16_mode
            else model.replace(".onnx", ".engine")
            for model in onnx_models
        ]
        engines = []

        if self.trt_engine_prefix is not None:
            for i, engine in enumerate(trt_engines):
                engine_path = os.path.join(self.trt_engine_prefix, engine)
                if not os.path.exists(engine_path):
                    raise FileNotFoundError("The {} engine file does not exist!".format(engine))
                else:
                    print("loading the {} ...".format(engine_path))
                engines.append(self.load_engine(engine_path))
        elif self.onnx_file_prefix is not None:
            for i, model in enumerate(onnx_models):
                onnx_model_path = os.path.join(self.onnx_file_prefix, model)
                engine_path = os.path.join(self.onnx_file_prefix, trt_engines[i])
                if os.path.exists(engine_path):
                    print("The {} fonud, skip building!".format(engine_path))
                    engines.append(self.load_engine(engine_path))
                    continue

                # check onnx model
                onnx_model = onnx.load(onnx_model_path)
                onnx.checker.check_model(onnx_model)
                print("The {} model is valid! building the engine...".format(onnx_model_path))

                engine = self.build_engine(onnx_model_path, self.fp16_mode)
                self.save_engine(engine, engine_path)
                print("The {} is saved!".format(engine_path))

                engines.append(self.load_engine(engine_path))
        else:
            print("Please specify the path to the TRT engine or ONNX model!")

        return engines

    def build_engine(self, onnx_file_path, fp16_mode=False):
        """
        Build a TensorRT engine from an ONNX file.
        """
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with (
            trt.Builder(self.trt_logger) as builder,
            builder.create_network(EXPLICIT_BATCH) as network,
            trt.OnnxParser(network, self.trt_logger) as parser,
        ):
            # Parse the ONNX model
            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # Configure the builder (optional settings)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 1 << 30
            )  # Set workspace size to 1GB

            # Optionally, set other configurations like precision
            # For example, to use FP16 precision:
            if builder.platform_has_fast_fp16 and fp16_mode:
                # config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.BF16)
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

            if "memory_attention" in onnx_file_path:
                # 设置动态输入尺寸
                profile = builder.create_optimization_profile()
                input_info = network.get_input

                # for i in range(network.num_inputs):
                #     print("Input {} {}: {}".format(i, input_info(i).name, input_info(i).shape))

                input_shape = input_info(2).shape
                profile.set_shape(
                    "maskmem_feats",
                    (input_shape[0], 1, input_shape[2], input_shape[3]),  # 最小尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]),  # 优化尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]),
                )  # 最大尺寸

                input_shape = input_info(3).shape
                profile.set_shape(
                    "memory_pos_embed",
                    (input_shape[0], 1, input_shape[2], input_shape[3]),  # 最小尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]),  # 优化尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]),
                )  # 最大尺寸

                input_shape = input_info(4).shape
                profile.set_shape(
                    "obj_ptrs",
                    (1, input_shape[1], input_shape[2]),  # 最小尺寸
                    (16, input_shape[1], input_shape[2]),  # 优化尺寸
                    (16, input_shape[1], input_shape[2]),
                )  # 最大尺寸

                input_shape = input_info(5).shape
                profile.set_shape(
                    "obj_pos",
                    (1,),  # 最小尺寸
                    (16,),  # 优化尺寸
                    (16,),
                )  # 最大尺寸
                config.add_optimization_profile(profile)

            # Build the engine
            engine = builder.build_serialized_network(network, config)

            return engine

    def save_engine(self, engine, engine_file_path):
        """
        Serialize the engine and save it to a file.
        """
        with open(engine_file_path, "wb") as f:
            f.write(engine)

    def load_engine(self, engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, engine, context=None, profile_idx=None):
        """
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        """
        inputs_buffer = []
        outputs_buffer = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)

            if context is not None:
                shape = context.get_tensor_shape(tensor_name)
            elif profile_idx is not None:
                shape = engine.get_tensor_profile_shape(tensor_name, profile_idx)[-1]
            else:
                shape = engine.get_tensor_shape(tensor_name)

            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(
                    f'Binding "{tensor_name}" has dynamic shape, but no profile was specified.'
                )

            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(
                size, dtype
            )  # page-locked memory buffer (won't swapped to disk)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. When cast to int, it's a linear index into the context's memory (like memory address). See https://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list.
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs_buffer.append(
                    {
                        "name": tensor_name,
                        "host_mem": host_mem,
                        "device_mem": device_mem,
                        "shape": shape,
                    }
                )
            else:
                outputs_buffer.append(
                    {
                        "name": tensor_name,
                        "host_mem": host_mem,
                        "device_mem": device_mem,
                        "shape": shape,
                    }
                )

        return inputs_buffer, outputs_buffer, bindings, stream

    def inference(self, input_datas, engine, context, buffers):
        """
        Perform inference on the TensorRT engine.
        """

        inputs_buffer, outputs_buffer, bindings, stream = buffers

        for i in range(len(inputs_buffer)):
            np.copyto(inputs_buffer[i]["host_mem"], input_datas[i].ravel())

        # 数据传输与执行
        [
            cuda.memcpy_htod_async(
                inputs_buffer[i]["device_mem"], inputs_buffer[i]["host_mem"], stream
            )
            for i in range(len(inputs_buffer))
        ]

        for i in range(engine.num_io_tensors):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

        context.execute_async_v3(stream_handle=stream.handle)

        [
            cuda.memcpy_dtoh_async(
                outputs_buffer[i]["host_mem"], outputs_buffer[i]["device_mem"], stream
            )
            for i in range(len(outputs_buffer))
        ]

        stream.synchronize()

        outputs_data = [
            output["host_mem"].reshape(output["shape"]).copy() for output in outputs_buffer
        ]

        return outputs_data

    def image_encoder_inference(self, input_image):
        """
        Image encoder inference.
        """
        return self.inference(
            [
                input_image,
            ],
            self.engines[0],
            self.contexts[0],
            self.buffers[0],
        )

    def memory_attention_inference(self, frame_idx, vision_feats, vision_pos):
        """
        Memory attention inference.
        """
        memmask_features = [self.memory_bank[0]["maskmem_features"].copy()]
        memmask_pos_enc = [self.memory_bank[0]["maskmem_pos_enc"] + self.maskmem_tpos_enc[6]]
        object_ptrs = [self.memory_bank[0]["obj_ptr"].copy()]
        ## samurai----------------------------------------------- ##
        valid_indices = []
        if frame_idx > 1:
            for i in range(frame_idx - 1, 0, -1):  # Iterate backwards through previous frames
                # Check the number of valid indices
                if len(valid_indices) >= self.max_obj_ptrs_in_encoder - 1:
                    break
                iou_score = self.memory_bank[i]["best_iou_score"]  # Get mask affinity score
                obj_score = self.memory_bank[i]["obj_score_logits"]  # Get object score
                kf_score = self.memory_bank[i]["kf_score"]  # Get motion score if available
                # Check if the scores meet the criteria for being a valid index
                if (
                    iou_score > self.memory_bank_iou_threshold
                    and obj_score > self.memory_bank_obj_score_threshold
                    and (kf_score is None or kf_score > self.memory_bank_kf_score_threshold)
                ):
                    valid_indices.insert(0, i)
                # valid_indices.insert(0, i)

        # print("valid_indices: ", valid_indices, '\nprev_frame_idx : ', end='')
        # 最近6帧的memmask_features
        for prev_frame_idx in valid_indices[::-1]:
            # print(prev_frame_idx, end=', ')
            mem = self.memory_bank.get(prev_frame_idx, None)
            if mem is not None:
                memmask_features.insert(1, mem["maskmem_features"].copy())
                memmask_pos_enc.insert(1, mem["maskmem_pos_enc"].copy())
            if len(memmask_features) >= self.num_maskmem:
                break
        # print()
        ## samurai----------------------------------------------- ##

        obj_pos_enc = np.arange(1, frame_idx)[:15]
        obj_pos_enc = np.insert(obj_pos_enc, 0, frame_idx).astype(np.int32)
        obj_pos_enc = obj_pos_enc[: self.max_obj_ptrs_in_encoder]

        for i in range(frame_idx - 15, frame_idx):
            if i < 1:
                continue
            if len(object_ptrs) >= self.max_obj_ptrs_in_encoder:
                break
            mem = self.memory_bank.get(i, None)
            if mem is not None:
                # print(i, end=', ')
                object_ptrs.append(mem["obj_ptr"].copy())

        for i, pos_enc in enumerate(reversed(memmask_pos_enc[1:])):
            pos_enc[:] = pos_enc[:] + self.maskmem_tpos_enc[i]

        memory = np.concatenate(memmask_features, axis=0)
        memory_pos_embed = np.concatenate(memmask_pos_enc, axis=0)
        memory = memory.reshape(-1, len(memmask_features), memory.shape[-2], memory.shape[-1])
        memory_pos_embed = memory_pos_embed.reshape(
            -1, len(memmask_pos_enc), memory_pos_embed.shape[-2], memory_pos_embed.shape[-1]
        )

        object_ptrs = object_ptrs[0:1] + object_ptrs[1:][::-1]
        object_ptrs = np.stack(object_ptrs, axis=0)

        self.contexts[1].set_input_shape("maskmem_feats", memory.shape)
        self.contexts[1].set_input_shape("memory_pos_embed", memory_pos_embed.shape)
        self.contexts[1].set_input_shape("obj_ptrs", object_ptrs.shape)
        self.contexts[1].set_input_shape("obj_pos", obj_pos_enc.shape)
        self.buffers[1] = self.allocate_buffers(self.engines[1], context=self.contexts[1])

        memory_attention_outputs = self.inference(
            [vision_feats, vision_pos, memory, memory_pos_embed, object_ptrs, obj_pos_enc],
            self.engines[1],
            self.contexts[1],
            self.buffers[1],
        )

        return memory_attention_outputs

    def memory_encoder_inference(
        self, vision_feats, high_res_feats, obj_score_logits, isMaskFromPts
    ):
        """
        Memory encoder inference.
        """
        return self.inference(
            [vision_feats, high_res_feats, obj_score_logits, isMaskFromPts],
            self.engines[2],
            self.contexts[2],
            self.buffers[2],
        )

    def mask_decoder_inference(
        self, input_points, input_labels, pixel_feat_with_memory, high_res_feats0, high_res_feats1
    ):
        """
        Mask decoder inference.
        """
        return self.inference(
            [input_points, input_labels, pixel_feat_with_memory, high_res_feats0, high_res_feats1],
            self.engines[3],
            self.contexts[3],
            self.buffers[3],
        )

    def _add_first_frame_prompts(self, image, prompt_coords, prompt_labels):
        self.init_state()

        input_image = cv2.resize(image, (self.image_size, self.image_size))
        input_image = input_image.astype(np.float32)[np.newaxis, ...]

        image_encoder_outputs = self.image_encoder_inference(input_image)
        high_res_features0, high_res_features1, low_res_features, _, pix_feat_with_mem = (
            image_encoder_outputs
        )

        mask_decoder_outputs = self.mask_decoder_inference(
            prompt_coords, prompt_labels, pix_feat_with_mem, high_res_features0, high_res_features1
        )
        _low_res_multimasks, ious, obj_ptrs, object_score_logits, self.maskmem_tpos_enc = (
            mask_decoder_outputs
        )

        pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score = self._forward_sam_head(
            mask_decoder_outputs
        )

        # memory_encoder predict
        is_mask_from_pts = np.array([self.frame_idx == 0]).astype(bool)
        memory_encoder_outputs = self.memory_encoder_inference(
            low_res_features, high_res_masks_for_mem, object_score_logits, is_mask_from_pts
        )
        maskmem_features, maskmem_pos_enc = memory_encoder_outputs

        # save to memory bank
        self.memory_bank[0] = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "obj_ptr": obj_ptrs[0, best_iou_inds],
            "best_iou_score": ious[0, best_iou_inds],
            "obj_score_logits": object_score_logits,
            "kf_score": kf_score,
        }

        mask = pred_mask.squeeze()
        mask = cv2.resize(mask, (self.video_W, self.video_H))
        return mask

    def add_first_frame_points(self, image, first_frame_points):
        """
        Add the first points when the frame_idx is 0.
        """
        point_coords = np.array(first_frame_points).reshape((1, 2, -1))

        if point_coords.shape[-1] >= 2:
            point_coords = point_coords[..., :2]
        else:
            point_coords = np.concatenate([point_coords, point_coords], axis=2)

        point_labels = np.array([1, 1]).reshape((1, 2))

        point_coords = point_coords / np.array([self.video_W, self.video_H])

        point_coords = (point_coords * self.image_size).astype(np.float32)
        point_labels = point_labels.astype(np.int32)
        return self._add_first_frame_prompts(image, point_coords, point_labels)

    def add_first_frame_bbox(self, image, first_frame_bbox):
        """
        Add the first bbox when the frame_idx is 0.
        """
        box_coords = np.array(first_frame_bbox).reshape((1, 2, 2))
        box_labels = np.array([2, 3]).reshape((1, 2))

        box_coords = box_coords / np.array([self.video_W, self.video_H])

        box_coords = (box_coords * self.image_size).astype(np.float32)
        box_labels = box_labels.astype(np.int32)
        return self._add_first_frame_prompts(image, box_coords, box_labels)

    def track_step(self, frame_idx, image):

        # step 1:image_encoder predict, get image feature

        self.frame_idx += 1
        frame_idx = self.frame_idx

        input_image = cv2.resize(image, (self.image_size, self.image_size))
        input_image = input_image.astype(np.float32)[np.newaxis, ...]

        image_encoder_outputs = self.image_encoder_inference(input_image)
        high_res_features0, high_res_features1, low_res_features, vision_pos_embeds, _ = (
            image_encoder_outputs
        )

        # step 2: memory_attention predict
        memory_attention_outputs = self.memory_attention_inference(
            frame_idx, low_res_features, vision_pos_embeds
        )
        pix_feat_with_mem = memory_attention_outputs[0]

        # step 3 : mask decoder predict
        input_points = np.zeros((1, 2, 2), dtype=np.float32)
        input_labels = -np.ones((1, 2), dtype=np.int32)
        mask_decoder_outputs = self.mask_decoder_inference(
            input_points, input_labels, pix_feat_with_mem, high_res_features0, high_res_features1
        )
        _, ious, obj_ptrs, object_score_logits, _ = mask_decoder_outputs

        pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score = self._forward_sam_head(
            mask_decoder_outputs
        )

        # step 4 : memory_encoder predict, save maskmem to memory bank
        is_mask_from_pts = np.array([frame_idx == 0]).astype(bool)
        memory_encoder_outputs = self.memory_encoder_inference(
            low_res_features, high_res_masks_for_mem, object_score_logits, is_mask_from_pts
        )
        maskmem_features, maskmem_pos_enc = memory_encoder_outputs

        self.memory_bank[frame_idx] = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "obj_ptr": obj_ptrs[0, best_iou_inds],
            "best_iou_score": ious[0, best_iou_inds],
            "obj_score_logits": object_score_logits,
            "kf_score": kf_score,
        }

        mask = pred_mask.squeeze()
        mask = cv2.resize(mask, (self.video_W, self.video_H))
        return mask

    def _forward_sam_head(self, mask_decoder_outputs):
        low_res_multimasks, ious, _, _, _ = mask_decoder_outputs
        # high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        high_res_multimasks = cv2.resize(
            low_res_multimasks[0].transpose(1, 2, 0), (self.image_size, self.image_size)
        )
        high_res_multimasks = high_res_multimasks.transpose(2, 0, 1)[None, ...]

        ## samurai ---------------------------------------------------------------------##
        B = 1
        kf_ious = None
        if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
            best_iou_inds = np.argmax(ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            non_zero_indices = np.argwhere(high_res_masks[0][0] > 0.0)
            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                high_res_bbox = [x_min, y_min, x_max, y_max]
            self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))

            self.stable_frames += 1
        elif self.stable_frames < self.stable_frames_threshold:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            best_iou_inds = np.argmax(ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            non_zero_indices = np.argwhere(high_res_masks[0][0] > 0.0)
            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                high_res_bbox = [x_min, y_min, x_max, y_max]
            if ious[0][best_iou_inds] > self.stable_ious_threshold:
                self.kf_mean, self.kf_covariance = self.kf.update(
                    self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox)
                )
                self.stable_frames += 1
            else:
                self.stable_frames = 0
        else:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            high_res_multibboxes = []
            batch_inds = np.arange(B)
            for i in range(ious.shape[1]):
                non_zero_indices = np.argwhere(high_res_multimasks[batch_inds, i][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_multibboxes.append([0, 0, 0, 0])
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    high_res_multibboxes.append([x_min, y_min, x_max, y_max])
            # compute the IoU between the predicted bbox and the high_res_multibboxes
            kf_ious = np.array(self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes))
            # weighted iou
            weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
            best_iou_inds = np.argmax(weighted_ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]

            if ious[0][best_iou_inds] < self.stable_ious_threshold:
                self.stable_frames = 0
            else:
                self.kf_mean, self.kf_covariance = self.kf.update(
                    self.kf_mean,
                    self.kf_covariance,
                    self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds.item()]),
                )

        ## sam2 ---------------------------------------------------------------------##
        # best_iou_inds = np.argmax(ious, axis=-1)
        # batch_inds = np.arange(1)
        # low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
        # high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]

        kf_score = kf_ious[best_iou_inds] if kf_ious is not None else None
        pred_mask = low_res_masks
        high_res_masks_for_mem = high_res_masks

        return pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score


def load_txt(gt_path):
    with open(gt_path, "r") as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(","))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts
