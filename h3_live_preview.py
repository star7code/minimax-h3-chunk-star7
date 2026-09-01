import base64
import hashlib
import io
import logging
import os
import threading
import time
import urllib.request

import torch
import torch.nn.functional as F
from PIL import Image

import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sd
import comfy.utils
import folder_paths

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


LOG_PREFIX = "[H3 Live Preview - Star7]"
EVENT_NAME = "star7_h3_live_preview"
TAEH3_FILENAME = "taeh3.safetensors"
# Accept the official filename and the decoder-suffixed alias used by some
# ComfyUI model packs. The official name remains the preferred one.
TAEH3_FILENAMES = (
    "taeh3.safetensors",
    "taeh3_decoder.safetensors",
)
TAEH3_OFFICIAL_URL = (
    "https://raw.githubusercontent.com/madebyollin/taehv/"
    "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/"
    "safetensors/taeh3.safetensors"
)
TAEH3_URLS = (
    "https://hf-mirror.com/suanyu/taeh3-star7/resolve/main/taeh3.safetensors",
    TAEH3_OFFICIAL_URL,
    "https://huggingface.co/suanyu/taeh3-star7/resolve/main/taeh3.safetensors",
    f"https://ghproxy.net/{TAEH3_OFFICIAL_URL}",
    f"https://gh-proxy.com/{TAEH3_OFFICIAL_URL}",
)
# Compatibility name for integrations that imported the original constant.
TAEH3_URL = TAEH3_OFFICIAL_URL
TAEH3_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"


class TAEH3BackgroundDownload:
    def __init__(self):
        self._lock = threading.Lock()
        self._started = False
        self._done = False
        self._error = None

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        logging.info("%s taeh3.safetensors missing; background download started", LOG_PREFIX)
        threading.Thread(
            target=self._run,
            name="star7_taeh3_download",
            daemon=True,
        ).start()

    def state(self):
        with self._lock:
            return self._done, self._error

    def _run(self):
        failures = []
        try:
            directories = folder_paths.get_folder_paths("vae_approx")
            if not directories:
                raise RuntimeError("ComfyUI has no models/vae_approx directory")
            directory = directories[0]
            os.makedirs(directory, exist_ok=True)
            final_path = os.path.join(directory, TAEH3_FILENAME)

            # Race the preferred domestic mirror and the official source. Each
            # attempt has its own temporary file; only the first SHA-256-valid
            # result is atomically moved into place.
            winner = threading.Event()
            winner_lock = threading.Lock()

            def download_one(index, url):
                part_path = f"{final_path}.star7-download-{index}"
                try:
                    digest = hashlib.sha256()
                    with urllib.request.urlopen(url, timeout=30) as response, open(
                        part_path, "wb"
                    ) as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            digest.update(chunk)
                    actual = digest.hexdigest().lower()
                    if actual != TAEH3_SHA256:
                        raise RuntimeError(
                            f"SHA256 mismatch (got {actual[:12]}…)"
                        )
                    with winner_lock:
                        if winner.is_set():
                            return False
                        os.replace(part_path, final_path)
                        winner.set()
                    logging.info("%s decoder downloaded from %s", LOG_PREFIX, url)
                    return True
                except Exception as error:
                    failures.append(f"{url}: {error}")
                    return False
                finally:
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass

            preferred = list(TAEH3_URLS[:2])
            workers = []
            for index, url in enumerate(preferred):
                worker = threading.Thread(
                    target=download_one, args=(index, url), daemon=True
                )
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()

            # If both preferred sources failed, try the remaining mirrors in
            # priority order before reporting a non-critical preview failure.
            if not winner.is_set():
                for index, url in enumerate(TAEH3_URLS[2:], start=2):
                    if download_one(index, url):
                        break
                else:
                    raise RuntimeError(
                        "all download sources failed; " + "; ".join(failures)
                    )
            with self._lock:
                self._done = True
            logging.info("%s taeh3.safetensors background download completed", LOG_PREFIX)
        except Exception as error:
            with self._lock:
                self._error = str(error)
            logging.warning("%s taeh3.safetensors download failed: %s", LOG_PREFIX, error)


_TAEH3_DOWNLOAD = TAEH3BackgroundDownload()


def temporal_indices(length, count):
    count = min(int(count), int(length))
    if count <= 0:
        return []
    if count == 1:
        return [0]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def preview_latent_size(height, width, resolution):
    longest = max(int(height), int(width))
    target = max(1, int(resolution) // 16)
    scale = min(1.0, target / longest)
    return max(1, round(height * scale)), max(1, round(width * scale))


def extract_video_latent(x0, latent_shapes):
    if isinstance(x0, comfy.nested_tensor.NestedTensor):
        streams = x0.unbind()
        video = streams[0] if streams else None
    elif latent_shapes and len(latent_shapes) > 1:
        video = comfy.utils.unpack_latents(x0, latent_shapes)[0]
    else:
        video = x0
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != 24:
        shape = getattr(video, "shape", None)
        raise ValueError(f"expected H3 video latent [B, 24, T, H, W], got {shape}")
    return video


class LatestPreviewWorker:
    def __init__(self, node_id, run_id):
        self.node_id = node_id
        self.run_id = run_id
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._failed = False
        self._warned = False
        self._thread = threading.Thread(target=self._run, name="star7_h3_preview", daemon=True)
        self._thread.start()

    def submit(self, frames, step, total_steps):
        with self._condition:
            if self._closed or self._failed:
                return
            self._pending = (frames, step, total_steps)
            self._condition.notify()

    @property
    def failed(self):
        with self._condition:
            return self._failed

    def finish(self):
        with self._condition:
            self._closed = True
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    return
                job = self._pending
                self._pending = None
            try:
                self._encode_and_send(*job)
            except Exception as error:
                with self._condition:
                    self._failed = True
                    self._pending = None
                if not self._warned:
                    self._warned = True
                    logging.warning("%s preview encoding/transport disabled: %s", LOG_PREFIX, error)
                return

    def _encode_and_send(self, frames, step, total_steps):
        images = [Image.fromarray(frame, mode="RGB") for frame in frames]
        buffer = io.BytesIO()
        images[0].save(
            buffer,
            format="WEBP",
            save_all=True,
            append_images=images[1:],
            duration=167,
            loop=0,
            quality=76,
            method=1,
        )
        payload = {
            "node_id": self.node_id,
            "run_id": self.run_id,
            "step": step,
            "total": total_steps,
            "width": images[0].width,
            "height": images[0].height,
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        PromptServer.instance.send_sync(EVENT_NAME, payload, PromptServer.instance.client_id)


def send_status(node_id, run_id, status, message=None, total=None):
    if PromptServer is None or node_id is None:
        return
    payload = {"node_id": node_id, "run_id": run_id, "status": status}
    if message is not None:
        payload["message"] = message
    if total is not None:
        payload["total"] = total
    PromptServer.instance.send_sync(EVENT_NAME, payload, PromptServer.instance.client_id)


def load_taeh3():
    paths = []
    for filename in TAEH3_FILENAMES:
        path = folder_paths.get_full_path("vae_approx", filename)
        if path is not None and path not in paths:
            paths.append(path)
    if not paths:
        names = " or ".join(TAEH3_FILENAMES)
        raise FileNotFoundError(f"{names} not found in models/vae_approx")

    last_error = None
    for path in paths:
        try:
            state_dict = comfy.utils.load_torch_file(path, safe_load=True)
            vae = comfy.sd.VAE(sd=state_dict)
            vae.throw_exception_if_invalid()
            if vae.latent_channels != 24 or vae.first_stage_model.__class__.__name__ != "TAEHV":
                raise ValueError("decoder is not a 24-channel TAEHV decoder")
            return vae
        except Exception as error:
            last_error = error
    raise ValueError(f"no compatible 24-channel TAEHV decoder found ({last_error})")


def load_taeh3_or_start_download():
    try:
        return load_taeh3(), None
    except FileNotFoundError:
        _TAEH3_DOWNLOAD.start()
        done, error = _TAEH3_DOWNLOAD.state()
        if error is not None:
            raise RuntimeError(f"taeh3.safetensors automatic download failed: {error}")
        if done:
            return load_taeh3(), None
        return None, "Downloading taeh3.safetensors in the background"


def decode_preview(vae, video, frame_count, resolution):
    indices = temporal_indices(video.shape[2], frame_count)
    index = torch.tensor(indices, device=video.device, dtype=torch.long)

    # index_select makes an independent allocation. The sampler-owned x0 and audio stream
    # remain untouched even though the TAE decoder uses in-place activations internally.
    selected = video[:1].index_select(2, index).clone()
    height, width = preview_latent_size(selected.shape[-2], selected.shape[-1], resolution)
    frames = selected.movedim(2, 1).reshape(len(indices), 24, selected.shape[-2], selected.shape[-1])
    if frames.shape[-2:] != (height, width):
        frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    frames = frames.unsqueeze(2).contiguous()

    memory_required = vae.memory_used_decode(frames.shape, vae.vae_dtype)
    loaded = comfy.model_management.loaded_models(only_currently_used=True)
    loaded.append(vae.patcher)
    comfy.model_management.load_models_gpu(loaded, memory_required=memory_required)

    rgb = vae.decode(frames)
    if rgb.ndim == 5:
        rgb = rgb[:, rgb.shape[1] // 2]
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"TAEH3 returned unexpected shape {tuple(rgb.shape)}")
    return rgb.detach().float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()


class H3LivePreviewWrapper:
    def __init__(self, preview_frames, preview_resolution, first_step_only, node_id):
        self.preview_frames = int(preview_frames)
        self.preview_resolution = int(preview_resolution)
        self.first_step_only = bool(first_step_only)
        self.node_id = str(node_id) if node_id is not None else None

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        run_id = str(time.monotonic_ns())
        total = max(0, len(sigmas) - 1)
        worker = None
        vae = None
        enabled = True
        warned = False
        preview_sent = False

        try:
            if PromptServer is None:
                raise RuntimeError("ComfyUI PromptServer is unavailable")
            worker = LatestPreviewWorker(self.node_id, run_id)
            vae, pending_message = load_taeh3_or_start_download()
            send_status(
                self.node_id,
                run_id,
                "downloading" if vae is None else "start",
                pending_message,
                total,
            )
        except Exception as error:
            enabled = False
            warned = True
            logging.warning("%s %s", LOG_PREFIX, error)
            try:
                send_status(self.node_id, run_id, "error", str(error), total)
            except Exception:
                pass

        def preview_callback(step, x0, x, total_steps):
            nonlocal enabled, warned, preview_sent, vae
            if callback is not None:
                callback(step, x0, x, total_steps)
            if not enabled:
                return
            if self.first_step_only and step > 0:
                return
            final_step = step >= total_steps - 1
            # Normally the final step is left to the regular output path.  On
            # first install, however, a decoder that finishes downloading only
            # at the final callback may emit one useful preview because no
            # earlier preview exists to duplicate.
            if final_step and preview_sent:
                return
            if worker.failed:
                enabled = False
                return
            try:
                if vae is None:
                    vae, pending_message = load_taeh3_or_start_download()
                    if vae is None:
                        return
                    send_status(self.node_id, run_id, "start", total=total_steps)
                video = extract_video_latent(x0, latent_shapes)
                frames = decode_preview(vae, video, self.preview_frames, self.preview_resolution)
                worker.submit(frames, step + 1, total_steps)
                preview_sent = True
            except Exception as error:
                enabled = False
                if not warned:
                    warned = True
                    logging.warning("%s preview disabled: %s", LOG_PREFIX, error)
                try:
                    send_status(self.node_id, run_id, "error", str(error), total_steps)
                except Exception:
                    pass

        try:
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                preview_callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            if worker is not None:
                worker.finish()


class MiniMaxH3LivePreviewStar7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preview_frames": ("INT", {"default": 25, "min": 4, "max": 64, "step": 1}),
                "preview_resolution": (["256", "384", "512"], {"default": "512"}),
                "first_step_only": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = "Shows a lightweight TAEH3 animation across the full H3 video timeline after each eligible sampling step."

    def patch(
        self,
        model,
        preview_frames,
        preview_resolution,
        first_step_only=False,
        unique_id=None,
    ):
        # Some older frontend builds briefly serialize a fresh INT widget as zero.
        # Keep the backend default safe without changing valid saved workflows.
        preview_frames = int(preview_frames)
        if preview_frames < 4:
            preview_frames = 25
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "star7_h3_live_preview",
            H3LivePreviewWrapper(
                preview_frames,
                preview_resolution,
                first_step_only,
                unique_id,
            ),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3LivePreviewStar7": MiniMaxH3LivePreviewStar7}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3LivePreviewStar7": "MiniMax H3 Live Preview - Star7"}
