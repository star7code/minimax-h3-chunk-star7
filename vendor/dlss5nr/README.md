# DLSS Neural Rendering bridge

This directory contains the small native bridge used by the Star7 ComfyUI node.
Its source and binaries are adapted from the MIT-licensed `ComfyUI-DLSS5-NR`
project; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.

The NVIDIA model runtime is not redistributed by this repository. Users place
exactly one file at:

`ComfyUI/models/upscale_models/nvngx_dlssnr.dll`

At runtime the node creates a private hard link in `runtime/` because NVIDIA NGX
expects the model beside the caller shim. A hard link does not duplicate the
model's disk contents. The private link is ignored by Git.

This single-model integration performs same-resolution Neural Rendering. When
the output scale exceeds 1.0, the node first resizes each frame with Lanczos and
then applies Neural Rendering at the target resolution; it does not claim native
DLSS Super Resolution.

V2 uses a versioned bridge DLL so an update can be installed while an older
bridge is still locked by a running ComfyUI process. Restart ComfyUI once after
updating to load V2. Its upload/readback conversion uses hardware F16C where
available and keeps the baseline scalar path as a compatibility fallback; this
changes transfer overhead, not the Neural Rendering result.
