import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3LivePreviewStar7";
const EVENT_NAME = "star7_h3_live_preview";
const PREVIEW_FRAME_DURATION_MS = 167;

function installPreviewScrubberStyle() {
    if (document.getElementById("star7-h3-preview-scrubber-style")) return;
    const style = document.createElement("style");
    style.id = "star7-h3-preview-scrubber-style";
    style.textContent = `
        .star7-h3-preview-media:hover .star7-h3-preview-scrubber,
        .star7-h3-preview-scrubber:focus-visible,
        .star7-h3-preview-scrubber.star7-dragging {
            opacity: 1;
        }
        .star7-h3-preview-scrubber {
            appearance: none;
            -webkit-appearance: none;
            opacity: 0;
            transition: opacity 120ms ease;
        }
        .star7-h3-preview-scrubber:disabled {
            display: none;
        }
        .star7-h3-preview-scrubber::-webkit-slider-runnable-track {
            height: 3px;
            border-radius: 3px;
            background: linear-gradient(
                to right,
                rgba(255,255,255,.92) 0 var(--star7-progress, 0%),
                rgba(255,255,255,.28) var(--star7-progress, 0%) 100%
            );
        }
        .star7-h3-preview-scrubber::-webkit-slider-thumb {
            appearance: none;
            -webkit-appearance: none;
            width: 9px;
            height: 9px;
            margin-top: -3px;
            border: 0;
            border-radius: 50%;
            background: #fff;
            box-shadow: 0 0 0 1px rgba(0,0,0,.35);
        }
        .star7-h3-preview-scrubber::-moz-range-track {
            height: 3px;
            border: 0;
            border-radius: 2px;
            background: rgba(255,255,255,.28);
        }
        .star7-h3-preview-scrubber::-moz-range-progress {
            height: 3px;
            border-radius: 2px;
            background: rgba(255,255,255,.92);
        }
        .star7-h3-preview-scrubber::-moz-range-thumb {
            width: 9px;
            height: 9px;
            border: 0;
            border-radius: 50%;
            background: #fff;
        }
    `;
    document.head.appendChild(style);
}

function localizeNode(node, chinese) {
    if (!chinese) return;
    node.title = "MiniMax H3 实时预览 - Star7";

    const modelInput = node.inputs?.find((input) => input.name === "model");
    const modelOutput = node.outputs?.find((output) => output.name === "model");
    if (modelInput) modelInput.label = "模型";
    if (modelOutput) modelOutput.label = "模型";

    const framesWidget = node.widgets?.find((widget) => widget.name === "preview_frames");
    const resolutionWidget = node.widgets?.find((widget) => widget.name === "preview_resolution");
    const firstStepWidget = node.widgets?.find((widget) => widget.name === "first_step_only");
    if (framesWidget) framesWidget.label = "时间轴采样帧数";
    if (resolutionWidget) resolutionWidget.label = "预览长边";
    if (firstStepWidget) firstStepWidget.label = "只显示第一步预览";
}

function findNode(graph, qualifiedId) {
    const parts = String(qualifiedId).split(":");
    let current = graph;
    for (let index = 0; index < parts.length - 1; index++) {
        const owner = current?.getNodeById?.(Number(parts[index]));
        if (!owner?.subgraph) return null;
        current = owner.subgraph;
    }
    return current?.getNodeById?.(Number(parts.at(-1))) ?? null;
}

api.addEventListener(EVENT_NAME, (event) => {
    const data = event.detail;
    const node = data?.node_id != null ? findNode(app.graph, data.node_id) : null;
    node?._star7H3Preview?.(data);
});

app.registerExtension({
    name: "Star7.MiniMaxH3LivePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const chinese = true;

            this.title = chinese ? "MiniMax H3 实时预览 - Star7" : "MiniMax H3 Live Preview - Star7";

            const originalConfigure = this.onConfigure;
            this.onConfigure = function () {
                const configureResult = originalConfigure?.apply(this, arguments);
                setTimeout(() => {
                    localizeNode(this, chinese);
                    this.setDirtyCanvas?.(true, true);
                }, 0);
                return configureResult;
            };

            setTimeout(() => {
                const framesWidget = this.widgets?.find((widget) => widget.name === "preview_frames");
                if (framesWidget && Number(framesWidget.value) < 4) framesWidget.value = 25;
                localizeNode(this, chinese);
                this.setDirtyCanvas?.(true, true);
            }, 250);

            const root = document.createElement("div");
            root.style.cssText = "width:100%;height:100%;min-height:210px;display:flex;flex-direction:column;background:#0b0b0b;border-radius:6px;overflow:hidden;";

            const status = document.createElement("div");
            status.textContent = chinese ? "等待采样…" : "Waiting for sampling…";
            status.style.cssText = "height:30px;box-sizing:border-box;padding:7px 10px;color:#d5d5d5;background:#191919;font:12px system-ui,sans-serif;";
            root.appendChild(status);

            const media = document.createElement("div");
            media.style.cssText = "position:relative;flex:1;min-height:180px;background:#050505;overflow:hidden;";
            media.className = "star7-h3-preview-media";
            root.appendChild(media);

            const placeholder = document.createElement("div");
            placeholder.textContent = chinese ? "首个采样步骤完成后显示动态预览" : "Animation appears after the first sampling step";
            placeholder.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;color:#777;text-align:center;font:12px system-ui,sans-serif;";
            media.appendChild(placeholder);

            const image = document.createElement("img");
            image.alt = "";
            image.draggable = false;
            image.style.cssText = "display:none;width:100%;height:100%;object-fit:contain;background:#050505;image-rendering:auto;";
            media.appendChild(image);

            const canvas = document.createElement("canvas");
            canvas.style.cssText = "display:none;width:100%;height:100%;object-fit:contain;background:#050505;image-rendering:auto;";
            media.appendChild(canvas);

            installPreviewScrubberStyle();
            const scrubber = document.createElement("input");
            scrubber.type = "range";
            scrubber.min = "0";
            scrubber.max = "0";
            scrubber.step = "1";
            scrubber.value = "0";
            scrubber.disabled = true;
            scrubber.className = "star7-h3-preview-scrubber";
            scrubber.setAttribute("aria-label", chinese ? "预览时间轴" : "Preview timeline");
            scrubber.style.cssText = "position:absolute;z-index:3;left:10px;right:10px;bottom:6px;width:calc(100% - 20px);height:14px;margin:0;padding:0;cursor:pointer;background:transparent;touch-action:none;";
            media.appendChild(scrubber);

            let objectUrl = null;
            let runId = null;
            let decodedFrames = [];
            let frameIndex = 0;
            let playbackTimer = null;
            let decodeGeneration = 0;
            let decodedPreviewGeneration = 0;
            let dragging = false;

            const closeFrames = (frames = decodedFrames) => {
                for (const frame of frames) frame?.close?.();
                if (frames === decodedFrames) decodedFrames = [];
            };
            const stopPlayback = () => {
                if (playbackTimer !== null) clearTimeout(playbackTimer);
                playbackTimer = null;
            };
            const updateScrubber = () => {
                const maximum = Math.max(0, decodedFrames.length - 1);
                scrubber.max = String(maximum);
                scrubber.value = String(Math.min(maximum, frameIndex));
                const percent = maximum > 0 ? frameIndex / maximum * 100 : 0;
                scrubber.style.setProperty("--star7-progress", `${percent}%`);
            };
            const drawFrame = (index) => {
                if (!decodedFrames.length) return;
                frameIndex = Math.min(decodedFrames.length - 1, Math.max(0, Number(index) || 0));
                const frame = decodedFrames[frameIndex];
                const context = canvas.getContext("2d", { alpha: false });
                if (!context) return;
                context.drawImage(frame, 0, 0, canvas.width, canvas.height);
                updateScrubber();
            };
            const schedulePlayback = () => {
                stopPlayback();
                if (dragging || decodedFrames.length < 2) return;
                playbackTimer = setTimeout(() => {
                    playbackTimer = null;
                    drawFrame((frameIndex + 1) % decodedFrames.length);
                    schedulePlayback();
                }, PREVIEW_FRAME_DURATION_MS);
            };
            const finishScrub = (event) => {
                if (!dragging) return;
                event?.stopPropagation?.();
                dragging = false;
                scrubber.classList.remove("star7-dragging");
                try {
                    if (event?.pointerId != null && scrubber.hasPointerCapture?.(event.pointerId)) {
                        scrubber.releasePointerCapture(event.pointerId);
                    }
                } catch (_) {
                    // Pointer capture is best-effort; playback recovery is not.
                }
                schedulePlayback();
            };
            scrubber.addEventListener("pointerdown", (event) => {
                if (!decodedFrames.length) return;
                event.stopPropagation?.();
                dragging = true;
                stopPlayback();
                scrubber.classList.add("star7-dragging");
                scrubber.setPointerCapture?.(event.pointerId);
            });
            scrubber.addEventListener("input", () => drawFrame(Number(scrubber.value)));
            scrubber.addEventListener("pointerup", finishScrub);
            scrubber.addEventListener("pointercancel", finishScrub);
            scrubber.addEventListener("lostpointercapture", finishScrub);
            scrubber.addEventListener("click", (event) => event.stopPropagation?.());

            const decodeAnimatedPreview = async (bytes, generation) => {
                const Decoder = globalThis.ImageDecoder;
                if (typeof Decoder !== "function" || typeof createImageBitmap !== "function") {
                    return false;
                }
                let decoder;
                const nextFrames = [];
                try {
                    decoder = new Decoder({ data: bytes, type: "image/webp" });
                    await decoder.tracks.ready;
                    const count = Number(decoder.tracks.selectedTrack?.frameCount) || 0;
                    if (count < 2) return false;
                    for (let index = 0; index < count; index += 1) {
                        const decoded = await decoder.decode({ frameIndex: index, completeFramesOnly: true });
                        const bitmap = await createImageBitmap(decoded.image);
                        decoded.image.close?.();
                        nextFrames.push(bitmap);
                        if (generation !== decodeGeneration) {
                            closeFrames(nextFrames);
                            return false;
                        }
                    }
                    if (generation !== decodeGeneration) {
                        closeFrames(nextFrames);
                        return false;
                    }
                    stopPlayback();
                    closeFrames();
                    decodedFrames = nextFrames;
                    decodedPreviewGeneration = generation;
                    frameIndex = 0;
                    canvas.width = decodedFrames[0].width;
                    canvas.height = decodedFrames[0].height;
                    canvas.style.display = "block";
                    image.style.display = "none";
                    scrubber.disabled = false;
                    drawFrame(0);
                    schedulePlayback();
                    return true;
                } catch (error) {
                    closeFrames(nextFrames);
                    console.warn("[Star7 H3 Preview] Timeline decoding unavailable; using animated WebP", error);
                    return false;
                } finally {
                    decoder?.close?.();
                }
            };

            this._star7H3Preview = (data) => {
                if (data.status === "downloading") {
                    runId = data.run_id;
                    status.style.color = "#d5d5d5";
                    status.textContent = chinese
                        ? "正在后台下载 TAEH3；采样继续运行…"
                        : "Downloading TAEH3 in background; sampling continues…";
                    return;
                }
                if (data.status === "start") {
                    runId = data.run_id;
                    status.style.color = "#d5d5d5";
                    status.textContent = chinese ? `等待第 1 / ${data.total} 步…` : `Waiting for Step 1 / ${data.total}…`;
                    return;
                }
                if (runId !== null && data.run_id !== runId) return;
                if (data.status === "error") {
                    status.textContent = data.message || (chinese ? "预览已跳过" : "Preview skipped");
                    status.style.color = "#ffb36b";
                    return;
                }
                if (typeof data.image !== "string") return;
                const bytes = Uint8Array.from(atob(data.image), (char) => char.charCodeAt(0));
                const generation = ++decodeGeneration;
                stopPlayback();
                closeFrames();
                decodedPreviewGeneration = 0;
                scrubber.disabled = true;
                const nextUrl = URL.createObjectURL(new Blob([bytes], { type: "image/webp" }));
                image.onload = () => {
                    if (generation !== decodeGeneration) {
                        URL.revokeObjectURL(nextUrl);
                        return;
                    }
                    if (objectUrl) URL.revokeObjectURL(objectUrl);
                    objectUrl = nextUrl;
                    placeholder.style.display = "none";
                    if (decodedPreviewGeneration !== generation) {
                        image.style.display = "block";
                        canvas.style.display = "none";
                    }
                };
                image.onerror = () => {
                    URL.revokeObjectURL(nextUrl);
                    if (generation !== decodeGeneration) return;
                    image.style.display = "none";
                    placeholder.style.display = "flex";
                    placeholder.textContent = chinese ? "预览图片加载失败，采样仍会继续" : "Preview image failed to load; sampling continues";
                };
                image.src = nextUrl;
                void decodeAnimatedPreview(bytes, generation);
                status.style.color = "#d5d5d5";
                status.textContent = chinese
                    ? `第 ${data.step} / ${data.total} 步 · ${data.width}×${data.height}`
                    : `Step ${data.step} / ${data.total} · ${data.width}×${data.height}`;
            };

            this.addDOMWidget("preview", "star7_h3_preview", root, { serialize: false });
            this.setSize([Math.max(this.size?.[0] ?? 340, 340), Math.max(this.size?.[1] ?? 360, 360)]);

            const removed = this.onRemoved;
            this.onRemoved = function () {
                decodeGeneration += 1;
                stopPlayback();
                closeFrames();
                if (objectUrl) URL.revokeObjectURL(objectUrl);
                this._star7H3Preview = null;
                return removed?.apply(this, arguments);
            };
            return result;
        };
    },
});
