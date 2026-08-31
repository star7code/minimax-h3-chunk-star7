import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3LivePreviewStar7";
const EVENT_NAME = "star7_h3_live_preview";

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

            let objectUrl = null;
            let runId = null;
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
                const nextUrl = URL.createObjectURL(new Blob([bytes], { type: "image/webp" }));
                image.onload = () => {
                    if (objectUrl) URL.revokeObjectURL(objectUrl);
                    objectUrl = nextUrl;
                    placeholder.style.display = "none";
                    image.style.display = "block";
                };
                image.onerror = () => {
                    URL.revokeObjectURL(nextUrl);
                    image.style.display = "none";
                    placeholder.style.display = "flex";
                    placeholder.textContent = chinese ? "预览图片加载失败，采样仍会继续" : "Preview image failed to load; sampling continues";
                };
                image.src = nextUrl;
                status.style.color = "#d5d5d5";
                status.textContent = chinese
                    ? `第 ${data.step} / ${data.total} 步 · ${data.width}×${data.height}`
                    : `Step ${data.step} / ${data.total} · ${data.width}×${data.height}`;
            };

            this.addDOMWidget("preview", "star7_h3_preview", root, { serialize: false });
            this.setSize([Math.max(this.size?.[0] ?? 340, 340), Math.max(this.size?.[1] ?? 360, 360)]);

            const removed = this.onRemoved;
            this.onRemoved = function () {
                if (objectUrl) URL.revokeObjectURL(objectUrl);
                this._star7H3Preview = null;
                return removed?.apply(this, arguments);
            };
            return result;
        };
    },
});
