import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3PromptImportStar7";
const ACCEPT = ".png,.webp,.jpg,.jpeg,.gif,.avif,.mp4,.mov,.mkv,.webm,.avi,.m4v,.json,image/*,video/*,application/json";
const EXTENSIONS = new Set(["png", "webp", "jpg", "jpeg", "gif", "avif", "mp4", "mov", "mkv", "webm", "avi", "m4v", "json"]);

function extensionOf(filename) {
    const parts = String(filename || "").toLowerCase().split(".");
    return parts.length > 1 ? parts.pop() : "";
}

function supported(file) {
    return !!file && EXTENSIONS.has(extensionOf(file.name));
}

function toast(message, success = false) {
    const element = document.createElement("div");
    element.textContent = message;
    Object.assign(element.style, {
        position: "fixed", right: "22px", bottom: "22px", zIndex: "100000",
        maxWidth: "460px", padding: "10px 14px", borderRadius: "7px",
        color: "#fff", background: success ? "#207a45" : "#823b34",
        boxShadow: "0 7px 24px rgba(0,0,0,.4)", font: "13px/1.45 sans-serif",
        whiteSpace: "pre-wrap",
    });
    document.body.appendChild(element);
    setTimeout(() => element.remove(), 3200);
}

function setPrompt(node, widget, text) {
    widget.value = String(text || "");
    widget.callback?.(widget.value);
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function overlay(title) {
    const backdrop = document.createElement("div");
    Object.assign(backdrop.style, {
        position: "fixed", inset: "0", zIndex: "99999", background: "rgba(0,0,0,.68)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: "24px",
    });
    const panel = document.createElement("div");
    Object.assign(panel.style, {
        width: "min(860px, 92vw)", maxHeight: "82vh", overflow: "auto",
        padding: "16px", borderRadius: "10px", background: "#222", color: "#eee",
        boxShadow: "0 14px 48px rgba(0,0,0,.55)", font: "13px/1.45 sans-serif",
    });
    const heading = document.createElement("div");
    heading.textContent = title;
    Object.assign(heading.style, { fontWeight: "700", fontSize: "16px", marginBottom: "10px" });
    panel.appendChild(heading);
    backdrop.appendChild(panel);
    backdrop.addEventListener("click", (event) => {
        if (event.target === backdrop) backdrop.remove();
    });
    document.body.appendChild(backdrop);
    return { backdrop, panel };
}

function showCandidates(node, widget) {
    const candidates = node._star7PromptCandidates || [];
    if (!candidates.length) {
        toast("当前没有候选词。请先导入文件。");
        return;
    }
    const { backdrop, panel } = overlay(`候选词（${candidates.length}）`);
    candidates.forEach((candidate, index) => {
        const button = document.createElement("button");
        button.type = "button";
        Object.assign(button.style, {
            display: "block", width: "100%", margin: "7px 0", padding: "10px 12px",
            border: index === 0 ? "1px solid #69bd8b" : "1px solid #505050",
            borderRadius: "7px", color: "#eee", background: index === 0 ? "#263b30" : "#303030",
            textAlign: "left", cursor: "pointer",
        });
        const meta = document.createElement("div");
        const length = Number(candidate.length || String(candidate.text || "").length);
        meta.textContent = `${index === 0 ? "首选 · " : ""}${candidate.label || "提示词"} · ${length} 字`;
        Object.assign(meta.style, { color: index === 0 ? "#91ddb0" : "#aaa", fontSize: "12px", marginBottom: "4px" });
        const preview = document.createElement("div");
        const text = String(candidate.text || "");
        preview.textContent = text.length > 700 ? `${text.slice(0, 700)}…` : text;
        Object.assign(preview.style, { whiteSpace: "pre-wrap", overflowWrap: "anywhere" });
        button.append(meta, preview);
        button.addEventListener("click", () => {
            setPrompt(node, widget, text);
            backdrop.remove();
        });
        panel.appendChild(button);
    });
}

async function importFile(node, widget, candidateButton, file) {
    if (!supported(file)) {
        toast("不支持该文件类型，原提示词未改变。");
        return;
    }
    const requestId = (node._star7PromptRequestId || 0) + 1;
    node._star7PromptRequestId = requestId;
    const original = String(widget.value ?? "");
    candidateButton.textContent = "识别中…";
    try {
        const form = new FormData();
        form.append("file", file, file.name);
        const response = await api.fetchApi("/minimax-h3-chunk-star7/prompt-import", { method: "POST", body: form });
        const result = await response.json();
        if (node._star7PromptRequestId !== requestId) return;
        node._star7PromptCandidates = Array.isArray(result.candidates) ? result.candidates : [];
        candidateButton.textContent = `候选词（${node._star7PromptCandidates.length}）`;
        if (!response.ok || !result.success || !String(result.prompt || "").trim()) {
            toast(result.message || "没有识别到可用提示词，原内容未改变。");
            return;
        }
        if (String(widget.value ?? "") !== original) {
            toast("识别期间提示词已被编辑；结果已保留在候选词中，没有自动覆盖。");
            return;
        }
        setPrompt(node, widget, result.prompt);
        toast(`已导入：${result.source || file.name}`, true);
    } catch (error) {
        candidateButton.textContent = `候选词（${node._star7PromptCandidates?.length || 0}）`;
        console.error("[MiniMaxH3Chunk-Star7 Prompt Import]", error);
        toast(`导入失败：${error.message || error}。原提示词未改变。`);
    }
}

function chooseFile(callback) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ACCEPT;
    input.addEventListener("change", () => {
        if (input.files?.[0]) callback(input.files[0]);
    });
    input.click();
}

function install(node) {
    if (node._star7PromptImportInstalled) return;
    node._star7PromptImportInstalled = true;
    node._star7PromptCandidates = [];
    const promptWidget = node.widgets?.find((widget) => widget.name === "prompt");
    if (!promptWidget) return;

    let candidateText = "候选词（0）";
    const actionWidget = {
        name: "star7_prompt_import_actions",
        type: "custom",
        value: "",
        options: { serialize: false },
        get textContent() {
            return candidateText;
        },
        set textContent(value) {
            candidateText = String(value || "候选词（0）");
            node.graph?.setDirtyCanvas?.(true, true);
        },
        computeSize(width) {
            return [width, 28];
        },
        draw(ctx, currentNode, width, y, height) {
            const margin = 10;
            const gap = 6;
            const buttonWidth = (width - margin * 2 - gap) / 2;
            const buttonHeight = Math.min(24, height - 2);
            const top = y + Math.max(0, (height - buttonHeight) / 2);
            ctx.save();
            ctx.font = "13px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            for (let index = 0; index < 2; index += 1) {
                const left = margin + index * (buttonWidth + gap);
                ctx.beginPath();
                ctx.roundRect(left, top, buttonWidth, buttonHeight, 5);
                ctx.fillStyle = globalThis.LiteGraph?.WIDGET_BGCOLOR || "#2d2d2d";
                ctx.fill();
                ctx.strokeStyle = globalThis.LiteGraph?.WIDGET_OUTLINE_COLOR || "#606060";
                ctx.stroke();
                ctx.fillStyle = globalThis.LiteGraph?.WIDGET_TEXT_COLOR || "#ddd";
                ctx.fillText(index === 0 ? "导入文件" : candidateText, left + buttonWidth / 2, top + buttonHeight / 2);
            }
            ctx.restore();
        },
        mouse(event, pos, currentNode) {
            if (event.type === "pointerdown" || event.type === "mousedown") {
                this._pressedSide = pos[0] >= currentNode.size[0] / 2 ? "right" : "left";
                return true;
            }
            if (event.type !== "pointerup" && event.type !== "mouseup") return false;
            const side = this._pressedSide || (pos[0] >= currentNode.size[0] / 2 ? "right" : "left");
            this._pressedSide = null;
            if (side === "right") {
                showCandidates(currentNode, promptWidget);
            } else {
                chooseFile((file) => void importFile(currentNode, promptWidget, actionWidget, file));
            }
            return true;
        },
        serializeValue() {
            return undefined;
        },
    };
    node.addCustomWidget(actionWidget);
    node._star7PromptImport = { promptWidget, candidateButton: actionWidget };

    const previousDragOver = node.onDragOver;
    const previousDragDrop = node.onDragDrop;
    node.onDragOver = function (event) {
        if (Array.from(event?.dataTransfer?.files || []).some(supported)) return true;
        return previousDragOver?.apply(this, arguments) ?? false;
    };
    node.onDragDrop = function (event) {
        const file = Array.from(event?.dataTransfer?.files || []).find(supported);
        if (!file) return previousDragDrop?.apply(this, arguments) ?? false;
        event.preventDefault?.();
        event.stopPropagation?.();
        event.stopImmediatePropagation?.();
        void importFile(this, promptWidget, candidateButton, file);
        return true;
    };

    requestAnimationFrame(() => {
        const computed = node.computeSize?.();
        // Workflow serialization already stores node.size. Keep the restored
        // user size and only enforce enough room for the widgets on new or
        // undersized nodes.
        const width = Math.max(
            Number(node.size?.[0] || 0), Number(computed?.[0] || 0), 340,
        );
        const height = Math.max(
            Number(node.size?.[1] || 0), Number(computed?.[1] || 0), 180,
        );
        node.setSize?.([width, height]);
    });
}

function promptNodeAtDrop(event) {
    const canvas = app.canvas;
    // Use this event's coordinates only. Falling back to graph_mouse can point
    // at a previously hovered node and would incorrectly consume a drop made
    // on empty canvas.
    const point = canvas?.convertEventToCanvasOffset?.(event);
    if (!Array.isArray(point) || point.length < 2) return null;
    const node = canvas.graph?.getNodeOnPos?.(point[0], point[1]) ?? null;
    return node?.comfyClass === NODE_NAME || node?.type === NODE_NAME ? node : null;
}

function installPromptNodeDropCapture() {
    if (window._star7ChunkPromptDropInstalled) return;
    window._star7ChunkPromptDropInstalled = true;

    document.addEventListener("dragover", (event) => {
        const file = Array.from(event.dataTransfer?.files || []).find(supported);
        const node = file ? promptNodeAtDrop(event) : null;
        if (!node) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
    }, true);

    document.addEventListener("drop", (event) => {
        const file = Array.from(event.dataTransfer?.files || []).find(supported);
        const node = file ? promptNodeAtDrop(event) : null;
        if (!node?._star7PromptImport) return;
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        void importFile(
            node,
            node._star7PromptImport.promptWidget,
            node._star7PromptImport.candidateButton,
            file,
        );
    }, true);
}

app.registerExtension({
    name: "Star7.MiniMaxH3.LightPromptImport",
    setup() {
        installPromptNodeDropCapture();
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            previous?.apply(this, arguments);
            install(this);
        };
    },
});
