import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3PromptImportStar7";
const ACCEPT = ".png,.webp,.jpg,.jpeg,.gif,.avif,.mp4,.mov,.mkv,.webm,.avi,.m4v,.json,image/*,video/*,application/json";
const EXTENSIONS = new Set(["png", "webp", "jpg", "jpeg", "gif", "avif", "mp4", "mov", "mkv", "webm", "avi", "m4v", "json"]);

const PROMPT_TEXT = {
    en: {
        title: "Prompt Load - Star7",
        prompt: "Prompt",
        output: "prompt",
        tooltip: "Edit directly, or drop a workflow image, video, or JSON file onto the node to import its prompt.",
        importFile: "Import file",
        candidates: (count) => `Candidates (${count})`,
        copy: "Copy",
        paste: "Paste",
        copied: "Prompt copied.",
        pasted: "Clipboard text pasted.",
        emptyClipboard: "The clipboard does not contain text.",
        copyFailed: (error) => `Copy failed: ${error}`,
        pasteFailed: (error) => `Paste failed: ${error}`,
        candidatesTitle: (count) => `Prompt candidates (${count})`,
        noCandidates: "No candidates are available. Import a file first.",
        preferred: "Preferred · ",
        promptFallback: "Prompt",
        chars: (count) => `${count} chars`,
        unsupported: "Unsupported file type. The current prompt was not changed.",
        recognizing: "Reading…",
        noPrompt: "No usable prompt was found. The current prompt was not changed.",
        edited: "The prompt was edited while the file was being read. The result remains available under Candidates and was not applied automatically.",
        imported: (source) => `Imported: ${source}`,
        failed: (error) => `Import failed: ${error}. The current prompt was not changed.`,
    },
    zh: {
        title: "提示词载入 - Star7",
        prompt: "提示词",
        output: "提示词",
        tooltip: "可直接编辑，或把带工作流的图片、视频、JSON 拖入节点自动导入提示词。",
        importFile: "导入文件",
        candidates: (count) => `候选词（${count}）`,
        copy: "复制",
        paste: "粘贴",
        copied: "提示词已复制。",
        pasted: "剪贴板内容已粘贴。",
        emptyClipboard: "剪贴板中没有文本内容。",
        copyFailed: (error) => `复制失败：${error}`,
        pasteFailed: (error) => `粘贴失败：${error}`,
        candidatesTitle: (count) => `候选词（${count}）`,
        noCandidates: "当前没有候选词。请先导入文件。",
        preferred: "首选 · ",
        promptFallback: "提示词",
        chars: (count) => `${count} 字`,
        unsupported: "不支持该文件类型，原提示词未改变。",
        recognizing: "识别中…",
        noPrompt: "没有识别到可用提示词，原内容未改变。",
        edited: "识别期间提示词已被编辑；结果已保留在候选词中，没有自动覆盖。",
        imported: (source) => `已导入：${source}`,
        failed: (error) => `导入失败：${error}。原提示词未改变。`,
    },
};

function language() {
    const locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? globalThis.navigator?.language
        ?? "en";
    return String(locale).toLowerCase().startsWith("zh") ? "zh" : "en";
}

function strings() {
    return PROMPT_TEXT[language()];
}

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

async function copyPrompt(widget) {
    const text = strings();
    try {
        await navigator.clipboard.writeText(String(widget.value ?? ""));
        toast(text.copied, true);
    } catch (error) {
        toast(text.copyFailed(error?.message || error));
    }
}

async function pastePrompt(node, widget) {
    const text = strings();
    try {
        const value = await navigator.clipboard.readText();
        if (!value) {
            toast(text.emptyClipboard);
            return;
        }
        setPrompt(node, widget, value);
        toast(text.pasted, true);
    } catch (error) {
        toast(text.pasteFailed(error?.message || error));
    }
}

function drawActionRow(ctx, width, y, height, labels) {
    const margin = 10;
    const gap = 5;
    const count = Math.max(1, labels.length);
    const buttonWidth = Math.max(1, (width - margin * 2 - gap * (count - 1)) / count);
    const buttonHeight = Math.min(24, Math.max(1, height - 2));
    const top = y + Math.max(0, (height - buttonHeight) / 2);
    ctx.save();
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let index = 0; index < count; index += 1) {
        const left = margin + index * (buttonWidth + gap);
        ctx.beginPath();
        ctx.roundRect(left, top, buttonWidth, buttonHeight, 5);
        ctx.fillStyle = globalThis.LiteGraph?.WIDGET_BGCOLOR || "#2d2d2d";
        ctx.fill();
        ctx.strokeStyle = globalThis.LiteGraph?.WIDGET_OUTLINE_COLOR || "#606060";
        ctx.stroke();
        ctx.fillStyle = globalThis.LiteGraph?.WIDGET_TEXT_COLOR || "#ddd";
        ctx.fillText(labels[index], left + buttonWidth / 2, top + buttonHeight / 2, Math.max(1, buttonWidth - 8));
    }
    ctx.restore();
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
    const text = strings();
    const candidates = node._star7PromptCandidates || [];
    if (!candidates.length) {
        toast(text.noCandidates);
        return;
    }
    const { backdrop, panel } = overlay(text.candidatesTitle(candidates.length));
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
        meta.textContent = `${index === 0 ? text.preferred : ""}${candidate.label || text.promptFallback} · ${text.chars(length)}`;
        Object.assign(meta.style, { color: index === 0 ? "#91ddb0" : "#aaa", fontSize: "12px", marginBottom: "4px" });
        const preview = document.createElement("div");
        const promptText = String(candidate.text || "");
        preview.textContent = promptText.length > 700 ? `${promptText.slice(0, 700)}…` : promptText;
        Object.assign(preview.style, { whiteSpace: "pre-wrap", overflowWrap: "anywhere" });
        button.append(meta, preview);
        button.addEventListener("click", () => {
            setPrompt(node, widget, promptText);
            backdrop.remove();
        });
        panel.appendChild(button);
    });
}

async function importFile(node, widget, candidateButton, file) {
    const text = strings();
    if (!supported(file)) {
        toast(text.unsupported);
        return;
    }
    const requestId = (node._star7PromptRequestId || 0) + 1;
    node._star7PromptRequestId = requestId;
    const original = String(widget.value ?? "");
    candidateButton.textContent = text.recognizing;
    try {
        const form = new FormData();
        form.append("file", file, file.name);
        const response = await api.fetchApi("/minimax-h3-chunk-star7/prompt-import", { method: "POST", body: form });
        const result = await response.json();
        if (node._star7PromptRequestId !== requestId) return;
        node._star7PromptCandidates = Array.isArray(result.candidates) ? result.candidates : [];
        candidateButton.textContent = text.candidates(node._star7PromptCandidates.length);
        if (!response.ok || !result.success || !String(result.prompt || "").trim()) {
            toast(text.noPrompt);
            return;
        }
        if (String(widget.value ?? "") !== original) {
            toast(text.edited);
            return;
        }
        setPrompt(node, widget, result.prompt);
        toast(text.imported(result.source || file.name), true);
    } catch (error) {
        candidateButton.textContent = text.candidates(node._star7PromptCandidates?.length || 0);
        console.error("[MiniMaxH3Chunk-Star7 Prompt Import]", error);
        toast(text.failed(error.message || error));
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

function localizePromptNode(node) {
    const text = strings();
    node.title = text.title;
    const promptWidget = node.widgets?.find((widget) => widget.name === "prompt");
    if (promptWidget) {
        promptWidget.label = text.prompt;
        promptWidget.localized_name = text.prompt;
        promptWidget.options ??= {};
        promptWidget.options.tooltip = text.tooltip;
    }
    const promptInput = node.inputs?.find((input) => input.name === "prompt");
    if (promptInput) promptInput.localized_name = text.prompt;
    const promptOutput = node.outputs?.find((output) => output.name === "prompt");
    if (promptOutput) promptOutput.localized_name = text.output;
    const candidateButton = node._star7PromptImport?.candidateButton;
    if (candidateButton) {
        candidateButton.textContent = text.candidates(node._star7PromptCandidates?.length || 0);
    }
    node.setDirtyCanvas?.(true, true);
}

function install(node) {
    if (node._star7PromptImportInstalled) return;
    node._star7PromptImportInstalled = true;
    node._star7PromptCandidates = [];
    const promptWidget = node.widgets?.find((widget) => widget.name === "prompt");
    if (!promptWidget) return;

    let candidateText = strings().candidates(0);
    const actionWidget = {
        name: "star7_prompt_import_actions",
        type: "custom",
        value: "",
        options: { serialize: false },
        get textContent() {
            return candidateText;
        },
        set textContent(value) {
            candidateText = String(value || strings().candidates(0));
            node.graph?.setDirtyCanvas?.(true, true);
        },
        computeSize(width) {
            return [width, 28];
        },
        draw(ctx, currentNode, width, y, height) {
            const text = strings();
            drawActionRow(ctx, width, y, height, [text.importFile, candidateText, text.copy, text.paste]);
        },
        mouse(event, pos, currentNode) {
            if (event.type === "pointerdown" || event.type === "mousedown") {
                this._pressedIndex = Math.min(3, Math.max(0, Math.floor(pos[0] / (currentNode.size[0] / 4))));
                return true;
            }
            if (event.type !== "pointerup" && event.type !== "mouseup") return false;
            const index = this._pressedIndex
                ?? Math.min(3, Math.max(0, Math.floor(pos[0] / (currentNode.size[0] / 4))));
            this._pressedIndex = null;
            if (index === 0) {
                chooseFile((file) => void importFile(currentNode, promptWidget, actionWidget, file));
            } else if (index === 1) {
                showCandidates(currentNode, promptWidget);
            } else if (index === 2) {
                void copyPrompt(promptWidget);
            } else {
                void pastePrompt(currentNode, promptWidget);
            }
            return true;
        },
        serializeValue() {
            return undefined;
        },
    };
    node.addCustomWidget(actionWidget);
    node._star7PromptImport = {
        promptWidget,
        candidateButton: actionWidget,
    };
    localizePromptNode(node);

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
        app.ui?.settings?.addEventListener?.("Comfy.Locale.change", () => {
            for (const node of app.graph?._nodes ?? []) {
                if (node.comfyClass === NODE_NAME || node.type === NODE_NAME) {
                    localizePromptNode(node);
                }
            }
        });
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        const text = strings();
        nodeData.display_name = text.title;
        const promptSpec = nodeData.input?.required?.prompt;
        if (Array.isArray(promptSpec)) {
            promptSpec[1] ??= {};
            promptSpec[1].display_name = text.prompt;
            promptSpec[1].tooltip = text.tooltip;
        }
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            previous?.apply(this, arguments);
            install(this);
        };
    },
});
