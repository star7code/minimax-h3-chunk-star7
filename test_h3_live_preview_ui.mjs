import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
let previewEvent;
const elements = [];
const drawnFrames = [];

class MockClassList {
    constructor() { this.values = new Set(); }
    add(value) { this.values.add(value); }
    remove(value) { this.values.delete(value); }
    contains(value) { return this.values.has(value); }
}

class MockElement {
    constructor(tagName) {
        this.tagName = tagName.toUpperCase();
        this.children = [];
        this.listeners = new Map();
        this.classList = new MockClassList();
        this.style = { setProperty(name, value) { this[name] = value; } };
        this.capturedPointers = new Set();
        elements.push(this);
    }
    appendChild(child) { this.children.push(child); return child; }
    addEventListener(name, callback) { this.listeners.set(name, callback); }
    dispatch(name, event = {}) { this.listeners.get(name)?.(event); }
    setAttribute(name, value) { this[name] = value; }
    setPointerCapture(id) { this.capturedPointers.add(id); }
    hasPointerCapture(id) { return this.capturedPointers.has(id); }
    releasePointerCapture(id) {
        this.capturedPointers.delete(id);
        this.dispatch("lostpointercapture", { pointerId: id });
    }
    getContext() {
        return { drawImage(frame) { drawnFrames.push(frame.id); } };
    }
    set src(value) {
        this._src = value;
        queueMicrotask(() => this.onload?.());
    }
    get src() { return this._src; }
}

const document = {
    head: new MockElement("head"),
    createElement(tagName) { return new MockElement(tagName); },
    getElementById(id) { return elements.find((element) => element.id === id) ?? null; },
};

class MockImageDecoder {
    constructor() {
        this.tracks = {
            ready: Promise.resolve(),
            selectedTrack: { frameCount: 3 },
        };
    }
    async decode({ frameIndex }) {
        return {
            image: {
                id: frameIndex,
                close() {},
            },
        };
    }
    close() {}
}

const source = fs.readFileSync(
    new URL("./web/h3_live_preview_star7.js", import.meta.url), "utf8",
).replace(/^import .*?;\r?\n/gm, "");

const graphNode = {
    id: 42,
    type: "MiniMaxH3LivePreviewStar7",
    comfyClass: "MiniMaxH3LivePreviewStar7",
    size: [340, 360],
    inputs: [{ name: "model" }],
    outputs: [{ name: "model" }],
    widgets: [
        { name: "preview_frames", value: 25 },
        { name: "preview_resolution", value: "512" },
        { name: "first_step_only", value: false },
    ],
    addDOMWidget(_name, _type, element) { this.previewRoot = element; },
    setSize(size) { this.size = size; },
    setDirtyCanvas() {},
};

const context = {
    api: {
        addEventListener(name, callback) {
            if (name === "star7_h3_live_preview") previewEvent = callback;
        },
    },
    app: {
        graph: { getNodeById(id) { return id === 42 ? graphNode : null; } },
        registerExtension(value) { extension = value; },
    },
    document,
    ImageDecoder: MockImageDecoder,
    createImageBitmap: async (image) => ({
        id: image.id,
        width: 320,
        height: 180,
        close() {},
    }),
    URL: {
        createObjectURL() { return `blob:preview-${Math.random()}`; },
        revokeObjectURL() {},
    },
    Blob,
    Uint8Array,
    Number,
    String,
    Math,
    Promise,
    Set,
    console,
    atob(value) { return Buffer.from(value, "base64").toString("binary"); },
    setTimeout,
    clearTimeout,
    queueMicrotask,
};
vm.runInNewContext(source, context);

class PreviewNodeType {}
PreviewNodeType.prototype.onNodeCreated = function () {};
await extension.beforeRegisterNodeDef(PreviewNodeType, { name: "MiniMaxH3LivePreviewStar7" });
Object.setPrototypeOf(graphNode, PreviewNodeType.prototype);
graphNode.onNodeCreated();

previewEvent({
    detail: {
        node_id: 42,
        run_id: "test-run",
        step: 1,
        total: 4,
        width: 320,
        height: 180,
        image: Buffer.from("animated-webp-placeholder").toString("base64"),
    },
});
await new Promise((resolve) => setTimeout(resolve, 20));

const media = elements.find((element) => element.className === "star7-h3-preview-media");
const scrubber = elements.find((element) => element.className === "star7-h3-preview-scrubber");
const canvas = elements.find((element) => element.tagName === "CANVAS");
assert.ok(media);
assert.ok(scrubber);
assert.equal(String(scrubber.style.cssText).includes("opacity:0"), false);
assert.equal(scrubber.disabled, false);
assert.equal(scrubber.max, "2");
assert.equal(canvas.style.display, "block");

scrubber.dispatch("pointerdown", { pointerId: 7 });
assert.equal(scrubber.classList.contains("star7-dragging"), true);
scrubber.value = "2";
scrubber.dispatch("input");
assert.equal(drawnFrames.at(-1), 2);
assert.equal(scrubber.style["--star7-progress"], "100%");
scrubber.dispatch("pointerup", { pointerId: 7 });
assert.equal(scrubber.classList.contains("star7-dragging"), false);

graphNode.onRemoved();
console.log("H3 live preview scrubber test passed");
