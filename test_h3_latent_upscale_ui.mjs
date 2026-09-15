import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
let eventHandler;
const app = {
    ui: { settings: { getSettingValue() { return "zh-CN"; } } },
    graph: { getNodeById() { return node; } },
    registerExtension(value) { extension = value; },
};
const api = { addEventListener(name, handler) { if (name === "star7-h3-hd-sigmas") eventHandler = handler; } };
const path = new URL("./web/h3_latent_upscale_star7.js", import.meta.url);
const source = fs.readFileSync(path, "utf8").replace(
    'import { app } from "../../scripts/app.js";',
    "const app = globalThis.__star7TestApp;",
).replace(
    'import { api } from "../../scripts/api.js";',
    "const api = globalThis.__star7TestApi;",
);
vm.runInNewContext(source, {
    __star7TestApp: app, __star7TestApi: api, navigator: { language: "zh-CN" },
});
assert.ok(extension);

class NodeType {}
await extension.beforeRegisterNodeDef(NodeType, { name: "MiniMaxH3OneClickHDStar7" });
const node = Object.create(NodeType.prototype);
node.widgets = [
    ["enable_hd", true],
    ["upscale_model", "minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    ["preset", "平衡高清"], ["target_megapixels", 1.0], ["refine_steps", 2],
    ["refine_strength", 0.25], ["seed", 0], ["enable_tiling", false],
    ["tile_count", 2], ["tile_overlap", 128], ["second_pass_attention", "继承一采"],
].map(([name, value]) => ({ name, value }));
node.inputs = [{ name: "sampled_av_latent" }, { name: "h3_context" }];
node.outputs = [{ name: "hd_av_latent" }, { name: "report" }];
node.properties = {};
node.comfyClass = "MiniMaxH3OneClickHDStar7";
node.size = [300, 200];
node.computeSize = () => [390, 260];
node.setSize = (value) => { node.size = value; };
node.setDirtyCanvas = () => {};
node.addWidget = (type, name, value, callback, options) => {
    const item = { type, name, value, callback, options };
    node.widgets.push(item);
    return item;
};
node.onNodeCreated();

assert.equal(node.title, "MiniMax H3 一键高清放大 - Star7");
assert.equal(node.inputs[0].label, "采样结果");
assert.equal(node.inputs[1].label, "采样上下文");
assert.equal(node.outputs[0].label, "采样结果");
const enableHD = node.widgets.find((item) => item.name === "enable_hd");
const enableTiling = node.widgets.find((item) => item.name === "enable_tiling");
const tileCount = node.widgets.find((item) => item.name === "tile_count");
const upscaleModel = node.widgets.find((item) => item.name === "upscale_model");
const preset = node.widgets.find((item) => item.name === "preset");
const steps = node.widgets.find((item) => item.name === "refine_steps");
const strength = node.widgets.find((item) => item.name === "refine_strength");
const secondPassAttention = node.widgets.find((item) => item.name === "second_pass_attention");
assert.equal(secondPassAttention.label, "二采注意力");
assert.equal(tileCount.label, "分格数量");
assert.equal(upscaleModel.label, "高清放大模型");
assert.equal(upscaleModel.value, "minimax_h3_latent_upscaler_3d_fp16.safetensors");
assert.equal(upscaleModel.options?.getOptionLabel, undefined);
assert.equal(typeof upscaleModel.draw, "function");
const paintedText = [];
const drawContext = {
    globalAlpha: 1, save() {}, restore() {}, beginPath() {}, roundRect() {},
    fill() {}, stroke() {}, moveTo() {}, lineTo() {},
    fillText(text) { paintedText.push(String(text)); },
};
upscaleModel.draw(drawContext, node, 390, 20, 20);
assert.ok(paintedText.includes("高清放大模型"));
assert.ok(paintedText.includes("minimax_…3d_fp16.safetensors"));
preset.value = "高质量";
preset.callback();
assert.equal(steps.value, 3);
assert.equal(strength.value, 0.30);
steps.value = 0;
steps.callback();
assert.equal(preset.value, "自定义");
assert.equal(strength.value, 0.0);
steps.value = 3;
steps.callback();
assert.equal(preset.value, "自定义");
assert.equal(strength.value, 0.18);
strength.value = 0.0;
strength.callback();
assert.equal(steps.value, 0);
assert.equal(tileCount.disabled, true);
enableTiling.value = true;
enableTiling.callback();
assert.equal(tileCount.disabled, false);
enableHD.value = false;
enableHD.callback();
assert.equal(preset.disabled, true);
assert.equal(enableTiling.disabled, true);
assert.equal(tileCount.disabled, true);
enableHD.value = true;
enableHD.callback();
assert.equal(preset.disabled, false);
assert.equal(enableTiling.disabled, false);
assert.equal(tileCount.disabled, false);
assert.ok(node.widgets.some((item) => item.name === "重置参数"));
const sigmaStatus = node.widgets.find((item) => item.__star7HDSigmaStatus);
assert.ok(sigmaStatus);
assert.equal(sigmaStatus.disabled, true);
assert.ok(sigmaStatus.name.includes("等待运行"));
eventHandler({ detail: {
    node_id: 376, sigmas: "0.7200,0.6000,0.4000,0.0000", shift: 6,
    steps: 3, strength: 0.30, profile: "Turbo/Distilled",
} });
assert.ok(sigmaStatus.name.includes("0.7200 → 0.6000 → 0.4000 → 0.0000"));
assert.ok(sigmaStatus.name.includes("Shift 6"));
assert.ok(sigmaStatus.name.includes("Turbo/Distilled"));
assert.equal(steps.value, 3);
assert.equal(strength.value, 0.30);
assert.ok(node.widgets.indexOf(sigmaStatus) < node.widgets.indexOf(node.__star7HDReset));
assert.ok(node.widgets.indexOf(sigmaStatus) > node.widgets.indexOf(secondPassAttention));
eventHandler({ detail: {
    node_id: 376, sigmas: "0.7500,0.6792,0.5714,0.3871,0.0000", shift: 12,
    steps: 4, strength: 0.20, profile: "Base",
} });
preset.value = "高质量";
preset.callback();
assert.equal(steps.value, 6);
assert.equal(strength.value, 0.25);

// A legacy workflow may contain a positional null left by an old dynamic
// widget. Invalid values are repaired, while valid saved user values persist.
node.properties.star7HDSavedValues = {
    enable_hd: true, upscale_model: "custom_3d.safetensors",
    preset: "自定义", target_megapixels: 1.35,
    refine_steps: 3, refine_strength: 0.22, seed: 42,
    enable_tiling: true, tile_count: 16, tile_overlap: 160,
    second_pass_attention: "comfy_kitchen_int8",
};
tileCount.value = null;
upscaleModel.value = null;
secondPassAttention.value = null;
node.onConfigure({});
assert.equal(tileCount.value, 16);
assert.equal(upscaleModel.value, "custom_3d.safetensors");
assert.equal(secondPassAttention.value, "comfy_kitchen_int8");
node.onSerialize({});
assert.equal(node.properties.star7HDSavedValues.tile_count, 16);

console.log("h3 latent upscale UI tests passed");
