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
const nodeData = {
    name: "MiniMaxH3OneClickHDStar7",
    input: { required: {
        sampled_av_latent: ["LATENT", {}], h3_context: ["STAR7_H3_REFINE_CONTEXT", {}],
        second_pass_lora: [["继承一采"], {}], second_pass_lora_strength: ["FLOAT", {}],
    } },
    output_name: ["hd_av_latent", "report"],
};
await extension.beforeRegisterNodeDef(NodeType, nodeData);
assert.equal(nodeData.display_name, "MiniMax H3 一键高清放大 - Star7");
assert.equal(nodeData.input.required.second_pass_lora[1].display_name, "二采 LoRA");
assert.equal(nodeData.input.required.second_pass_lora_strength[1].display_name, "二采 LoRA 强度");
assert.deepEqual(nodeData.output_name, ["采样结果", "运行报告"]);
const node = Object.create(NodeType.prototype);
node.widgets = [
    ["enable_hd", true],
    ["upscale_model", "minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    ["second_pass_lora", "继承一采"], ["second_pass_lora_strength", 1.0],
    ["second_pass_attention", "继承一采"],
    ["preset", "平衡高清"], ["target_megapixels", 1.0], ["refine_steps", 2],
    ["refine_strength", 0.25], ["seed", 0], ["enable_tiling", false],
    ["tile_count", 2], ["tile_overlap", 128],
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
const secondPassLora = node.widgets.find((item) => item.name === "second_pass_lora");
const secondPassLoraStrength = node.widgets.find((item) => item.name === "second_pass_lora_strength");
const secondPassAttention = node.widgets.find((item) => item.name === "second_pass_attention");
assert.equal(secondPassLora.label, "二采 LoRA");
assert.equal(secondPassLoraStrength.label, "二采 LoRA 强度");
assert.equal(secondPassAttention.label, "二采注意力");
assert.equal(tileCount.label, "分格数量");
tileCount.value = 5;
tileCount.callback();
assert.equal(tileCount.value, 5);
tileCount.value = 7;
tileCount.callback();
assert.equal(tileCount.value, 7);
tileCount.value = 9;
tileCount.callback();
assert.equal(tileCount.value, 9);
tileCount.value = 8;
tileCount.callback();
assert.equal(tileCount.value, 8);
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
assert.equal(steps.value, 1);
assert.equal(strength.value, 0.30);
steps.value = 3;
steps.callback();
assert.equal(preset.value, "自定义");
assert.equal(strength.value, 0.30);
strength.value = 0.0;
strength.callback();
assert.equal(steps.value, 3);
steps.value = 4;
steps.callback();
assert.equal(strength.value, 0.0);
assert.equal(Boolean(tileCount.disabled), false);
enableTiling.value = true;
enableTiling.callback();
assert.equal(Boolean(tileCount.disabled), false);
enableHD.value = false;
enableHD.callback();
assert.equal(Boolean(preset.disabled), false);
assert.equal(Boolean(enableTiling.disabled), false);
assert.equal(Boolean(tileCount.disabled), false);
enableHD.value = true;
enableHD.callback();
assert.equal(Boolean(preset.disabled), false);
assert.equal(Boolean(enableTiling.disabled), false);
assert.equal(Boolean(tileCount.disabled), false);
assert.ok(node.widgets.some((item) => item.name === "重置参数"));
const sigmaStatus = node.widgets.find((item) => item.__star7HDSigmaStatus);
assert.ok(sigmaStatus);
assert.equal(sigmaStatus.disabled, true);
assert.ok(sigmaStatus.name.includes("等待运行"));
steps.value = 2;
strength.value = 0.24;
eventHandler({ detail: {
    node_id: 376, sigmas: "0.7200,0.6000,0.4000,0.0000", shift: 6,
    steps: 3, strength: 0.30, profile: "Turbo/Distilled",
} });
assert.ok(sigmaStatus.name.includes("0.7200 → 0.6000 → 0.4000 → 0.0000"));
assert.ok(sigmaStatus.name.includes("Shift 6"));
assert.ok(sigmaStatus.name.includes("Turbo/Distilled"));
assert.equal(steps.value, 2);
assert.equal(strength.value, 0.24);
assert.ok(node.widgets.indexOf(sigmaStatus) < node.widgets.indexOf(node.__star7HDReset));
assert.ok(node.widgets.indexOf(sigmaStatus) > node.widgets.indexOf(secondPassAttention));
eventHandler({ detail: {
    node_id: 376, sigmas: "0.7500,0.6792,0.5714,0.3871,0.0000", shift: 12,
    steps: 4, strength: 0.20, profile: "Base",
} });
// Runtime results are display-only. In particular, bypass reports zeros and
// must not overwrite values which need to survive toggling and page reloads.
steps.value = 3;
strength.value = 0.27;
enableHD.value = false;
enableHD.callback();
eventHandler({ detail: {
    node_id: 376, sigmas: "off", steps: 0, strength: 0.0, profile: "disabled",
} });
assert.equal(steps.value, 3);
assert.equal(strength.value, 0.27);
enableHD.value = true;
enableHD.callback();
assert.equal(steps.value, 3);
assert.equal(strength.value, 0.27);
node.onSerialize({});
assert.equal(node.properties.star7HDSavedValues.refine_steps, 3);
assert.equal(node.properties.star7HDSavedValues.refine_strength, 0.27);
preset.value = "高质量";
preset.callback();
assert.equal(steps.value, 6);
assert.equal(strength.value, 0.25);

// A legacy workflow may contain a positional null left by an old dynamic
// widget. Invalid values are repaired, while valid saved user values persist.
node.properties.star7HDSavedValues = {
    enable_hd: true, upscale_model: "custom_3d.safetensors",
    second_pass_lora: "custom_lora.safetensors",
    second_pass_lora_strength: 0.72,
    preset: "自定义", target_megapixels: 1.35,
    refine_steps: 3, refine_strength: 0.22, seed: 42,
    enable_tiling: true, tile_count: 16, tile_overlap: 160,
    second_pass_attention: "comfy_kitchen_int8",
};
steps.value = 0;
tileCount.value = null;
upscaleModel.value = null;
secondPassAttention.value = null;
secondPassLora.value = null;
secondPassLoraStrength.value = null;
node.onConfigure({});
assert.equal(steps.value, 1);
assert.equal(tileCount.value, 16);
assert.equal(upscaleModel.value, "custom_3d.safetensors");
assert.equal(secondPassLora.value, "custom_lora.safetensors");
assert.equal(secondPassLoraStrength.value, 0.72);
assert.equal(secondPassAttention.value, "comfy_kitchen_int8");
node.onSerialize({});
assert.equal(node.properties.star7HDSavedValues.tile_count, 16);

// Current released workflows stored attention after the seed control and had
// no per-pass LoRA row. Migrate them by meaning, not by shifted positions.
node.onConfigure({ widgets_values: [
    true, "legacy_3d.safetensors", "自定义", 2.0, 4, 0.28, 77,
    "fixed", "vsa_sm75", true, 6, 192,
] });
assert.equal(upscaleModel.value, "legacy_3d.safetensors");
assert.equal(secondPassLora.value, "继承一采");
assert.equal(secondPassAttention.value, "vsa_sm75");
assert.equal(preset.value, "自定义");
assert.equal(tileCount.value, 6);

// Workflows from the immediately preceding layout already had per-pass LoRA
// and attention, but no strength row. Keep every following value aligned.
node.onConfigure({ widgets_values: [
    true, "current_3d.safetensors", "detail.safetensors", "ck_vsa_sm75",
    "自定义", 4.0, 5, 0.31, 88, "increment", true, 9, 224,
] });
assert.equal(upscaleModel.value, "current_3d.safetensors");
assert.equal(secondPassLora.value, "detail.safetensors");
assert.equal(secondPassLoraStrength.value, 1.0);
assert.equal(secondPassAttention.value, "ck_vsa_sm75");
assert.equal(preset.value, "自定义");
assert.equal(tileCount.value, 9);

console.log("h3 latent upscale UI tests passed");
