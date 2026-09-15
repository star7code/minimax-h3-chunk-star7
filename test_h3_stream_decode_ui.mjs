import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

for (const [locale, expectedTitle] of [
    ["zh-CN", "MiniMax H3 分块解码 - Star7"],
    ["en-US", "MiniMax H3 Chunked Decode - Star7"],
]) {
    let extension;
    const app = {
        ui: { settings: { getSettingValue() { return locale; } } },
        registerExtension(value) { extension = value; },
    };
    const source = fs.readFileSync(
        new URL("./web/h3_stream_decode_star7.js", import.meta.url), "utf8",
    ).replace(
        'import { app } from "../../scripts/app.js";',
        "const app = globalThis.__star7TestApp;",
    );
    vm.runInNewContext(source, {
        __star7TestApp: app, navigator: { language: locale },
    });
    class NodeType {}
    await extension.beforeRegisterNodeDef(NodeType, { name: "MiniMaxH3ChunkedDecodeStar7" });
    const node = Object.create(NodeType.prototype);
    node.inputs = [
        { name: "av_latent" }, { name: "video_vae" }, { name: "audio_vae" },
    ];
    node.outputs = [
        { name: "frames" }, { name: "audio" }, { name: "report" },
    ];
    node.onNodeCreated();
    assert.equal(node.title, expectedTitle);
    if (locale.startsWith("zh")) {
        assert.equal(node.inputs[0].label, "音视频潜空间");
        assert.equal(node.outputs[0].label, "视频画面");
    }
}

console.log("h3 chunked decode UI tests passed");
