import { app } from "../../scripts/app.js";

const NODE = "MiniMaxH3ChunkedDecodeStar7";

function chinese() {
    const locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? globalThis.navigator?.language ?? "en";
    return String(locale).toLowerCase().startsWith("zh");
}

app.registerExtension({
    name: "Star7.H3ChunkedDecode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = original?.apply(this, args);
            if (chinese()) {
                this.title = "MiniMax H3 分块解码 - Star7";
                const labels = {
                    av_latent: "音视频潜空间", video_vae: "视频 VAE", audio_vae: "音频 VAE",
                    frames: "视频画面", audio: "输出音频", report: "运行报告",
                };
                for (const item of [...(this.inputs ?? []), ...(this.outputs ?? [])]) {
                    if (labels[item.name]) item.label = item.localized_name = labels[item.name];
                }
            } else {
                this.title = "MiniMax H3 Chunked Decode - Star7";
            }
            return result;
        };
    },
});
