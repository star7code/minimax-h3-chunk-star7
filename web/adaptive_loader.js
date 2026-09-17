import { app } from "../../scripts/app.js";

const NODE = "MiniMaxH3ChunkEnhancedLoaderStar7";

function isChinese() {
    const locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? globalThis.navigator?.language
        ?? "en";
    return String(locale).toLowerCase().startsWith("zh");
}

function title() {
    return isChinese()
        ? "MiniMax H3 增强载入 - Star7"
        : "MiniMax H3 Enhanced Loader - Star7";
}

app.registerExtension({
    name: "Star7.MiniMaxH3ChunkEnhancedLoader",
    beforeRegisterNodeDef(_nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        nodeData.display_name = title();
        const model = nodeData.input?.required?.unet_name;
        if (Array.isArray(model)) {
            model[1] ??= {};
            model[1].display_name = isChinese() ? "H3 模型" : "H3 model";
        }
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE) return;
        const chinese = isChinese();
        node.title = title();
        const model = node.widgets?.find((item) => item.name === "unet_name");
        if (model) model.label = chinese ? "H3 模型" : "H3 model";
        const output = node.outputs?.find((item) => item.name === "model");
        if (output) output.localized_name = chinese ? "模型" : "model";
    },
});
