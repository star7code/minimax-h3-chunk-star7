import assert from "node:assert/strict";
import fs from "node:fs";

for (const name of [
    "MiniMax-H3-Activation-Chunk-Star7-HD.json",
    "MiniMax-H3-Activation-Chunk-Star7-HD-English.json",
]) {
    const workflow = JSON.parse(fs.readFileSync(new URL(`./examples/workflows/${name}`, import.meta.url)));
    const nodes = new Map(workflow.nodes.map((node) => [node.id, node]));
    const links = new Map(workflow.links.map((link) => [link[0], link]));
    const hd = nodes.get(376);
    const decode = nodes.get(377);
    const videoCombine = nodes.get(150);

    assert.equal(hd.type, "MiniMaxH3OneClickHDStar7");
    assert.equal(decode.type, "MiniMaxH3ChunkedDecodeStar7");
    assert.equal(nodes.has(123), false, "The separate Star7 decoder must replace core audio/video decode nodes");
    assert.equal(nodes.has(124), false, "Face Repair must not hide VAE decoding in the HD example");
    assert.equal(hd.inputs[0].name, "enable_hd");
    assert.equal(hd.inputs.at(-3).name, "enable_tiling");
    assert.equal(hd.widgets_values[0], true);
    assert.equal(hd.widgets_values.at(-3), false);
    assert.deepEqual(links.get(256).slice(1, 5), [127, 0, 376, 1]);
    assert.deepEqual(links.get(482).slice(1, 5), [275, 4, 376, 2]);
    assert.deepEqual(links.get(488).slice(1, 5), [376, 0, 377, 0]);
    assert.deepEqual(links.get(489).slice(1, 5), [121, 0, 377, 1]);
    assert.deepEqual(links.get(490).slice(1, 5), [122, 0, 377, 2]);
    assert.deepEqual(links.get(487).slice(1, 5), [377, 0, 150, 0]);
    assert.deepEqual(links.get(491).slice(1, 5), [377, 1, 150, 1]);
    assert.equal(videoCombine.inputs.find((input) => input.name === "images").link, 487);
    assert.equal(videoCombine.inputs.find((input) => input.name === "audio").link, 491);
}

console.log("H3 external-VAE HD workflow tests passed");
