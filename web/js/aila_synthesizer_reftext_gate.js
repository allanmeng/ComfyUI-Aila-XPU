import { app } from "../../../scripts/app.js";

/**
 * AilaSynthesizer: voice_clone_mode (ICL / XVECTOR_ONLY) 联动控制 ref_text 有效/无效。
 *
 * - XVECTOR_ONLY: ref_text 灰显禁用（无法编辑/无法新建连接；若已有连接则保持但视作无效）
 * - ICL: ref_text 恢复正常可连接
 */
const NODE_TYPE = "AilaSynthesizer";
const COMBO_WIDGET = "voice_clone_mode";
const REFTEXT_WIDGET = "ref_text";

function findWidget(node, name) {
    if (!node?.widgets) return null;
    return node.widgets.find((w) => w.name === name) || null;
}

function isICLMode(node) {
    const w = findWidget(node, COMBO_WIDGET);
    if (!w) return true; // 找不到 combo 时不拦截（保守）
    return String(w.value).toUpperCase().startsWith("ICL");
}

function applyRefTextGate(node) {
    const textW = findWidget(node, REFTEXT_WIDGET);
    const comboW = findWidget(node, COMBO_WIDGET);
    if (!textW || !comboW) return;

    const enabled = isICLMode(node);
    if (textW.disabled === !enabled) {
        // 状态已一致，无需重复处理
        return;
    }
    textW.disabled = !enabled;

    if (textW.options) {
        textW.options.disabled = !enabled;
    }

    if (node.setDirtyCanvas) {
        node.setDirtyCanvas(true, true);
    }
}

function setupNode(node) {
    const comboW = findWidget(node, COMBO_WIDGET);
    if (!comboW) return;

    // 1. combo 变化时同步 ref_text 状态
    const origCallback = comboW.callback;
    comboW.callback = function (...args) {
        if (typeof origCallback === "function") {
            origCallback.apply(this, args);
        }
        applyRefTextGate(node);
    };

    // 2. 初始同步
    applyRefTextGate(node);

    // 3. 配置加载（含 workflow 恢复）后再次同步
    const origConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        if (typeof origConfigure === "function") {
            origConfigure.apply(this, args);
        }
        applyRefTextGate(node);
    };
}

app.registerExtension({
    name: "Aila.AilaSynthesizer.RefTextGate",
    async nodeCreated(node) {
        const typeName = node.type || node.comfyClass || "";
        if (typeName !== NODE_TYPE && !String(typeName).endsWith(NODE_TYPE)) {
            return;
        }
        setupNode(node);
    },
});
