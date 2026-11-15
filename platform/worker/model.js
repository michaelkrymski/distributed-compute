import { fail, text } from "./security.js";
export const referenceModel = { vocab_size: 8192, width: 384, layers: 12, heads: 6, expansion: 4, context: 512, tied: true };
export function parameterCount(c) {
  return c.vocab_size * c.width * (c.tied ? 1 : 2) + c.layers * (4 * c.width ** 2 + 2 * c.width ** 2 * c.expansion + 2 * c.width) + c.width;
}
export function modelSpec(value = referenceModel) {
  const c = {};
  for (const key of ["vocab_size", "width", "layers", "heads", "expansion", "context"]) {
    if (!Number.isSafeInteger(value[key]) || value[key] <= 0) fail(`Model ${key} must be a positive integer.`);
    c[key] = value[key];
  }
  if (c.vocab_size < 257 || c.vocab_size > 65536 || c.width > 16384 || c.layers > 256 || c.expansion > 16 || c.context > 32768 || c.width % c.heads || (c.width / c.heads) % 2 || typeof value.tied !== "boolean") fail("Invalid decoder configuration. width / heads must be an even integer.");
  c.tied = value.tied;
  if (parameterCount(c) > 2000000000) fail("This release supports model configurations up to 2B parameters.");
  return c;
}
export function projectSpec(b) {
  const targetTokens = b.targetTokens ?? 500000000;
  if (!Number.isSafeInteger(targetTokens) || targetTokens < 1 || targetTokens > 100000000000000) fail("Invalid training token goal.");
  if (b.runtime === "external") {
    if (!Number.isSafeInteger(b.parameters) || b.parameters<1 || b.parameters>2000000000 || !Number.isSafeInteger(b.context) || b.context<1 || b.context>32768) fail("Specify the actual model parameter count and context length.");
    if (!b.adapterConfig || typeof b.adapterConfig!=="object" || Array.isArray(b.adapterConfig) || JSON.stringify(b.adapterConfig).length>16000) fail("Provide a small JSON object for adapter configuration.");
    return {name:text(b.name,"project name"),modelName:text(b.modelName || "External model","model name"),description:"",runtime:"external",model:{context:b.context},parameters:b.parameters,targetTokens,adapterConfig:b.adapterConfig};
  }
  const model = modelSpec(b.model);
  return { name: text(b.name || b.projectName || "My training project", "project name"), modelName: text(b.modelName || "Custom decoder", "model name"), description: typeof b.description === "string" ? b.description.slice(0, 2000) : "", model, parameters: parameterCount(model), targetTokens, runtime: "glm-decoder-v1" };
}
