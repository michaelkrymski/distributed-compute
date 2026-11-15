export type Poll = {
  id: string;
  title: string;
  description: string;
  category: string;
  options: string[];
};
export const initialPolls: Poll[] = [
  {
    id: "corpus",
    title: "Training corpus",
    category: "Data",
    description:
      "A focused corpus makes the first run easier to evaluate. Check licenses and deduplicate before splitting.",
    options: [
      "Simple English stories",
      "General English text",
      "Domain-specific text",
    ],
  },
  {
    id: "tokenizer",
    title: "Tokenizer algorithm",
    category: "Tokenizer",
    description:
      "Byte-level BPE is a practical baseline with full byte coverage. Train only on the training split.",
    options: [
      "Byte-level BPE",
      "SentencePiece Unigram",
      "Byte-level (256 tokens)",
    ],
  },
  {
    id: "vocab",
    title: "Vocabulary size",
    category: "Tokenizer",
    description:
      "More tokens mean a larger embedding table, but may shorten sequences. Compare on the same held-out text.",
    options: ["4,096 tokens", "8,192 tokens", "16,384 tokens", "32,768 tokens"],
  },
  {
    id: "width",
    title: "Model width",
    category: "Architecture",
    description:
      "Hidden width changes attention and feed-forward costs roughly quadratically. These choices support six attention heads.",
    options: ["288 hidden units", "384 hidden units", "480 hidden units"],
  },
  {
    id: "layers",
    title: "Transformer depth",
    category: "Architecture",
    description:
      "More layers add parameters and sequential work. Compare configurations inside the 25M budget.",
    options: ["8 layers", "12 layers", "16 layers"],
  },
  {
    id: "ffn",
    title: "Feed-forward expansion",
    category: "Architecture",
    description:
      "The lab models a two-matrix GELU feed-forward block. A gated block needs a different parameter formula.",
    options: ["3× hidden width", "4× hidden width"],
  },
  {
    id: "tied",
    title: "Share embedding weights?",
    category: "Architecture",
    description:
      "Weight tying reuses the input embedding table for next-token output scores. Untying adds V × d parameters.",
    options: ["Tied embeddings", "Separate output weights"],
  },
  {
    id: "context",
    title: "Context length",
    category: "Training",
    description:
      "With RoPE, context adds no learned position parameters. Attention compute and activation memory still grow.",
    options: ["256 tokens", "512 tokens", "1,024 tokens", "2,048 tokens"],
  },
  {
    id: "data",
    title: "Training token budget",
    category: "Training",
    description:
      "500M tokens gives 20 tokens per target parameter. Repeated epochs count toward tokens processed.",
    options: ["250M tokens", "500M tokens", "1B tokens"],
  },
  {
    id: "batch",
    title: "Effective token batch",
    category: "Training",
    description:
      "Tokens per optimizer step, across devices and gradient accumulation. This controls the estimated number of steps.",
    options: ["16,384 tokens", "32,768 tokens", "65,536 tokens"],
  },
  {
    id: "precision",
    title: "Training precision",
    category: "Training",
    description:
      "Choose based on GPU support. Precision affects memory and speed, not learned parameter count.",
    options: ["BF16 mixed precision", "FP16 mixed precision", "FP32"],
  },
  {
    id: "evaluation",
    title: "Evaluation plan",
    category: "Evaluation",
    description:
      "Use a held-out document split. Fixed prompts make generation quality easier to compare across checkpoints.",
    options: [
      "Validation loss + fixed prompts",
      "Validation loss + domain benchmark",
      "Validation loss only",
    ],
  },
];
export const phases = [
  {
    id: "data",
    title: "Prepare the data",
    detail:
      "Choose sources, verify licenses, deduplicate, and split documents.",
    tasks: ["corpus", "evaluation"],
  },
  {
    id: "tokenizer",
    title: "Build the tokenizer",
    detail: "Train, check round trips, and measure held-out token counts.",
    tasks: ["tokenizer", "vocab"],
  },
  {
    id: "model",
    title: "Implement the model",
    detail: "Count parameters, check shapes, and overfit a tiny batch.",
    tasks: ["width", "layers", "ffn", "tied"],
  },
  {
    id: "pilot",
    title: "Run a pilot",
    detail: "Benchmark throughput, inspect loss, and verify checkpoint resume.",
    tasks: ["context", "batch", "precision"],
  },
  {
    id: "training",
    title: "Train the baseline",
    detail: "Process the token budget with validation and checkpoints.",
    tasks: ["data"],
  },
  {
    id: "evaluate",
    title: "Evaluate & document",
    detail: "Compare samples and loss; publish the config and findings.",
    tasks: ["evaluation"],
  },
];
export type ModelConfig = {
  vocab: number;
  width: number;
  layers: number;
  expansion: number;
  tied: boolean;
  context: number;
  tokens: number;
  batch: number;
};
export const defaultConfig: ModelConfig = {
  vocab: 8192,
  width: 384,
  layers: 12,
  expansion: 4,
  tied: true,
  context: 512,
  tokens: 500_000_000,
  batch: 32768,
};
export function parameterBudget(c: ModelConfig) {
  const embeddings = c.vocab * c.width,
    output = c.tied ? 0 : embeddings;
  const attention = c.layers * 4 * c.width * c.width;
  const feedForward = c.layers * 2 * c.width * c.width * c.expansion;
  const norms = 2 * c.layers * c.width + c.width;
  return {
    embeddings,
    output,
    attention,
    feedForward,
    norms,
    total: embeddings + output + attention + feedForward + norms,
  };
}
export function applyDecision(
  c: ModelConfig,
  id: string,
  choice: number,
): ModelConfig {
  const values: Record<string, (number | boolean)[]> = {
    vocab: [4096, 8192, 16384, 32768],
    width: [288, 384, 480],
    layers: [8, 12, 16],
    ffn: [3, 4],
    tied: [true, false],
    context: [256, 512, 1024, 2048],
    data: [250e6, 500e6, 1e9],
    batch: [16384, 32768, 65536],
  };
  const key: Record<string, string> = { ffn: "expansion", data: "tokens" };
  return values[id] && choice >= 0 && choice < values[id].length
    ? { ...c, [key[id] ?? id]: values[id][choice] }
    : c;
}
