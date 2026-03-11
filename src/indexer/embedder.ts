import {
  type FeatureExtractionPipeline,
  pipeline,
} from "@huggingface/transformers";

const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
const BATCH_SIZE = 32;

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

async function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_NAME, {
      dtype: "fp32",
    });
  }
  return extractorPromise;
}

export async function loadModel(): Promise<void> {
  await getExtractor();
}

export async function embedTexts(texts: string[]): Promise<Float32Array[]> {
  const extractor = await getExtractor();

  const results: Float32Array[] = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const output = await extractor(batch, {
      pooling: "mean",
      normalize: true,
    });

    // output.data is a flat Float32Array of all embeddings concatenated
    const dim = 384;
    for (let j = 0; j < batch.length; j++) {
      const embedding = new Float32Array(dim);
      embedding.set(output.data.slice(j * dim, (j + 1) * dim) as Float32Array);
      results.push(embedding);
    }
  }

  return results;
}

export async function embedText(text: string): Promise<Float32Array> {
  const [embedding] = await embedTexts([text]);
  if (!embedding) throw new Error("Failed to generate embedding");
  return embedding;
}

export function composeEmbeddingText(
  message: string,
  filePaths: string[],
): string {
  const filesStr = filePaths.join(", ");
  return filesStr ? `${message}\n\nFiles: ${filesStr}` : message;
}
