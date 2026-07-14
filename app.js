import { pipeline } from '@huggingface/transformers';

// 1. Create the pipeline (it automatically downloads and caches the model)
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// 2. Compute embeddings
const output = await extractor('Acute myocardial infarction of anterolateral wall', {
    pooling: 'mean',
    normalize: true,
});

// 3. There is your raw math vector!
console.log(output.data);