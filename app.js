import { pipeline } from '@huggingface/transformers';

const ICD_DATABASE = [
    { code: "I21.9", description: "Acute myocardial infarction, unspecified (Heart Attack)" },
    { code: "J45.909", description: "Unspecified asthma, uncomplicated" },
    { code: "E11.9", description: "Type 2 diabetes mellitus without complications" },
    { code: "M54.50", description: "Low back pain, unspecified" },
    { code: "G43.909", description: "Migraine, unspecified, not intractable, without status migrainosus" }
];

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

let extractorInstance = null;

async function runMedicalRAG(userQuery) {
    const statusDiv = document.getElementById('statusMessage');
    const resultsDiv = document.getElementById('resultsContainer');

    if (!userQuery || !userQuery.trim()) {
        alert("Please enter a note or symptoms first!");
        return;
    }

    resultsDiv.innerHTML = "";
    statusDiv.innerText = "Processing symptoms via local AI model...";

    try {
        if (!extractorInstance) {
            statusDiv.innerText = "Downloading and configuring local model weights...";
            extractorInstance = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        }

        statusDiv.innerText = "Analyzing text vectors...";

        // Compute query tensor and convert it to a standard JS array
        const queryTensor = await extractorInstance(userQuery, { pooling: 'mean', normalize: true });
        const queryVector = queryTensor.tolist()[0]; 

        const results = [];
        for (const item of ICD_DATABASE) {
            const itemTensor = await extractorInstance(item.description, { pooling: 'mean', normalize: true });
            const itemVector = itemTensor.tolist()[0]; 

            const similarity = cosineSimilarity(queryVector, itemVector);
            results.push({ ...item, similarity });
        }

        results.sort((a, b) => b.similarity - a.similarity);
        statusDiv.innerText = "Analysis Ready!";

        // Generate the visual HTML cards directly into the container wrapper
        results.forEach(match => {
            const matchPercentage = (match.similarity * 100).toFixed(1);
            const card = document.createElement('div');
            
            card.style.border = "1px solid #ddd";
            card.style.padding = "12px";
            card.style.marginBottom = "10px";
            card.style.borderRadius = "4px";
            card.style.backgroundColor = match.similarity > 0.2 ? "#f0fdf4" : "#ffffff";

            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; font-family: sans-serif;">
                    <strong style="font-size: 18px; color: #1e1b4b;">Code: ${match.code}</strong>
                    <span style="background: #e0e7ff; padding: 2px 8px; border-radius: 12px; font-size: 14px; font-weight: bold; color: #4f46e5;">
                        ${matchPercentage}% Match
                    </span>
                </div>
                <p style="margin: 6px 0 0 0; color: #4b5563; font-family: sans-serif;">${match.description}</p>
            `;
            resultsDiv.appendChild(card);
        });

    } catch (err) {
        statusDiv.innerText = "Error extracting semantic similarities.";
        console.error(err);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('searchBtn');
    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            const noteValue = document.getElementById('clinicalNote').value;
            runMedicalRAG(noteValue);
        });
    }
});