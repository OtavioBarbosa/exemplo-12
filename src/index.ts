import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./document_processor.ts";
import { type PretrainedOptions } from "@huggingface/transformers";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { displayResults } from "./util.ts";

let _ne04j_vector_store: any = null;

async function clearAll(vector_store: Neo4jVectorStore, node_label: string) {
    await vector_store.query(`MATCH (n:${node_label}) DETACH DELETE n`);
}

try { 
    const documentProcessor = new DocumentProcessor(
        CONFIG.pdf.path,
        CONFIG.textSplitter
    );

    const documents = await documentProcessor.loadAndSplit();
    const embeddings = new HuggingFaceTransformersEmbeddings({
        model: CONFIG.embedding.modelName,
        pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions,
    });

    _ne04j_vector_store = await Neo4jVectorStore.fromExistingGraph(
        embeddings,
        CONFIG.neo4j,
    );

    clearAll(_ne04j_vector_store, CONFIG.neo4j.nodeLabel);

    for (const document of documents) {
        await _ne04j_vector_store.addDocuments([document])
    }

    const questions = [
        "O que significa treinar uma rede neural?",
    ];

    for (const question of questions) {
        const results = await _ne04j_vector_store.similaritySearch(question, CONFIG.similarity.topK);
        console.log(`\nQuestion: ${question}`);
        displayResults(results);
    }

}
catch (error) {
    console.error("An error occurred:", error);
}
finally {
    await _ne04j_vector_store?.close();
}