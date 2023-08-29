import { Milvus } from "langchain/vectorstores/milvus";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";


export class langchain {
    private static _chain;
    private static readonly collectionName = "pdf";

    public static async uploadFile(filePath: string) {
        // 1. Load
        const loader = new PDFLoader(filePath, {
            splitPages: false,
        });
        const docs = await loader.load();

        // 2. Transform
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const splittedDocs = await splitter.splitDocuments(docs);
        console.log("splittedDocs.length: " + splittedDocs.length);

        // 3. Embed
        const vectorStore = await Milvus.fromDocuments(
            splittedDocs,
            new OpenAIEmbeddings({
                azureOpenAIApiKey: process.env.azureOpenAIApiKey,
                azureOpenAIApiVersion: process.env.azureOpenAIApiVersion,
                azureOpenAIApiInstanceName: process.env.azureOpenAIApiInstanceName,
                azureOpenAIApiDeploymentName: process.env.azureOpenAIApiDeploymentName_Embeddings,
            }),
            {
                collectionName: this.collectionName,
                clientConfig: {
                    address: process.env.MILVUS_URL,
                    token: process.env.MILVUS_TOKEN,
                },
            }
        );
    }

    public static async askQuestion(query: string): Promise<string> {
        if (!this._chain) {
            const vectorStore = await Milvus.fromExistingCollection(
                new OpenAIEmbeddings({
                    azureOpenAIApiKey: process.env.azureOpenAIApiKey,
                    azureOpenAIApiVersion: process.env.azureOpenAIApiVersion,
                    azureOpenAIApiInstanceName: process.env.azureOpenAIApiInstanceName,
                    azureOpenAIApiDeploymentName: process.env.azureOpenAIApiDeploymentName_Embeddings,
                }),
                {
                    collectionName: this.collectionName,
                    clientConfig: {
                        address: process.env.MILVUS_URL,
                        token: process.env.MILVUS_TOKEN,
                    },
                }
            );

            const vectorStoreRetriever = vectorStore.asRetriever();

            const model = new OpenAI({
                azureOpenAIApiKey: process.env.azureOpenAIApiKey,
                azureOpenAIApiVersion: process.env.azureOpenAIApiVersion,
                azureOpenAIApiInstanceName: process.env.azureOpenAIApiInstanceName,
                azureOpenAIApiDeploymentName: process.env.azureOpenAIApiDeploymentName,
            });

            // const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);
            this._chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStoreRetriever,
                {
                    memory: new BufferMemory({
                        memoryKey: "chat_history", // Must be set to "chat_history"
                    }),
                });
        }

        const res = await this._chain.call({
            question: query,
        });

        return res.text;
    }
}