import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import * as pg from "pg";
import { parse } from "ts-command-line-args";
import { randomUUID } from "crypto";
import { logger } from "./logger.js";
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import type { Document } from "langchain";
import { Effect, pipe } from "effect";
import { OpenAIEmbeddings } from "@langchain/openai";

interface CliArguments {
  collectionName?: string,
  postgresHost?: string
  postgresPort?: number
  postgresUser?: string
  postgresPass?: string
  postgresDb?: string
  dirPath: string
  embeddingName: string
}

const args = parse<CliArguments>({
  dirPath: String,
  collectionName: { type: String, optional: true },
  postgresHost: { type: String, optional: true },
  postgresPort: { type: Number, optional: true, defaultValue: 5432 },
  postgresUser: { type: String, optional: true },
  postgresPass: { type: String, optional: true },
  postgresDb: { type: String, optional: true },
  embeddingName: { type: String, defaultValue: "text-embedding-3-large" }
});

type DefaultValueProvider<T> = T | (() => T);

interface ResolverOptions<T> {
  cliKey: keyof CliArguments;
  envKey: string;
  defaultValue: DefaultValueProvider<T>;
  transform?: (val: any) => T;
}

interface UndefinableResolverOption<T> {
  cliKey: keyof CliArguments;
  envKey: string;
  transform?: (val: any) => T;
}

function createUndefinableConfigResolver<T>(options: UndefinableResolverOption<T>): () => T | undefined {
  return () => {
    // 1. Check CLI Arguments
    if (args[options.cliKey] !== undefined) {
      const val = args[options.cliKey];
      return options.transform ? options.transform(val) : (val as unknown as T);
    }

    // 2. Check Environment Variables
    if (process.env[options.envKey] !== undefined) {
      const val = process.env[options.envKey];
      return options.transform ? options.transform(val) : (val as unknown as T);
    }

    return undefined;

  };
}

function createConfigResolver<T>(options: ResolverOptions<T>): () => T {
  return () => {
    // 1. Check CLI Arguments
    if (args[options.cliKey] !== undefined) {
      const val = args[options.cliKey];
      return options.transform ? options.transform(val) : (val as unknown as T);
    }

    // 2. Check Environment Variables
    if (process.env[options.envKey] !== undefined) {
      const val = process.env[options.envKey];
      return options.transform ? options.transform(val) : (val as unknown as T);
    }

    // 3. Fall back to Default Value (Value or Function)
    if (typeof options.defaultValue === "function") {
      return (options.defaultValue as () => T)();
    }
    return options.defaultValue;
  };
}

const exitWithMissingError = (paramName: string, envName: string) => () => {
  console.log(`${paramName} is not set. Either set environment variable \`${envName}\` or set command line argument \`--${paramName}\`.`);
  process.exit(1);
};

const postgresUser = createUndefinableConfigResolver<string>({
  cliKey: "postgresUser",
  envKey: "POSTGRES_USER",
});


const postgresHost = createUndefinableConfigResolver<string>({
  cliKey: "postgresHost",
  envKey: "POSTGRES_HOST",
});

const postgresPort = createUndefinableConfigResolver<number>({
  cliKey: "postgresPort",
  envKey: "POSTGRES_PORT",
  transform: (val) => {
    // Enforce base-10 conversion (especially useful for the raw process.env string)
    const num = typeof val === "string" ? parseInt(val, 10) : Number(val);

    if (!Number.isInteger(num)) {
      throw new Error(`${val} is not an integer`);
    }
    return num;
  },
});

const collectionName = createConfigResolver<string>({
  cliKey: "collectionName",
  envKey: "COLLECTION_NAME",
  defaultValue: exitWithMissingError("collectionName", "COLLECTION_NAME"),
});

const postgresPassword = createUndefinableConfigResolver<string>({
  cliKey: "postgresPass",
  envKey: "POSTGRES_PASS",
});


const postgresDatabase = createUndefinableConfigResolver<string>({
  cliKey: "postgresDb",
  envKey: "POSTGRES_DB",
})

const embeddings = new OpenAIEmbeddings({
  model: args.embeddingName
});

const config = {
  postgresConnectionOptions: {
    type: "postgres",
    host: postgresHost(),
    port: postgresPort(),
    user: postgresUser(),
    password: postgresPassword(),
    database: postgresDatabase(),
  } as pg.PoolConfig,
  tableName: "langchain_pg_embedding",
  collectionTableName: "langchain_pg_collection",
  collectionName: collectionName(),
  schemaName: "public",
  columns: {
    idColumnName: "id",
    vectorColumnName: "embedding",
    contentColumnName: "document",
    metadataColumnName: "cmetadata"
  },
  verbose: true
}

async function loadPDFDocumentsFromDirectory(path: string) {

  const document_loader = new DirectoryLoader(path, {
    ".pdf": (path) => {
      logger.debug(`Loading PDF document: ${path}`);
      return new PDFLoader(path);
    }
  }, true);

  return await document_loader.load();
}

function addFileIdAndNameInDocumentMetaData(docs: Document<Record<string, any>>[]): Document<Record<string, any>>[] {
  return docs.map((doc) => {
    const filePath = doc.metadata.source || "unknown";

    const fileId = randomUUID();

    doc.metadata.file_id = fileId;
    doc.metadata.file_name = filePath.split('/').pop();

    return doc
  })
}

async function splitDocuments(docs: Document<Record<string, any>>[]) {

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
  })

  const splits = await splitter.splitDocuments(docs);

  return splits.map(doc => {
    const trimmed = doc.pageContent ? doc.pageContent.trim() : "";

    // Check if the chunk is completely empty or lacks alphanumeric text
    const hasAlphaNumeric = /[a-zA-Z0-9]/.test(trimmed);

    if (trimmed.length === 0 || !hasAlphaNumeric) {
      // Retain the structural meaning by appending a structural context token
      // This forces the embedding model to return a valid, multi-dimensional vector
      doc.pageContent = `[Structure/Formatting] ${doc.pageContent}`;
    }

    return doc;
  });
}

function createDocumentIdentifier(document: Document<Record<string, any>>, index: number) {
  return `${document.metadata.file_id}_chunk_${index}`;
}

function performBatch<T, R>(batchSize: number, values: T[], callback: (values: T[], index: number) => R): R[] {
  let result: R[] = [];
  for (let i = 0; i < values.length; i += batchSize) {
    const batchSplits = values.slice(i, i + batchSize);
    result.push(callback(batchSplits, i));
  }
  return result;
}

function zip<T, U>(array1: T[], array2: U[]): [T, U][] {
  const length = Math.min(array1.length, array2.length);
  const result: [T, U][] = new Array(length);

  for (let i = 0; i < length; i++) {
    result[i] = [array1[i]!, array2[i]!];
  }

  return result;
}


async function shouldContinueFromUser(yer_or_no_message: string): Promise<boolean> {

  const rl = readline.createInterface({ input, output });

  try {
    const yesOrNo = await rl.question(`${yer_or_no_message}: [Y/n] `);

    switch (yesOrNo) {
      case 'y':
      case 'yes':
      case 'Y':
        return true;
      case 'n':
      case 'N':
      case 'no':
        return false;
      default:
        console.log('Invalid input. Please enter Y or N.');
        return shouldContinueFromUser(yer_or_no_message);
    }
  } finally {
    rl.close()
  }
}


async function main() {
  const getDocuments = pipe(
    args.dirPath,
    (path) => Effect.promise(() => loadPDFDocumentsFromDirectory(path)),
    Effect.map(addFileIdAndNameInDocumentMetaData),
    Effect.andThen(splitDocuments),
  );

  const docs = await Effect.runPromise(getDocuments);
  const ids = docs.map(createDocumentIdentifier);

  const BATCH_SIZE = 1024;

  const batchDocs = performBatch(BATCH_SIZE, docs, (docs, _i) => docs);
  const batchIds = performBatch(BATCH_SIZE, ids, (ids) => ids);

  const vectorStore = new PGVectorStore(embeddings, config);

  let index: number = 0;
  for (const [docs, ids] of zip(batchDocs, batchIds)) {

    try {
      await vectorStore.addDocuments(docs, { ids });
    } catch (err) {
      console.error(`❌ Critical error in batch starting at index ${index}:`, err);
      // Log problematic contents to find out if a chunk is malicious/unsupported
      console.error("Sample batch content:", docs.map(s => s.pageContent.substring(0, 60)));

      console.log("Ids:", [...new Set(docs.map((doc) => doc.metadata.file_name))]);

      if (await shouldContinueFromUser("Should continue with other batch?")) {
        continue;
      }

      throw err;
    }

    // Brief pause to prevent hitting Google AI Studio RPM limits
    if (index + BATCH_SIZE < docs.length) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }

    index++;
  }

}

await main();
