import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import * as pg from "pg";
import { parse } from "ts-command-line-args";
import { randomUUID } from "crypto";

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
console.log(config);
const vectorStore = new PGVectorStore(embeddings, config);

const document_loader = new DirectoryLoader(args.dirPath, {
  ".pdf": (path) => new PDFLoader(path)
}, true);

const docs = await document_loader.load();

console.log(`Total documents: ${docs.length}`);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200
})

const allSplits = await splitter.splitDocuments(docs);
console.log(`Split documents into ${allSplits.length} sub-documents`);


const ids = allSplits.map(() => randomUUID());
await vectorStore.addDocuments(allSplits, { ids: ids });
