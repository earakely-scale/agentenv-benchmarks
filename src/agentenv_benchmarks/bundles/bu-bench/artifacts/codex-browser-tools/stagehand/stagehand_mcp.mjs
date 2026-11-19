#!/usr/bin/env node
// Stdio MCP server exposing Stagehand v3 (navigate, act, extract, observe, screenshot, get_url) on an existing
// browser. Mirrors the tool surface of Stagehand's own MCP server (@browserbasehq/mcp-server-browserbase) without
// its Browserbase session tools. Stagehand attaches to STAGEHAND_CDP_URL and only disconnects on shutdown.
//
// Env: STAGEHAND_CDP_URL (CDP websocket), STAGEHAND_MODEL_KEY_FILE (file holding the model API key),
// STAGEHAND_LOG_DIR (logs; stdout is the MCP channel), STAGEHAND_MODEL (default openai/gpt-6.1-terra),
// STAGEHAND_REASONING_EFFORT (default medium), STAGEHAND_BASE_URL (optional OpenAI-compatible endpoint; AgentEnv's
// model endpoint, `<LITELLM_BASE_URL>/openai/v1`), STAGEHAND_HEADERS (optional JSON object of headers sent on each model
// call: the endpoint's LITELLM_EXTRA_HEADERS). Otherwise as the bake-off ran it (harnesses/codex-browser-tools).
import fs from "node:fs";
import path from "node:path";
import { format } from "node:util";

const required = (name) => {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
};
const cdpUrl = required("STAGEHAND_CDP_URL");
const logDir = required("STAGEHAND_LOG_DIR");
const apiKey = fs.readFileSync(required("STAGEHAND_MODEL_KEY_FILE"), "utf8").trim();
if (!apiKey) throw new Error("STAGEHAND_MODEL_KEY_FILE is empty");
const modelName = process.env.STAGEHAND_MODEL || "openai/gpt-6.1-terra";
const reasoningEffort = process.env.STAGEHAND_REASONING_EFFORT || "medium";
const baseURL = process.env.STAGEHAND_BASE_URL || undefined;
// Like AgentEnv's codex_config: headers that aren't a JSON object are ignored, never fatal for the arm's only tool.
let headers = {};
try {
  const parsed = JSON.parse(process.env.STAGEHAND_HEADERS || "{}");
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) headers = parsed;
} catch {
  // ignored, as above
}

// Everything except the MCP transport goes to the log file: Stagehand falls back to console.log outside an instance
// context, and any stray stdout byte corrupts the protocol.
fs.mkdirSync(logDir, { recursive: true });
const log = fs.createWriteStream(path.join(logDir, "stagehand-mcp.log"), { flags: "a" });
const write = (level, text) => log.write(`${new Date().toISOString()} ${level} ${text}\n`);
for (const level of ["log", "info", "debug", "warn", "error"]) {
  console[level] = (...args) => write(level, format(...args));
}
// Stagehand's flow logger reads this at import time and writes per-session JSONL event logs under it.
process.env.BROWSERBASE_CONFIG_DIR = logDir;

const { Stagehand } = await import("@browserbasehq/stagehand");
const { McpServer } = await import("@modelcontextprotocol/sdk/server/mcp.js");
const { StdioServerTransport } = await import("@modelcontextprotocol/sdk/server/stdio.js");
const { z } = await import("zod");

let stagehand = null;
async function getStagehand() {
  if (stagehand?.context) return stagehand;
  const instance = new Stagehand({
    env: "LOCAL",
    localBrowserLaunchOptions: { cdpUrl },
    // keepAlive skips every browser cleanup path; attaching via cdpUrl already never launches or kills Chrome.
    keepAlive: true,
    model: { modelName, apiKey, reasoningEffort, ...(baseURL ? { baseURL } : {}),
             ...(Object.keys(headers).length ? { headers } : {}) },
    verbose: 1,
    disablePino: true,
    logger: (line) => write(`stagehand.${line.level ?? 1}`, JSON.stringify(line)),
  });
  await instance.init();
  write("info", `attached model=${modelName} reasoningEffort=${reasoningEffort}`);
  stagehand = instance;
  return stagehand;
}

async function activePage() {
  return (await getStagehand()).context.awaitActivePage();
}

async function pageLine() {
  const page = await activePage();
  return `URL: ${page.url()}\nTitle: ${await page.title()}`;
}

// One page, one model: run tool calls one at a time even if the client sends them in parallel.
let queue = Promise.resolve();
function tool(handler) {
  return (args) => {
    const run = queue.then(async () => {
      try {
        return await handler(args);
      } catch (error) {
        write("error", error?.stack || String(error));
        return { isError: true, content: [{ type: "text", text: `Error: ${error?.message || error}` }] };
      }
    });
    queue = run.catch(() => {});
    return run;
  };
}
const text = (value) => ({ content: [{ type: "text", text: value }] });

// Stagehand sends extract schemas to OpenAI in strict mode, which rejects open objects and optional properties.
function strictSchema(schema) {
  if (Array.isArray(schema)) return schema.map(strictSchema);
  if (!schema || typeof schema !== "object") return schema;
  const out = {};
  for (const [key, value] of Object.entries(schema)) {
    const named = ["properties", "$defs", "definitions"].includes(key) && value && typeof value === "object";
    const map = (entries) => Object.fromEntries(Object.entries(entries).map(([k, v]) => [k, strictSchema(v)]));
    out[key] = named ? map(value) : strictSchema(value);
  }
  if (out.type === "object" || out.properties) {
    out.properties ??= {};
    out.required = Object.keys(out.properties);
    out.additionalProperties = false;
  }
  return out;
}

// The agent's JSON Schema as a zod 3 schema. Stagehand runs on zod 3 because, with zod 4 and AI SDK 5, the nullable
// object in its own act schema loses additionalProperties:false and OpenAI strict mode rejects every act call.
function toZod(schema) {
  if (!schema || typeof schema !== "object") return z.any();
  const types = Array.isArray(schema.type) ? schema.type : [schema.type];
  const nullable = types.includes("null");
  const options = (schema.anyOf || schema.oneOf || []).filter((s) => s?.type !== "null");
  let out;
  if (options.length) out = options.length === 1 ? toZod(options[0]) : z.union(options.map(toZod));
  else if (schema.enum) out = schema.enum.every((v) => typeof v === "string") ? z.enum(schema.enum) : z.any();
  else if (types.includes("object") || schema.properties) {
    out = z.object(Object.fromEntries(Object.entries(schema.properties || {}).map(([k, v]) => [k, toZod(v)]))).strict();
  } else if (types.includes("array")) out = z.array(toZod(schema.items));
  else if (types.includes("string")) out = z.string();
  else if (types.includes("integer")) out = z.number().int();
  else if (types.includes("number")) out = z.number();
  else if (types.includes("boolean")) out = z.boolean();
  else out = z.any();
  if (schema.description) out = out.describe(schema.description);
  return nullable || (schema.anyOf || schema.oneOf || []).some((s) => s?.type === "null") ? out.nullable() : out;
}

const server = new McpServer({ name: "stagehand", version: "3.7.3" });

server.registerTool("navigate", {
  description: "Navigate the current tab to a URL.",
  inputSchema: { url: z.string().describe("The URL to navigate to") },
}, tool(async ({ url }) => {
  await (await activePage()).goto(url, { waitUntil: "domcontentloaded" });
  return text(`Navigated to: ${url}\n${await pageLine()}`);
}));

server.registerTool("act", {
  description: "Perform a single action on the page (e.g., click, type).",
  inputSchema: {
    action: z.string().describe("The action to perform. Should be as atomic and specific as possible, i.e. "
      + "'Click the sign in button' or 'Type 'hello' into the search input'."),
    variables: z.record(z.string(), z.string()).optional().describe("Variables used in the action template. "
      + "ONLY use variables if you're dealing with sensitive data or dynamic content. When using variables, you MUST "
      + "have the variable key in the action template. ie: {\"action\": \"Fill in the password with %password%\", "
      + "\"variables\": {\"password\": \"123456\"}}"),
  },
}, tool(async ({ action, variables }) => {
  const result = await (await getStagehand()).act(action, variables ? { variables } : undefined);
  const output = `${JSON.stringify(result)}\n${await pageLine()}`;
  return { isError: !result.success, content: [{ type: "text", text: output }] };
}));

server.registerTool("extract", {
  description: "Extract structured data or text from the current page using an instruction. Without an "
    + "instruction, returns the page's text.",
  inputSchema: {
    instruction: z.string().optional().describe("The specific instruction for what information to extract from the "
      + "current page, e.g. 'Extract all product names and prices from the listing page'."),
    schema: z.record(z.string(), z.unknown()).optional().describe("Optional JSON Schema (type object) the extracted "
      + "data must match. Every property is required; use a type like [\"string\", \"null\"] for missing values."),
  },
}, tool(async ({ instruction, schema }) => {
  const sh = await getStagehand();
  let data;
  if (!instruction) data = await sh.extract();
  else if (schema) data = await sh.extract(instruction, toZod(strictSchema(schema)));
  else data = await sh.extract(instruction);
  return text(`Extracted content:\n${JSON.stringify(data, null, 2)}`);
}));

server.registerTool("observe", {
  description: "Find interactive elements on the page from an instruction; returns candidate actions.",
  inputSchema: {
    instruction: z.string().describe("Specific instruction for what elements to observe, e.g. 'Find the red login "
      + "button in the top right corner' or 'Locate the search input field with placeholder text'."),
  },
}, tool(async ({ instruction }) => {
  return text(`Observations: ${JSON.stringify(await (await getStagehand()).observe(instruction))}`);
}));

server.registerTool("screenshot", {
  description: "Capture a screenshot of the current viewport.",
  inputSchema: {},
}, tool(async () => {
  const image = await (await activePage()).screenshot({ type: "png" });
  return { content: [{ type: "text", text: await pageLine() },
    { type: "image", data: image.toString("base64"), mimeType: "image/png" }] };
}));

server.registerTool("get_url", {
  description: "Return the current page URL (full URL with query/fragment) and title.",
  inputSchema: {},
}, tool(async () => text(await pageLine())));

// Disconnect only: with keepAlive and an attached cdpUrl, close() just drops the CDP websocket.
let closing = false;
async function shutdown(reason) {
  if (closing) return;
  closing = true;
  write("info", `shutdown: ${reason}`);
  try {
    await stagehand?.close();
  } catch (error) {
    write("error", `close: ${error?.message || error}`);
  }
  log.end(() => process.exit(0));
}
process.stdin.on("end", () => shutdown("stdin closed"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));

await server.connect(new StdioServerTransport());
write("info", `ready cdp=${cdpUrl.replace(/\?.*$/, "")}`);
