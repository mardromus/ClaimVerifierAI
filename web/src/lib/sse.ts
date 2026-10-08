export interface SSEMessage {
  event: string;
  data: string;
}

/** Incremental parser for a `text/event-stream` body (handles chunks split anywhere). */
export class SSEParser {
  private buffer = "";

  push(chunk: string): SSEMessage[] {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const messages: SSEMessage[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const msg = parseBlock(block);
      if (msg) messages.push(msg);
      boundary = this.buffer.indexOf("\n\n");
    }
    return messages;
  }

  flush(): SSEMessage[] {
    const rest = this.buffer.trim();
    this.buffer = "";
    const msg = rest ? parseBlock(rest) : null;
    return msg ? [msg] : [];
  }
}

function parseBlock(block: string): SSEMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}
