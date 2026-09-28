#!/usr/bin/env node
// Rewrites a .vsix (a zip) so the same sources always give the same bytes: entries sorted by name, one fixed
// timestamp, no per-machine attributes, each entry deflated again at a fixed level. vsce stamps every entry with the
// time the file was compiled or packed and lists them in filesystem order, so two builds of one commit used to differ
// only in those. Also writes <vsix>.manifest.json: the sha256 of every packed file, to compare contents across
// machines or vsce versions. Node's standard library only.
"use strict";

const fs = require("fs");
const zlib = require("zlib");
const crypto = require("crypto");

// 1980-01-01 00:00:00, the earliest time a zip entry can carry.
const DOS_TIME = 0;
const DOS_DATE = (0 << 9) | (1 << 5) | 1;

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function readEntries(buf) {
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("not a zip file (no end-of-central-directory record)");
  const count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  // A zip64 archive keeps its real counts and offsets elsewhere: refuse it rather than misread it.
  if (count === 0xffff || p === 0xffffffff || buf.readUInt32LE(eocd + 12) === 0xffffffff) {
    throw new Error("zip64 archives are not handled");
  }
  const entries = [];
  for (let n = 0; n < count; n++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error("broken central directory");
    const flags = buf.readUInt16LE(p + 8);
    const method = buf.readUInt16LE(p + 10);
    const compSize = buf.readUInt32LE(p + 20);
    const size = buf.readUInt32LE(p + 24);
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    const localOff = buf.readUInt32LE(p + 42);
    const name = buf.toString("utf8", p + 46, p + 46 + nameLen);
    if (flags & 0x1) throw new Error(`${name}: encrypted entries are not handled`);
    if (compSize === 0xffffffff || size === 0xffffffff || localOff === 0xffffffff) {
      throw new Error(`${name}: zip64 entries are not handled`);
    }
    const lNameLen = buf.readUInt16LE(localOff + 26);
    const lExtraLen = buf.readUInt16LE(localOff + 28);
    const start = localOff + 30 + lNameLen + lExtraLen;
    const raw = buf.subarray(start, start + compSize);
    let data;
    if (method === 0) data = Buffer.from(raw);
    else if (method === 8) data = zlib.inflateRawSync(raw);
    else throw new Error(`${name}: compression method ${method} is not handled`);
    if (data.length !== size) throw new Error(`${name}: ${data.length} bytes read, ${size} expected`);
    entries.push({ name, data });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

function writeZip(entries) {
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const e of entries) {
    const name = Buffer.from(e.name, "utf8");
    const comp = zlib.deflateRawSync(e.data, { level: 9, memLevel: 8, strategy: zlib.constants.Z_DEFAULT_STRATEGY });
    const crc = crc32(e.data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6); // UTF-8 names
    local.writeUInt16LE(8, 8);
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(comp.length, 18);
    local.writeUInt32LE(e.data.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);
    locals.push(local, name, comp);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4); // made by: MS-DOS, version 2.0
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(8, 10);
    central.writeUInt16LE(DOS_TIME, 12);
    central.writeUInt16LE(DOS_DATE, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(comp.length, 20);
    central.writeUInt32LE(e.data.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(0, 38); // no external attributes
    central.writeUInt32LE(offset, 42);
    centrals.push(central, name);
    offset += 30 + name.length + comp.length;
  }
  const cd = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cd.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, end]);
}

function main() {
  const path = process.argv[2];
  if (!path) {
    console.error("usage: node scripts/normalize-vsix.js <file.vsix>");
    process.exit(2);
  }
  const entries = readEntries(fs.readFileSync(path));
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const out = writeZip(entries);
  fs.writeFileSync(path, out);
  const sha = (b) => crypto.createHash("sha256").update(b).digest("hex");
  const manifest = {
    vsix_sha256: sha(out),
    files: Object.fromEntries(entries.map((e) => [e.name, { sha256: sha(e.data), bytes: e.data.length }])),
  };
  fs.writeFileSync(`${path}.manifest.json`, JSON.stringify(manifest, null, 2) + "\n");
  console.log(`normalized ${path}: ${entries.length} files, sha256 ${manifest.vsix_sha256}`);
}

main();
