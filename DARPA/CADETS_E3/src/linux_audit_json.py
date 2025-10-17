#!/usr/bin/env python3
"""
Convert Linux Audit logs into DARPA TC CDM JSONL suitable for KAIROS ingestion.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

AUDIT_MSG_RE = re.compile(r"audit\((?P<ts>[0-9]+\.[0-9]+):(?P<serial>[0-9]+)\)")
KV_PAIR_RE = re.compile(r'''(?P<key>[a-zA-Z0-9_]+)=(?P<val>"[^"]*"|\S+)''')

SYSCALL_EVENT_MAP = {
    "59": "EVENT_EXECUTE",      # execve
    "2": "EVENT_OPEN",          # open
    "87": "EVENT_EXECUTE",      # unlink
    "4": "EVENT_WRITE",         # write
    "0": "EVENT_READ",          # read
    "322": "EVENT_CONNECT",     # connect
    "44": "EVENT_ACCEPT",       # accept
}


@dataclass
class AuditRecord:
    record_type: str
    header: Dict[str, str]
    fields: Dict[str, str]


@dataclass
class AggregatedEvent:
    serial: str
    timestamp_ns: int
    syscall: Optional[str] = None
    records: Dict[str, List[AuditRecord]] = field(default_factory=lambda: {})

    def add_record(self, record: AuditRecord) -> None:
        self.records.setdefault(record.record_type, []).append(record)
        if record.record_type == "SYSCALL":
            self.syscall = record.fields.get("syscall")


def parse_line(line: str) -> Optional[AuditRecord]:
    line = line.strip()
    if not line or not line.startswith("type="):
        return None

    parts = line.split(" ", 1)
    record_type = parts[0].split("=")[1]
    remainder = parts[1] if len(parts) > 1 else ""

    header_match = AUDIT_MSG_RE.search(remainder)
    if not header_match:
        return None

    header = header_match.groupdict()
    kv_pairs = dict(
        (m.group("key"), m.group("val").strip('"'))
        for m in KV_PAIR_RE.finditer(remainder)
    )
    return AuditRecord(record_type=record_type, header=header, fields=kv_pairs)


def assemble_events(lines: List[str]) -> List[AggregatedEvent]:
    events: Dict[str, AggregatedEvent] = {}
    for line in lines:
        record = parse_line(line)
        if not record:
            continue

        serial = record.header["serial"]
        if serial not in events:
            seconds = float(record.header["ts"])
            events[serial] = AggregatedEvent(
                serial=serial,
                timestamp_ns=int(seconds * 1_000_000_000),
            )
        events[serial].add_record(record)
    return list(events.values())


def build_subject(syscall_rec: AuditRecord) -> Dict[str, object]:
    pid = syscall_rec.fields.get("pid")
    exe = syscall_rec.fields.get("exe")
    comm = syscall_rec.fields.get("comm")
    user = syscall_rec.fields.get("uid")
    uuid = f"process-{pid}-{syscall_rec.header['serial']}"
    return {
        "type": "SUBJECT_PROCESS",
        "uuid": uuid,
        "properties": {
            "pid": int(pid) if pid else None,
            "ppid": int(syscall_rec.fields.get("ppid", "0")),
            "userId": int(user) if user else None,
            "commandLine": comm,
            "exePath": exe,
        },
    }


def build_predicates(event: AggregatedEvent) -> List[Dict[str, object]]:
    predicates: List[Dict[str, object]] = []
    for path_record in event.records.get("PATH", []):
        path = path_record.fields.get("name") or path_record.fields.get("obj")
        if not path:
            continue
        uuid = f"file-{path_record.fields.get('inode', 'unknown')}"
        predicates.append(
            {
                "type": "OBJECT_FILE",
                "uuid": uuid,
                "properties": {
                    "path": path,
                    "inode": path_record.fields.get("inode"),
                    "mode": path_record.fields.get("mode"),
                    "dev": path_record.fields.get("dev"),
                },
            }
        )
    for sock_record in event.records.get("SOCKADDR", []):
        ip = sock_record.fields.get("addr")
        port = sock_record.fields.get("port")
        uuid = f"netflow-{ip}-{port}"
        predicates.append(
            {
                "type": "OBJECT_NETWORK_FLOW",
                "uuid": uuid,
                "properties": {
                    "remoteAddress": ip,
                    "remotePort": int(port, 16) if port else None,
                    "family": sock_record.fields.get("fam"),
                },
            }
        )
    return predicates


def event_type_from_syscall(syscall: Optional[str]) -> str:
    if syscall and syscall in SYSCALL_EVENT_MAP:
        return SYSCALL_EVENT_MAP[syscall]
    return "EVENT_GENERIC"


def convert_event(event: AggregatedEvent) -> Optional[Dict[str, object]]:
    syscall_records = event.records.get("SYSCALL")
    if not syscall_records:
        return None
    syscall_rec = syscall_records[0]
    predicate_objects = build_predicates(event)

    return {
        "eventId": event.serial,
        "timestampNanos": event.timestamp_ns,
        "eventType": event_type_from_syscall(event.syscall),
        "subject": build_subject(syscall_rec),
        "predicateObjects": predicate_objects,
        "properties": {
            "syscall": syscall_rec.fields.get("syscall"),
            "success": syscall_rec.fields.get("success") == "yes",
            "exitCode": int(syscall_rec.fields.get("exit", "0")),
            "auditKey": syscall_rec.fields.get("key"),
        },
    }


def convert_file(src: Path, dst: Path) -> None:
    lines = src.read_text().splitlines()
    aggregated = assemble_events(lines)

    with dst.open("w") as fout:
        for event in aggregated:
            cdm_event = convert_event(event)
            if not cdm_event:
                continue
            fout.write(json.dumps(cdm_event))
            fout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Linux Audit logs to DARPA TC CDM JSONL."
    )
    parser.add_argument("input", type=Path, help="Path to raw audit log")
    parser.add_argument("output", type=Path, help="Destination JSONL file")
    args = parser.parse_args()

    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()