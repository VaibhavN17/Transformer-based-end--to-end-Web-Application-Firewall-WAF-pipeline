"""
log_parser.py  –  Parse Apache Combined Log Format and Nginx access logs.

Yields dicts ready to be scored by the WAF model.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


# ── Log Line Patterns ────────────────────────────────────────────────────────

APACHE_COMBINED = re.compile(
    r'(?P<ip>\S+)\s+'           # Remote host
    r'\S+\s+'                   # ident (-)
    r'(?P<user>\S+)\s+'         # auth user (-)
    r'\[(?P<time>[^\]]+)\]\s+'  # [time]
    r'"(?P<request>[^"]*)"\s+'  # "GET /path HTTP/1.1"
    r'(?P<status>\d{3})\s+'     # status code
    r'(?P<size>\S+)'            # response size
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<ua>[^"]*)")?'  # optional referer + UA
)

NGINX_ERROR = re.compile(
    r'(?P<time>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\s+'
    r'\[(?P<level>\w+)\]\s+'
    r'.*?client:\s*(?P<ip>[\d.]+).*?'
    r'request:\s*"(?P<request>[^"]*)"'
)


@dataclass
class LogEntry:
    ip: str
    time: str
    method: str
    path: str
    protocol: str
    status: int
    size: str
    user_agent: str
    referer: str
    raw_request: str

    def to_waf_input(self) -> str:
        """Compose a single string for the WAF model to analyse."""
        return f"{self.method} {self.path} {self.user_agent} {self.referer}".strip()

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_request(request: str) -> tuple[str, str, str]:
    """Split 'GET /path HTTP/1.1' → (method, path, protocol)."""
    parts = request.split(" ", 2)
    method   = parts[0] if len(parts) > 0 else "-"
    path     = parts[1] if len(parts) > 1 else "/"
    protocol = parts[2] if len(parts) > 2 else "HTTP/1.1"
    return method, path, protocol


def parse_apache_line(line: str) -> LogEntry | None:
    m = APACHE_COMBINED.match(line.strip())
    if not m:
        return None
    method, path, protocol = _parse_request(m.group("request"))
    return LogEntry(
        ip=m.group("ip"),
        time=m.group("time"),
        method=method,
        path=path,
        protocol=protocol,
        status=int(m.group("status")),
        size=m.group("size"),
        user_agent=m.group("ua") or "-",
        referer=m.group("referer") or "-",
        raw_request=m.group("request"),
    )


def parse_nginx_error_line(line: str) -> LogEntry | None:
    m = NGINX_ERROR.match(line.strip())
    if not m:
        return None
    method, path, protocol = _parse_request(m.group("request"))
    return LogEntry(
        ip=m.group("ip"),
        time=m.group("time"),
        method=method,
        path=path,
        protocol=protocol,
        status=0,
        size="-",
        user_agent="-",
        referer="-",
        raw_request=m.group("request"),
    )


def parse_log_file(filepath: str | Path, fmt: str = "apache") -> Iterator[LogEntry]:
    """
    Stream-parse a log file and yield LogEntry objects.

    fmt: 'apache' | 'nginx'
    """
    parser = parse_apache_line if fmt == "apache" else parse_nginx_error_line
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            entry = parser(line)
            if entry:
                yield entry


# ── Demo / CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sample_log = """
127.0.0.1 - frank [10/Oct/2024:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326 "-" "Mozilla/5.0"
10.0.0.1 - - [10/Oct/2024:13:55:37 -0700] "GET /search?q=' OR '1'='1 HTTP/1.1" 200 512 "-" "curl/7.64.1"
192.168.1.1 - - [10/Oct/2024:13:55:38 -0700] "GET /page?id=<script>alert(1)</script> HTTP/1.1" 200 128 "-" "python-requests/2.28"
172.16.0.1 - - [10/Oct/2024:13:55:39 -0700] "GET /../../../etc/passwd HTTP/1.1" 404 0 "-" "Nikto/2.1.6"
""".strip()

    print("Demo: parsing sample Apache log lines\n")
    for raw_line in sample_log.splitlines():
        entry = parse_apache_line(raw_line)
        if entry:
            print(f"  IP={entry.ip}  Method={entry.method}  Path={entry.path[:60]}")
            print(f"  WAF input: {entry.to_waf_input()[:80]}")
            print()
