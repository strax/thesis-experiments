import sys


def dprint(message: str):
    print(f"[*] {message}", flush=True, file=sys.stdout)


def iprint(message: str):
    print(f"[+] {message}", flush=True, file=sys.stdout)


def wprint(message: str):
    print(f"[!] {message}", flush=True, file=sys.stderr)


def eprint(message: str):
    print(f"[-] {message}", flush=True, file=sys.stderr)


__all__ = ["dprint", "iprint", "wprint", "eprint"]
