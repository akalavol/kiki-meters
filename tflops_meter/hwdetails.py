"""Configuration détaillée (Windows) : lecteurs, disques physiques, RAM, carte mère, OS.

Les lecteurs viennent de l'API Windows (instantané). Le reste vient d'un seul appel PowerShell/CIM
(1 à 3 s), exécuté en arrière-plan au démarrage ; en cas d'échec, les champs restent simplement vides.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys

DRIVE_FIXED, DRIVE_REMOVABLE = 3, 2
MEMORY_TYPES = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 34: "DDR5", 35: "LPDDR5"}
BUS_TYPES = {"NVMe": "NVMe", "SATA": "SATA", "USB": "USB", "SAS": "SAS", "RAID": "RAID", "SD": "SD"}


def drives() -> list[dict]:
    """Lecteurs locaux (disques fixes et amovibles) avec espace libre."""
    if sys.platform != "win32":
        return []
    k32 = ctypes.windll.kernel32
    buf = ctypes.create_unicode_buffer(512)
    k32.GetLogicalDriveStringsW(512, buf)
    out = []
    for root in buf[:].split("\x00"):
        if not root:
            continue
        t = k32.GetDriveTypeW(root)
        if t not in (DRIVE_FIXED, DRIVE_REMOVABLE):
            continue
        free, total = ctypes.c_ulonglong(), ctypes.c_ulonglong()
        if not k32.GetDiskFreeSpaceExW(root, ctypes.byref(free), ctypes.byref(total), None):
            continue
        label = ctypes.create_unicode_buffer(261)
        fs = ctypes.create_unicode_buffer(261)
        k32.GetVolumeInformationW(root, label, 261, None, None, None, fs, 261)
        out.append({"letter": root[0], "root": root, "label": label.value, "fs": fs.value,
                    "total_gb": total.value / 1e9, "free_gb": free.value / 1e9,
                    "removable": t == DRIVE_REMOVABLE})
    return out


_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$r = [ordered]@{}
$r.board = Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer, Product
$r.os = (Get-CimInstance Win32_OperatingSystem | Select-Object Caption, BuildNumber)
$r.ram = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed, ConfiguredClockSpeed, SMBIOSMemoryType, Manufacturer, PartNumber)
$r.gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
$r.disks = @(Get-PhysicalDisk | Select-Object DeviceId, FriendlyName, @{n='MediaType';e={"$($_.MediaType)"}}, @{n='BusType';e={"$($_.BusType)"}}, Size)
$r.parts = @(Get-Partition | Where-Object DriveLetter | Select-Object @{n='Letter';e={"$($_.DriveLetter)"}}, DiskNumber)
$r | ConvertTo-Json -Depth 4 -Compress
"""


def collect() -> dict:
    """Détails matériels (peut prendre quelques secondes)."""
    info = {"drives": drives()}
    if sys.platform != "win32":
        return info
    try:
        flags = 0x08000000  # CREATE_NO_WINDOW : pas de fenêtre console dans l'exécutable
        p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS],
                           capture_output=True, text=True, timeout=25, creationflags=flags,
                           encoding="utf-8", errors="replace")
        raw = json.loads(p.stdout.strip() or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return info

    board = raw.get("board") or {}
    if board.get("Product"):
        info["board"] = f"{(board.get('Manufacturer') or '').strip()} {board['Product'].strip()}".strip()
    osd = raw.get("os") or {}
    if osd.get("Caption"):
        info["os"] = f"{osd['Caption'].replace('Microsoft ', '')} (build {osd.get('BuildNumber')})"
    mods = [m for m in raw.get("ram") or [] if m.get("Capacity")]
    if mods:
        total = sum(int(m["Capacity"]) for m in mods) / 1024 ** 3
        speed = max((m.get("ConfiguredClockSpeed") or m.get("Speed") or 0) for m in mods)
        rated = max((m.get("Speed") or 0) for m in mods)
        kind = MEMORY_TYPES.get(mods[0].get("SMBIOSMemoryType"), "")
        sizes = {round(int(m["Capacity"]) / 1024 ** 3) for m in mods}
        layout = f"{len(mods)} × {sizes.pop()} Go" if len(sizes) == 1 else f"{len(mods)} barrettes"
        info["ram"] = {"total_gb": total, "type": kind, "speed_mts": speed, "rated_mts": rated,
                       "modules": len(mods), "layout": layout,
                       "part": (mods[0].get("PartNumber") or "").strip()}
    gpus = raw.get("gpu") or []
    if gpus:
        info["gpu_drivers"] = {g.get("Name"): g.get("DriverVersion") for g in gpus if g.get("Name")}
    disks = {str(d.get("DeviceId")): d for d in raw.get("disks") or []}
    phys = []
    for d in disks.values():
        media = d.get("MediaType") or ""
        bus = d.get("BusType") or ""
        kind = "SSD NVMe" if bus == "NVMe" else "SSD" if media == "SSD" else "HDD" if media == "HDD" else bus or "?"
        phys.append({"id": str(d.get("DeviceId")), "model": (d.get("FriendlyName") or "").strip(),
                     "kind": kind, "bus": BUS_TYPES.get(bus, bus), "size_gb": (d.get("Size") or 0) / 1e9,
                     "letters": []})
    by_id = {p["id"]: p for p in phys}
    letter_disk = {}
    for part in raw.get("parts") or []:
        dn = str(part.get("DiskNumber"))
        if dn in by_id and part.get("Letter"):
            by_id[dn]["letters"].append(part["Letter"])
            letter_disk[part["Letter"]] = by_id[dn]
    info["disks"] = sorted(phys, key=lambda p: p["id"])
    for dr in info["drives"]:
        d = letter_disk.get(dr["letter"])
        if d:
            dr["model"], dr["kind"] = d["model"], d["kind"]
    return info


def drive_label(dr: dict) -> str:
    """« C: · Samsung SSD 990 PRO 2TB (SSD NVMe) »"""
    model = dr.get("model") or dr.get("label") or "disque"
    kind = f" ({dr['kind']})" if dr.get("kind") else ""
    return f"{dr['letter']}: · {model}{kind}"


def test_path(dr: dict) -> str:
    """Fichier de test : %TEMP% pour le lecteur système (la racine demande les droits admin), sinon la racine."""
    temp = os.environ.get("TEMP") or os.environ.get("TMP") or ""
    if temp and temp[0].upper() == dr["letter"].upper():
        return os.path.join(temp, "TflopsMeter_test.tmp")
    return f"{dr['letter']}:\\TflopsMeter_test.tmp"
