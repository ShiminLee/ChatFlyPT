"""Temporary: WMI shim installed into the body venv by setup-body-env.ps1 (kept in repo)."""
import os
import platform

_ARCHITECTURE_NAMES = {'AMD64': 'AMD64', 'ARM64': 'ARM64', 'X86': 'x86',
                       'IA64': 'ia64', 'ARM': 'ARM'}


def _machine_from_environment():
    # WOW64 processes mask the native architecture, so prefer the W6432 variable.
    architecture = (os.environ.get('PROCESSOR_ARCHITEW6432')
                    or os.environ.get('PROCESSOR_ARCHITECTURE') or '').upper()
    return _ARCHITECTURE_NAMES.get(architecture, architecture or 'AMD64')


if hasattr(platform, '_get_machine_win32'):
    platform._get_machine_win32 = _machine_from_environment
