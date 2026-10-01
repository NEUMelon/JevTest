"""Finish the already authorized collection after its process exits once.

This is a one-shot job continuation, not a recurring monitor. A collection stop
does not trigger automatic restart, and incomplete groups remain blocked.
"""
import argparse
import ctypes
import json
from pathlib import Path
from .api_missing import reobserve
from .llm_aggregate import aggregate
from .progress_report import main as report

BASE=Path(__file__).resolve().parents[2]


def main(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:raise OSError(ctypes.get_last_error(),'Cannot observe collection process')
    print(f'FINISH_API_WAITING_FOR_COLLECTION pid={pid}',flush=True)
    try:
        while True:
            state=kernel.WaitForSingleObject(handle,1000)
            if state==0:break
            if state!=258:raise OSError('WaitForSingleObject failed')
    finally:kernel.CloseHandle(handle)
    collection=BASE/'runs/api_llm_recovery_20260930'
    repairs=BASE/'runs/api_llm_missing_20260930'
    manifest=json.loads((collection/'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] not in ['COLLECTED_NOT_FROZEN','COLLECTED_WITH_MISSING_NOT_FROZEN']:
        report();print('FINISH_API_COLLECTION_STOPPED_REVIEW_REQUIRED',flush=True);return
    try:
        reobserve(collection,repairs)
        aggregate(collection,BASE/'runs/llm_recovered_20260930',repairs)
    finally:report()
    print('FINISH_API_EVALUATION_SAVED_NOT_FROZEN',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);a=ap.parse_args();main(a.wait_pid)
