# app/keep_alive.py
import asyncio
import httpx
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI 

# --- Keep-Alive Configuration ---
RENDER_EXTERNAL_URL = os.getenv('RENDER_EXTERNAL_URL')
SELF_KEEP_ALIVE_TARGET_URL = f"{RENDER_EXTERNAL_URL}/" if RENDER_EXTERNAL_URL else "http://localhost:8000/" # Assuming root path for ping
KEEP_ALIVE_INTERVAL_SECONDS = 14 * 60  # Ping every 14 minutes

keep_alive_task_ref: asyncio.Task | None = None # Holds the task reference

async def _self_keep_alive_ping_task_loop():
    """Periodically pings this app's own root URL to keep it alive on Render."""
    async with httpx.AsyncClient(timeout=30.0) as client: # Set a timeout for requests
        while True:
            try:
                print(f"[Keep-Alive-Self] Sending self-ping to {SELF_KEEP_ALIVE_TARGET_URL}...")
                response = await client.get(SELF_KEEP_ALIVE_TARGET_URL)
                response.raise_for_status()  
                print(f"[Keep-Alive-Self] Self-ping to {SELF_KEEP_ALIVE_TARGET_URL} successful: Status {response.status_code}")
            except httpx.RequestError as exc:
                print(f"[Keep-Alive-Self] Self-ping to {SELF_KEEP_ALIVE_TARGET_URL} failed (RequestError): {exc}")
            except httpx.HTTPStatusError as exc:
                print(f"[Keep-Alive-Self] Self-ping to {SELF_KEEP_ALIVE_TARGET_URL} received error status: {exc.response.status_code} - {exc}")
            except Exception as exc: # Catch any other unexpected errors during the ping
                print(f"[Keep-Alive-Self] An unexpected error occurred in _self_keep_alive_ping_task_loop: {exc}")
            
            # Check if task was cancelled (e.g., during shutdown)
            try:
                # Sleep for the interval, but allow cancellation to interrupt
                await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                print("[Keep-Alive-Self] Ping loop task cancelled, exiting.")
                break


@asynccontextmanager
async def lifespan_manager(app: FastAPI): # app parameter is conventional for lifespan
    # --- Code to run on startup ---
    global keep_alive_task_ref
    is_production_env = os.getenv('RENDER_INSTANCE_ID') or os.getenv('WORKER_CLASS') or not os.getenv('RELOADER_MAIN_PID')
    
    if is_production_env:
        print(f"[Keep-Alive-Self] Production-like environment detected. Starting self keep-alive task for URL: {SELF_KEEP_ALIVE_TARGET_URL} with interval {KEEP_ALIVE_INTERVAL_SECONDS}s.")
        keep_alive_task_ref = asyncio.create_task(_self_keep_alive_ping_task_loop())
    else:
        print("[Keep-Alive-Self] Development environment (Uvicorn reloader likely active). Self keep-alive task skipped.")
    
    yield # This is where the application runs

    # --- Code to run on shutdown ---
    if keep_alive_task_ref and not keep_alive_task_ref.done():
        print("[Keep-Alive-Self] Application shutting down. Cancelling keep-alive task...")
        keep_alive_task_ref.cancel()
        try:
            await keep_alive_task_ref # Wait for the task to acknowledge cancellation
        except asyncio.CancelledError:
            print("[Keep-Alive-Self] Keep-alive task successfully cancelled.")
        except Exception as e: # Log other potential errors during task cleanup
            print(f"[Keep-Alive-Self] Error during keep-alive task cancellation/cleanup: {e}")