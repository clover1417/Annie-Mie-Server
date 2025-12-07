#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from websocket_server import AnnieMieServer

def main():
    server = AnnieMieServer()
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
