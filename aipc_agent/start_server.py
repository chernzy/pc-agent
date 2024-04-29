import uvicorn
from multiprocessing import cpu_count, freeze_support

def start_server(host="0.0.0.0",
                 port=8080,
                 num_workers=4,
                 loop="asyncio",
                 reload=False):
    uvicorn.run("main:app",
                host=host,
                port=port,
                workers=num_workers,
                loop=loop,
                reload=reload)

if __name__ == "__main__":
    freeze_support()
    # num_workers = int(cpu_count() * 0.5)cd
    # num_workers = int(cpu_count() * 0.2)
    num_workers = 1
    # uvicorn.run(app=app, host="0.0.0.0", port=8000)
    start_server(num_workers=num_workers)
    # start_server()