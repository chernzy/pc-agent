```
app.spec.example:
sample config for build EXE file while using langchain / langchain_community. if not use this, it'll throw err `ModuleNotFound: langchain_community.*.*`
```

Usage:
```
# step 1:
# package start server as an exe
cd aipc_agent
pyinstaller start_server.spec --noconfirm
# step 2:
# package win_service as an exe
cd script
pyinstaller win_service.spec --noconfirm
# step 3:
# run the bat
win_service.bat
```