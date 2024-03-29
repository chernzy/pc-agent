"""
import win32serviceutil
import win32service
import win32event
import servicemanager
import winerror
import os
import sys
import threading
import subprocess
import time

class AIPCService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AIPCService"
    _svc_display_name_ = "ISS AIPC Service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.is_alive = True
        
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_alive = False
        
    def SvcDoRun(self):
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING, winerror.NO_ERROR, waitHint=120000)
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE, servicemanager.PYS_SERVICE_STARTED, (self._svc_name_, ''))
        servicemanager.LogInfoMsg("Starting main...")
        threading.Thread(target=self.main).start()
        servicemanager.LogInfoMsg("Main started.")

    def main(self):
        exe_path = r'D:\\Lucky\\Code\\Python\\aipc-agent\\aipc_agent\\dist\\main.exe'
        try:
            servicemanager.LogInfoMsg("Starting subprocess...")
            subprocess.Popen([exe_path], shell=True)
            servicemanager.LogInfoMsg("Subprocess started.")
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        except Exception as e:
            servicemanager.LogInfoMsg(str(e))
        
if __name__ == "__main__":
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AIPCService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(AIPCService)
"""

import win32serviceutil
import win32service
import win32event
import winerror
import os
import sys
import threading
import subprocess
import time

class AIPCService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AIPCService"
    _svc_display_name_ = "ISS AIPC Service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.is_alive = True
        
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_alive = False
        
    def SvcDoRun(self):
        self.timeout = 10000
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING, winerror.NO_ERROR, waitHint=120000)
        self.main()
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
    
    def main(self):
        exe_path = r'D:\\Lucky\\Code\\Python\\aipc-agent\\aipc_agent\\dist\\main.exe'
        try:
            subprocess.Popen([exe_path], shell=True)
        except Exception as e:
            # 在这里可以记录日志或者处理异常
            pass
        
if __name__ == "__main__":
    if len(sys.argv) == 1:
        try:
            servicename = AIPCService._svc_name_
            desc = AIPCService._svc_display_name_
            handle = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
            handle_service = win32service.CreateService(handle, servicename, desc, win32service.SERVICE_ALL_ACCESS, win32service.SERVICE_WIN32_OWN_PROCESS, win32service.SERVICE_AUTO_START, win32service.SERVICE_ERROR_NORMAL, r"D:\Lucky\Code\Python\aipc-agent\aipc_agent\dist\main.exe", None, 0, None, None, None)
            win32service.StartService(handle_service, None)
        except Exception as e:
            # 在这里处理创建服务时的异常
            pass
    else:
        win32serviceutil.HandleCommandLine(AIPCService)