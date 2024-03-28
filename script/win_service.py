import win32serviceutil
import win32service
import win32event
import servicemanager
import winerror
import os
import sys

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
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING, winerror.NO_ERROR, waitHint=60000)
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE, servicemanager.PYS_SERVICE_STARTED, (self._svc_name_, ''))
        self.main()
        
    def main(self):
        exe_path = r'D:\\Lucky\\Code\\Python\\aipc-agent\\aipc_agent\\dist\\main.exe'
        os.system(exe_path)
        
if __name__ == "__main__":
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AIPCService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(AIPCService)