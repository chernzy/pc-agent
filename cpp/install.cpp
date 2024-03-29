#include <windows.h>
#include <iostream>

bool InstallService(const char *servicePath, const char *serviceName, const char *displayName)
{
	SC_HANDLE schSCManager = OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);
	if (!schSCManager)
	{
		std::cout << "OpenSCManager failed. Error: " << GetLastError() << std::endl;
		return FALSE;
	}

	SC_HANDLE schService = CreateService(schSCManager, serviceName, displayName, SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS,
										 SERVICE_AUTO_START, SERVICE_ERROR_NORMAL, servicePath, NULL, NULL, NULL, NULL, NULL);

	if (!schService)
	{
		std::cout << "CreateService failed. Error: " << GetLastError() << std::endl;
		CloseServiceHandle(schSCManager);
		return FALSE;
	}

	CloseServiceHandle(schService);
	CloseServiceHandle(schSCManager);

	return TRUE;
}

int main()
{
	const char *servicePath = "D:\\Lucky\\Code\\Python\\aipc-agent\\aipc_agent\\dist\\main.exe";
	const char *serviceName = "AIPCService";
	const char *displayName = "ISS AIPC Service";

	if (InstallService(servicePath, serviceName, displayName))
	{
		std::cout << "Service Install successfully" << std::endl;
	}
	else
	{
		std::cout << "Failed to install service." << std::endl;
	}

	return 0;
}