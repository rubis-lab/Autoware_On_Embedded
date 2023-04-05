#include "../../SerialPort.h"
#include <windows.h>
#include <string>
#include <cstdio>

namespace IL {

	SerialPort::SerialPort()
		: hCom(INVALID_HANDLE_VALUE)
		, timeout(1000)
	{
	}

	SerialPort::~SerialPort()
	{
		close();
	}

	int SerialPort::open(const char* url)
	{
		std::string urlStr(url);
		std::string pathStr;
		std::string baudrate;
		size_t pos = urlStr.find(':');
		if (pos != std::string::npos) {
			baudrate = urlStr.substr(pos + 1);
			pathStr = urlStr.substr(0, pos);
		}
		else {
			baudrate = "115200";
			pathStr = urlStr;
		}
		hCom = CreateFileA(pathStr.c_str(), GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, nullptr);
		if (hCom == INVALID_HANDLE_VALUE) return -static_cast<int>(GetLastError());
		COMMTIMEOUTS timeouts = { MAXDWORD,MAXDWORD,MAXDWORD,1,1 };
		timeouts.ReadTotalTimeoutConstant = timeout;
		if (!SetCommTimeouts(hCom, &timeouts)) return 1;
		COMMPROP commProp = {};
		if (!GetCommProperties(hCom, &commProp)) return 2;
		if (!(commProp.dwSettableBaud & BAUD_USER))	return 3;
		_DCB config;
		config.DCBlength = sizeof(config);
		if (!GetCommState(hCom, &config)) return 4;
		if (!BuildCommDCBA(("baud=" + baudrate + " parity=N data=8 stop=1 to=off xon=off odsr=off octs=off dtr=off rts=off idsr=off").c_str(), &config)) return 5;
		if (!SetCommState(hCom, &config)) return -static_cast<int>(GetLastError());
		return 0;
	}

	bool SerialPort::isOpen()
	{
		return hCom != INVALID_HANDLE_VALUE;
	}

	void SerialPort::close()
	{
		if (isOpen())
		{
			CloseHandle(hCom);
			hCom = INVALID_HANDLE_VALUE;
		}
	}

	int SerialPort::read(char* buf, unsigned int size)
	{
		DWORD bytesRead = 0;
		if (!ReadFile(hCom, buf, size, &bytesRead, nullptr)) return -static_cast<int>(GetLastError());
		return bytesRead;
	}

	int SerialPort::write(char* buf, unsigned int size)
	{
		DWORD bytesWritten = 0;
		if (!WriteFile(hCom, buf, size, reinterpret_cast<LPDWORD>(&bytesWritten), nullptr)) return -static_cast<int>(GetLastError());
		return bytesWritten;
	}
}