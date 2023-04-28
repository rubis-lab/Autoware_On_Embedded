#include <unistd.h>
#include <poll.h>
#include <termios.h>
#include <fcntl.h>
#include <cerrno>
#include <string>
#include <cstdio>
#include "../../SerialPort.h"

namespace IL {

	SerialPort::SerialPort()
		: fd(-1)
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
		tcflag_t baudrate;
		size_t pos = urlStr.find(':');
		if (pos != std::string::npos) {
			std::string param = urlStr.substr(pos + 1);
			if (param == "4800") baudrate = B4800;
			else if (param == "9600") baudrate = B9600;
			else if (param == "19200") baudrate = B19200;
			else if (param == "38400") baudrate = B38400;
			else if (param == "57600") baudrate = B57600;
			else if (param == "115200") baudrate = B115200;
			else if (param == "230400") baudrate = B230400;
			else if (param == "460800") baudrate = B460800;
			else if (param == "921600") baudrate = B921600;
			else if (param == "2000000") baudrate = B2000000;
			else  return 2;
			pathStr = urlStr.substr(0, pos);
		}
		else {
			baudrate = B115200;
			pathStr = urlStr;
		}
		fd = ::open(pathStr.c_str(), O_NOCTTY | O_RDWR);
		if (fd < 0) return -errno;
		struct termios config;
		if (tcgetattr(fd, &config) < 0)	return 1;
		cfmakeraw(&config);
		config.c_cflag = baudrate & ~(CSIZE | PARENB);
		config.c_cflag |= CS8 | CREAD | CLOCAL;
		config.c_lflag |= IEXTEN;
		if (tcsetattr(fd, TCSANOW, &config) < 0) return 3;
		return 0;
	}

	bool SerialPort::isOpen()
	{
		return fd >= 0;
	}

	void SerialPort::close()
	{
		if (isOpen())
		{
			::close(fd);
			fd = -1;
		}
	}

	int SerialPort::read(char* buf, unsigned int size)
	{
		if (!isatty(fd)) return -1;
		pollfd fds;
		fds.fd = fd;
		fds.events = POLLIN;
		fds.revents = 0;
		int result = poll(&fds, 1, timeout);
		if (result < 0)	return -errno;
		if (!result) return 0;
		result = ::read(fd, buf, size);
		if (result < 0)	return -errno;
		return result;
	}

	int SerialPort::write(char* buf, unsigned int size)
	{
		int result = ::write(fd, buf, size);
		if (result < 0)	return -errno;
		return result;
	}
}