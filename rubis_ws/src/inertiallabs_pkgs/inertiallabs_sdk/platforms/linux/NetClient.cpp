#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <cerrno>
#include <string>
#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "../../NetClient.h"

namespace IL {
	NetClient::NetClient()
		: fd(-1)
		, timeout(1000)
		, type(None)
		, addr(nullptr)
	{
	}

	NetClient::~NetClient()
	{
		close();
		if (addr) {
			delete addr;
		}
	}

	int NetClient::open(const char* url)
	{
		if (!addr) {
			addr = new sockaddr_in;
			memset(addr, 0, sizeof(sockaddr_in));
		}
		std::string urlStr(url);
		size_t pos = urlStr.find(':');
		if (pos == std::string::npos) return 1;
		std::string addrStr = urlStr.substr(pos + 1);
		std::string protoStr = urlStr.substr(0, pos);
		pos = addrStr.find(':');
		if (pos == std::string::npos) return 1;
		std::string portStr = addrStr.substr(pos + 1);
		std::string hostStr = addrStr.substr(0, pos);
		struct addrinfo hints = {};
		struct addrinfo* result, * rp;
		hints.ai_family = AF_INET;
		if (protoStr == "tcp") {
			hints.ai_socktype = SOCK_STREAM;
			type = Tcp;
		} else if (protoStr == "udp") {
			hints.ai_socktype = SOCK_DGRAM;
			type = Udp;
		}
		else return 1;
		hints.ai_flags = 0;
		hints.ai_protocol = 0;
		int res = getaddrinfo(hostStr.c_str(), portStr.c_str(), &hints, &result);
		if (res != 0) return -res;
		for (rp = result; rp != NULL; rp = rp->ai_next) {
			fd = socket(rp->ai_family, rp->ai_socktype,	rp->ai_protocol);
			if (fd == -1) continue;

			if (type == Tcp) {
				if (connect(fd, rp->ai_addr, rp->ai_addrlen) != -1)	break;
				::close(fd);
				fd = -1;
			}
		}
		freeaddrinfo(result);

		if (type == Udp) {
			addr->sin_family = AF_INET;
			addr->sin_port = htons(stoi(portStr));
			addr->sin_addr.s_addr = INADDR_ANY;
		}
		if (fd < 0)	return 2;
		return 0;
	}

	bool NetClient::isOpen()
	{
		return fd >= 0;
	}

	void NetClient::close()
	{
		if (isOpen()) {
			::close(fd);
			fd = -1;
		}
	}

	int NetClient::read(char* buf, unsigned int size)
	{
		if (fd < 0) return -1;
		pollfd fds;
		fds.fd = fd;
		fds.events = POLLIN  | POLLHUP | POLLERR;
		fds.revents = 0;
		int result = poll(&fds, 1, timeout);
		if (result < 0) return -errno;
		if (!result) return 0;
		if (type == Udp) {
			socklen_t slen = sizeof(sockaddr_in);
			result = ::recvfrom(fd, buf, size, 0, (sockaddr*)addr, &slen);
		} else {
			result = ::read(fd, buf, size);
		}
		if (result < 0)	return -errno;
		return result;
	}

	int NetClient::write(char* buf, unsigned int size)
	{
		int result = -1;
		if (type == Udp) {
			socklen_t slen = sizeof(sockaddr_in);
			result = ::sendto(fd, buf, size, 0 , (struct sockaddr *) addr, slen);
		} else {
			result = ::write(fd, buf, size);
		}
		if (result < 0)	return -errno;
		return result;
	}
}
