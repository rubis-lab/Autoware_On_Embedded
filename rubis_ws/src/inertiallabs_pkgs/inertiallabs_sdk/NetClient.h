#pragma once
#include "Transport.h"

struct sockaddr_in;

namespace IL {
    class NetClient :
        public Transport
    {
    public:
        enum Type {
            None,
            Tcp,
            Udp,
        };
        NetClient();
        virtual ~NetClient();
        virtual int open(const char* url);
        virtual bool isOpen();
        virtual void close();
        virtual int read(char* buf, unsigned int size);
        virtual int write(char* buf, unsigned int size);

    private:
        int fd;
        int64_t hCom;
        int timeout;
        Type type;
        sockaddr_in *addr;
    };
}

