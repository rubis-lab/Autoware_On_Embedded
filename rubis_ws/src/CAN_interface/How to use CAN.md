# All share
## Precheck
### Get Avaliable can
```
ls /sys/class/net
```
You can see some avaliable networks.
```
can0  can1  can2  can3  can4  docker0  eno1  lo
```

## Install

```
sudo apt-get install can-utils
```

## CAN Set-up
```
sudo ip link set can0 up type can bitrate 500000
```

## Check CAN
```
cansend
```
You can see the example like this.
```
Usage: cansend - simple command line tool to send CAN-frames via CAN_RAW sockets.
Usage: cansend <device> <can_frame>.
<can_frame>:
 <can_id>#{R|data}          for CAN 2.0 frames
 <can_id>##<flags>{data}    for CAN FD frames

<can_id>:
 can have 3 (SFF) or 8 (EFF) hex chars
{data}:
 has 0..8 (0..64 CAN FD) ASCII hex-values (optionally separated by '.')
<flags>:
 a single ASCII Hex value (0 .. F) which defines canfd_frame.flags

Examples:
  5A1#11.2233.44556677.88 / 123#DEADBEEF / 5AA# / 123##1 / 213##311
  1F334455#1122334455667788 / 123#R for remote transmission request.
```

You are ready to send.

You need to do all of these on the client and the host.

## Host side
```
candump can0
```
=> listen

## Client Side
```
cansend can0 5A1#11.2233.44556677.88
```
=> send

## Host side
```
can0  5A1   [8]  11 22 33 44 55 66 77 88
```