#!/bin/sh
# softflowd delivers packets only when the libpcap ring block fills unless the
# handle is in immediate mode; the shim sets immediate mode (see shim/pcapimm.c).
export LD_PRELOAD=/opt/lab/shim/pcapimm.so
softflowd -i vh -v 9 -n 127.0.0.1:9995 -p /opt/lab/run/sf1.pid -c /opt/lab/run/sf1.ctl
softflowd -i vh -v 9 -n 127.0.0.1:9996 -p /opt/lab/run/sf2.pid -c /opt/lab/run/sf2.ctl \
  -t maxlife=1800 -t general=15 -t tcp=15 -t udp=15 -t icmp=15 -t tcp.fin=1 -t tcp.rst=1 -t expint=1
