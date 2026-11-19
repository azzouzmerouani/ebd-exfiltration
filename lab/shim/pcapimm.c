/* LD_PRELOAD shim for softflowd: switch the capture handle to immediate mode
   before activation, so packets are delivered as they arrive. Without it,
   libpcap's TPACKET_V3 ring holds packets until a block fills, which on a
   quiet lab link never happens. */
#define _GNU_SOURCE
#include <pcap/pcap.h>
#include <dlfcn.h>
int pcap_activate(pcap_t *p) {
    static int (*real)(pcap_t *) = 0;
    if (!real) real = (int (*)(pcap_t *))dlsym(RTLD_NEXT, "pcap_activate");
    pcap_set_immediate_mode(p, 1);
    return real(p);
}
