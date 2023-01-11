#ifndef MGT_CORE_H_
#define MGT_CORE_H_

#ifndef PROFAPI
#define MGT_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI

#endif