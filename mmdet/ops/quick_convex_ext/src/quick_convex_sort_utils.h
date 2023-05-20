#pragma once

#if defined(__CUDACC__) || __HCC__ == 1 || __HIP__ == 1
// Designates functions callable from the host (CPU) and the device (GPU)
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__
#else
#define HOST_DEVICE
#define HOST_DEVICE_INLINE HOST_DEVICE inline
#endif
#define pp(p, i) p + (i) * 2
#define px(p, i) *(pp(p, i) + 0)
#define py(p, i) *(pp(p, i) + 1)

#define p_set(p, n, d, i) px(p, n) = dx(d, i); \
                          py(p, n) = dy(d, i)  \


#define dd(d, i) d + (i) * 3
#define dx(d, i) *(dd(d, i) + 0)
#define dy(d, i) *(dd(d, i) + 1)
#define di(d, i) *(dd(d, i) + 2)

#define d_set(d, i, p, n)  dx(d, i) = px(p, n); \
                           dy(d, i) = py(p, n); \
                           di(d, i) = n \

#define EPS 1e-6

template <typename T>
HOST_DEVICE_INLINE T is_left(const T* p0, const T* p1, const T* p2) {
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
}

// (nbs, 3 * (2n + 1))
template<typename T>
HOST_DEVICE_INLINE void simple_hull_2d(T* p, T* d, int64_t* ind, const int& n, const bool inplace=false) {
    int bot = n - 2, top = bot + 3;
    d_set(d, bot, p, 2);
    d_set(d, top, p, 2);

    if (is_left<T>(pp(p, 0), pp(p, 1), pp(p, 2)) >= EPS) {
        d_set(d, bot+1, p, 0);
        d_set(d, bot+2, p, 1);
    } else {
        d_set(d, bot+1, p, 1);
        d_set(d, bot+2, p, 0);
    }

    for (auto i = 3; i < n; i++) {
        if ((is_left<T>(dd(d, bot), dd(d, bot+1), pp(p, i)) >= EPS) && 
            (is_left<T>(dd(d, top-1), dd(d, top), pp(p, i)) >= EPS)) continue;

        while (is_left<T>(dd(d, bot), dd(d, bot+1), pp(p, i)) < EPS) ++bot;
        bot--;
        d_set(d, bot, p, i);

        while (is_left<T>(dd(d, top-1), dd(d, top), pp(p, i)) < EPS) --top;
        top++;
        d_set(d, top, p, i);
    }

    int tmp = 0;
    for (int h = 0; h < (top-bot); h++) {
        tmp = bot+h+1;
        ind[h] = (int64_t)di(d, tmp);
        if (inplace && ind[h] != h) p_set(p, h, d, tmp);
    }
}
